"""VLM transport and token accounting, shared by every module that calls a judge.

Split out of `scoring.card_scorer` so that module holds the SSR and rubric
logic it is named for. Nothing here knows what a greeting card is: it resolves a
provider, sends one image-plus-text request, retries, and records what the call
cost.

`USAGE` is a process-wide singleton because a labelling run makes on the order
of 25k calls across worker threads and the totals are only meaningful pooled.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)

# Providers downscale above this and bill the reduced size, so pixels beyond it
# are paid for by nobody and seen by nobody.
IMAGE_LONG_EDGE_CAP = 1568

# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
# Providers return exact usage per call; recording it answers two questions
# that would otherwise be arithmetic over assumed image dimensions. Whether
# images already sit under the cap decides if downscaling reclaims waste or
# discards detail the judge sees. Whether cache_write/read stay zero across a
# run is the only signal that a cached prefix fell below the model's minimum
# and cache_control was ignored silently.
#
# Scoring runs in worker threads, so the counters take a lock.
@dataclass
class Usage:
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    embed_calls: int = 0
    embed_tokens: int = 0
    images: int = 0
    long_edge_sum: int = 0
    long_edge_min: int = 0
    long_edge_max: int = 0
    images_over_cap: int = 0
    # Which upstream actually served each call. Gateways load-balance across
    # hosts running different quantisations of the same nominal weights, so an
    # unrecorded route makes a run unreproducible and its noise unattributable.
    served_by: Counter = field(default_factory=Counter)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_call(
        self,
        *,
        inp: int,
        out: int,
        cache_write: int = 0,
        cache_read: int = 0,
        served_by: str | None = None,
    ) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.cache_write_tokens += cache_write
            self.cache_read_tokens += cache_read
            if served_by:
                self.served_by[served_by] += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed_calls += 1

    def record_embedding(self, tokens: int) -> None:
        with self._lock:
            self.embed_calls += 1
            self.embed_tokens += tokens

    def record_image(self, width: int, height: int) -> None:
        edge = max(width, height)
        with self._lock:
            self.images += 1
            self.long_edge_sum += edge
            self.long_edge_max = max(self.long_edge_max, edge)
            self.long_edge_min = min(self.long_edge_min or edge, edge)
            if edge > IMAGE_LONG_EDGE_CAP:
                self.images_over_cap += 1

    def report(self, cards: int = 0, project_to: int = 0) -> str:
        with self._lock:
            lines = [
                "Token usage",
                f"  calls              {self.calls}  ({self.failed_calls} failed)",
                f"  input tokens       {self.input_tokens:,}",
                f"  output tokens      {self.output_tokens:,}",
                f"  cache write        {self.cache_write_tokens:,}",
                f"  cache read         {self.cache_read_tokens:,}",
                f"  embedding tokens   {self.embed_tokens:,}  ({self.embed_calls} calls)",
            ]
            if not (self.cache_write_tokens or self.cache_read_tokens):
                lines.append("  (no caching active on this run)")

            if self.served_by:
                lines += ["", "Served by"]
                lines += [f"  {n:34s} {c} calls" for n, c in self.served_by.most_common()]
                if len(self.served_by) > 1:
                    lines.append(
                        "  more than one upstream served this run; pin with "
                        "--route before treating the scores as reproducible."
                    )

            if self.images:
                over = self.images_over_cap
                lines += [
                    "",
                    "Source images",
                    f"  count              {self.images}",
                    f"  long edge          min {self.long_edge_min}  "
                    f"mean {self.long_edge_sum / self.images:.0f}  max {self.long_edge_max}",
                    f"  above {IMAGE_LONG_EDGE_CAP}px cap    {over} ({over / self.images:.0%})",
                ]
                if not over:
                    lines.append(
                        f"  every image is already under the {IMAGE_LONG_EDGE_CAP}px cap, "
                        "so downscaling\n  would discard detail the model currently sees "
                        "rather than waste."
                    )

            if cards:
                inp, out = self.input_tokens / cards, self.output_tokens / cards
                lines += [
                    "",
                    f"Per card            {inp:,.0f} in / {out:,.0f} out"
                    f"  ({self.calls / cards:.1f} calls)",
                ]
                if project_to:
                    lines.append(
                        f"Projected {project_to} cards  "
                        f"{inp * project_to / 1e6:,.1f}M in / "
                        f"{out * project_to / 1e6:,.1f}M out"
                    )
            return "\n".join(lines)


USAGE = Usage()


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------
# Judges reachable through the OpenAI chat-completions shape, as
# provider -> (base_url, settings attribute holding the key, default model).
# One transport covers hosted OpenAI, Z.ai, OpenRouter, Gemini's own endpoint
# and a local vLLM server, so comparing judges is a flag rather than a new code
# path.
OPENAI_COMPATIBLE: dict[str, tuple[str | None, str, str | None]] = {
    "openai": (None, "openai_api_key", "gpt-4o"),
    "glm": ("https://api.z.ai/api/paas/v4", "glm_api_key", "glm-4.6v"),
    "openrouter": ("https://openrouter.ai/api/v1", "openrouter_api_key", None),
    # Google's own OpenAI-compatible endpoint. Direct rather than via a gateway,
    # so there is no upstream to pin and no gateway margin.
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini_api_key",
        "gemini-3.5-flash-lite",
    ),
}

# Clients are cheap to reuse and not free to build; a full labelling run makes
# ~25k calls, and constructing a client per call also discards connection reuse.
_clients: dict[tuple[str, str | None], Any] = {}
_client_lock = threading.Lock()


def openai_client(api_key: str, base_url: str | None = None) -> Any:
    with _client_lock:
        client = _clients.get((api_key, base_url))
        if client is None:
            from openai import OpenAI

            client = _clients[(api_key, base_url)] = OpenAI(
                api_key=api_key, base_url=base_url
            )
        return client


def _anthropic_client(api_key: str | None) -> Any:
    with _client_lock:
        client = _clients.get((api_key or "", "anthropic"))
        if client is None:
            import anthropic

            client = _clients[(api_key or "", "anthropic")] = anthropic.Anthropic(
                api_key=api_key
            )
        return client

# ---------------------------------------------------------------------------
# VLM transport
# ---------------------------------------------------------------------------
def image_to_b64(image: Image.Image) -> str:
    USAGE.record_image(*image.size)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def openrouter_route(spec: str | None) -> dict | None:
    """Build a gateway routing constraint from a CLI spec.

    A bare name pins one upstream and forbids fallback, which is what a
    reproducible run needs. A JSON object is passed through untouched so the
    gateway's full routing vocabulary stays reachable without this function
    having to track it.
    """
    if not spec or not spec.strip():
        return None
    spec = spec.strip()
    return json.loads(spec) if spec.startswith("{") else {
        "order": [spec],
        "allow_fallbacks": False,
    }


def _resolve(provider: str, model: str | None) -> tuple[str | None, str | None, str]:
    """(base_url, api_key, model) for a provider, or raise.

    Resolved before any request: a missing key or unknown provider is a
    configuration error, and discovering it inside the retry loop would burn
    the retries and return "", which the caller records as a card that failed
    to score rather than a misconfiguration.
    """
    if provider == "anthropic":
        return None, settings.anthropic_api_key, model or settings.llm_model
    if provider not in OPENAI_COMPATIBLE:
        raise ValueError(
            f"unknown provider {provider!r}; expected 'anthropic' or one of "
            f"{sorted(OPENAI_COMPATIBLE)}"
        )
    base_url, key_attr, default_model = OPENAI_COMPATIBLE[provider]
    api_key = getattr(settings, key_attr)
    if not api_key:
        raise RuntimeError(f"{key_attr.upper()} is required for --provider {provider}")
    chosen = model or default_model
    if chosen is None:
        raise RuntimeError(f"--provider {provider} has no default model; pass --model")
    return base_url, api_key, chosen


def _call_openai_compatible(
    image_b64: str,
    system_prompt: str,
    user_text: str,
    *,
    base_url: str | None,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    route: dict | None,
) -> str:
    resp = openai_client(api_key, base_url).chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        **({"extra_body": {"provider": route}} if route else {}),
    )
    if resp.usage is not None:
        details = getattr(resp.usage, "prompt_tokens_details", None)
        USAGE.record_call(
            inp=resp.usage.prompt_tokens or 0,
            out=resp.usage.completion_tokens or 0,
            cache_read=getattr(details, "cached_tokens", 0) or 0,
            served_by=(resp.model_extra or {}).get("provider"),
        )
    if not resp.choices:
        return ""
    message = resp.choices[0].message
    # Some reasoning models leave `content` empty and put the answer in a vendor
    # field, which reads as a failed call rather than a short reply.
    text = message.content or ""
    if text.strip():
        return text
    extra = message.model_extra or {}
    return extra.get("reasoning_content") or extra.get("reasoning") or ""


# Newer Anthropic models reject `temperature` outright rather than ignoring it,
# so a run against one fails every call with a 400 and scores nothing. Recorded
# per model, and the first occurrence is logged, because dropping the parameter
# is a real deviation: SSR specifies an elicitation temperature of 0.5 and the
# rubric judge is meant to be deterministic at 0. A judge run without it is
# still a judge, but it is not the same instrument, and any writeup using it has
# to say so.
_NO_TEMPERATURE: set[str] = set()


def _call_anthropic(
    image_b64: str,
    system_prompt: str,
    user_text: str,
    *,
    api_key: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    kwargs = {} if model in _NO_TEMPERATURE else {"temperature": temperature}
    try:
        return _anthropic_message(
            image_b64, system_prompt, user_text,
            api_key=api_key, model=model, max_tokens=max_tokens, **kwargs,
        )
    except Exception as e:
        if "temperature" not in str(e).lower() or model in _NO_TEMPERATURE:
            raise
        _NO_TEMPERATURE.add(model)
        log.warning(
            f"{model} rejects `temperature`; retrying without it and for the rest "
            f"of this run. Sampling is at the model default, not the requested "
            f"{temperature}."
        )
        return _anthropic_message(
            image_b64, system_prompt, user_text,
            api_key=api_key, model=model, max_tokens=max_tokens,
        )


def _anthropic_message(
    image_b64: str,
    system_prompt: str,
    user_text: str,
    *,
    api_key: str | None,
    model: str,
    max_tokens: int,
    **extra,
) -> str:
    msg = _anthropic_client(api_key).messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        **extra,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )
    USAGE.record_call(
        inp=getattr(msg.usage, "input_tokens", 0) or 0,
        out=getattr(msg.usage, "output_tokens", 0) or 0,
        cache_write=getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        cache_read=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def call_vlm(
    image_b64: str,
    system_prompt: str,
    user_text: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 5,
    route: dict | None = None,
) -> str:
    """One VLM call returning raw text, with exponential backoff. "" on failure."""
    base_url, api_key, chosen = _resolve(provider, model)

    for attempt in range(retries):
        try:
            if provider == "anthropic":
                return _call_anthropic(
                    image_b64,
                    system_prompt,
                    user_text,
                    api_key=api_key,
                    model=chosen,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return _call_openai_compatible(
                image_b64,
                system_prompt,
                user_text,
                base_url=base_url,
                api_key=api_key,
                model=chosen,
                temperature=temperature,
                max_tokens=max_tokens,
                route=route,
            )
        except Exception as e:
            USAGE.record_failure()
            log.warning(f"{provider} call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(min(2**attempt, 30))
    return ""
