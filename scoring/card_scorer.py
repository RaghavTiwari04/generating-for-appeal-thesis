"""LLM scoring for greeting cards, on five dimensions.

One implementation, used everywhere a card is scored: labelling scraped
listings, reranking generated candidates, and the system evaluation. Each of
those previously had its own prompts, API clients and parsers, so training
targets and evaluation measured the same constructs by different instruments.

Two methods, each following its source:

  purchase_intent — Semantic Similarity Rating (Maier et al. 2025,
      arXiv:2510.08338; github.com/pymc-labs/semantic-similarity-rating).
      Synthetic consumers answer in free text; responses are embedded and
      compared against reference anchor statements, giving a PMF over a
      5-point Likert scale. Direct numeric self-reports are what SSR exists
      to avoid, so the personas are told not to give ratings.

      SSR is a method for reproducing a population's *distribution* of survey
      responses. Collapsing the persona samples to one number is an adaptation,
      so the paper's KS>0.85 validation does not carry over to this use. Averaging
      the per-persona expectations equals the expectation of the averaged PMF,
      which is what the reference aggregates, so the point estimate itself is
      consistent with it.

  occasion_fit, aesthetic, emotional_resonance, distinctiveness —
      rubric-guided judge (Zheng et al. 2023, MT-Bench;
      github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). Explanation
      first, then a 1-10 score in "[[rating]]" form, at temperature 0.

Cost: one card is 6 SSR calls (3 consumer profiles x 2 samples each, per the
paper's n=2) plus 4 rubric calls, so 10 VLM calls, plus one short embedding
call per persona reply. The anchor statements are embedded once per process.
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)

DIMS = (
    "purchase_intent",
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
)
RUBRIC_DIMS = DIMS[1:]

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


def _openai_client(api_key: str, base_url: str | None = None) -> Any:
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
# SSR — purchase intent
# ---------------------------------------------------------------------------
# Six anchor sets, averaged per response, following the paper: "we use m=6
# sets […] all similar but not identical, and designed to capture the different
# ways a consumer may express their purchase intent", with pmfs "averaged over
# six different statement sets for every response".
#
# pymc-labs' API takes one reference_set_id per call, so the averaging lives
# here rather than in the library — but the behaviour is the reference's, not
# an extension of ours.
SSR_REFERENCE_SETS: tuple[tuple[str, ...], ...] = (
    (
        "I would never buy this card",
        "I probably would not buy this card",
        "I am unsure whether I would buy this card",
        "I would likely buy this card",
        "I would definitely buy this card",
    ),
    (
        "This card has no commercial appeal",
        "This card has weak commercial appeal",
        "This card has moderate commercial appeal",
        "This card has strong commercial appeal",
        "This card has excellent commercial appeal",
    ),
    (
        "No one would want to purchase this",
        "Few people would consider purchasing this",
        "Some people might consider purchasing this",
        "Many people would consider purchasing this",
        "Most people would want to purchase this",
    ),
    (
        "I would leave this on the shelf",
        "I would be reluctant to pick this one",
        "I could go either way on this one",
        "I would be inclined to pick this one",
        "I would take this one straight to the till",
    ),
    (
        "This is not worth the money",
        "This is poor value for the money",
        "This is fair value for the money",
        "This is good value for the money",
        "This is excellent value for the money",
    ),
    (
        "I would not send this card to anyone",
        "I can hardly think of anyone I would send this to",
        "I might send this to one or two people",
        "I can think of several people I would send this to",
        "I would send this to almost anyone",
    ),
)

# Synthetic consumers. The reference does not prescribe personas, so the
# spread of age, income and region is our choice.
CONSUMER_PROFILES: tuple[dict, ...] = (
    {"age": 28, "gender": "female", "income": "moderate", "region": "urban UK"},
    {"age": 45, "gender": "male", "income": "above average", "region": "suburban UK"},
    {"age": 62, "gender": "female", "income": "comfortable", "region": "rural UK"},
)

# Elicitation settings from the paper: T_LLM = 0.5, n = 2 samples per prompt,
# "which we found was sufficient to obtain stable results".
#
# The paper also sets top_p = 0.9. We cannot: the Anthropic API rejects a request
# carrying both temperature and top_p, and temperature is the parameter the
# paper's own sensitivity analysis varies. Deviation is noted here rather than
# hidden.
SSR_ELICITATION_TEMPERATURE = 0.5
SSR_SAMPLES_PER_PERSONA = 2

# The rubric judge must be deterministic (MT-Bench scores at temperature 0).
JUDGE_TEMPERATURE = 0.0

# The embedder is the instrument SSR measures with — it is what turns a free-text
# reply into a position between the anchors — so it follows the paper rather than
# being a free choice: "OpenAI's model 'text-embedding-3-small'".
_EMBED_MODEL = "text-embedding-3-small"

_anchors: np.ndarray | None = None
_anchor_lock = threading.Lock()


def _embed(texts: list[str], retries: int = 4) -> np.ndarray:
    """Embed texts as unit row vectors. Raises rather than degrading silently.

    A failed embedding cannot be defaulted: every SSR score is a cosine against
    these vectors, so a placeholder would be a fabricated score rather than a
    missing one.
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is required for SSR embeddings ({_EMBED_MODEL})"
        )
    client = _openai_client(settings.openai_api_key)

    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
            USAGE.record_embedding(getattr(resp.usage, "total_tokens", 0) or 0)
            vecs = np.asarray([d.embedding for d in resp.data], dtype=float)
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        except Exception as e:
            last = e
            log.warning(f"Embedding failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Embedding failed after {retries} attempts") from last


def anchor_embeddings() -> np.ndarray:
    """(sets, 5, dim) unit vectors for the reference statements.

    Embedded once per process, in a single call, under a lock — concurrent
    scoring threads would otherwise each re-embed the same statements.
    """
    global _anchors
    with _anchor_lock:
        if _anchors is None:
            flat = [s for refs in SSR_REFERENCE_SETS for s in refs]
            vecs = _embed(flat)
            _anchors = vecs.reshape(len(SSR_REFERENCE_SETS), -1, vecs.shape[1])
        return _anchors


def similarities_to_pmf(sims: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """Similarities to a PMF by min-subtraction and normalisation.

    Reference divides by (cos_sum - n*cos_min + epsilon); adding epsilon to the
    minimum element and dividing by the sum is the same quantity.
    """
    s = np.asarray(sims, dtype=float)
    s = s - s.min()
    if epsilon > 0:
        s[int(s.argmin())] += epsilon
    total = s.sum()
    return s / total if total else np.full(len(s), 1 / len(s))


def scale_pmf(pmf: np.ndarray, temperature: float) -> np.ndarray:
    """Raise a PMF to 1/temperature and renormalise; one-hot at 0."""
    if temperature == 1.0:
        return pmf
    if temperature == 0.0:
        out = np.zeros_like(pmf)
        out[int(pmf.argmax())] = 1.0
        return out
    scaled = pmf ** (1.0 / temperature)
    return scaled / scaled.sum()


def ssr_score(response: str, temperature: float = 1.0, epsilon: float = 0.0) -> dict:
    """Map one free-text response to a Likert PMF and a 0-1 score."""
    emb = _embed([response])[0]
    # Cosine rescaled from [-1,1] to [0,1], as the reference does. Vectors are
    # unit-norm, so the dot product is the cosine.
    sims = (1 + anchor_embeddings() @ emb) / 2
    pmfs = np.stack([similarities_to_pmf(s, epsilon) for s in sims])

    # Temperature is applied after averaging the anchor sets. The reference
    # scales a single set's PMF; with one set the two are identical.
    pmf = scale_pmf(pmfs.mean(axis=0), temperature)
    expected = float(pmf @ np.arange(1, 6))
    return {"pmf": pmf.tolist(), "likert": expected, "score": (expected - 1) / 4}


def persona_prompt(profile: dict) -> str:
    return (
        f"You are a {profile['age']}-year-old {profile['gender']} living in "
        f"{profile['region']} with {profile['income']} income. You regularly buy "
        f"greeting cards for friends and family. Reply briefly and naturally to any "
        f"questions posed to you. Do not use numerical ratings or scales — just "
        f"share your honest reaction in your own words."
    )


# ---------------------------------------------------------------------------
# Rubric judge — the four quality dimensions
# ---------------------------------------------------------------------------
# MT-Bench's format instruction, kept in one place so every dimension asks for
# the score identically and the extraction regex stays valid.
_RUBRIC_TEMPLATE = """\
You are an impartial expert judge evaluating a greeting card. You will see one \
card image plus metadata.

Evaluate {name}: {question}

Scoring rubric (1-10):
{rubric}

First provide a brief explanation of your assessment (2-3 sentences). Then \
output your score strictly in this format: [[rating]]
Example: Rating: [[7]]"""

_RUBRICS: dict[str, tuple[str, str, str]] = {
    "occasion_fit": (
        "OCCASION FIT",
        "How well does the card's imagery, text, colour palette, and overall theme "
        "match the stated occasion?",
        """\
  1-2: Completely wrong occasion. A sympathy card that looks celebratory, or a birthday card with no birthday-relevant imagery whatsoever.
  3-4: Weak connection. Generic design that could apply to any occasion. The occasion is not clearly communicated by the visual or text.
  5-6: Acceptable match. The card communicates the correct occasion but through generic or overused means. A reasonable but uninspired choice.
  7-8: Strong match. The imagery, text, and design clearly and effectively communicate the occasion. The recipient would immediately understand what the card is for and feel it was well-chosen.
  9-10: Perfect match. The card captures the spirit of the occasion in a way that feels both specific and thoughtful. Every element reinforces the occasion naturally.""",
    ),
    "aesthetic": (
        "AESTHETIC QUALITY",
        "Consider composition, colour harmony, typography, illustration quality, "
        "whitespace usage, and overall professionalism.",
        """\
  1-2: Unprofessional. Poor resolution, clashing colours, illegible text, clip-art quality, broken layout. Would not pass for a commercial product.
  3-4: Below average. Identifiable quality issues — awkward composition, jarring colour palette, amateurish illustration, or typography problems.
  5-6: Average commercial quality. Competent design that meets basic standards but lacks polish or distinction. Acceptable for a budget range.
  7-8: High quality. Well-composed, harmonious colours, clean typography, polished illustration. Looks professional and gift-worthy. Standard to premium card shop quality.
  9-10: Exceptional. Museum-quality illustration, masterful composition, stunning colour work. Could anchor a premium card collection or win design awards.""",
    ),
    "emotional_resonance": (
        "EMOTIONAL RESONANCE",
        "Would this card make the recipient feel something? Consider warmth, humour, "
        "sentiment, joy, tenderness, or any genuine emotional response it evokes.",
        """\
  1-2: No emotional impact. Cold, corporate, or confusing. The recipient would feel nothing or feel the sender put in no effort.
  3-4: Minimal emotional impact. Generic sentiment without genuine feeling. The recipient might glance at it and set it aside.
  5-6: Moderate emotional impact. The card communicates care adequately. The recipient would appreciate the gesture but not be particularly moved.
  7-8: Strong emotional impact. The card evokes genuine warmth, laughter, or sentiment. The recipient would likely keep it displayed for a while or share it with someone.
  9-10: Deeply moving. The card captures something truly heartfelt or genuinely funny. The recipient might tear up, laugh out loud, or keep it for years.""",
    ),
    "distinctiveness": (
        "DISTINCTIVENESS",
        "How original and unique is this card compared to typical mass-market "
        "greeting cards? Does it have a distinctive artistic voice or creative concept?",
        """\
  1-2: Completely generic. Indistinguishable from thousands of stock greeting cards. Cookie-cutter design with no creative thought.
  3-4: Mostly generic. One or two mildly unusual elements but the overall concept and execution are familiar and predictable.
  5-6: Somewhat distinctive. Has identifiable creative choices that separate it from pure stock designs, but still within well-trodden territory.
  7-8: Distinctive. Clear artistic voice or creative concept that would catch a shopper's eye. Memorable design that stands apart from typical offerings.
  9-10: Highly original. Boldly creative concept or artistic approach unlike typical greeting cards. Would stand out immediately in any card shop.""",
    ),
}

# No score-distribution instruction. An earlier version told the judge that
# "most cards should score 4-7", which compresses scores toward the middle —
# and the headline result is a TOST equivalence test at delta=0.02, so anything
# that shrinks between-condition differences makes equivalence easier to
# demonstrate whether or not it holds. MT-Bench uses a plain assistant system
# prompt; the domain framing is kept, the anchoring is not.
JUDGE_SYSTEM_PROMPT = (
    "You are an expert greeting card designer and market analyst serving as an "
    "impartial judge. Be objective. Do not let verbosity or elaborateness bias "
    "your judgment."
)

# FastChat's patterns: "[[7]]" preferred, bare "[7]" as fallback.
_SCORE_RE = re.compile(r"\[\[(\d+\.?\d*)\]\]")
_SCORE_RE_FALLBACK = re.compile(r"\[(\d+\.?\d*)\]")


def rubric_prompt(dim: str) -> str:
    name, question, rubric = _RUBRICS[dim]
    return _RUBRIC_TEMPLATE.format(name=name, question=question, rubric=rubric)


def extract_rating(text: str) -> float | None:
    """Pull the 1-10 rating out of a judge response, or None if absent.

    Out-of-range values are rejected rather than clamped: the bare-bracket
    fallback also matches things like a year in "[2024]", and clamping would
    turn that into a confident 10.
    """
    m = _SCORE_RE.search(text or "") or _SCORE_RE_FALLBACK.search(text or "")
    if not m:
        return None
    value = float(m.group(1))
    return value if 1.0 <= value <= 10.0 else None


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
    resp = _openai_client(api_key, base_url).chat.completions.create(
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
    msg = _anthropic_client(api_key).messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
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


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
@dataclass
class CardScorer:
    """Score a card on all five dimensions."""

    provider: str = "anthropic"
    model: str | None = None
    profiles: tuple[dict, ...] = CONSUMER_PROFILES
    route: dict | None = None
    samples_per_persona: int = SSR_SAMPLES_PER_PERSONA
    # The paper restricts its study to epsilon = 0 and T = 1; these match.
    ssr_temperature: float = 1.0
    ssr_epsilon: float = 0.0

    def _call(self, b64: str, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        return call_vlm(
            b64,
            system,
            user,
            provider=self.provider,
            model=self.model,
            route=self.route,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _purchase_intent(self, b64: str, occasion: str, headline: str) -> dict:
        """SSR over every persona sample that produced usable text."""
        context = f"a greeting card for {occasion}" if occasion else "a greeting card"
        if headline:
            context += f' with the headline "{headline}"'
        question = (
            f"You see {context} in a shop. Would you buy it? "
            f"How appealing is it as a purchase?"
        )

        pmfs, scores, replies = [], [], []
        for profile in self.profiles:
            for _ in range(self.samples_per_persona):
                reply = self._call(
                    b64,
                    persona_prompt(profile),
                    question,
                    temperature=SSR_ELICITATION_TEMPERATURE,
                    max_tokens=200,
                )
                # A failed call must not be stood in for: SSR would score the
                # substitute, turning silence into a confident mid-scale number.
                if not reply.strip():
                    continue
                r = ssr_score(reply, self.ssr_temperature, self.ssr_epsilon)
                pmfs.append(r["pmf"])
                scores.append(r["score"])
                replies.append(reply)

        if not scores:
            log.warning("No usable SSR replies; omitting purchase_intent")
            return {"ssr_responses": []}
        return {
            "purchase_intent": float(np.mean(scores)),
            "purchase_intent_std": float(np.std(scores)),
            "purchase_intent_pmf": np.mean(pmfs, axis=0).tolist(),
            "purchase_intent_n": len(scores),
            "ssr_responses": replies,
        }

    def score(
        self,
        image: Image.Image,
        *,
        occasion: str = "",
        headline: str = "",
        inside_message: str = "",
    ) -> dict:
        """Return {dimension: 0-1 score} plus SSR detail and explanations.

        Dimensions whose call fails are omitted rather than defaulted, so a
        failure cannot masquerade as a low score.
        """
        b64 = image_to_b64(image)
        out: dict = self._purchase_intent(b64, occasion, headline)

        meta = (
            f"Occasion: {occasion}\nHeadline: {headline}\n"
            f"Inside message: {inside_message}\n\n"
        )
        explanations: dict[str, str] = {}
        for dim in RUBRIC_DIMS:
            reply = self._call(
                b64,
                JUDGE_SYSTEM_PROMPT,
                meta + rubric_prompt(dim),
                temperature=JUDGE_TEMPERATURE,
                max_tokens=1024,
            )
            rating = extract_rating(reply)
            if rating is None:
                log.warning(f"No rating extracted for {dim}; omitting")
                continue
            out[dim] = (rating - 1) / 9
            out[f"{dim}_raw_1_10"] = rating
            explanations[dim] = reply

        out["explanations"] = explanations
        return out
