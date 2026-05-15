"""Inside-message generator.

Same LLM as the brief generator, but separate prompt template and
optional distinctiveness-aware regeneration loop:

If the predictor's distinctiveness head scores the message below a threshold
(against the cover features), regenerate up to `max_retries` times. The
predictor scoring is cheap (CLIP text + frozen MLP).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "message_v1.txt"
PROMPT_VERSION = "message_v1"


COMMON_CLICHES = [
    "wishing you all the best",
    "have a great one",
    "hope this card finds you well",
    "thinking of you in this difficult time",
    "another year older, another year wiser",
    "they're in a better place",
]


@dataclass
class InsideMessage:
    primary: str
    alternatives: list[str]


def _render(occasion: str, tone: str, concept: str, headline: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    cliches = "\n".join(f"- {c}" for c in COMMON_CLICHES)
    return (
        template.replace("{{occasion}}", occasion)
        .replace("{{tone}}", tone)
        .replace("{{concept}}", concept)
        .replace("{{headline}}", headline)
        .replace("{{cliches}}", cliches)
    )


def _call_llm(prompt: str) -> str:
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.llm_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""
    raise ValueError(f"Unknown LLM provider: {provider}")


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in LLM response: {text[:200]}")
    return json.loads(text[start : end + 1])


def generate_message(*, occasion: str, tone: str, concept: str, headline: str) -> InsideMessage:
    prompt = _render(occasion, tone, concept, headline)
    raw = _call_llm(prompt)
    payload = _extract_json(raw)
    return InsideMessage(
        primary=payload["primary"],
        alternatives=list(payload.get("alternatives", []))[:3],
    )
