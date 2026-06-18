"""Inside-message generator.

Same LLM as the brief generator, but separate prompt template and
optional distinctiveness-aware regeneration loop:

If the predictor's distinctiveness head scores the message below a threshold
(against the cover features), regenerate up to `max_retries` times. The
predictor scoring is cheap (CLIP text + frozen MLP).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.llm import call_llm, extract_json
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


def generate_message(*, occasion: str, tone: str, concept: str, headline: str) -> InsideMessage:
    prompt = _render(occasion, tone, concept, headline)
    raw = call_llm(prompt, max_tokens=512, temperature=0.8)
    payload = extract_json(raw)
    primary = payload.get("primary")
    if not primary:
        raise ValueError(f"LLM response missing 'primary' key: {list(payload.keys())}")
    return InsideMessage(
        primary=primary,
        alternatives=list(payload.get("alternatives", []))[:3],
    )
