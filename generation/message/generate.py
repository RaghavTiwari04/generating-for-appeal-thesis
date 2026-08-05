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


# A primary message of up to four sentences plus a sign-off, and up to three
# alternatives of the same length, is roughly 240 words before JSON overhead.
# 512 tokens left no margin: two evaluation runs each lost a card when the model
# wrote at the wordier end and the response was cut mid-string, which is invalid
# JSON and takes the whole card down. The cost of the headroom is nothing —
# billing is on tokens produced, not on the ceiling.
MESSAGE_MAX_TOKENS = 1536


def generate_message(*, occasion: str, tone: str, concept: str, headline: str) -> InsideMessage:
    prompt = _render(occasion, tone, concept, headline)
    raw = call_llm(prompt, max_tokens=MESSAGE_MAX_TOKENS, temperature=0.8)
    payload = extract_json(raw)
    primary = payload.get("primary")
    if not primary:
        raise ValueError(f"LLM response missing 'primary' key: {list(payload.keys())}")
    return InsideMessage(
        primary=primary,
        alternatives=list(payload.get("alternatives", []))[:3],
    )
