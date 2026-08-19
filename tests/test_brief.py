"""Unit tests for brief generation helpers (no LLM call needed)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from common.llm import extract_json as _extract_json
from generation.brief.schema import Brief, validate_request

_VALID_BRIEF_JSON = {
    "concept": "Wildflowers and a cup of tea",
    "tone": "warm-sincere",
    "headline": "Thanks For Everything, Mum",
    "inside_message": "Happy Birthday, Mum. Thank you for everything. Love always.",
    "visual_prompt": "Watercolour wildflowers, uncluttered band across the top",
    "negative_prompt": "photorealistic, harsh shadows",
    "style_tags": ["watercolour", "floral"],
    "target_price_band": "premium",
}


def test_extract_json_clean() -> None:
    raw = json.dumps(_VALID_BRIEF_JSON)
    parsed = _extract_json(raw)
    assert parsed["headline"] == _VALID_BRIEF_JSON["headline"]


def test_extract_json_with_prose() -> None:
    raw = "Here is the brief:\n" + json.dumps(_VALID_BRIEF_JSON) + "\nLet me know!"
    parsed = _extract_json(raw)
    assert parsed["concept"] == _VALID_BRIEF_JSON["concept"]


def test_extract_json_fenced() -> None:
    raw = "```json\n" + json.dumps(_VALID_BRIEF_JSON) + "\n```"
    parsed = _extract_json(raw)
    assert parsed["target_price_band"] == "premium"


def test_extract_json_malformed_raises() -> None:
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _extract_json("no JSON here at all")


def test_brief_validate_valid() -> None:
    brief = Brief.model_validate(_VALID_BRIEF_JSON)
    assert brief.target_price_band == "premium"
    assert len(brief.style_tags) <= 4


def test_brief_headline_max_length() -> None:
    data = {**_VALID_BRIEF_JSON, "headline": "x" * 91}
    with pytest.raises(ValueError):
        Brief.model_validate(data)


def test_validate_request_bad_tone() -> None:
    with pytest.raises(ValueError):
        validate_request({
            "occasion": "birthday/general",
            "tone": "not-a-valid-tone",
        })


def test_request_without_a_tone_is_valid() -> None:
    """No tone means the brief picks one, which is the evaluation's path.

    Generated cards then inherit the tone mix of the bestsellers they are drawn
    from — the same corpus condition D is sampled from — instead of a rotation
    that had to guess that mix without tone labels to check against.
    """
    req = validate_request({"occasion": "birthday/general"})
    assert req.tone is None


def test_a_pinned_tone_overrides_what_the_model_returned() -> None:
    """The site's tone picker is a promise, not a suggestion."""
    payload = dict(_VALID_BRIEF_JSON, tone="funny-irreverent")
    with (
        patch("common.llm.call_llm", return_value=json.dumps(payload)),
        patch("generation.brief.market_signals.gather", return_value=MagicMock(
            top_tropes=[], coverage_gaps=[], longevity_caution=""
        )),
    ):
        from generation.brief.generate import generate_brief

        brief = generate_brief({"occasion": "birthday/general", "tone": "sentimental"})
    assert brief.tone == "sentimental"

def test_unrecognised_tone_raises_rather_than_defaulting() -> None:
    """An unparseable tone must not silently become `warm-sincere`.

    `warm-sincere` is TONES[0] and was the old fallback. It is also the value
    Section 4.4 reports as never chosen when the generator picks freely, which
    is the evidence that it does not collapse to a default. Writing it on a
    parse failure makes a code fault indistinguishable from a model choice,
    and inverts the reported result.
    """
    from generation.brief.generate import generate_brief

    payload = dict(_VALID_BRIEF_JSON)
    payload["tone"] = "whimsical-nonsense"
    with patch("generation.brief.generate.call_llm",
               return_value=json.dumps(payload)), \
         patch("generation.brief.generate._render_template", return_value="x"):
        with pytest.raises(ValueError, match="not one of"):
            generate_brief({"occasion": "birthday/general"})


def test_a_pinned_tone_still_overrides_the_model() -> None:
    """Pinning bypasses the check: the picker is a promise to the customer."""
    from generation.brief.generate import generate_brief

    payload = dict(_VALID_BRIEF_JSON)
    payload["tone"] = "whimsical-nonsense"
    with patch("generation.brief.generate.call_llm",
               return_value=json.dumps(payload)), \
         patch("generation.brief.generate._render_template", return_value="x"):
        brief = generate_brief(
            {"occasion": "birthday/general", "tone": "warm-humorous"})
    assert brief.tone == "warm-humorous"
