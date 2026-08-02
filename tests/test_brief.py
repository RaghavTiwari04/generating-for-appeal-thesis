"""Unit tests for brief generation helpers (no LLM call needed)."""

from __future__ import annotations

import json

import pytest

from common.llm import extract_json as _extract_json
from generation.brief.schema import Brief, validate_request

_VALID_BRIEF_JSON = {
    "concept": "Wildflowers and a cup of tea",
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
