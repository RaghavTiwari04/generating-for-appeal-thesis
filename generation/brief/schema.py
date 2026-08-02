"""Pydantic schemas for the brief generator's input/output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from common.occasions import ACTIVE_OCCASIONS, RELATIONSHIPS, TONES


class BriefRequest(BaseModel):
    occasion: Literal[ACTIVE_OCCASIONS] = Field(..., description="From the canonical taxonomy")  # type: ignore[valid-type]
    relationship: str | None = Field(None, description="e.g. 'mum', 'partner', 'friend'")
    tone: str = Field(..., description="From TONES")
    constraints: dict = Field(default_factory=dict)


class Brief(BaseModel):
    concept: str
    # Cover lettering, not a sentence. 90 characters allowed briefs like
    # "Here's to you - every wonderfully ridiculous bit of you.", which no image
    # model can letter legibly, so the card fell back to a typographic overlay
    # every time. Commercial fronts carry two or three words; the sentiment goes
    # in `inside_message`.
    headline: str = Field(max_length=30)
    inside_message: str
    visual_prompt: str
    negative_prompt: str
    style_tags: list[str] = Field(max_length=4)
    target_price_band: Literal["budget", "standard", "premium", "luxury"]


def validate_request(req: dict) -> BriefRequest:
    if req.get("relationship") and req["relationship"] not in RELATIONSHIPS:
        # Soft validation — we accept free-form relationships, just log it
        pass
    if req.get("tone") not in TONES:
        raise ValueError(f"Tone must be one of {TONES}")
    return BriefRequest.model_validate(req)
