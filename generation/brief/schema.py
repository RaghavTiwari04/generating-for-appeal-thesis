"""Pydantic schemas for the brief generator's input/output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from common.occasions import ACTIVE_OCCASIONS, RELATIONSHIPS, TONES


class BriefRequest(BaseModel):
    occasion: Literal[ACTIVE_OCCASIONS] = Field(..., description="From the canonical taxonomy")  # type: ignore[valid-type]
    relationship: str | None = Field(None, description="e.g. 'mum', 'partner', 'friend'")
    # Optional. Supplied, it pins the register — which is what a customer
    # choosing a tone on the site expects. Omitted, the brief picks one to suit
    # the concept it draws from BESTSELLER_SUBJECTS, so generated cards inherit
    # the tone distribution of the scraped corpus.
    #
    # That matters for the evaluation: condition D is sampled from that corpus
    # and carries whatever tone its designers chose, and the corpus has no tone
    # labels to match against. Rotating a fixed list across A/B/C only
    # approximated the mix; deriving it from the same bestsellers D is drawn
    # from matches it by construction.
    tone: str | None = Field(None, description="From TONES; the brief chooses when omitted")
    constraints: dict = Field(default_factory=dict)


class Brief(BaseModel):
    concept: str
    # Echoed when the request pinned one, chosen by the brief otherwise. The
    # inside-message generator and the font palette both read it from here, so
    # cover and message share a register either way.
    tone: str
    # Cover lettering, not a sentence. 90 characters allowed briefs like
    # "Here's to you - every wonderfully ridiculous bit of you.", which no image
    # model can letter legibly, so the card fell back to a typographic overlay
    # every time. Commercial fronts carry two or three words; the sentiment goes
    # in `inside_message`.
    headline: str = Field(max_length=30)
    inside_message: str
    visual_prompt: str
    # Inert on the default backend. FLUX is guidance-distilled and takes no
    # negative prompt; only the SDXL path forwards this. Kept because the
    # schema is backend-independent, not because it is currently doing work.
    negative_prompt: str
    style_tags: list[str] = Field(max_length=4)
    target_price_band: Literal["budget", "standard", "premium", "luxury"]


def validate_request(req: dict) -> BriefRequest:
    if req.get("relationship") and req["relationship"] not in RELATIONSHIPS:
        # Soft validation — we accept free-form relationships, just log it
        pass
    # Only checked when supplied: a request with no tone asks the brief to pick
    # one, which is the evaluation's path. A bad tone is still rejected, so the
    # site's picker cannot send a value the prompt does not understand.
    if req.get("tone") is not None and req["tone"] not in TONES:
        raise ValueError(f"Tone must be one of {TONES}")
    return BriefRequest.model_validate(req)
