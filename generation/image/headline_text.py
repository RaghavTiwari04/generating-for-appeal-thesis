"""Render the headline into the artwork.

Commercial cards integrate their lettering into the design. The original
pipeline instead reserved a fixed region (top-left, 55% x 25%), inpainted it to
blank with a second Flux Fill pass, and drew text into the emptiness — so every
card shared one layout and the type never belonged to the artwork.

Flux letters these cards well, so it does it directly: the greeting leads the
prompt and the model draws it as part of the design.

An OCR check used to gate that, falling back to the overlay when it could not
read the headline back. It is off by default now. Tesseract is built for
documents and cannot read brush-script lettering, which is what card fronts use
— it returned an empty read on four covers whose headlines were large, correct
and plainly legible. So the gate sent every card to the overlay and reported
that the model had lettered none of them.

The OCR score is still computed and recorded, as a lower bound on legibility.
HEADLINE_VERIFY=1 restores the gate and the overlay fallback.

Used by conditions A, B and C alike: card format must not differ between
conditions, or it confounds the comparison.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from common.logging import get_logger
from generation.image.headline_mask import LayoutMaskSpec, build_headline_mask
from generation.layout.compose import compose

log = get_logger(__name__)

# Share of headline words that must survive OCR for the render to count.
MATCH_THRESHOLD = 0.8

# Whether that check gates the result. Off: tesseract is built for documents and
# cannot read the brush-script lettering commercial cards use, so it returned an
# empty read on four covers whose headline was large, correctly spelled and
# plainly legible. Every card was then replaced by a typographic overlay in a
# fixed reserved region — the look integrated lettering exists to avoid — and
# text_in_image reported 0% for a model that had rendered all four correctly.
#
# HEADLINE_VERIFY=1 restores the gate, for a run where garbled text matters more
# than the overlay's uniform layout.
VERIFY_HEADLINE = os.environ.get("HEADLINE_VERIFY", "0") == "1"

# How lettering is described, to the image model at generation time and to the
# LoRA in its training captions. One constant because the two must agree: the
# LoRA is conditioned on the words that describe its training images, so
# wording used at generation but never during training asks for something the
# model was never shown.
LETTERING_STYLE = "hand-lettering integrated into the artwork"


@dataclass
class RenderedCard:
    image: Image.Image
    text_in_image: bool          # True when the model rendered the headline
    match_score: float


def augment_prompt(visual_prompt: str, headline: str) -> str:
    """Ask for the greeting as designed lettering rather than an overlay.

    The lettering leads. Appended after the scene description it sat at the end
    of ninety words of composition, palette and lighting detail, and the model
    drew the artwork and little or no text — OCR read back 0.00 on three of
    four cards and 0.33 on the fourth. Diffusion text rendering is far more
    reliable when the words are the first thing asked for.
    """
    return (
        f'A greeting card with the words "{headline}" in large, clearly legible, '
        f"correctly spelled {LETTERING_STYLE}. {visual_prompt}"
    )


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def match_score(ocr_text: str, headline: str) -> float:
    """Share of headline words present in the OCR output.

    Order-insensitive and forgiving of surrounding text: diffusion often adds
    flourishes, and the card may carry incidental words. What matters is
    whether the intended greeting is legibly there.
    """
    want = _words(headline)
    if not want:
        return 0.0
    got = set(_words(ocr_text))
    return sum(w in got for w in want) / len(want)


def _ocr_preview(image: Image.Image, limit: int = 80) -> str:
    """What OCR actually read, for the failure log. Empty when unavailable."""
    try:
        from data.features.ocr import ocr_image

        return " ".join(ocr_image(image).text.split())[:limit]
    except Exception:
        return ""


def verify_headline(image: Image.Image, headline: str) -> float:
    """Score how much of the headline OCR can read back from the image."""
    try:
        from data.features.ocr import ocr_image

        return match_score(ocr_image(image).text, headline)
    except Exception as e:
        # Missing tesseract must not fail generation; fall back to the overlay.
        log.warning(f"OCR verification unavailable ({e}); using overlay")
        return 0.0


def _save_rejected(cover: Image.Image, headline: str, score: float) -> None:
    """Keep the cover that failed verification, when REJECTED_DIR is set.

    Otherwise this image is discarded and only its OCR score survives, which
    cannot distinguish a card the model left blank from one carrying lettering
    tesseract could not parse. The overlay pass then replaces it, so the
    finished card shows overlay text either way and answers nothing.

    Off unless the environment asks for it: a full run generates hundreds of
    these and they are only wanted while diagnosing.
    """
    dest = os.environ.get("REJECTED_DIR")
    if not dest:
        return
    try:
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", headline.lower()).strip("-")[:40]
        digest = hashlib.sha256(cover.tobytes()).hexdigest()[:8]
        cover.save(out / f"{slug}_{score:.2f}_{digest}.png")
    except Exception as e:
        log.debug(f"Could not save rejected cover: {e}")


def render_card(
    runner,
    *,
    visual_prompt: str,
    headline: str,
    tone: str,
    style_tags: list[str],
    occasion: str | None,
    seed: int | None,
    negative_prompt: str = "",
    threshold: float = MATCH_THRESHOLD,
    verify: bool = VERIFY_HEADLINE,
) -> RenderedCard:
    """Generate one card, lettering the headline into the artwork.

    `verify` gates that on OCR reading the headline back, falling back to a
    typographic overlay when it cannot. Off by default: tesseract cannot read
    the brush-script lettering these cards are made of, so it scored 0.00 on
    covers whose headline was plainly legible and correct, and every card was
    replaced by an overlay it did not need.

    The OCR score is still recorded, as a lower bound on legibility rather than
    a control. Leaving it as a gate meant shipping the reserved-region layout
    for every card, which is the look the integrated lettering exists to avoid.
    """
    # Attempt 1: no mask, so no Fill pass — the model letters the card itself.
    images = runner.generate(
        prompt=augment_prompt(visual_prompt, headline),
        negative_prompt=negative_prompt,
        occasion=occasion,
        seed=seed,
        n=1,
        mask_image=None,
    )
    cover = images[0]
    score = verify_headline(cover, headline)
    # `score` gates nothing unless `verify` is set, which it is not by default
    # (HEADLINE_VERIFY). Tesseract's page segmentation discards brush-script
    # lettering before recognition, so the score reads zero on correctly
    # lettered cards and gating on it substituted the overlay for good output.
    # It is carried on RenderedCard.match_score as a recorded lower bound, and
    # the line below is its only consumer. Do not add a second: any new
    # pass/fail read of this value reproduces the defect.
    if not verify:
        log.info(f"Headline lettered by the model (ocr match={score:.2f}, unverified)")
        return RenderedCard(image=cover, text_in_image=True, match_score=score)
    if score >= threshold:
        log.info(f"Headline rendered in image (match={score:.2f})")
        return RenderedCard(image=cover, text_in_image=True, match_score=score)

    # Attempt 2: reserve and blank a region, then set type into it.
    #
    # The OCR read is logged, not just the score: a score of zero cannot
    # distinguish "the model drew nothing" from "the model drew garbled
    # glyphs", and those call for opposite fixes — more prompt weight on the
    # text versus less, or a different LoRA scale.
    log.info(
        f"Headline not legible (match={score:.2f}, ocr={_ocr_preview(cover)!r}); "
        "falling back to overlay"
    )
    _save_rejected(cover, headline, score)
    spec = LayoutMaskSpec(width=runner.cfg.width, height=runner.cfg.height)
    mask, _ = build_headline_mask(spec)
    images = runner.generate(
        prompt=visual_prompt,
        negative_prompt=negative_prompt,
        occasion=occasion,
        seed=seed,
        n=1,
        mask_image=mask,
    )
    composed = compose(
        cover=images[0], headline=headline, tone=tone, style_tags=style_tags
    )
    return RenderedCard(image=composed.image, text_in_image=False, match_score=score)
