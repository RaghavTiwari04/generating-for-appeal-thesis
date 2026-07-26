"""Render the headline into the artwork, falling back to a typographic overlay.

Commercial cards integrate their lettering into the design. The original
pipeline instead reserved a fixed region (top-left, 55% x 25%), inpainted it to
blank with a second Flux Fill pass, and drew text into the emptiness — so every
card shared one layout and the type never belonged to the artwork.

Flux renders text well enough to try the direct route first:

  1. Ask the model for the greeting as part of the design.
  2. OCR the result and check the headline actually came out.
  3. Only if it did not, fall back to the reserved-region overlay.

Verification matters because diffusion text fails ungracefully — misspelt or
garbled glyphs look worse than a clean overlay. The fallback also costs one
extra generation, but successes skip the Fill pass entirely, which was roughly
half the time spent per image.

Used by conditions A, B and C alike: card format must not differ between
conditions, or it confounds the comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image

from common.logging import get_logger
from generation.image.headline_mask import LayoutMaskSpec, build_headline_mask
from generation.layout.compose import compose

log = get_logger(__name__)

# Share of headline words that must survive OCR for the render to count.
MATCH_THRESHOLD = 0.8


@dataclass
class RenderedCard:
    image: Image.Image
    text_in_image: bool          # True when the model rendered the headline
    match_score: float


def augment_prompt(visual_prompt: str, headline: str) -> str:
    """Ask for the greeting as designed lettering rather than an overlay."""
    return (
        f'{visual_prompt}. The greeting "{headline}" is lettered into the design '
        f"as part of the artwork, clearly legible and correctly spelled"
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


def verify_headline(image: Image.Image, headline: str) -> float:
    """Score how much of the headline OCR can read back from the image."""
    try:
        from data.features.ocr import ocr_image

        return match_score(ocr_image(image).text, headline)
    except Exception as e:
        # Missing tesseract must not fail generation; fall back to the overlay.
        log.warning(f"OCR verification unavailable ({e}); using overlay")
        return 0.0


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
) -> RenderedCard:
    """Generate one card, preferring lettering rendered into the artwork."""
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
    if score >= threshold:
        log.info(f"Headline rendered in image (match={score:.2f})")
        return RenderedCard(image=cover, text_in_image=True, match_score=score)

    # Attempt 2: reserve and blank a region, then set type into it.
    log.info(f"Headline not legible (match={score:.2f}); falling back to overlay")
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
