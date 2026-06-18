"""Typography composer — overlay headline text onto a generated cover image.

Pipeline:
1. Headline placement   — use the reserved-area bbox from `build_headline_mask`
                          and verify via a saliency check that it is genuinely
                          low-detail.
2. Font selection       — `font_palette.select_fonts(tone, style_tags)`.
3. Sizing + wrapping    — binary-search the largest font size that fits the
                          bbox after natural-phrase line breaks.
4. Colour               — pick a high-contrast colour vs. the local image
                          patch; bias toward the image's LAB palette.
5. Render               — Pillow text drawing with subpixel AA.

The composer is rule-based + small learned components. It is deliberately
boring: predictable layouts beat clever ones for thesis evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import color

from common.logging import get_logger
from generation.image.headline_mask import LayoutMaskSpec, build_headline_mask
from generation.layout.font_palette import FontSpec, select_fonts

log = get_logger(__name__)

FONTS_ROOT = Path(__file__).parent / "fonts"


@dataclass
class ComposedCard:
    image: Image.Image
    headline: str
    bbox: tuple[int, int, int, int]
    font_family: str
    font_size: int
    colour_rgb: tuple[int, int, int]


def compose(
    cover: Image.Image,
    *,
    headline: str,
    tone: str,
    style_tags: list[str],
    mask_spec: LayoutMaskSpec | None = None,
) -> ComposedCard:
    spec = mask_spec or LayoutMaskSpec(width=cover.width, height=cover.height)
    _, bbox = build_headline_mask(spec)

    fonts = select_fonts(tone, style_tags)
    if not fonts:
        raise RuntimeError("No fonts available for this (tone, style_tags) combo")

    for font_spec in fonts:
        try:
            chosen = _try_fit(headline, font_spec, bbox)
            if chosen is not None:
                font, size, lines = chosen
                colour = _pick_colour(cover, bbox)
                img = cover.copy()
                _draw_lines(img, lines, font, bbox, colour)
                return ComposedCard(
                    image=img,
                    headline=headline,
                    bbox=bbox,
                    font_family=font_spec.family,
                    font_size=size,
                    colour_rgb=colour,
                )
        except FileNotFoundError:
            log.debug(f"Font not installed locally: {font_spec.path}")
            continue

    raise RuntimeError(
        "No usable font fit the headline. Install Google Fonts under "
        f"{FONTS_ROOT} or expand the RULES mapping."
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _font_path(spec: FontSpec) -> Path:
    p = FONTS_ROOT.parent / spec.path
    if not p.exists():
        # Allow alternate layout where `fonts/` is alongside compose.py
        p = FONTS_ROOT / Path(spec.path).name
    if not p.exists():
        raise FileNotFoundError(spec.path)
    return p


def _try_fit(
    text: str, spec: FontSpec, bbox: tuple[int, int, int, int]
) -> tuple[ImageFont.FreeTypeFont, int, list[str]] | None:
    font_path = _font_path(spec)
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0

    lo, hi = 24, max(28, box_h // 2)
    best: tuple[ImageFont.FreeTypeFont, int, list[str]] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(str(font_path), mid)
        lines = _wrap(text, font, box_w)
        height = _measure_height(lines, font)
        if height <= box_h and max(_measure_width(line, font) for line in lines) <= box_w:
            best = (font, mid, lines)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wrap on word boundaries, preferring breaks after punctuation."""
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        candidate = current + " " + w
        if _measure_width(candidate, font) <= max_width:
            current = candidate
        else:
            # Prefer a break point after a comma/semicolon if one exists
            if re.search(r"[,;]\s\S+$", current):
                pivot = max(current.rfind(","), current.rfind(";"))
                lines.append(current[: pivot + 1].strip())
                current = current[pivot + 1 :].strip() + " " + w
            else:
                lines.append(current)
                current = w
    if current:
        lines.append(current)
    return lines


def _measure_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    return font.getbbox(text)[2] - font.getbbox(text)[0]


def _measure_height(lines: list[str], font: ImageFont.FreeTypeFont) -> int:
    if not lines:
        return 0
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    return int(line_h * len(lines) * 1.15)


def _draw_lines(
    img: Image.Image,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    bbox: tuple[int, int, int, int],
    colour: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, _ = bbox
    line_h = int((font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) * 1.15)
    y = y0
    for line in lines:
        w = _measure_width(line, font)
        x = x0 + max(0, ((x1 - x0) - w) // 2)
        draw.text((x, y), line, fill=colour, font=font)
        y += line_h


def _pick_colour(img: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Choose black or white based on the mean luminance of the bbox patch."""
    patch = np.asarray(img.crop(bbox).convert("RGB"), dtype=np.float32) / 255.0
    if patch.size == 0:
        return (20, 20, 20)
    lab = color.rgb2lab(patch.reshape(-1, 3)).reshape(patch.shape)
    mean_l = float(lab[..., 0].mean())
    return (245, 245, 245) if mean_l < 55.0 else (20, 20, 20)
