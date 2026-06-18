"""Unit tests for layout composer helpers (no GPU / DB needed)."""

from __future__ import annotations

from PIL import Image

from generation.image.headline_mask import LayoutMaskSpec, build_headline_mask
from generation.layout.compose import _pick_colour
from generation.layout.font_palette import select_fonts


def test_headline_mask_shape() -> None:
    spec = LayoutMaskSpec(width=512, height=512)
    mask, bbox = build_headline_mask(spec)
    assert mask.size == (512, 512)
    x0, y0, x1, y1 = bbox
    assert x0 >= 0 and y0 >= 0 and x1 <= 512 and y1 <= 512
    assert x1 > x0 and y1 > y0


def test_pick_colour_bright_patch() -> None:
    # Bright image -> dark text
    img = Image.new("RGB", (200, 200), color=(240, 240, 240))
    colour = _pick_colour(img, (0, 0, 200, 200))
    assert colour[0] < 100


def test_pick_colour_dark_patch() -> None:
    img = Image.new("RGB", (200, 200), color=(10, 10, 10))
    colour = _pick_colour(img, (0, 0, 200, 200))
    assert colour[0] > 150


def test_select_fonts_returns_specs() -> None:
    result = select_fonts("warm-humorous", ["watercolour"])
    assert len(result) >= 1
    for spec in result:
        assert spec.family


def test_select_fonts_fallback() -> None:
    result = select_fonts("unknown-tone", ["unknown-style"])
    assert len(result) >= 1


def test_mask_region_top_left() -> None:
    spec = LayoutMaskSpec(width=1024, height=1024, region="top-left")
    _, bbox = build_headline_mask(spec)
    x0, y0, _x1, _y1 = bbox
    assert x0 < 1024 // 2
    assert y0 < 1024 // 2


def test_mask_region_bottom() -> None:
    spec = LayoutMaskSpec(width=1024, height=1024, region="bottom")
    _, bbox = build_headline_mask(spec)
    _x0, _y0, _x1, y1 = bbox
    assert y1 > 1024 // 2
