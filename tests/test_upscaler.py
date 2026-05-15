"""Unit tests for upscaler — Lanczos backend only (no GPU/weights needed)."""

from __future__ import annotations

import pytest
from PIL import Image

from generation.image.upscaler import (
    PRINT_H,
    PRINT_W,
    _upscale_lanczos,
    upscale_to_print,
)


def _make_img(w: int = 1024, h: int = 1024) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 64, 32))


class TestLanczosUpscaler:
    def test_output_dimensions(self) -> None:
        img = _make_img(1024, 1024)
        out = _upscale_lanczos(img, PRINT_W, PRINT_H)
        assert out.size == (PRINT_W, PRINT_H)

    def test_custom_target(self) -> None:
        img = _make_img(512, 512)
        out = _upscale_lanczos(img, 800, 600)
        assert out.size == (800, 600)

    def test_mode_preserved(self) -> None:
        img = _make_img()
        out = _upscale_lanczos(img, PRINT_W, PRINT_H)
        assert out.mode == "RGB"

    def test_small_input_still_upscales(self) -> None:
        img = _make_img(64, 64)
        out = _upscale_lanczos(img, PRINT_W, PRINT_H)
        assert out.size == (PRINT_W, PRINT_H)


class TestUpscaleToPrint:
    def test_lanczos_backend_explicit(self) -> None:
        img = _make_img()
        out = upscale_to_print(img, backend="lanczos")
        assert out.size == (PRINT_W, PRINT_H)

    def test_realesrgan_falls_back_to_lanczos_if_unavailable(self) -> None:
        """Real-ESRGAN weights won't be present in CI — should fall back gracefully."""
        img = _make_img()
        # realesrgan backend selected but weights missing → Lanczos fallback
        out = upscale_to_print(img, backend="realesrgan")
        assert out.size == (PRINT_W, PRINT_H)

    def test_default_backend_produces_correct_size(self) -> None:
        img = _make_img()
        out = upscale_to_print(img)
        assert out.size == (PRINT_W, PRINT_H)

    def test_portrait_aspect_ratio(self) -> None:
        """Output is portrait (taller than wide) — correct for A6 card."""
        out = upscale_to_print(_make_img())
        assert out.height > out.width

    def test_does_not_mutate_input(self) -> None:
        img = _make_img()
        orig_size = img.size
        upscale_to_print(img, backend="lanczos")
        assert img.size == orig_size
