"""Tests for image complexity feature."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from data.features.image_complexity import (
    _canny_density,
    _frequency_entropy,
    compute_complexity,
)


def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), "L")


def test_solid_image_low_complexity(solid_white_img: Image.Image) -> None:
    score = compute_complexity(solid_white_img)
    assert score < 0.15, f"Solid image should be low complexity, got {score}"


def test_checkerboard_high_complexity() -> None:
    arr = np.indices((256, 256)).sum(axis=0) % 2 * 255
    img = Image.fromarray(arr.astype(np.uint8), "L").convert("RGB")
    score = compute_complexity(img)
    assert score > 0.5, f"Checkerboard should be high complexity, got {score}"


def test_complexity_in_range(gradient_img: Image.Image) -> None:
    score = compute_complexity(gradient_img)
    assert 0.0 <= score <= 1.0


def test_complexity_accepts_bytes(solid_white_img: Image.Image) -> None:
    import io
    buf = io.BytesIO()
    solid_white_img.save(buf, format="PNG")
    score = compute_complexity(buf.getvalue())
    assert 0.0 <= score <= 1.0


def test_edge_density_flat() -> None:
    gray = np.zeros((128, 128), dtype=np.float32)
    assert _canny_density(gray) == pytest.approx(0.0, abs=1e-4)


def test_frequency_entropy_positive(gradient_img: Image.Image) -> None:
    arr = np.asarray(gradient_img.convert("L"), dtype=np.float32)
    arr = arr[:256, :256]
    entropy = _frequency_entropy(arr)
    assert entropy > 0.0


def test_more_complex_image_higher_score(solid_white_img: Image.Image) -> None:
    """Natural image should score higher than flat white."""
    # Simulate a noisy image
    arr = np.random.default_rng(0).integers(0, 255, (256, 256)).astype(np.uint8)
    noisy = Image.fromarray(arr, "L").convert("RGB")
    score_noisy = compute_complexity(noisy)
    score_flat = compute_complexity(solid_white_img)
    assert score_noisy > score_flat
