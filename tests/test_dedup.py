"""Unit tests for dedup helpers (pHash, hamming, union-find)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from data.features.dedup import (
    PHASH_HAMMING_THRESHOLD,
    UnionFind,
    compute_phash,
    hamming,
)


def _solid_img(colour: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (128, 128), color=colour)


def test_phash_identical_images() -> None:
    img = _solid_img((128, 64, 32))
    h1 = compute_phash(img)
    h2 = compute_phash(img.copy())
    assert hamming(h1, h2) == 0


def test_phash_very_different_images() -> None:
    # Solid images have no frequency content → similar pHash regardless of colour.
    # Use texturally-opposite images: noise vs uniform grey.
    rng = np.random.default_rng(0)
    noise_arr = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr).convert("RGB")
    flat_img = Image.new("RGB", (128, 128), color=(128, 128, 128))
    h1 = compute_phash(noise_img)
    h2 = compute_phash(flat_img)
    assert hamming(h1, h2) > PHASH_HAMMING_THRESHOLD


def test_hamming_symmetry() -> None:
    assert hamming(0b1010, 0b1100) == hamming(0b1100, 0b1010)


def test_hamming_zero() -> None:
    assert hamming(42, 42) == 0


def test_union_find_basic() -> None:
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("d") != uf.find("a")


