"""Unit tests for dedup helpers (pHash, hamming, TF-IDF, union-find)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from data.features.dedup import (
    PHASH_HAMMING_THRESHOLD,
    UnionFind,
    compute_phash,
    hamming,
    tfidf_duplicates,
)


def _solid_img(colour: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (128, 128), color=colour)


def test_phash_identical_images() -> None:
    img = _solid_img((128, 64, 32))
    h1 = compute_phash(img)
    h2 = compute_phash(img.copy())
    assert hamming(h1, h2) == 0


def test_phash_very_different_images() -> None:
    h1 = compute_phash(_solid_img((0, 0, 0)))
    h2 = compute_phash(_solid_img((255, 255, 255)))
    # Very different images should differ on many bits
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


def test_tfidf_duplicates_exact() -> None:
    rows = [("id1", "happy birthday card for mum"), ("id2", "happy birthday card for mum")]
    pairs = list(tfidf_duplicates(rows, threshold=0.8))
    assert len(pairs) == 1
    assert set(pairs[0][:2]) == {"id1", "id2"}


def test_tfidf_duplicates_dissimilar() -> None:
    rows = [
        ("id1", "happy birthday card flowers"),
        ("id2", "sympathy bereavement condolences funeral"),
    ]
    pairs = list(tfidf_duplicates(rows, threshold=0.8))
    assert len(pairs) == 0
