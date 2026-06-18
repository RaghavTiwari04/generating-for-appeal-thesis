"""Shared pytest fixtures.

All fixtures that require no network/DB/GPU so tests run offline.
DB-dependent fixtures gated behind `--integration` flag (not run by default).
"""

from __future__ import annotations

from textwrap import dedent

import numpy as np
import pytest
from PIL import Image

# ── Image fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def solid_white_img() -> Image.Image:
    return Image.new("RGB", (512, 512), color=(240, 240, 240))


@pytest.fixture
def solid_dark_img() -> Image.Image:
    return Image.new("RGB", (512, 512), color=(15, 15, 15))


@pytest.fixture
def gradient_img() -> Image.Image:
    arr = np.linspace(0, 255, 512 * 512, dtype=np.uint8).reshape(512, 512)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, "RGB")


@pytest.fixture
def print_res_img() -> Image.Image:
    """1240x1748 px - print-resolution card size."""
    return Image.new("RGB", (1240, 1748), color=(200, 180, 160))


# ── HTML fixture helpers ──────────────────────────────────────────────────────

ETSY_LISTING_HTML = dedent("""\
    <html><head>
    <script type="application/ld+json">
    {
      "@type": "Product",
      "name": "Happy Birthday Mum — Watercolour Floral Card",
      "description": "A beautiful watercolour card for mum",
      "brand": {"name": "FloralPaperCo"},
      "offers": {"price": "3.99", "priceCurrency": "GBP"},
      "aggregateRating": {"reviewCount": 127, "ratingValue": 4.9},
      "image": ["https://i.etsystatic.com/123/card.jpg"]
    }
    </script>
    </head><body>
    <h1 data-buy-box-listing-title>Happy Birthday Mum — Watercolour Floral Card</h1>
    <a href="/shop/FloralPaperCo">FloralPaperCo</a>
    <div data-buy-box-region="price">£3.99</div>
    <span data-favorites-count>2,341</span>
    <span data-bestseller-badge>Bestseller</span>
    </body></html>
""")

REDBUBBLE_LISTING_HTML = dedent("""\
    <html><body>
    <h1>Birthday Watercolour Card</h1>
    <div data-testid="work-description">A lovely birthday card with watercolour art</div>
    <a href="/people/artshop">artshop</a>
    <div data-testid="product-price">$4.50</div>
    <img src="https://ih1.redbubble.net/image.123456/card.jpg">
    </body></html>
""")


@pytest.fixture
def etsy_html() -> str:
    return ETSY_LISTING_HTML


@pytest.fixture
def redbubble_html() -> str:
    return REDBUBBLE_LISTING_HTML


# ── Embedding fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def random_clip_emb() -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.standard_normal(768).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def dummy_card_batch() -> list[dict]:
    rng = np.random.default_rng(42)
    return [
        {
            "image_emb": rng.standard_normal(768).astype(np.float32),
            "text_emb": rng.standard_normal(768).astype(np.float32),
            "occasion": "birthday/general",
            "price_rel": float(rng.standard_normal()),
        }
        for _ in range(4)
    ]


# ── Palette fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_palette() -> list[dict]:
    return [
        {"L": 80.0, "a": -10.0, "b": 20.0, "weight": 0.40},
        {"L": 50.0, "a": 5.0,   "b": -5.0, "weight": 0.30},
        {"L": 30.0, "a": 0.0,   "b": 0.0,  "weight": 0.20},
        {"L": 90.0, "a": 2.0,   "b": 8.0,  "weight": 0.07},
        {"L": 10.0, "a": 1.0,   "b": 1.0,  "weight": 0.03},
    ]
