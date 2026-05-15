"""Ablation: pipeline without layout/typography composer.

Generates cards with the full pipeline but skips the `compose()` step —
the raw diffusion output is used as the cover (headline text overlaid
naively at a fixed position without font-palette selection, contrast
check, or binary-search sizing).

Condition tag: 'ablation_no_layout'. Compares saleability against condition C.

Usage:
    python -m eval.ablations.no_layout
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import typer
from PIL import Image, ImageDraw, ImageFont

from common.logging import get_logger
from generation.layout.compose import ComposedCard, LayoutMaskSpec

log = get_logger(__name__)
CONDITION_TAG = "ablation_no_layout"

DEFAULT_OCCASIONS = [
    "birthday/general",
    "christmas/general",
    "mothers_day",
    "valentines_day",
    "sympathy/bereavement",
]


def _naive_compose(
    cover: Image.Image,
    *,
    headline: str,
    tone: str,
    style_tags: list[str],
    mask_spec: LayoutMaskSpec | None = None,
) -> ComposedCard:
    """Overlay headline at fixed top-left with default system font."""
    img = cover.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 30), headline, fill=(245, 245, 245), font=font)
    return ComposedCard(
        image=img,
        headline=headline,
        bbox=(30, 30, 30 + len(headline) * 20, 70),
        font_family="system-default",
        font_size=36,
        colour_rgb=(245, 245, 245),
    )


def run(
    occasions: str = ",".join(DEFAULT_OCCASIONS),
    n: int = 8,
    top_k: int = 3,
    seed: int = 2000,
) -> None:
    occ_list = [o.strip() for o in occasions.split(",") if o.strip()]

    with patch("pipeline.orchestrator.compose", side_effect=_naive_compose):
        from pipeline.orchestrator import OrchestratorConfig, generate
        for i, occ in enumerate(occ_list):
            cfg = OrchestratorConfig(
                n_candidates=n,
                top_k=top_k,
                image_seed_base=seed + i * 100,
                condition_tag=CONDITION_TAG,
            )
            ranked = generate({"occasion": occ, "tone": "warm-sincere"}, cfg)
            scores = [c.scores.get("saleability_calibrated", 0) for c in ranked if c.scores]
            mean_s = sum(scores) / len(scores) if scores else 0
            log.info(f"no_layout {occ}: mean_saleability={mean_s:.3f}")


if __name__ == "__main__":
    typer.run(run)
