"""Ablation: reranker ignores distinctiveness head.

Uses the full pipeline + rerank but sorts candidates by saleability head
alone, with the distinctiveness signal zeroed out of the composite score.

Concretely: sort key = saleability_calibrated (unchanged), but we also
record what the ranking would look like if we used:
    composite = 0.7 * saleability + 0.3 * distinctiveness   (default, approx)
vs:
    composite = 1.0 * saleability                            (this ablation)

Both orderings are persisted so downstream analysis can compare.

Usage:
    python -m eval.ablations.no_distinctiveness
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from common.db import connection
from common.logging import get_logger

log = get_logger(__name__)
CONDITION_TAG = "ablation_no_distinctiveness"

DEFAULT_OCCASIONS = [
    "birthday/general",
    "christmas/general",
    "mothers_day",
    "valentines_day",
    "sympathy/bereavement",
]


def _rerank_no_distinctiveness(candidates, *, predictor, embedder, top_k=None):
    """Rerank by saleability alone, ignoring distinctiveness."""
    from pipeline.rerank import rerank as _rerank
    ranked = _rerank(candidates, predictor=predictor, embedder=embedder)
    # Re-sort ignoring distinctiveness (saleability_calibrated only)
    ranked.sort(
        key=lambda c: c.scores.get("saleability_calibrated", c.scores.get("saleability", 0.0)),
        reverse=True,
    )
    return ranked[:top_k] if top_k else ranked


def run(
    occasions: str = ",".join(DEFAULT_OCCASIONS),
    n: int = 8,
    top_k: int = 3,
    seed: int = 3000,
) -> None:
    from unittest.mock import patch
    occ_list = [o.strip() for o in occasions.split(",") if o.strip()]

    with patch("pipeline.orchestrator.rerank", side_effect=_rerank_no_distinctiveness):
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
            log.info(f"no_distinctiveness {occ}: mean_saleability={mean_s:.3f}")


if __name__ == "__main__":
    typer.run(run)
