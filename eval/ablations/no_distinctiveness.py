"""Ablation: reranker ignores distinctiveness head.

Default reranker uses composite = 0.7*PI + 0.3*distinctiveness. This ablation
zeros the distinctiveness weight so reranking uses purchase_intent alone.

Usage:
    python -m eval.ablations.no_distinctiveness
"""

from __future__ import annotations

import typer

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
    """Rerank by purchase_intent alone, ignoring distinctiveness."""
    from pipeline.rerank import rerank as _rerank
    ranked = _rerank(candidates, predictor=predictor, embedder=embedder)
    for c in ranked:
        if c.scores:
            pi = c.scores.get("purchase_intent_calibrated", c.scores.get("purchase_intent", 0.0))
            c.scores["saleability_calibrated"] = pi
    ranked.sort(
        key=lambda c: c.scores.get("saleability_calibrated", 0.0),
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
            scores = [c.scores["saleability_calibrated"] for c in ranked if c.scores]
            mean_s = sum(scores) / len(scores) if scores else 0
            log.info(f"no_distinctiveness {occ}: mean_sale={mean_s:.3f} (PI only, no dist)")


if __name__ == "__main__":
    typer.run(run)
