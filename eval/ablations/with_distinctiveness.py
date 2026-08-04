"""Ablation: rerank on a blend of purchase intent and distinctiveness.

The pipeline ranks candidates by predicted purchase intent alone, because that
is what condition D's human cards are selected on and ranking both by one
objective keeps the comparison clean. It used to blend in distinctiveness at
0.3, and the argument for that is real: distinctiveness is what stops best-of-N
converging on the safest design, which matters for a generator asked to produce
a catalogue rather than a single card.

This runs the blend so the cost of dropping it is measurable rather than
asserted — whether the chosen cards score lower on purchase intent, and whether
they come out more varied.

Usage:
    python -m eval.ablations.with_distinctiveness
"""

from __future__ import annotations

import typer

from common.logging import get_logger

log = get_logger(__name__)
CONDITION_TAG = "ablation_with_distinctiveness"

DIST_WEIGHT = 0.3

DEFAULT_OCCASIONS = [
    "birthday/general",
    "christmas/general",
    "mothers_day",
    "valentines_day",
    "sympathy/bereavement",
]


def _rerank_blended(candidates, *, predictor, embedder, top_k=None):
    """Rerank by (1 - w) * purchase intent + w * distinctiveness."""
    from pipeline.rerank import rerank as _rerank

    ranked = _rerank(candidates, predictor=predictor, embedder=embedder)
    for c in ranked:
        if c.scores:
            pi = c.scores.get("purchase_intent_calibrated", c.scores.get("purchase_intent", 0.0))
            dist = c.scores.get("distinctiveness", 0.0)
            c.scores["saleability_calibrated"] = (1 - DIST_WEIGHT) * pi + DIST_WEIGHT * dist
    ranked.sort(key=lambda c: c.scores.get("saleability_calibrated", 0.0), reverse=True)
    return ranked[:top_k] if top_k else ranked


def run(
    occasions: str = ",".join(DEFAULT_OCCASIONS),
    n: int = 8,
    top_k: int = 3,
    seed: int = 3000,
) -> None:
    from unittest.mock import patch

    occ_list = [o.strip() for o in occasions.split(",") if o.strip()]

    with patch("pipeline.orchestrator.rerank", side_effect=_rerank_blended):
        from pipeline.orchestrator import OrchestratorConfig, generate

        for i, occ in enumerate(occ_list):
            cfg = OrchestratorConfig(
                n_candidates=n,
                top_k=top_k,
                image_seed_base=seed + i * 100,
                condition_tag=CONDITION_TAG,
            )
            ranked = generate({"occasion": occ, "tone": "warm-sincere"}, cfg)
            scored = [c.scores for c in ranked if c.scores]
            if not scored:
                continue
            mean_blend = sum(s["saleability_calibrated"] for s in scored) / len(scored)
            # Reported apart from the blend: its whole point is trading purchase
            # intent for variety, and only the two together show whether it did.
            mean_pi = sum(
                s.get("purchase_intent_calibrated", s.get("purchase_intent", 0.0))
                for s in scored
            ) / len(scored)
            mean_dist = sum(s.get("distinctiveness", 0.0) for s in scored) / len(scored)
            log.info(
                f"with_distinctiveness {occ}: blend={mean_blend:.3f} "
                f"purchase_intent={mean_pi:.3f} distinctiveness={mean_dist:.3f}"
            )


if __name__ == "__main__":
    typer.run(run)
