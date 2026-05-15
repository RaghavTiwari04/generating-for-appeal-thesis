"""Ablation: pipeline without LoRA conditioning.

Generates cards using the full pipeline except no occasion-specific LoRA is
loaded. Saves to `generated_cards` with condition_tag='ablation_no_lora'.
Compare mean saleability score against condition C (pipeline + rerank) to
quantify the LoRA contribution.

Usage:
    python -m eval.ablations.no_lora --occasions birthday/general,christmas/general
"""

from __future__ import annotations

import typer
from common.logging import get_logger
from pipeline.orchestrator import OrchestratorConfig, generate

log = get_logger(__name__)

CONDITION_TAG = "ablation_no_lora"
DEFAULT_OCCASIONS = [
    "birthday/general",
    "christmas/general",
    "mothers_day",
    "valentines_day",
    "sympathy/bereavement",
]


def _patch_no_lora():
    """Monkey-patch DiffusionRunner.generate so it never loads LoRAs."""
    from generation.image import diffusion as _diff
    original_apply = _diff.DiffusionRunner._apply_loras

    def _noop(self, occasion):
        pass  # Skip LoRA loading

    _diff.DiffusionRunner._apply_loras = _noop
    return original_apply


def run(
    occasions: str = ",".join(DEFAULT_OCCASIONS),
    n: int = 8,
    top_k: int = 3,
    seed: int = 1000,
) -> None:
    occ_list = [o.strip() for o in occasions.split(",") if o.strip()]
    original_apply = _patch_no_lora()

    try:
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
            log.info(f"no_lora {occ}: mean_saleability={mean_s:.3f} ({len(ranked)} cards)")
    finally:
        from generation.image import diffusion as _diff
        _diff.DiffusionRunner._apply_loras = original_apply


if __name__ == "__main__":
    typer.run(run)
