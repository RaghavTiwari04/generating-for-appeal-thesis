"""Predictor inference helpers.

Public entrypoint: `score_cards(rows) -> list[dict]` where each row carries
the cached features needed by the model. Used by `pipeline.rerank` and the
evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common.occasions import ACTIVE_OCCASIONS as OCCASIONS
from models.predictor.architecture import HEAD_NAMES, PredictorConfig, SaleabilityPredictor
from models.predictor.calibrate import load as load_isotonic

OCCASION_TO_IDX = {occ: i for i, occ in enumerate(OCCASIONS)}


@dataclass
class CardFeatures:
    image_emb: np.ndarray              # (image_dim,)
    text_emb: np.ndarray               # (text_dim,)
    occasion: str
    price_rel: float = 0.0             # log-price relative to occasion median


class PredictorRunner:
    def __init__(self, ckpt_path: str | Path, calib_path: str | Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(ckpt_path, map_location=self.device)
        # Checkpoints record the architecture they were trained with; the sweep
        # varies trunk/head width, so assuming defaults would fail on shapes for
        # any swept model.
        arch = state.get("arch")
        if arch:
            arch = dict(arch)
            arch["head_names"] = tuple(arch.get("head_names") or HEAD_NAMES)
            cfg = PredictorConfig(**arch)
        else:
            cfg = PredictorConfig()
        self.model = SaleabilityPredictor(cfg).to(self.device).eval()
        self.model.load_state_dict(state["state_dict"])
        self.isotonic = load_isotonic(calib_path) if calib_path else None

    @torch.inference_mode()
    def score(self, features: list[CardFeatures]) -> list[dict[str, float]]:
        if not features:
            return []
        image_emb = torch.from_numpy(np.stack([f.image_emb for f in features])).float().to(self.device)
        text_emb = torch.from_numpy(np.stack([f.text_emb for f in features])).float().to(self.device)
        occ_idx = torch.tensor(
            [OCCASION_TO_IDX.get(f.occasion, 0) for f in features], dtype=torch.long
        ).to(self.device)
        price_rel = torch.tensor(
            [[f.price_rel] for f in features], dtype=torch.float32
        ).to(self.device)

        out = self.model(image_emb, text_emb, occ_idx, price_rel)
        scores: list[dict[str, float]] = []
        pi = out["purchase_intent"].cpu().numpy()
        if self.isotonic is not None:
            pi_cal = self.isotonic.predict(pi)
        else:
            pi_cal = pi
        for i in range(len(features)):
            row = {name: float(out[name][i].cpu()) for name in HEAD_NAMES}
            row["purchase_intent_calibrated"] = float(pi_cal[i])
            scores.append(row)
        return scores
