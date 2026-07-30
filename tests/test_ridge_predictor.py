"""The ridge predictor, which reranking now uses in place of the MLP.

It only has to do two things: fit per-head on the training split and expose the
same `.score()` the pipeline already calls, so `pipeline.rerank` cannot tell the
two models apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from models.predictor.architecture import HEAD_NAMES
from models.predictor.dataset import OCCASION_TO_IDX, SplitConfig, split_by_seller
from models.predictor.infer import CardFeatures
from models.predictor.ridge import RidgePredictor, fit

DIM = 32


@pytest.fixture
def frame() -> pd.DataFrame:
    """Cards whose judge scores are a linear function of the embedding."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=DIM)
    occasions = list(OCCASION_TO_IDX)[:4]
    rows = []
    for i in range(200):
        emb = rng.normal(size=DIM)
        target = float(1 / (1 + np.exp(-(emb @ w) / 4)))
        occ = occasions[i % 4]
        rows.append({
            "listing_id": f"l{i}",
            "seller_id": f"s{i // 4}",
            "occasion": occ,
            "occasion_idx": OCCASION_TO_IDX[occ],
            "clip_embedding": list(emb),
            "extracted_text": "happy birthday",
            "vlm_raw": {name: target for name in HEAD_NAMES},
        })
    return pd.DataFrame(rows)


def _fit(frame: pd.DataFrame, tmp_path: Path) -> Path:
    out = tmp_path / "ridge.npz"
    splits = split_by_seller(frame, SplitConfig(seed=42))
    with (
        patch("models.predictor.ridge.load_training_frame", return_value=frame),
        patch("models.predictor.ridge.split_by_seller", return_value=splits),
    ):
        fit(out_path=str(out))
    return out


def test_fit_learns_a_recoverable_signal(frame: pd.DataFrame, tmp_path: Path) -> None:
    out = _fit(frame, tmp_path)
    metrics = json.loads(out.with_suffix(".json").read_text())

    assert out.exists()
    # Targets are a deterministic function of the embedding, so a linear fit
    # that cannot rank them is broken rather than merely weak.
    assert metrics["spearman_purchase_intent"] > 0.5
    assert set(metrics) == {f"spearman_{name}" for name in HEAD_NAMES}


def test_score_matches_the_runner_interface(frame: pd.DataFrame, tmp_path: Path) -> None:
    """rerank calls `.score()` and reads these keys; nothing else is required."""
    predictor = RidgePredictor.load(_fit(frame, tmp_path))
    rng = np.random.default_rng(1)
    feats = [
        CardFeatures(
            image_emb=rng.normal(size=DIM),
            text_emb=rng.normal(size=DIM),
            occasion="birthday/general",
        )
        for _ in range(3)
    ]

    scores = predictor.score(feats)

    assert len(scores) == 3
    for row in scores:
        assert set(row) == {*HEAD_NAMES, "purchase_intent_calibrated"}
        # Ridge is unbounded but the pipeline compares these against judge
        # scores in [0, 1], so predictions are clipped to that range.
        assert all(0.0 <= row[name] <= 1.0 for name in HEAD_NAMES)


def test_empty_input_returns_empty(frame: pd.DataFrame, tmp_path: Path) -> None:
    assert RidgePredictor.load(_fit(frame, tmp_path)).score([]) == []
