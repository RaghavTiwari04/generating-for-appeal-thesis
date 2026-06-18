"""Unit tests for survey.analysis.bradley_terry.

Offline (no DB). Tests use synthetic pair data with known ground-truth scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survey.analysis.bradley_terry import (
    fit_bradley_terry,
    to_dataframe,
)


def _simulate_pairs(
    true_scores: np.ndarray,
    n_pairs_per_combo: int,
    *,
    tie_prob: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic 2AFC pairs under BT with the given true scores."""
    rng = np.random.default_rng(seed)
    n = len(true_scores)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            for _ in range(n_pairs_per_combo):
                p_i = np.exp(true_scores[i]) / (
                    np.exp(true_scores[i]) + np.exp(true_scores[j])
                )
                u = rng.random()
                if u < tie_prob:
                    winner = "T"
                elif rng.random() < p_i:
                    winner = "L"
                else:
                    winner = "R"
                rows.append(
                    {
                        "left_key": f"c{i}",
                        "right_key": f"c{j}",
                        "winner_side": winner,
                        "attention_check_pass": True,
                    }
                )
    return pd.DataFrame(rows)


def test_recovers_known_ranking_no_ties():
    true_s = np.array([-1.5, -0.5, 0.5, 1.5])
    df = _simulate_pairs(true_s, n_pairs_per_combo=80, seed=42)
    result = fit_bradley_terry(df, prior_strength=0.1)

    # Rank order must match
    order = np.argsort(result.scores)
    assert list(order) == [0, 1, 2, 3]

    # Score differences should approximately recover ground-truth gaps
    # (up to mean-centring and noise; check Spearman correlation = 1 and
    # Pearson > 0.95 on the centred ground truth).
    centred_true = true_s - true_s.mean()
    pearson = np.corrcoef(result.scores, centred_true)[0, 1]
    assert pearson > 0.95


def test_handles_ties():
    true_s = np.array([-1.0, 0.0, 1.0])
    df = _simulate_pairs(true_s, n_pairs_per_combo=120, tie_prob=0.15, seed=7)
    result = fit_bradley_terry(df, prior_strength=0.1)
    assert result.converged
    # Rank-order preserved
    assert np.argsort(result.scores).tolist() == [0, 1, 2]


def test_prior_keeps_extreme_finite():
    """A card that wins every comparison must still get a finite score."""
    df = pd.DataFrame(
        [
            {"left_key": "winner", "right_key": "loser", "winner_side": "L",
             "attention_check_pass": True}
            for _ in range(30)
        ]
    )
    result = fit_bradley_terry(df, prior_strength=1.0)
    assert np.isfinite(result.scores).all()
    assert result.sale_scores[result.card_keys.index("winner")] > 0.5
    assert result.sale_scores[result.card_keys.index("loser")] < 0.5


def test_empty_raises():
    with pytest.raises(ValueError):
        fit_bradley_terry(pd.DataFrame(columns=["left_key", "right_key", "winner_side"]))


def test_sale_scores_in_unit_interval():
    true_s = np.array([-2.0, 0.0, 2.0])
    df = _simulate_pairs(true_s, n_pairs_per_combo=50, seed=11)
    result = fit_bradley_terry(df, prior_strength=0.1)
    assert (result.sale_scores >= 0).all()
    assert (result.sale_scores <= 1).all()


def test_to_dataframe_shape():
    true_s = np.array([-1.0, 0.0, 1.0])
    df = _simulate_pairs(true_s, n_pairs_per_combo=20, seed=3)
    result = fit_bradley_terry(df, prior_strength=0.1)
    out = to_dataframe(result)
    assert list(out.columns) == ["card_key", "bt_score", "sale_score"]
    assert len(out) == 3
    assert out["card_key"].tolist() == result.card_keys
