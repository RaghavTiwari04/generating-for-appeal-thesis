"""Tests for survey analysis helpers (no DB required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from survey.analysis.icc import ICCResult, compute_icc
from survey.analysis.survey_loader import (
    LIKERT_DIMENSIONS,
    aggregate_ratings,
    response_time_filter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ratings(n_items: int = 20, n_raters: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for item_i in range(n_items):
        base = rng.integers(2, 6)  # item "true" score
        for rater_j in range(n_raters):
            noise = rng.integers(-1, 2)
            score = int(np.clip(base + noise, 1, 7))
            rows.append({
                "card_key": f"card_{item_i}",
                "participant_id": f"p_{rater_j}",
                "purchase_intent": score,
                "occasion_fit": max(1, min(7, score + rng.integers(-1, 2))),
                "aesthetic": max(1, min(7, score + rng.integers(-1, 2))),
                "emotional_resonance": max(1, min(7, score + rng.integers(-1, 2))),
                "distinctiveness": max(1, min(7, score + rng.integers(-1, 2))),
                "response_time_ms": rng.integers(3000, 30000),
                "listing_id": f"card_{item_i}",
                "attention_check_pass": True,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ICC tests
# ---------------------------------------------------------------------------

class TestICC:
    def test_icc_returns_result(self) -> None:
        df = _make_ratings()
        result = compute_icc(df, item_col="card_key")
        assert isinstance(result, ICCResult)

    def test_icc31_in_range(self) -> None:
        df = _make_ratings()
        result = compute_icc(df, item_col="card_key")
        assert -1.0 <= result.icc31 <= 1.0

    def test_icc3k_ge_icc31(self) -> None:
        df = _make_ratings()
        result = compute_icc(df, item_col="card_key")
        assert result.icc3k >= result.icc31 - 1e-6

    def test_perfect_agreement_high_icc(self) -> None:
        rows = []
        for i in range(15):
            for j in range(10):
                rows.append({"card_key": f"c{i}", "participant_id": f"p{j}", "purchase_intent": i % 7 + 1})
        df = pd.DataFrame(rows)
        result = compute_icc(df, item_col="card_key")
        assert result.icc3k > 0.9

    def test_random_agreement_low_icc(self) -> None:
        rng = np.random.default_rng(99)
        rows = []
        for i in range(20):
            for j in range(8):
                rows.append({
                    "card_key": f"c{i}",
                    "participant_id": f"p{j}",
                    "purchase_intent": int(rng.integers(1, 8)),
                })
        df = pd.DataFrame(rows)
        result = compute_icc(df, item_col="card_key")
        assert result.icc3k < 0.5

    def test_ci_ordered(self) -> None:
        df = _make_ratings()
        result = compute_icc(df, item_col="card_key")
        assert result.ci_low <= result.icc3k <= result.ci_high

    def test_n_items_n_raters(self) -> None:
        df = _make_ratings(n_items=15, n_raters=8)
        result = compute_icc(df, item_col="card_key")
        assert result.n_items == 15
        assert result.n_raters == 8


# ---------------------------------------------------------------------------
# Survey loader helpers
# ---------------------------------------------------------------------------

class TestSurveyLoader:
    def test_aggregate_means_in_range(self) -> None:
        df = _make_ratings()
        agg = aggregate_ratings(df)
        for dim in LIKERT_DIMENSIONS:
            col = f"{dim}_mean"
            if col in agg.per_card.columns:
                assert agg.per_card[col].between(1, 7).all()

    def test_aggregate_counts_correct(self) -> None:
        df = _make_ratings(n_items=5, n_raters=10)
        agg = aggregate_ratings(df)
        count_col = "purchase_intent_count"
        if count_col in agg.per_card.columns:
            assert (agg.per_card[count_col] == 10).all()

    def test_n_items(self) -> None:
        df = _make_ratings(n_items=12, n_raters=5)
        agg = aggregate_ratings(df)
        assert agg.n_items == 12

    def test_n_raters_total(self) -> None:
        df = _make_ratings(n_items=5, n_raters=7)
        agg = aggregate_ratings(df)
        assert agg.n_raters_total == 7

    def test_response_time_filter_removes_fast(self) -> None:
        df = _make_ratings()
        df.loc[0, "response_time_ms"] = 100   # too fast
        df.loc[1, "response_time_ms"] = 2999  # just under threshold
        filtered = response_time_filter(df, min_ms=3000)
        assert len(filtered) == len(df) - 2

    def test_response_time_filter_keeps_null(self) -> None:
        df = _make_ratings(n_items=3, n_raters=3)
        df.loc[0, "response_time_ms"] = None
        filtered = response_time_filter(df, min_ms=3000)
        assert len(filtered) == len(df)

    def test_aggregate_index_is_card_key(self) -> None:
        df = _make_ratings(n_items=5, n_raters=4)
        agg = aggregate_ratings(df)
        assert agg.per_card.index.name == "card_key"
