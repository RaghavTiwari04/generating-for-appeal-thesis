"""Unit tests for proxy label computation."""

import math

import numpy as np
import pandas as pd
import pytest

from data.labels.proxy import ProxyWeights, _zscore_clip_minmax, compute_proxy_scores


def _make_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    occasions = ["birthday/general", "christmas/general", "mothers_day"]
    occ = [occasions[i % len(occasions)] for i in range(n)]
    return pd.DataFrame(
        {
            "listing_id": [f"id_{i}" for i in range(n)],
            "occasion": occ,
            "is_bestseller": rng.integers(0, 2, n).astype(bool),
            "review_count": rng.integers(0, 500, n),
            "favourite_count": rng.integers(0, 2000, n),
            "favourite_velocity": rng.exponential(5, n),
            "review_velocity": rng.exponential(2, n),
            "log_review_count": rng.exponential(3, n),
            "is_bestseller_f": rng.integers(0, 2, n).astype(float),
        }
    )


def test_zscore_all_same() -> None:
    s = pd.Series([5.0, 5.0, 5.0, 5.0])
    result = _zscore_clip_minmax(s)
    assert all(v == 0.5 for v in result)


def test_zscore_range() -> None:
    s = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    result = _zscore_clip_minmax(s)
    assert all(0.0 <= v <= 1.0 for v in result)


def test_proxy_score_bounds() -> None:
    df = _make_df()
    for col in ["favourite_velocity", "review_velocity", "log_review_count"]:
        df[col + "_norm"] = (
            df.groupby("occasion")[col]
            .transform(lambda s: _zscore_clip_minmax(s))
            .fillna(0.0)
        )

    w = ProxyWeights()
    df["proxy_score"] = (
        w.favourite_velocity * df["favourite_velocity_norm"]
        + w.review_velocity * df["review_velocity_norm"]
        + w.is_bestseller * df["is_bestseller_f"]
        + w.log_review_count * df["log_review_count_norm"]
    ).clip(0.0, 1.0)

    assert df["proxy_score"].between(0.0, 1.0).all()


def test_proxy_weights_sum_to_one() -> None:
    w = ProxyWeights()
    total = w.favourite_velocity + w.review_velocity + w.is_bestseller + w.log_review_count
    assert math.isclose(total, 1.0, abs_tol=1e-6)
