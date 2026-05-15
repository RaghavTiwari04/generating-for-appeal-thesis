"""Unit tests for pricing model (no DB needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.pricing.price_model import (
    BANDS,
    _band,
    predict,
    train,
)


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    occasions = ["birthday/general", "christmas/general", "mothers_day", "sympathy/bereavement"]
    return pd.DataFrame(
        {
            "occasion": [occasions[i % len(occasions)] for i in range(n)],
            "aesthetic_score": rng.uniform(0.3, 0.9, n),
            "distinctiveness_score": rng.uniform(0.2, 0.8, n),
            "image_complexity": rng.uniform(0.1, 0.7, n),
            "marketplace": rng.choice(["etsy", "redbubble", "zazzle"], n),
            "price_gbp": rng.uniform(1.5, 12.0, n),
        }
    )


FEATURE_COLS = [
    "occasion",
    "aesthetic_score",
    "distinctiveness_score",
    "image_complexity",
    "marketplace",
]
CAT_COLS = ["occasion", "marketplace"]


def test_train_returns_bundle() -> None:
    df = _make_df()
    bundle = train(df, feature_cols=FEATURE_COLS, categorical_cols=CAT_COLS)
    assert bundle.median is not None
    assert bundle.lower is not None
    assert bundle.upper is not None


def test_predict_shape() -> None:
    df = _make_df()
    bundle = train(df, feature_cols=FEATURE_COLS, categorical_cols=CAT_COLS)
    preds = predict(bundle, df)
    assert len(preds) == len(df)
    assert set(preds.columns) >= {"price_gbp_median", "price_gbp_p10", "price_gbp_p90", "band"}


def test_prediction_intervals_ordered() -> None:
    df = _make_df()
    bundle = train(df, feature_cols=FEATURE_COLS, categorical_cols=CAT_COLS)
    preds = predict(bundle, df)
    assert (preds["price_gbp_p10"] <= preds["price_gbp_median"]).all()
    assert (preds["price_gbp_median"] <= preds["price_gbp_p90"]).all()


def test_predict_positive_prices() -> None:
    df = _make_df()
    bundle = train(df, feature_cols=FEATURE_COLS, categorical_cols=CAT_COLS)
    preds = predict(bundle, df)
    assert (preds["price_gbp_median"] > 0).all()


def test_band_assignment() -> None:
    assert _band(1.5) == "budget"
    assert _band(3.0) == "standard"
    assert _band(4.99) == "standard"
    assert _band(5.0) == "premium"
    assert _band(7.99) == "premium"
    assert _band(8.0) == "luxury"
    assert _band(20.0) == "luxury"


def test_all_bands_covered() -> None:
    test_prices = [1.0, 4.0, 6.0, 10.0]
    band_names = {_band(p) for p in test_prices}
    assert band_names == {"budget", "standard", "premium", "luxury"}


def test_band_output_in_predictions() -> None:
    df = _make_df()
    bundle = train(df, feature_cols=FEATURE_COLS, categorical_cols=CAT_COLS)
    preds = predict(bundle, df)
    valid = {"budget", "standard", "premium", "luxury"}
    assert preds["band"].isin(valid).all()


def test_save_load_roundtrip(tmp_path) -> None:
    df = _make_df()
    bundle = train(df, feature_cols=FEATURE_COLS, categorical_cols=CAT_COLS)
    bundle.save(tmp_path / "model")
    from models.pricing.price_model import PriceModelBundle
    loaded = PriceModelBundle.load(tmp_path / "model")
    preds_orig = predict(bundle, df)["price_gbp_median"].to_numpy()
    preds_loaded = predict(loaded, df)["price_gbp_median"].to_numpy()
    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)
