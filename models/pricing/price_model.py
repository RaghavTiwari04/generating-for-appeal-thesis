"""LightGBM price-band regressor.

Target: log of GBP price. Features: occasion, predicted aesthetic +
distinctiveness scores, image complexity, marketplace, seller-tier proxy.

Output: point estimate + 80% prediction interval -> band:
  budget (< £3) / standard (£3–5) / premium (£5–8) / luxury (£8+).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

BANDS = (
    ("budget", 0.0, 3.0),
    ("standard", 3.0, 5.0),
    ("premium", 5.0, 8.0),
    ("luxury", 8.0, float("inf")),
)


@dataclass
class PriceModelBundle:
    median: lgb.Booster
    lower: lgb.Booster  # 10th percentile
    upper: lgb.Booster  # 90th percentile
    feature_cols: list[str]
    categorical_cols: list[str]

    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self.median.save_model(str(d / "median.txt"))
        self.lower.save_model(str(d / "lower.txt"))
        self.upper.save_model(str(d / "upper.txt"))
        (d / "meta.json").write_text(
            json.dumps(
                {"feature_cols": self.feature_cols, "categorical_cols": self.categorical_cols},
                indent=2,
            )
        )

    @classmethod
    def load(cls, dir_path: str | Path) -> PriceModelBundle:
        d = Path(dir_path)
        meta = json.loads((d / "meta.json").read_text())
        return cls(
            median=lgb.Booster(model_file=str(d / "median.txt")),
            lower=lgb.Booster(model_file=str(d / "lower.txt")),
            upper=lgb.Booster(model_file=str(d / "upper.txt")),
            feature_cols=meta["feature_cols"],
            categorical_cols=meta["categorical_cols"],
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(df: pd.DataFrame, *, feature_cols: list[str], categorical_cols: list[str]) -> PriceModelBundle:
    """Train quantile-regression LightGBM at 10%/50%/90% for prediction intervals."""
    assert "price_gbp" in df.columns, "df must contain price_gbp (numeric, GBP)"
    y = np.log1p(df["price_gbp"].astype(float).clip(lower=0.0))
    X = df[feature_cols].copy()
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    common = {
        "objective": "quantile",
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    def _fit(alpha: float) -> lgb.Booster:
        params = {**common, "alpha": alpha}
        ds = lgb.Dataset(X, label=y, categorical_feature=categorical_cols, free_raw_data=False)
        return lgb.train(params, ds, num_boost_round=500)

    return PriceModelBundle(
        median=_fit(0.50),
        lower=_fit(0.10),
        upper=_fit(0.90),
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(bundle: PriceModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    X = df[bundle.feature_cols].copy()
    for col in bundle.categorical_cols:
        X[col] = X[col].astype("category")

    med = np.expm1(bundle.median.predict(X))
    lo = np.expm1(bundle.lower.predict(X))
    hi = np.expm1(bundle.upper.predict(X))
    bands = [_band(m) for m in med]

    return pd.DataFrame(
        {
            "price_gbp_p10": lo,
            "price_gbp_median": med,
            "price_gbp_p90": hi,
            "band": bands,
        }
    )


def _band(price_gbp: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= price_gbp < hi:
            return name
    return "luxury"


def save_joblib(bundle: PriceModelBundle, path: str | Path) -> None:
    """Convenience single-file save (uses joblib of the bundle)."""
    joblib.dump(bundle, path)
