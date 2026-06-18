"""Train and evaluate the pricing model.

Pulls listing data (price, occasion, predictor scores, complexity) from
Postgres, normalises prices to GBP via fx_rates, trains three LightGBM
quantile regressors (P10 / P50 / P90), reports RMSE + band accuracy,
saves bundle to artifacts/pricing/.

Usage:
    python -m models.pricing.train_pricing
    make train-pricing
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from common.db import engine
from common.fx_rates import normalise_price_column
from common.logging import get_logger
from models.pricing.price_model import PriceModelBundle, predict, train

log = get_logger(__name__)

FEATURE_COLS = [
    "occasion",
    "aesthetic_score",
    "distinctiveness_score",
    "image_complexity",
    "marketplace",
]
CAT_COLS = ["occasion", "marketplace"]

_QUERY = """
SELECT
    l.listing_id,
    l.source AS marketplace,
    lf.occasion,
    l.price_minor_units,
    l.currency,
    lf.image_complexity,
    (lf.predictor_scores->>'aesthetic')::float      AS aesthetic_score,
    (lf.predictor_scores->>'distinctiveness')::float AS distinctiveness_score
FROM listings l
JOIN listing_features lf USING (listing_id)
WHERE l.price_minor_units IS NOT NULL
  AND l.currency IS NOT NULL
  AND lf.occasion IS NOT NULL
  AND lf.predictor_scores IS NOT NULL;
"""


def load_pricing_data() -> pd.DataFrame:
    df = pd.read_sql(_QUERY, engine())
    normalise_price_column(df, "price_minor_units", "currency", "price_gbp")
    df = df.dropna(subset=["price_gbp"])
    df = df[df["price_gbp"].between(0.50, 30.0)]  # sanity range
    df["aesthetic_score"] = df["aesthetic_score"].fillna(0.5)
    df["distinctiveness_score"] = df["distinctiveness_score"].fillna(0.5)
    df["image_complexity"] = df["image_complexity"].fillna(0.3)
    return df.reset_index(drop=True)


def evaluate(bundle: PriceModelBundle, df: pd.DataFrame) -> dict:
    preds = predict(bundle, df)
    y = df["price_gbp"].to_numpy()
    y_hat = preds["price_gbp_median"].to_numpy()

    rmse = float(np.sqrt(((y - y_hat) ** 2).mean()))
    mae = float(np.abs(y - y_hat).mean())

    # Band accuracy
    from models.pricing.price_model import _band
    true_bands = [_band(v) for v in y]
    pred_bands = preds["band"].tolist()
    band_acc = float(sum(t == p for t, p in zip(true_bands, pred_bands, strict=False)) / len(y))

    return {"rmse_gbp": rmse, "mae_gbp": mae, "band_accuracy": band_acc, "n": len(y)}


def run(out_dir: str = "./artifacts/pricing") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_pricing_data()
    log.info(f"Pricing training set: {len(df)} listings")
    if len(df) < 100:
        log.warning("Very few pricing examples — results unreliable")

    # 80/20 split (random; no seller leakage risk for pricing)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[idx[:n_train]].reset_index(drop=True)
    test_df = df.iloc[idx[n_train:]].reset_index(drop=True)

    bundle = train(train_df, feature_cols=FEATURE_COLS, categorical_cols=CAT_COLS)
    metrics = evaluate(bundle, test_df)

    log.info(
        f"Pricing test RMSE=£{metrics['rmse_gbp']:.2f} "
        f"MAE=£{metrics['mae_gbp']:.2f} "
        f"band_acc={metrics['band_accuracy']:.2%}"
    )

    bundle.save(out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"Pricing model saved to {out}")


if __name__ == "__main__":
    typer.run(run)
