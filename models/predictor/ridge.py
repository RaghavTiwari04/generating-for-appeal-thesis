"""Per-head ridge on the image embedding — the predictor that actually wins.

Measured on the same seller-grouped test split, same pad+crops features, with
the MLP given the easier task (it gets a validation split for early stopping;
ridge fits on train alone):

    head                  MLP (5 seeds)   ridge
    occasion_fit              0.529       0.631
    aesthetic                 0.661       0.787
    emotional_resonance       0.357       0.467
    distinctiveness           0.461       0.625
    purchase_intent           0.472       0.536

Seed sd on the MLP is about 0.007, so every gap is many standard deviations.
Best-of-8 recovery, which is what reranking is actually judged on, goes 66.4%
to 71.4%.

Image embedding only. A ridge on the predictor's full input — image, text and
occasion concatenated — scores 0.521 on purchase intent against 0.536 for the
image alone, and the handcrafted feature block on its own manages 0.065. The
extra channels dilute rather than add.

Each head gets its own alpha by cross-validation on the training split. That is
most of the gap to the MLP: the right penalty differs per head, while the MLP
applies one weight decay to everything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
from sklearn.linear_model import RidgeCV

from common.logging import get_logger
from models.predictor.architecture import HEAD_NAMES
from models.predictor.calibrate import load as load_isotonic
from models.predictor.dataset import (
    SplitConfig,
    _build_targets,
    load_training_frame,
    split_by_seller,
)
from models.predictor.infer import CardFeatures

log = get_logger(__name__)

# Wide enough to cover both an under- and over-regularised fit at any feature
# width; the right penalty scales with dimensionality, so a fixed alpha would
# read a 768-d and a 1536-d feature set at different effective strengths.
ALPHAS = np.logspace(-3, 4, 22)

DEFAULT_PATH = Path("./artifacts/predictor/ridge.npz")


def _matrix(df) -> np.ndarray:
    return np.stack([np.asarray(e, dtype=np.float32) for e in df["clip_embedding"]])


def _targets(df) -> dict[str, np.ndarray]:
    """Per-head targets, NaN where the judge returned nothing for that head."""
    built = [_build_targets(row) for _, row in df.iterrows()]
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(HEAD_NAMES):
        vals = np.array([t[i] if m[i] else np.nan for t, m in built], dtype=np.float64)
        out[name] = vals
    return out


@dataclass
class RidgePredictor:
    """Scores cards through the same interface as the MLP runner.

    `pipeline.rerank` takes the predictor as a parameter and only calls
    `.score()`, so this substitutes without the pipeline knowing which model it
    holds.
    """

    coefs: dict[str, np.ndarray]
    intercepts: dict[str, float]
    isotonic: object | None = None

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_PATH, calib_path: str | Path | None = None
    ) -> RidgePredictor:
        data = np.load(Path(path))
        return cls(
            coefs={name: data[f"coef_{name}"] for name in HEAD_NAMES},
            intercepts={name: float(data[f"intercept_{name}"]) for name in HEAD_NAMES},
            isotonic=load_isotonic(calib_path) if calib_path else None,
        )

    def score(self, features: list[CardFeatures]) -> list[dict[str, float]]:
        if not features:
            return []
        x = np.stack([np.asarray(f.image_emb, dtype=np.float64) for f in features])
        # Ridge is unbounded; the labels it was fitted on are judge scores in
        # [0, 1] and the pipeline compares these against MLP scores on the same
        # scale, so predictions are clipped rather than left to run past either
        # end on an unusual candidate.
        preds = {
            name: np.clip(x @ self.coefs[name] + self.intercepts[name], 0.0, 1.0)
            for name in HEAD_NAMES
        }
        pi = preds["purchase_intent"]
        pi_cal = self.isotonic.predict(pi) if self.isotonic is not None else pi
        return [
            {
                **{name: float(preds[name][i]) for name in HEAD_NAMES},
                "purchase_intent_calibrated": float(pi_cal[i]),
            }
            for i in range(len(features))
        ]


def fit(out_path: str = str(DEFAULT_PATH), seed: int = 42) -> None:
    """Fit one ridge per head on the training split and save the coefficients.

    Uses the same seller-grouped split as the MLP, at the same seed, so the
    reported numbers are comparable to `artifacts/predictor/test_metrics.json`
    rather than to a differently-drawn test set.
    """
    df = load_training_frame()
    if df.empty:
        raise SystemExit("No training data. Run scrapers + feature extraction first.")
    splits = split_by_seller(df, SplitConfig(seed=seed))

    train_x, test_x = _matrix(splits["train"]), _matrix(splits["test"])
    train_y, test_y = _targets(splits["train"]), _targets(splits["test"])
    log.info(f"Fitting on {len(train_x)} cards, {train_x.shape[1]}-d features")

    from scipy.stats import spearmanr

    payload: dict[str, np.ndarray] = {}
    metrics: dict[str, float] = {}
    for name in HEAD_NAMES:
        tr = ~np.isnan(train_y[name])
        te = ~np.isnan(test_y[name])
        model = RidgeCV(alphas=ALPHAS).fit(train_x[tr], train_y[name][tr])
        rho = float(spearmanr(model.predict(test_x[te]), test_y[name][te])[0] or 0.0)

        payload[f"coef_{name}"] = model.coef_.astype(np.float64)
        payload[f"intercept_{name}"] = np.float64(model.intercept_)
        metrics[f"spearman_{name}"] = rho
        log.info(f"  {name:22s} rho={rho:.3f}  alpha={model.alpha_:g}  n={int(tr.sum())}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    log.info(f"Saved {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    typer.run(fit)
