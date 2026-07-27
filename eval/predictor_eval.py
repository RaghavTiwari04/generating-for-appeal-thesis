"""Predictor evaluation.

Metrics:
- Spearman ρ vs held-out reference purchase intent  (primary)
- AUC top-quartile classification
- Per-head Spearman vs the corresponding reference dimension
- Calibration: ECE + reliability plot

Baselines:
- Random
- CLIP similarity to "a popular greeting card"
- Generic aesthetic predictor
- Linear regression on hand-crafted features
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from common.logging import get_logger
from models.predictor.calibrate import (
    expected_calibration_error,
    fit_isotonic,
    reliability_plot,
    report_json,
)
from models.predictor.infer import CardFeatures, PredictorRunner

log = get_logger(__name__)


@dataclass
class PredictorEvalReport:
    spearman_purchase_intent: float
    auc_top_quartile: float
    per_head_spearman: dict[str, float]
    ece: float
    baselines: dict[str, float]


def _top_quartile_auc(predictions: np.ndarray, targets: np.ndarray) -> float:
    q3 = np.quantile(targets, 0.75)
    y = (targets >= q3).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, predictions))


def evaluate(
    predictor: PredictorRunner,
    features: list[CardFeatures],
    *,
    reference_purchase_intent: np.ndarray,
    per_head_targets: dict[str, np.ndarray],
    baseline_features: pd.DataFrame | None = None,
    out_dir: str | Path = "./artifacts/predictor_eval",
) -> PredictorEvalReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scored = predictor.score(features)
    sale_pred = np.array(
        [s.get("purchase_intent_calibrated", s["purchase_intent"]) for s in scored], dtype=np.float64
    )

    rho_pi, _ = spearmanr(sale_pred, reference_purchase_intent)
    auc = _top_quartile_auc(sale_pred, reference_purchase_intent)

    per_head: dict[str, float] = {}
    for head_name, tgt in per_head_targets.items():
        preds_head = np.array([s.get(head_name, np.nan) for s in scored], dtype=np.float64)
        m = ~np.isnan(preds_head) & ~np.isnan(tgt)
        if m.sum() >= 5:
            rho, _ = spearmanr(preds_head[m], tgt[m])
            per_head[head_name] = float(rho or 0.0)
        else:
            per_head[head_name] = float("nan")

    cal_report = expected_calibration_error(sale_pred, reference_purchase_intent)
    report_json(cal_report, out / "calibration.json")
    reliability_plot(cal_report, out / "reliability.png")

    iso = fit_isotonic(sale_pred, reference_purchase_intent)
    import joblib
    joblib.dump(iso, out / "isotonic.joblib")

    baselines = _baselines(
        sale_pred,
        reference_purchase_intent,
        baseline_features=baseline_features,
    )

    report = PredictorEvalReport(
        spearman_purchase_intent=float(rho_pi or 0.0),
        auc_top_quartile=auc,
        per_head_spearman=per_head,
        ece=cal_report.ece,
        baselines=baselines,
    )
    (out / "report.json").write_text(
        json.dumps(
            {
                "spearman_purchase_intent": report.spearman_purchase_intent,
                "auc_top_quartile": report.auc_top_quartile,
                "per_head_spearman": report.per_head_spearman,
                "ece": report.ece,
                "baselines": report.baselines,
            },
            indent=2,
        )
    )
    return report


def _baselines(
    model_pred: np.ndarray,
    targets: np.ndarray,
    *,
    baseline_features: pd.DataFrame | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(0)
    random_pred = rng.random(len(targets))
    out["random_spearman"] = float(spearmanr(random_pred, targets)[0] or 0.0)

    if baseline_features is not None and not baseline_features.empty:
        ridge = Ridge(alpha=1.0)
        ridge.fit(baseline_features.values, targets)
        pred = ridge.predict(baseline_features.values)
        out["ridge_handcrafted_spearman"] = float(spearmanr(pred, targets)[0] or 0.0)
    return out
