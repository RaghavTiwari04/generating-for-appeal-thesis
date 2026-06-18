"""Calibrate the purchase_intent head via isotonic regression.

Fit on validation set predictions vs. ground-truth human BT purchase intent.
Report Expected Calibration Error and a reliability diagram.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationReport:
    ece: float
    bin_centres: list[float]
    bin_observed: list[float]
    bin_predicted: list[float]
    bin_counts: list[int]


def fit_isotonic(predictions: np.ndarray, targets: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(predictions, targets)
    return iso


def expected_calibration_error(
    predictions: np.ndarray, targets: np.ndarray, *, n_bins: int = 10
) -> CalibrationReport:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres, observed, predicted, counts = [], [], [], []
    ece = 0.0
    n = len(predictions)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (predictions >= lo) & (predictions < hi if i < n_bins - 1 else predictions <= hi)
        if mask.sum() == 0:
            continue
        p_bin = float(predictions[mask].mean())
        t_bin = float(targets[mask].mean())
        c = int(mask.sum())
        ece += (c / n) * abs(p_bin - t_bin)
        centres.append(float((lo + hi) / 2))
        predicted.append(p_bin)
        observed.append(t_bin)
        counts.append(c)
    return CalibrationReport(
        ece=float(ece),
        bin_centres=centres,
        bin_observed=observed,
        bin_predicted=predicted,
        bin_counts=counts,
    )


def save(iso: IsotonicRegression, path: str | Path) -> None:
    joblib.dump(iso, path)


def load(path: str | Path) -> IsotonicRegression:
    return joblib.load(path)


def reliability_plot(report: CalibrationReport, out_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(report.bin_predicted, report.bin_observed, "o-", label=f"ECE={report.ece:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def report_json(report: CalibrationReport, out_path: str | Path) -> None:
    Path(out_path).write_text(
        json.dumps(
            {
                "ece": report.ece,
                "bin_centres": report.bin_centres,
                "bin_observed": report.bin_observed,
                "bin_predicted": report.bin_predicted,
                "bin_counts": report.bin_counts,
            },
            indent=2,
        )
    )
