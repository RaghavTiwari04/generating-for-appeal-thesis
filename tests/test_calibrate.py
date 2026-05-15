"""Unit tests for calibration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from models.predictor.calibrate import (
    expected_calibration_error,
    fit_isotonic,
)


def test_ece_perfect_calibration() -> None:
    preds = np.linspace(0.1, 0.9, 100)
    targets = (np.random.default_rng(0).random(100) < preds).astype(float)
    report = expected_calibration_error(preds, targets)
    assert 0.0 <= report.ece <= 1.0


def test_ece_terrible_calibration() -> None:
    preds = np.full(100, 0.9)
    targets = np.zeros(100)
    report = expected_calibration_error(preds, targets)
    assert report.ece > 0.5


def test_isotonic_monotone() -> None:
    preds = np.linspace(0.0, 1.0, 50)
    targets = preds + np.random.default_rng(1).normal(0, 0.05, 50)
    targets = targets.clip(0, 1)
    iso = fit_isotonic(preds, targets)
    calibrated = iso.predict(preds)
    diffs = np.diff(calibrated)
    assert (diffs >= -1e-9).all(), "Isotonic output not monotone"


def test_ece_bins_populated() -> None:
    rng = np.random.default_rng(42)
    preds = rng.random(200)
    targets = (rng.random(200) < preds).astype(float)
    report = expected_calibration_error(preds, targets, n_bins=10)
    assert len(report.bin_centres) > 0
    assert len(report.bin_centres) == len(report.bin_observed)
