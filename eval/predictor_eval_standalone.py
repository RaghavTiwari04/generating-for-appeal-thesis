"""Standalone predictor evaluation script.

Loads a trained checkpoint, runs it over the held-out test split (seller-id
split, same seed as training), computes all metrics, writes report + figures.

Usage:
    python -m eval.predictor_eval_standalone \
        --ckpt artifacts/predictor/best.ckpt \
        --calib artifacts/predictor/isotonic.joblib \
        --study-id main_v1 \
        --out-dir artifacts/predictor_eval
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from common.logging import get_logger
from data.features.clip_embed import CLIPEmbedder
from models.predictor.architecture import HEAD_NAMES
from models.predictor.dataset import (
    PredictorDataset,
    SplitConfig,
    load_training_frame,
    split_by_seller,
)
from models.predictor.infer import CardFeatures, PredictorRunner

log = get_logger(__name__)


def _dataset_to_features(ds: PredictorDataset, embedder: CLIPEmbedder) -> tuple[
    list[CardFeatures], np.ndarray, dict[str, np.ndarray]
]:
    """Convert dataset rows → CardFeatures + survey targets."""
    from common.occasions import ACTIVE_OCCASIONS as OCCASIONS

    idx_to_occasion = {i: o for i, o in enumerate(OCCASIONS)}

    features: list[CardFeatures] = []
    pi_targets: list[float] = []
    head_targets: dict[str, list[float]] = {n: [] for n in HEAD_NAMES}
    head_masks: dict[str, list[float]] = {n: [] for n in HEAD_NAMES}

    for i in range(len(ds)):
        item = ds[i]
        occ_idx = int(item["occasion_idx"].item())
        features.append(CardFeatures(
            image_emb=item["image_emb"].numpy(),
            text_emb=item["text_emb"].numpy(),
            occasion=idx_to_occasion.get(occ_idx, "birthday/general"),
        ))
        tgts = item["targets"].numpy()
        mask = item["mask"].numpy()
        pi_idx = HEAD_NAMES.index("purchase_intent")
        pi_targets.append(float(tgts[pi_idx]) if mask[pi_idx] else float("nan"))
        for j, name in enumerate(HEAD_NAMES):
            head_targets[name].append(float(tgts[j]) if mask[j] else float("nan"))
            head_masks[name].append(float(mask[j]))

    return features, np.array(pi_targets), {
        k: np.array(v) for k, v in head_targets.items()
    }


def _baseline_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """Hand-crafted features for ridge baseline: log price, occasion OHE."""
    try:
        from sklearn.preprocessing import LabelEncoder
        enc = LabelEncoder()
        occ_enc = enc.fit_transform(df["occasion"].fillna("birthday/general"))
        feats = pd.DataFrame({
            "occ_enc": occ_enc,
            "log_review": np.log1p(df["review_count"].fillna(0).astype(float)),
            "log_fav": np.log1p(df["favourite_count"].fillna(0).astype(float)),
        })
        return feats
    except Exception:
        return None


def run(
    ckpt: Path = Path("./artifacts/predictor/best.ckpt"),
    calib: Path | None = Path("./artifacts/predictor/isotonic.joblib"),
    study_id: str = "main_v1",
    out_dir: Path = Path("./artifacts/predictor_eval"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Run `make train-predictor` first.")

    calib_path = calib if (calib and calib.exists()) else None

    log.info("Loading training frame (test split)...")
    df = load_training_frame()
    splits = split_by_seller(df, SplitConfig(seed=42))
    test_df = splits["test"]
    log.info(f"Test set: {len(test_df)} listings")

    ds = PredictorDataset(test_df)
    predictor = PredictorRunner(ckpt, calib_path)
    embedder = CLIPEmbedder()

    log.info("Extracting features...")
    features, pi_targets, head_targets = _dataset_to_features(ds, embedder)

    # Filter to rows with survey PI labels
    pi_mask = ~np.isnan(pi_targets)
    log.info(f"Cards with survey PI labels: {pi_mask.sum()} / {len(features)}")

    features_pi = [f for f, m in zip(features, pi_mask, strict=False) if m]
    pi_valid = pi_targets[pi_mask]

    from eval.predictor_eval import evaluate

    report = evaluate(
        predictor=predictor,
        features=features_pi,
        survey_purchase_intent=pi_valid,
        per_head_targets={
            name: vals[pi_mask]
            for name, vals in head_targets.items()
        },
        baseline_features=_baseline_features(test_df[pi_mask]),
        out_dir=out_dir,
    )

    log.info(
        f"\nResults:\n"
        f"  Spearman vs purchase intent : {report.spearman_purchase_intent:.3f}\n"
        f"  AUC top-quartile            : {report.auc_top_quartile:.3f}\n"
        f"  ECE                         : {report.ece:.3f}\n"
        f"  Random baseline rho         : {report.baselines.get('random_spearman', float('nan')):.3f}"
    )

    # Generate reliability plot if matplotlib available
    try:
        import json as _json

        from eval.reports.figures import fig4_reliability
        cal = _json.loads((out_dir / "calibration.json").read_text())
        fig4_reliability(cal, out_dir / "reliability.png")
        log.info("Reliability plot saved")
    except Exception as e:
        log.debug(f"Skipped reliability plot: {e}")


if __name__ == "__main__":
    typer.run(run)
