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

import json
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

_PI = HEAD_NAMES.index("purchase_intent")


def _dataset_to_features(ds: PredictorDataset) -> tuple[
    list[CardFeatures], np.ndarray, dict[str, np.ndarray]
]:
    """Convert dataset rows → CardFeatures + reference targets."""
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


def _baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Occasion one-hot, as the floor the predictor has to beat.

    One-hot rather than an integer code: a linear model reads 0..3 as an
    ordering, so label encoding would understate what occasion alone can do.

    This previously also used price, review and favourite counts. The training
    query does not select them, so the lookup raised and a bare except returned
    None — the baseline silently vanished from the report rather than being
    reported as weak.
    """
    return pd.get_dummies(
        df["occasion"].fillna("birthday/general"), prefix="occ", dtype=float
    ).reset_index(drop=True)


def _seed_spread(
    ckpt_dir: Path,
    calib_path: Path | None,
    features: list,
    targets: np.ndarray,
) -> dict[str, float]:
    """Best-of-N recovery across every seed the training run produced.

    The report's headline recovery comes from `best.ckpt`, which is one seed —
    whichever validated best. Purchase-intent Spearman varies by about 0.03 sd
    across seeds, so a single checkpoint's recovery is a one-sample draw. Read
    against a deterministic ridge it looked decisive in both directions on
    consecutive runs: 77.0% versus 76.5% one session, 67.1% versus 71.4% the
    next. Neither gap meant anything without this spread.
    """
    from eval.predictor_eval import best_of_n

    ckpts = sorted(ckpt_dir.glob("seed_*.ckpt"))
    if len(ckpts) < 2:
        return {}

    recovered = []
    for path in ckpts:
        runner = PredictorRunner(path, calib_path)
        preds = np.array([s["purchase_intent"] for s in runner.score(features)])
        result = best_of_n(preds, targets)
        if result:
            recovered.append(result["recovered"])

    if not recovered:
        return {}
    vals = np.array(recovered)
    return {
        "seeds": float(len(vals)),
        "recovered_mean": float(vals.mean()),
        "recovered_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "recovered_min": float(vals.min()),
        "recovered_max": float(vals.max()),
    }


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

    # The text embedder must be wired here as it is in training. Without it the
    # dataset yields zero text vectors, and the model would be evaluated with a
    # third of its input blanked — understating it for a reason invisible in
    # the report.
    embedder = CLIPEmbedder()
    ds = PredictorDataset(test_df, text_embedder=embedder.embed_texts)
    predictor = PredictorRunner(ckpt, calib_path)
    train_ds = PredictorDataset(splits["train"], text_embedder=embedder.embed_texts)

    log.info("Extracting features...")
    features, pi_targets, head_targets = _dataset_to_features(ds)

    pi_mask = ~np.isnan(pi_targets)
    log.info(f"Cards with a purchase-intent label: {pi_mask.sum()} / {len(features)}")

    features_pi = [f for f, m in zip(features, pi_mask, strict=False) if m]
    pi_valid = pi_targets[pi_mask]

    from eval.predictor_eval import BaselineTrainingData, evaluate

    # Baselines are fitted on the training split, the same data the predictor
    # saw, so the comparison is like for like.
    def _stack(ds, key):
        return np.stack([ds[i][key].numpy() for i in range(len(ds))])

    def _concat(images, texts, occ_onehot):
        return np.hstack([images, texts, occ_onehot])

    train_occ = _baseline_features(splits["train"]).to_numpy(dtype=float)
    test_occ = _baseline_features(test_df[pi_mask]).to_numpy(dtype=float)
    train_baseline = BaselineTrainingData(
        image_embeddings=_stack(train_ds, "image_emb"),
        occasion_features=train_occ,
        targets=np.array(
            [float(train_ds[i]["targets"][_PI].item()) for i in range(len(train_ds))]
        ),
        test_image_embeddings=np.stack([f.image_emb for f in features_pi]),
        full_features=_concat(
            _stack(train_ds, "image_emb"), _stack(train_ds, "text_emb"), train_occ
        ),
        test_full_features=_concat(
            np.stack([f.image_emb for f in features_pi]),
            np.stack([f.text_emb for f in features_pi]),
            test_occ,
        ),
        head_targets={
            name: np.array(
                [
                    float(train_ds[i]["targets"][j])
                    if train_ds[i]["mask"][j]
                    else float("nan")
                    for i in range(len(train_ds))
                ]
            )
            for j, name in enumerate(HEAD_NAMES)
        },
    )

    report = evaluate(
        predictor=predictor,
        features=features_pi,
        reference_purchase_intent=pi_valid,
        per_head_targets={
            name: vals[pi_mask]
            for name, vals in head_targets.items()
        },
        baseline_features=_baseline_features(test_df[pi_mask]),
        train_baseline=train_baseline,
        out_dir=out_dir,
    )

    # Written after `evaluate`, which owns report.json — this needs every seed
    # checkpoint, which only exists once training has finished.
    spread = _seed_spread(ckpt.parent, calib_path, features_pi, pi_valid)
    if spread:
        ridge = report.best_of_n.get("ridge_recovered", float("nan"))
        log.info(
            f"Best-of-8 recovery over {int(spread['seeds'])} seeds: "
            f"{spread['recovered_mean']:.1%} +/- {spread['recovered_sd']:.1%} "
            f"(range {spread['recovered_min']:.1%}-{spread['recovered_max']:.1%}) "
            f"vs ridge {ridge:.1%}"
        )
        path = out_dir / "report.json"
        data = json.loads(path.read_text())
        data["best_of_n_seed_spread"] = spread
        path.write_text(json.dumps(data, indent=2))

    log.info(
        f"\nResults:\n"
        f"  Spearman vs purchase intent : {report.spearman_purchase_intent:.3f}\n"
        f"  AUC top-quartile            : {report.auc_top_quartile:.3f}\n"
        f"  ECE                         : {report.ece:.3f}\n"
        f"  Random baseline rho         : {report.baselines.get('random_spearman', float('nan')):.3f}"
    )

    # Generate reliability plot if matplotlib available
    try:
        from eval.reports.figures import fig4_reliability
        cal = json.loads((out_dir / "calibration.json").read_text())
        fig4_reliability(cal, out_dir / "reliability.png")
        log.info("Reliability plot saved")
    except Exception as e:
        log.debug(f"Skipped reliability plot: {e}")


if __name__ == "__main__":
    typer.run(run)
