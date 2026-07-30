"""Train the multi-head saleability predictor.

Loss: masked weighted MSE per head. All five heads are supervised by the VLM
labels — the rubric judge for the quality dimensions, SSR for purchase intent
— so all five see the same cards. The purchase-intent head is upweighted
because it is the head the pipeline ranks on, not because it has fewer labels.

Masked multi-task loss (Ruder 2017): mask[head][sample] = 1 where a label
exists, 0 otherwise, so a dimension the judge failed to return contributes no
gradient rather than an imputed target.

Optimiser: AdamW, lr 1e-4, weight decay 1e-2, cosine schedule.
Backbone frozen; only the trunk + heads train (features are pre-cached).

Split: by `seller_id` 70/15/15. Sampler: weighted by occasion.

Usage:
    python -m models.predictor.train --epochs 30 --batch-size 64
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from common.config import settings
from common.logging import get_logger
from models.predictor.architecture import (
    HEAD_NAMES,
    SaleabilityPredictor,
    head_loss_weights,
)
from models.predictor.dataset import (
    PredictorDataset,
    SplitConfig,
    load_training_frame,
    make_occasion_sampler,
    split_by_seller,
)

log = get_logger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    purchase_intent_loss_factor: float = 2.0
    early_stop_patience: int = 5
    seed: int = 42
    out_dir: str = "./artifacts/predictor"
    wandb_enabled: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(
    pred: dict[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    head_weights: dict[str, float],
) -> torch.Tensor:
    total = torch.zeros((), device=targets.device, dtype=targets.dtype)
    denom = 0.0
    for i, name in enumerate(HEAD_NAMES):
        m = mask[:, i]
        if m.sum() == 0:
            continue
        diff = (pred[name] - targets[:, i]) ** 2 * m
        total = total + head_weights[name] * diff.sum()
        denom += float(head_weights[name] * m.sum())
    if denom == 0.0:
        return (pred[HEAD_NAMES[0]] * 0.0).sum()
    return total / denom


@torch.inference_mode()
def evaluate(
    model: SaleabilityPredictor, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    head_preds: dict[str, list[float]] = {n: [] for n in HEAD_NAMES}
    head_tgts: dict[str, list[float]] = {n: [] for n in HEAD_NAMES}

    for batch in loader:
        out = model(
            batch["image_emb"].to(device),
            batch["text_emb"].to(device),
            batch["occasion_idx"].to(device),
        )
        targets = batch["targets"].numpy()
        mask = batch["mask"].numpy()
        for i, name in enumerate(HEAD_NAMES):
            keep = mask[:, i].astype(bool)
            if keep.any():
                head_preds[name].extend(out[name].cpu().numpy()[keep].tolist())
                head_tgts[name].extend(targets[:, i][keep].tolist())

    metrics: dict[str, float] = {}
    for name in HEAD_NAMES:
        if len(head_preds[name]) >= 5:
            rho, _ = spearmanr(head_preds[name], head_tgts[name])
            metrics[f"spearman_{name}"] = float(rho or 0.0)
        else:
            metrics[f"spearman_{name}"] = float("nan")
    metrics["spearman_primary"] = metrics.get("spearman_purchase_intent", float("nan"))
    return metrics


def _log_mask_coverage(df: pd.DataFrame, split_name: str) -> None:
    """Log how many samples have labels per head. Confirms masking works."""
    from models.predictor.dataset import _build_targets

    n = len(df)
    counts = {name: 0 for name in HEAD_NAMES}
    for _, row in df.iterrows():
        _, mask = _build_targets(row)
        for i, name in enumerate(HEAD_NAMES):
            counts[name] += int(mask[i])
    parts = [f"{name}={counts[name]}/{n}" for name in HEAD_NAMES]
    log.info(f"[{split_name}] label coverage: {', '.join(parts)}")


def _train_once(
    *,
    cfg: TrainConfig,
    arch_cfg,
    splits: dict[str, pd.DataFrame],
    text_embedder,
    device: torch.device,
    out: Path,
    wandb_run: Any,
) -> tuple[dict[str, float], float]:
    """One training run. Returns (test metrics, best val score)."""
    set_seed(cfg.seed)

    train_ds = PredictorDataset(splits["train"], text_embedder=text_embedder)
    val_ds = PredictorDataset(splits["val"], text_embedder=text_embedder)
    test_ds = PredictorDataset(splits["test"], text_embedder=text_embedder)

    sampler = make_occasion_sampler(splits["train"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = SaleabilityPredictor(arch_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weights = head_loss_weights(cfg.purchase_intent_loss_factor)

    best_metric = -float("inf")
    best_state: dict | None = None
    epochs_since_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            pred = model(
                batch["image_emb"].to(device),
                batch["text_emb"].to(device),
                batch["occasion_idx"].to(device),
            )
            loss = masked_mse(pred, batch["targets"].to(device), batch["mask"].to(device), weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()

        train_loss = running / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device)
        log.info(
            f"  seed={cfg.seed} epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_pi_rho={val_metrics['spearman_purchase_intent']:.3f}"
        )
        if wandb_run is not None:
            wandb_run.log({"train_loss": train_loss, "epoch": epoch, **val_metrics})

        primary = val_metrics["spearman_primary"]
        if not np.isnan(primary) and primary > best_metric:
            best_metric = primary
            # Kept in memory: only the best seed's weights are written, so
            # writing every improvement of every seed would be wasted IO.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= cfg.early_stop_patience:
                log.info("  early stopping")
                break

    if best_state is None:
        raise SystemExit(
            "No checkpoint: validation Spearman was never finite. Check that the "
            "val split carries purchase_intent labels."
        )
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    torch.save(
        {
            "state_dict": best_state,
            "config": asdict(cfg),
            # Without this the sweep's trunk_hidden/head_hidden/dropout are
            # unrecoverable, and PredictorRunner rebuilds a default model whose
            # shapes do not match the saved weights.
            "arch": asdict(arch_cfg),
        },
        out / f"seed_{cfg.seed}.ckpt",
    )
    return test_metrics, best_metric


def _summarise(runs: list[dict[str, float]]) -> dict[str, float]:
    """Mean and sd per metric across seeds.

    A single run is not a measurement here: identical configurations have
    differed by 0.12 on a head. Reporting the spread is what makes a change
    distinguishable from noise.
    """
    out: dict[str, float] = {}
    for key in runs[0]:
        vals = np.array([r[key] for r in runs], dtype=float)
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            out[key], out[f"{key}_sd"] = float("nan"), float("nan")
            continue
        out[key] = float(vals.mean())
        out[f"{key}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    purchase_intent_loss_factor: float = 2.0,
    trunk_hidden: int = 512,
    head_hidden: int = 128,
    dropout: float = 0.1,
    occasion_emb_dim: int = 32,
    skip_connection: bool = False,
    input_norm: bool = False,
    early_stop_patience: int = 5,
    seed: int = 42,
    seeds: int = 1,
    out_dir: str = "./artifacts/predictor",
    wandb: bool = True,
    embed_text: bool = True,
) -> None:
    cfg = TrainConfig(
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay,
        purchase_intent_loss_factor=purchase_intent_loss_factor,
        early_stop_patience=early_stop_patience,
        seed=seed, out_dir=out_dir, wandb_enabled=wandb,
    )
    # Arch overrides so the sweep can vary model size
    from models.predictor.architecture import PredictorConfig
    arch_cfg = PredictorConfig(
        trunk_hidden=trunk_hidden,
        head_hidden=head_hidden,
        dropout=dropout,
        occasion_emb_dim=occasion_emb_dim,
        skip_connection=skip_connection,
        input_norm=input_norm,
    )

    if cfg.wandb_enabled and settings.wandb_api_key:
        import wandb as _wandb

        _wandb.init(project=settings.wandb_project, entity=settings.wandb_entity, config=asdict(cfg))
        wandb_run: Any = _wandb
    else:
        wandb_run = None

    df = load_training_frame()
    if df.empty:
        raise SystemExit("No training data available. Run scrapers + feature extraction first.")
    # The split is seeded separately from training, so every seed sees the same
    # cards; the spread then measures training variance rather than split luck.
    splits = split_by_seller(df, SplitConfig(seed=cfg.seed))

    # Feature width comes from the data, not a constant: a larger backbone or a
    # second one concatenated changes it, and the checkpoint records whatever
    # was used so inference rebuilds the matching shape.
    image_dim = len(df["clip_embedding"].iloc[0])
    if image_dim != arch_cfg.image_dim:
        log.info(f"Image features are {image_dim}-d; adjusting from {arch_cfg.image_dim}")
        arch_cfg = replace(arch_cfg, image_dim=image_dim)

    # Real text vectors, matching what rerank serves. Training on zeros would
    # leave the text block of the trunk untrained.
    text_embedder = None
    if embed_text:
        try:
            from data.features.clip_embed import CLIPEmbedder

            text_embedder = CLIPEmbedder().embed_texts
            log.info("Text embedder wired: SigLIP text tower (extracted_text → text_emb)")
        except Exception as e:
            log.warning(f"Text embedder unavailable ({e}); falling back to zero text_emb")

    _log_mask_coverage(splits["train"], "train")
    _log_mask_coverage(splits["val"], "val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, float]] = []
    best_val, best_seed = -float("inf"), cfg.seed
    for i in range(seeds):
        run_cfg = replace(cfg, seed=cfg.seed + i)
        log.info(f"--- seed {run_cfg.seed} ({i + 1}/{seeds}) ---")
        metrics, val_score = _train_once(
            cfg=run_cfg,
            arch_cfg=arch_cfg,
            splits=splits,
            text_embedder=text_embedder,
            device=device,
            out=out,
            wandb_run=wandb_run,
        )
        runs.append(metrics)
        if val_score > best_val:
            best_val, best_seed = val_score, run_cfg.seed

    # Downstream loads best.ckpt; point it at the seed that validated best.
    shutil.copyfile(out / f"seed_{best_seed}.ckpt", out / "best.ckpt")

    summary = _summarise(runs)
    if seeds > 1:
        log.info(f"Test metrics over {seeds} seeds (best val: seed {best_seed}):")
        for name in HEAD_NAMES:
            k = f"spearman_{name}"
            log.info(f"  {name:22s} {summary[k]:.3f} ± {summary[f'{k}_sd']:.3f}")
    else:
        log.info(f"Test metrics: {runs[0]}")

    (out / "test_metrics.json").write_text(
        json.dumps({**summary, "seeds": seeds, "per_seed": runs}, indent=2)
    )
    if wandb_run is not None:
        wandb_run.log({f"test_{k}": v for k, v in summary.items()})
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(train)
