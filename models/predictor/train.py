"""Train the multi-head saleability predictor.

Loss: masked weighted MSE per head. Heads 1-4 (VLM-labelled) train on
all ~2,377 cards. Head 5 (purchase_intent, human BT) trains only on
the ~500-card subsample. Purchase-intent head weighted 2× to compensate
for smaller label set.

This is the standard masked multi-task loss approach (Ruder 2017):
mask[head][sample] = 1 where label exists, 0 otherwise. Zero-masked
samples contribute zero gradient to that head.

Optimiser: AdamW, lr 1e-4, weight decay 1e-2, cosine schedule.
Backbone frozen; only the trunk + heads train (features are pre-cached).

Split: by `seller_id` 70/15/15. Sampler: weighted by occasion.

Usage:
    python -m models.predictor.train --epochs 30 --batch-size 64
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
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
    seed: int = 42,
    out_dir: str = "./artifacts/predictor",
    wandb: bool = True,
    embed_text: bool = True,
) -> None:
    cfg = TrainConfig(
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay,
        purchase_intent_loss_factor=purchase_intent_loss_factor,
        seed=seed, out_dir=out_dir, wandb_enabled=wandb,
    )
    # Apply arch overrides so W&B sweep can vary model size
    from models.predictor.architecture import PredictorConfig
    _arch_cfg = PredictorConfig(
        trunk_hidden=trunk_hidden,
        head_hidden=head_hidden,
        dropout=dropout,
        occasion_emb_dim=occasion_emb_dim,
    )
    set_seed(cfg.seed)

    if cfg.wandb_enabled and settings.wandb_api_key:
        import wandb as _wandb

        _wandb.init(project=settings.wandb_project, entity=settings.wandb_entity, config=asdict(cfg))
        wandb_run: Any = _wandb
    else:
        wandb_run = None

    df = load_training_frame()
    if df.empty:
        raise SystemExit("No training data available. Run scrapers + feature extraction first.")
    splits = split_by_seller(df, SplitConfig(seed=cfg.seed))

    # Wire a text embedder so text_emb is real (not zeros). The predictor is
    # served real SigLIP text vectors at rerank time (pipeline/rerank.py), so
    # training on zeros would create a train/serve mismatch and leave the text
    # block of the trunk untrained. `embed_texts` L2-normalises to match rerank.
    text_embedder = None
    if embed_text:
        try:
            from data.features.clip_embed import CLIPEmbedder

            _embedder = CLIPEmbedder()
            text_embedder = _embedder.embed_texts
            log.info("Text embedder wired: SigLIP text tower (extracted_text → text_emb)")
        except Exception as e:
            log.warning(f"Text embedder unavailable ({e}); falling back to zero text_emb")

    train_ds = PredictorDataset(splits["train"], text_embedder=text_embedder)
    val_ds = PredictorDataset(splits["val"], text_embedder=text_embedder)
    test_ds = PredictorDataset(splits["test"], text_embedder=text_embedder)

    # Log label coverage per head — confirms masked multi-task setup
    _log_mask_coverage(splits["train"], "train")
    _log_mask_coverage(splits["val"], "val")

    sampler = make_occasion_sampler(splits["train"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SaleabilityPredictor(_arch_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weights = head_loss_weights(cfg.purchase_intent_loss_factor)

    best_metric = -float("inf")
    epochs_since_improvement = 0
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_batches = 0
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
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_pi_rho={val_metrics['spearman_purchase_intent']:.3f}"
        )
        if wandb_run is not None:
            wandb_run.log({"train_loss": train_loss, "epoch": epoch, **val_metrics})

        primary = val_metrics["spearman_primary"]
        if not np.isnan(primary) and primary > best_metric:
            best_metric = primary
            epochs_since_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    # Without this the sweep's trunk_hidden/head_hidden/dropout
                    # are unrecoverable, and PredictorRunner rebuilds a default
                    # model whose shapes do not match the saved weights.
                    "arch": asdict(_arch_cfg),
                },
                out / "best.ckpt",
            )
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= cfg.early_stop_patience:
                log.info("Early stopping triggered")
                break

    # Final test eval
    ckpt = out / "best.ckpt"
    if not ckpt.exists():
        # Only written when val Spearman improves, so an all-NaN primary metric
        # (no purchase_intent labels in val) leaves nothing to load.
        raise SystemExit(
            "No checkpoint written: validation Spearman was never finite. "
            "Check that the val split carries purchase_intent labels."
        )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    log.info(f"Test metrics: {test_metrics}")
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    if wandb_run is not None:
        wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(train)
