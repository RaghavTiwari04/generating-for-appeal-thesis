"""Train the multi-head saleability predictor.

Loss: weighted MSE per head with the saleability head weighted 2x. Mask
out heads with no ground truth on a given sample.

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
import torch
import typer
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from common.config import settings
from common.logging import get_logger
from models.predictor.architecture import HEAD_NAMES, PredictorConfig, SaleabilityPredictor, head_loss_weights
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
    saleability_loss_factor: float = 2.0
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
    total = pred[HEAD_NAMES[0]].new_zeros(())
    denom = pred[HEAD_NAMES[0]].new_zeros(())
    for i, name in enumerate(HEAD_NAMES):
        m = mask[:, i]
        if m.sum() == 0:
            continue
        diff = (pred[name] - targets[:, i]) ** 2 * m
        total = total + head_weights[name] * diff.sum()
        denom = denom + head_weights[name] * m.sum()
    return total / denom.clamp_min(1.0)


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
            batch["price_rel"].to(device),
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
    metrics["spearman_primary"] = metrics.get("spearman_saleability", float("nan"))
    return metrics


def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    saleability_loss_factor: float = 2.0,
    trunk_hidden: int = 512,
    head_hidden: int = 128,
    dropout: float = 0.1,
    occasion_emb_dim: int = 32,
    seed: int = 42,
    out_dir: str = "./artifacts/predictor",
    wandb: bool = True,
) -> None:
    cfg = TrainConfig(
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay,
        saleability_loss_factor=saleability_loss_factor,
        seed=seed, out_dir=out_dir, wandb_enabled=wandb,
    )
    # Apply arch overrides so W&B sweep can vary model size
    from models.predictor.architecture import PredictorConfig as _PC
    _arch_cfg = _PC(
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

    train_ds = PredictorDataset(splits["train"])
    val_ds = PredictorDataset(splits["val"])
    test_ds = PredictorDataset(splits["test"])

    sampler = make_occasion_sampler(splits["train"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SaleabilityPredictor(_arch_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weights = head_loss_weights(cfg.saleability_loss_factor)

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
                batch["price_rel"].to(device),
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
            f"val_saleability_rho={val_metrics['spearman_saleability']:.3f}"
        )
        if wandb_run is not None:
            wandb_run.log({"train_loss": train_loss, "epoch": epoch, **val_metrics})

        primary = val_metrics["spearman_primary"]
        if not np.isnan(primary) and primary > best_metric:
            best_metric = primary
            epochs_since_improvement = 0
            torch.save(
                {"state_dict": model.state_dict(), "config": asdict(cfg)},
                out / "best.ckpt",
            )
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= cfg.early_stop_patience:
                log.info("Early stopping triggered")
                break

    # Final test eval
    state = torch.load(out / "best.ckpt", map_location=device)
    model.load_state_dict(state["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    log.info(f"Test metrics: {test_metrics}")
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    if wandb_run is not None:
        wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(train)
