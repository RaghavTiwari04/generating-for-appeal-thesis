"""Train the multi-head saleability predictor.

Loss: masked weighted MSE per head. All five heads are supervised by the VLM
labels — the rubric judge for the quality dimensions, SSR for purchase intent
— so all five see the same cards. The purchase-intent head is upweighted
because it is the head the pipeline ranks on, not because it has fewer labels.

Masked multi-task loss (Ruder 2017): mask[head][sample] = 1 where a label
exists, 0 otherwise, so a dimension the judge failed to return contributes no
gradient rather than an imputed target.

Optimiser: AdamW with a cosine schedule, backbone frozen — only the trunk and
heads train, since features are pre-cached.

The defaults are the measured best. 1,792 training cards at batch 64 is 28
steps an epoch, so the epoch count is really a step budget: the original 30
epochs at lr 1e-4 was ~840 steps and reached 0.510 on purchase intent, while
1500 at lr 1e-2 reaches 0.586. Shrinking the model or adding regularisation
made every head worse, so the constraint was training length, not capacity.

Split: by `seller_id` 70/15/15. Sampler: weighted by occasion.

Usage:
    python -m models.predictor.train --seeds 5
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
    PredictorConfig,
    SaleabilityPredictor,
    head_loss_weights,
)
from models.predictor.dataset import (
    PredictorDataset,
    SplitConfig,
    _build_targets,
    load_training_frame,
    make_occasion_sampler,
    split_by_seller,
)

log = get_logger(__name__)

LOG_EVERY = 50  # epochs between console progress lines


@dataclass
class TrainConfig:
    epochs: int = 1500
    batch_size: int = 64
    lr: float = 1e-2
    weight_decay: float = 1e-2
    purchase_intent_loss_factor: float = 2.0
    early_stop_patience: int = 150
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
    """Weighted mean squared error over the labelled entries only.

    Computed across heads at once. The per-head loop this replaces called
    `.sum()` and `float()` on device tensors twice per head per batch, and each
    of those forces a GPU synchronisation — around 200k of them over a
    1500-epoch run whose actual arithmetic is trivial.
    """
    preds = torch.stack([pred[name] for name in HEAD_NAMES], dim=1)
    weights = torch.as_tensor(
        [head_weights[name] for name in HEAD_NAMES],
        device=targets.device,
        dtype=targets.dtype,
    )
    weighted_mask = mask * weights
    total = ((preds - targets) ** 2 * weighted_mask).sum()
    # Every weight is positive, so the denominator is zero only when nothing in
    # the batch is labelled — and then the numerator is zero too, giving a real
    # zero loss that still carries a gradient path.
    return total / weighted_mask.sum().clamp(min=1e-12)


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
    """Per-head label counts, so a head silently training on nothing is visible."""
    counts = np.sum([_build_targets(row)[1] for _, row in df.iterrows()], axis=0)
    parts = [f"{name}={int(c)}/{len(df)}" for name, c in zip(HEAD_NAMES, counts, strict=True)]
    log.info(f"[{split_name}] label coverage: {', '.join(parts)}")


def _train_once(
    *,
    cfg: TrainConfig,
    arch_cfg: PredictorConfig,
    loaders: dict[str, DataLoader],
    device: torch.device,
    out: Path,
    wandb_run: Any,
) -> tuple[dict[str, float], float]:
    """One training run. Returns (test metrics, best val score).

    Loaders are built once by the caller: they hold no per-seed state, and
    rebuilding them would re-run the text encoder over every split for each
    seed. Seeding here still varies the sampler, which draws at iteration time.
    """
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = (
        loaders["train"], loaders["val"], loaders["test"]
    )

    model = SaleabilityPredictor(arch_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weights = head_loss_weights(cfg.purchase_intent_loss_factor)

    best_metric = -float("inf")
    best_state: dict | None = None
    epochs_since_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        # Accumulated on device and read once per epoch; `.item()` per batch
        # would synchronise ~42k times over a full run for a logged average.
        running = torch.zeros((), device=device)
        for batch in train_loader:
            pred = model(
                batch["image_emb"].to(device),
                batch["text_emb"].to(device),
                batch["occasion_idx"].to(device),
            )
            loss = masked_mse(pred, batch["targets"].to(device), batch["mask"].to(device), weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.detach()
        sched.step()

        train_loss = float(running) / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)
        # wandb gets every epoch; the console gets a sample, because 1500 epochs
        # times five seeds is 7,500 lines to scroll through in a SLURM log.
        if epoch % LOG_EVERY == 0 or epoch == cfg.epochs - 1:
            log.info(
                f"  seed={cfg.seed} epoch={epoch:04d} train_loss={train_loss:.4f} "
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
                log.info(
                    f"  seed={cfg.seed} early stop at epoch {epoch} "
                    f"(best val_pi_rho={best_metric:.3f})"
                )
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

    A single run is not a measurement: seed-to-seed sd on purchase intent is
    about 0.013, so a difference smaller than roughly 0.03 between two configs
    is noise. Reporting the spread is what makes that judgeable.
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
    epochs: int = 1500,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 1e-2,
    purchase_intent_loss_factor: float = 2.0,
    trunk_hidden: int = 512,
    head_hidden: int = 128,
    dropout: float = 0.1,
    occasion_emb_dim: int = 32,
    skip_connection: bool = False,
    input_norm: bool = False,
    early_stop_patience: int = 150,
    seed: int = 42,
    seeds: int = 5,
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
            log.info("Text embedder wired: SigLIP text tower (extracted_text -> text_emb)")
        except Exception as e:
            log.warning(f"Text embedder unavailable ({e}); falling back to zero text_emb")

    _log_mask_coverage(splits["train"], "train")
    _log_mask_coverage(splits["val"], "val")

    # Built once, not per seed: the text encoder runs over every split at
    # construction, and the splits do not change between seeds.
    datasets = {
        name: PredictorDataset(rows, text_embedder=text_embedder)
        for name, rows in splits.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            sampler=make_occasion_sampler(splits["train"]),
        ),
        "val": DataLoader(datasets["val"], batch_size=cfg.batch_size),
        "test": DataLoader(datasets["test"], batch_size=cfg.batch_size),
    }

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
            loaders=loaders,
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
            log.info(f"  {name:22s} {summary[k]:.3f} +/- {summary[f'{k}_sd']:.3f}")
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
