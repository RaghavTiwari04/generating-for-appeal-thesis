"""Predictor learning curves, parsed from the training job's own log.

The claim these support is in Section~\\ref{sec:r-predictor}: training loss falls
below 0.001 within about fifty epochs while validation rank correlation
plateaus, so the limit is generalisation rather than capacity. That is the
argument for preferring a linear probe, and it was text-only.

The data is already on disk. `models.predictor.train` prints a progress line
every LOG_EVERY = 50 epochs:

    seed=42 epoch=0150 train_loss=0.0009 val_pi_rho=0.548

so the SLURM log of the training job carries a sampled curve per seed and no
W&B export is needed. Per-epoch history exists in W&B too, but needs an API key
and adds nothing at this resolution.

    python -m eval.reports.training_curves logs/slurm-*-predictor.out
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("report/figures")

LINE = re.compile(
    r"seed=(?P<seed>\d+)\s+epoch=(?P<epoch>\d+)\s+"
    r"train_loss=(?P<loss>[\d.eE+-]+)\s+val_pi_rho=(?P<rho>[-\d.eE+]+)"
)


def parse(paths: list[Path]) -> dict[int, list[tuple[int, float, float]]]:
    runs: dict[int, list[tuple[int, float, float]]] = {}
    for p in paths:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = LINE.search(line)
            if not m:
                continue
            runs.setdefault(int(m["seed"]), []).append(
                (int(m["epoch"]), float(m["loss"]), float(m["rho"]))
            )
    for seed in runs:
        runs[seed].sort()
    return runs


def plot(runs: dict[int, list[tuple[int, float, float]]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for seed, rows in sorted(runs.items()):
        ep = [r[0] for r in rows]
        ax1.plot(ep, [r[1] for r in rows], lw=1.3, alpha=0.85, label=f"seed {seed}")
        ax2.plot(ep, [r[2] for r in rows], lw=1.3, alpha=0.85)

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.axhline(1e-3, color="0.45", ls="--", lw=0.9)
    ax1.text(ax1.get_xlim()[1], 1.1e-3, "$10^{-3}$ ", ha="right", va="bottom",
             fontsize=7.5, color="0.4")
    ax1.grid(alpha=0.3)
    ax1.set_axisbelow(True)
    if len(runs) > 1:
        ax1.legend(fontsize=7, frameon=False, ncol=2)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Validation Spearman $\rho$ (purchase intent)")
    ax2.grid(alpha=0.3)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'training_curves.pdf'}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        args = [str(p) for p in Path("logs").glob("slurm-*-predictor.out")]
    paths = [Path(a) for a in args if Path(a).exists()]
    if not paths:
        raise SystemExit("no training logs found; pass the slurm .out path")

    runs = parse(paths)
    if not runs:
        raise SystemExit(
            f"no progress lines matched in {[p.name for p in paths]}. "
            "Expected lines like 'seed=42 epoch=0150 train_loss=... val_pi_rho=...'"
        )
    for seed, rows in sorted(runs.items()):
        first_below = next((e for e, loss, _ in rows if loss < 1e-3), None)
        best = max(r[2] for r in rows)
        print(f"  seed {seed}: {len(rows)} points, epochs {rows[0][0]}-{rows[-1][0]}, "
              f"loss<1e-3 from epoch {first_below}, best val rho {best:.3f}")
    plot(runs)


if __name__ == "__main__":
    main()
