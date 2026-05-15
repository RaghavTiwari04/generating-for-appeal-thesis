"""Publication-quality figure generation for thesis evaluation chapters.

All figures follow a consistent style (single-column NeurIPS-ish, 300 DPI,
no chartjunk). Call `generate_all(out_dir)` to produce every figure, or
individual functions for specific plots.

Figures produced:
  fig1_condition_means.pdf    — Main result: mean purchase intent ± 95% CI per condition
  fig2_per_occasion.pdf       — Per-occasion breakdown (conditions as lines)
  fig3_best_of_n.pdf          — Best-of-N saturation curve
  fig4_reliability.pdf        — Predictor calibration reliability diagram
  fig5_per_head_spearman.pdf  — Per-head Spearman ρ bar chart
  fig6_ablation_heatmap.pdf   — Ablation results table as heatmap
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

CONDITION_LABELS = {
    "A_naive_ai":             "A: Naive AI",
    "B_pipeline_no_rerank":   "B: Pipeline",
    "C_pipeline_rerank":      "C: Pipeline+Rerank",
    "D_human_bestseller":     "D: Human bestsellers",
}
CONDITION_COLOURS = {
    "A_naive_ai":             "#9ca3af",
    "B_pipeline_no_rerank":   "#60a5fa",
    "C_pipeline_rerank":      "#2563eb",
    "D_human_bestseller":     "#16a34a",
}
CONDITIONS_ORDER = list(CONDITION_LABELS.keys())


def _fig_ax(w: float = 3.5, h: float = 2.8) -> tuple[Any, Any]:
    return plt.subplots(figsize=(w, h))


# ---------------------------------------------------------------------------
# Fig 1: Condition means ± 95% CI
# ---------------------------------------------------------------------------

def fig1_condition_means(
    means: dict[str, float],
    stderr: dict[str, float],
    out: Path,
) -> None:
    conds = [c for c in CONDITIONS_ORDER if c in means]
    mu = [means[c] for c in conds]
    se = [stderr.get(c, 0.0) * 1.96 for c in conds]
    labels = [CONDITION_LABELS.get(c, c) for c in conds]
    colours = [CONDITION_COLOURS.get(c, "#888") for c in conds]

    fig, ax = _fig_ax(3.5, 2.8)
    x = np.arange(len(conds))
    bars = ax.bar(x, mu, color=colours, width=0.55, zorder=3)
    ax.errorbar(x, mu, yerr=se, fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean purchase intent (1–7)")
    ax.set_ylim(1, 7)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.axhline(4.0, ls="--", color="#6b7280", linewidth=0.8, label="Neutral (4)")
    ax.set_title("Purchase intent by condition")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: Per-occasion breakdown
# ---------------------------------------------------------------------------

def fig2_per_occasion(per_occasion: pd.DataFrame, out: Path) -> None:
    # per_occasion: index=occasion, columns=condition_tags
    conds = [c for c in CONDITIONS_ORDER if c in per_occasion.columns]
    occs = per_occasion.index.tolist()

    fig, ax = _fig_ax(5.0, 3.5)
    x = np.arange(len(occs))
    width = 0.18
    for i, cond in enumerate(conds):
        vals = per_occasion[cond].reindex(occs).fillna(0).to_numpy()
        ax.bar(x + i * width, vals, width=width,
               color=CONDITION_COLOURS.get(cond, "#888"),
               label=CONDITION_LABELS.get(cond, cond),
               zorder=3)
    ax.set_xticks(x + width * (len(conds) - 1) / 2)
    occ_labels = [o.replace("/", "\n").replace("_", " ") for o in occs]
    ax.set_xticklabels(occ_labels, fontsize=7)
    ax.set_ylabel("Mean purchase intent (1–7)")
    ax.set_ylim(1, 7)
    ax.set_title("Purchase intent by occasion and condition")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: Best-of-N saturation curve
# ---------------------------------------------------------------------------

def fig3_best_of_n(curve_data: list[dict], out: Path) -> None:
    ns = [d["n"] for d in curve_data]
    scores = [d["score"] for d in curve_data]

    fig, ax = _fig_ax(3.0, 2.5)
    ax.plot(ns, scores, "o-", color="#2563eb", linewidth=1.5, markersize=5, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("N (candidates)")
    ax.set_ylabel("Mean saleability (calibrated)")
    ax.set_title("Best-of-N saturation curve")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: Reliability diagram (calibration)
# ---------------------------------------------------------------------------

def fig4_reliability(calibration_json: dict, out: Path) -> None:
    fig, ax = _fig_ax(3.0, 3.0)
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=0.8, label="Perfect calibration")
    ax.plot(
        calibration_json["bin_predicted"],
        calibration_json["bin_observed"],
        "o-", color="#2563eb", linewidth=1.5, markersize=5, zorder=3,
        label=f"Model (ECE={calibration_json['ece']:.3f})",
    )
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Mean observed score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram — saleability head")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5: Per-head Spearman ρ bar chart
# ---------------------------------------------------------------------------

def fig5_per_head_spearman(per_head: dict[str, float], baselines: dict[str, float], out: Path) -> None:
    heads = list(per_head.keys())
    rhos = list(per_head.values())
    colours = ["#2563eb" if not np.isnan(r) else "#d1d5db" for r in rhos]

    fig, ax = _fig_ax(4.0, 2.5)
    x = np.arange(len(heads))
    ax.bar(x, [r if not np.isnan(r) else 0 for r in rhos], color=colours, width=0.55, zorder=3)

    if "random_spearman" in baselines:
        ax.axhline(baselines["random_spearman"], ls=":", color="#6b7280", linewidth=1,
                   label=f"Random (ρ={baselines['random_spearman']:.2f})")
    if "ridge_handcrafted_spearman" in baselines:
        ax.axhline(baselines["ridge_handcrafted_spearman"], ls="--", color="#f59e0b", linewidth=1,
                   label=f"Ridge baseline (ρ={baselines['ridge_handcrafted_spearman']:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels([h.replace("_", "\n") for h in heads], fontsize=8)
    ax.set_ylabel("Spearman ρ vs survey")
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Per-head Spearman ρ")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_all(
    artifacts_dir: str | Path = "./artifacts",
    out_dir: str | Path = "./artifacts/figures",
) -> None:
    arts = Path(artifacts_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # System eval report
    sys_report_path = arts / "system_eval" / "report.json"
    if sys_report_path.exists():
        sys_report = json.loads(sys_report_path.read_text())
        fig1_condition_means(
            sys_report["condition_means"],
            sys_report["condition_stderr"],
            out / "fig1_condition_means.pdf",
        )
        print("fig1 ✓")

        per_occ_path = arts / "system_eval" / "per_occasion.csv"
        if per_occ_path.exists():
            per_occ = pd.read_csv(per_occ_path, index_col=0)
            fig2_per_occasion(per_occ, out / "fig2_per_occasion.pdf")
            print("fig2 ✓")

    # Best-of-N
    bon_path = arts / "ablations" / "best_of_n_curve.json"
    if bon_path.exists():
        curve = json.loads(bon_path.read_text())
        fig3_best_of_n(curve, out / "fig3_best_of_n.pdf")
        print("fig3 ✓")

    # Calibration
    cal_path = arts / "predictor" / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        fig4_reliability(cal, out / "fig4_reliability.pdf")
        print("fig4 ✓")

    # Predictor eval
    pred_path = arts / "predictor_eval" / "report.json"
    if pred_path.exists():
        pred = json.loads(pred_path.read_text())
        fig5_per_head_spearman(
            pred["per_head_spearman"],
            pred.get("baselines", {}),
            out / "fig5_per_head_spearman.pdf",
        )
        print("fig5 ✓")

    print(f"\nFigures saved to {out}")


if __name__ == "__main__":
    import typer
    typer.run(generate_all)
