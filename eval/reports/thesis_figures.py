"""Figures for the dissertation, generated from the evaluation's own ratings.

Reads `raw_ratings.csv` as written by `eval.llm_system_eval` and writes PDFs to
`report/figures/`. Everything plotted here is measured; nothing is illustrative.

    python -m eval.reports.thesis_figures

Condition A note. The ratings file retains the 40 superseded A cards scored
before the naive-baseline headline was corrected, because the cache is keyed on
card_key and those rows were never deleted. They are dropped here by taking the
last 40 A rows, which the incremental cache appends. The check is arithmetic
rather than positional faith: the superseded set means 0.407 and the corrected
set 0.624, and the loader asserts the split reproduces those.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMS = ["purchase_intent", "occasion_fit", "aesthetic", "emotional_resonance", "distinctiveness"]
DIM_LABEL = {
    "purchase_intent": "Purchase intent",
    "occasion_fit": "Occasion fit",
    "aesthetic": "Aesthetic",
    "emotional_resonance": "Emotional resonance",
    "distinctiveness": "Distinctiveness",
}
COND_LABEL = {
    "A_naive_ai": "A: naive AI",
    "B_pipeline_no_rerank": "B: pipeline",
    "C_pipeline_rerank": "C: pipeline + rerank",
    "D_human_reference": "D: human reference",
    "D_human_bestseller": "D: human reference",
}
ORDER = ["A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank", "D_human_bestseller"]
COLOUR = {
    "A_naive_ai": "#9ca3af",
    "B_pipeline_no_rerank": "#60a5fa",
    "C_pipeline_rerank": "#2563eb",
    "D_human_bestseller": "#16a34a",
}
DELTA = 0.02

OUT = Path("report/figures")


def load(path: str = "artifacts/llm_system_eval/raw_ratings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    a = df[df.condition == "A_naive_ai"]
    if len(a) > 40:
        superseded, current = a.head(len(a) - 40), a.tail(40)
        assert abs(superseded.purchase_intent.mean() - 0.407) < 0.01, "unexpected superseded A mean"
        assert abs(current.purchase_intent.mean() - 0.624) < 0.01, "unexpected corrected A mean"
        df = pd.concat([df[df.condition != "A_naive_ai"], current], ignore_index=True)
    return df


def _boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _hodges_lehmann(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(a[:, None] - b[None, :]))


def fig_condition_means(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, cond in enumerate(ORDER):
        x = df[df.condition == cond].purchase_intent.to_numpy()
        lo, hi = _boot_ci(x)
        ax.errorbar(i, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]],
                    fmt="o", ms=7, capsize=5, lw=1.6, color=COLOUR[cond])
        ax.annotate(f"{x.mean():.3f}", (i, x.mean()), textcoords="offset points",
                    xytext=(11, -3), fontsize=9)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([COND_LABEL[c] for c in ORDER], fontsize=9)
    ax.set_ylabel("Purchase intent")
    ax.set_ylim(0.55, 0.76)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "condition_means.pdf")
    plt.close(fig)


def fig_dimension_means(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    width = 0.2
    idx = np.arange(len(DIMS))
    for j, cond in enumerate(ORDER):
        sub = df[df.condition == cond]
        vals = [sub[d].mean() for d in DIMS]
        ax.bar(idx + (j - 1.5) * width, vals, width,
               label=COND_LABEL[cond], color=COLOUR[cond])
    ax.set_xticks(idx)
    ax.set_xticklabels([DIM_LABEL[d] for d in DIMS], fontsize=8.5)
    ax.set_ylabel("Mean judge score")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "dimension_means.pdf")
    plt.close(fig)


def fig_equivalence(df: pd.DataFrame) -> None:
    """Hodges-Lehmann shift per contrast against the equivalence margin."""
    pairs = [
        ("B_pipeline_no_rerank", "D_human_bestseller", "B vs D"),
        ("B_pipeline_no_rerank", "C_pipeline_rerank", "B vs C"),
        ("C_pipeline_rerank", "D_human_bestseller", "C vs D"),
        ("A_naive_ai", "B_pipeline_no_rerank", "A vs B"),
        ("A_naive_ai", "C_pipeline_rerank", "A vs C"),
        ("A_naive_ai", "D_human_bestseller", "A vs D"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axvspan(-DELTA, DELTA, color="#16a34a", alpha=0.12, zorder=0)
    ax.axvline(0, color="#444", lw=0.8, zorder=1)
    for i, (ca, cb, _label) in enumerate(pairs):
        a = df[df.condition == ca].purchase_intent.to_numpy()
        b = df[df.condition == cb].purchase_intent.to_numpy()
        hl = _hodges_lehmann(a, b)
        inside = abs(hl) <= DELTA
        ax.plot(hl, i, "o", ms=7,
                color="#16a34a" if inside else "#dc2626", zorder=3)
        ax.annotate(f"{hl:+.4f}", (hl, i), textcoords="offset points",
                    xytext=(9, -3), fontsize=8)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p[2] for p in pairs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"Hodges--Lehmann shift in purchase intent (shaded: $\pm\delta = 0.02$)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "equivalence.pdf")
    plt.close(fig)


def fig_distributions(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    data = [df[df.condition == c].purchase_intent.to_numpy() for c in ORDER]
    parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.8)
    for body, cond in zip(parts["bodies"], ORDER, strict=True):
        body.set_facecolor(COLOUR[cond])
        body.set_alpha(0.45)
    rng = np.random.default_rng(0)
    for i, (x, cond) in enumerate(zip(data, ORDER, strict=True), start=1):
        ax.scatter(i + rng.uniform(-0.07, 0.07, len(x)), x, s=7,
                   color=COLOUR[cond], alpha=0.75, linewidths=0)
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels([COND_LABEL[c] for c in ORDER], fontsize=9)
    ax.set_ylabel("Purchase intent")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "score_distributions.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    counts = df.groupby("condition").size().to_dict()
    print("cards per condition:", counts)
    fig_condition_means(df)
    fig_dimension_means(df)
    fig_equivalence(df)
    fig_distributions(df)
    print(f"wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
