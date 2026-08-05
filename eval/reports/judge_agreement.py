"""Cross-judge agreement over the evaluated cards.

The robustness runs give three independent scorings of the same 159 cards, so
the disagreement between judges can be measured rather than cited. Two questions
matter and they have different answers:

  per-card    do the judges order individual cards the same way?
  per-condition  do they order the four conditions the same way?

A judge set can agree well on the second while agreeing poorly on the first, and
it is the second that the thesis's claims rest on.

The original run's ratings are keyed on card_key and the robustness runs on
cover_path, with no mapping between them outside the database, so the original
judge enters this comparison at condition level only.

    python -m eval.reports.judge_agreement
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.logging import get_logger

log = get_logger(__name__)

RATINGS_DIR = Path("artifacts/judge_robustness")
ORIGINAL = Path("artifacts/llm_system_eval/raw_ratings.csv")
OUT = Path("report/figures")
ORDER = ["A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank", "D_human_reference"]
SHORT = {"A_naive_ai": "A", "B_pipeline_no_rerank": "B",
         "C_pipeline_rerank": "C", "D_human_reference": "D"}


def _load_original() -> pd.DataFrame:
    from eval.reports.thesis_figures import load

    df = load(str(ORIGINAL))
    df["condition"] = df.condition.replace({"D_human_bestseller": "D_human_reference"})
    return df


def load_all() -> dict[str, pd.DataFrame]:
    judges: dict[str, pd.DataFrame] = {}
    for path in sorted(RATINGS_DIR.glob("ratings_*.csv")):
        name = path.stem.replace("ratings_", "")
        judges[name] = pd.read_csv(path)
    judges["original"] = _load_original()
    return judges


def per_card_agreement(judges: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spearman between judges over individual cards, joined on cover_path."""
    keyed = {
        n: d.set_index("cover_path").purchase_intent
        for n, d in judges.items()
        if "cover_path" in d.columns
    }
    rows = []
    for a, b in itertools.combinations(sorted(keyed), 2):
        joined = pd.concat([keyed[a].rename("a"), keyed[b].rename("b")], axis=1).dropna()
        rho, p = spearmanr(joined.a, joined.b)
        rows.append({"judge_a": a, "judge_b": b, "n": len(joined),
                     "spearman": round(float(rho), 3), "p": float(p)})
    return pd.DataFrame(rows)


def per_condition_means(judges: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, d in judges.items():
        m = d.dropna(subset=["purchase_intent"]).groupby("condition").purchase_intent.mean()
        rows[name] = {SHORT.get(c, c): round(v, 4) for c, v in m.items()}
    out = pd.DataFrame(rows).T[["A", "B", "C", "D"]]
    out["ordering"] = [
        " < ".join(r.sort_values().index) for _, r in out[["A", "B", "C", "D"]].iterrows()
    ]
    return out


def condition_rank_agreement(means: pd.DataFrame) -> pd.DataFrame:
    """Spearman between judges over the four condition means."""
    cols = ["A", "B", "C", "D"]
    rows = []
    for a, b in itertools.combinations(means.index, 2):
        rho, _ = spearmanr(means.loc[a, cols].astype(float), means.loc[b, cols].astype(float))
        rows.append({"judge_a": a, "judge_b": b, "spearman_over_4_conditions": round(float(rho), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    judges = load_all()
    for n, d in judges.items():
        log.info(f"{n}: {len(d)} cards")

    means = per_condition_means(judges)
    print("\n=== condition means by judge ===")
    print(means.to_string())

    print("\n=== per-card agreement (robustness runs only) ===")
    pc = per_card_agreement(judges)
    print(pc.to_string(index=False))

    print("\n=== agreement over the four condition means ===")
    print(condition_rank_agreement(means).to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(4)
    for judge in means.index:
        ax.plot(x, means.loc[judge, ["A", "B", "C", "D"]].astype(float),
                marker="o", ms=5, linewidth=1.7, label=judge)
    ax.set_xticks(x)
    ax.set_xticklabels(["A: naive", "B: pipeline", "C: +rerank", "D: human"], fontsize=9)
    ax.set_ylabel("Mean purchase intent")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "judge_agreement.pdf")
    plt.close(fig)
    print(f"\nwrote {OUT / 'judge_agreement.pdf'}")


if __name__ == "__main__":
    main()
