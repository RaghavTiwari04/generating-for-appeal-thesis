"""Two explanatory figures for the methods chapter.

Drawn with matplotlib rather than TikZ so they can be rendered and inspected
before they reach the document, and so the LaTeX carries no drawing code that
has not been executed.

  corpus_funnel.pdf   the corpus from scrape to training split, in one view.
                      Those numbers currently sit across two tables and three
                      paragraphs, which is where the arithmetic is easiest to
                      lose track of.
  ssr_pipeline.pdf    how purchase intent is produced. It is the primary
                      outcome measure and the only one that never asks the
                      model for a number, so it is worth showing.

    python -m eval.reports.methods_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = Path("report/figures")

# Measured by scripts/verify_corpus_numbers.py against the delivered database.
FUNNEL = [
    ("Scraped listings", 3906, "Greetings Island 2,033 + Redbubble 1,873"),
    ("With a birthday subtype", 3491, "415 left unclassified"),
    ("Distinct designs", 2795, "1,111 redundant copies collapsed"),
    ("Labelled by the judge", 2468, "one representative per cluster"),
    ("Predictor training split", 1727, "plus 370 validation, remainder test"),
]

BLUE = "#2563eb"


def fig_corpus_funnel() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    top = FUNNEL[0][1]
    for i, (label, n, note) in enumerate(FUNNEL):
        frac = n / top
        half = frac / 2
        y = len(FUNNEL) - 1 - i
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.5 - half, y - 0.32), frac, 0.64,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor=BLUE, alpha=0.25 + 0.13 * i))
        ax.text(0.5, y + 0.06, f"{label}: {n:,}", ha="center", va="center",
                fontsize=9.5, weight="bold")
        ax.text(0.5, y - 0.17, note, ha="center", va="center",
                fontsize=7.5, color="0.35")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(FUNNEL) - 0.4)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "corpus_funnel.pdf")
    plt.close(fig)
    print("wrote corpus_funnel.pdf")


def fig_ssr_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    steps = [
        ("Card image\n+ occasion", "#e5e7eb"),
        ("3 personas\n$\\times$ 2 samples", "#dbeafe"),
        ("Free-text\nreplies", "#dbeafe"),
        ("Embed each\nreply", "#bfdbfe"),
        ("Cosine vs six\nanchor sets", "#bfdbfe"),
        ("PMF over\n5-point scale", "#93c5fd"),
        ("Expectation\n$\\rightarrow$ score", "#60a5fa"),
    ]
    w, gap = 0.116, 0.026
    for i, (label, colour) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.34), w, 0.34, boxstyle="round,pad=0.012",
            facecolor=colour, edgecolor="0.45", linewidth=0.8))
        ax.text(x + w / 2, 0.51, label, ha="center", va="center", fontsize=7.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap, 0.51), xytext=(x + w, 0.51),
                        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.35"))
    # Mark where the model stops and the instrument begins.
    split = 3 * (w + gap) - gap / 2
    ax.plot([split, split], [0.24, 0.78], ls=(0, (3, 3)), lw=1, color="0.5")
    ax.text(split - 0.012, 0.20, "model", ha="right", fontsize=7.5,
            style="italic", color="0.4")
    ax.text(split + 0.012, 0.20, "instrument", ha="left", fontsize=7.5,
            style="italic", color="0.4")
    ax.text(0.5, 0.86, "The persona is instructed not to give a rating; "
                       "the number is derived, never asked for.",
            ha="center", fontsize=7.6, color="0.3", transform=ax.transAxes)
    ax.set_xlim(-0.01, len(steps) * (w + gap))
    ax.set_ylim(0.12, 0.92)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "ssr_pipeline.pdf")
    plt.close(fig)
    print("wrote ssr_pipeline.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_corpus_funnel()
    fig_ssr_pipeline()


if __name__ == "__main__":
    main()
