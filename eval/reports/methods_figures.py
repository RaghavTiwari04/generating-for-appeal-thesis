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

BLUE = "#2563eb"

# Measured by scripts/verify_corpus_numbers.py against the delivered database.
#
# Three tracks, not one funnel. The quantity changes units twice going down the
# chain -- listings, then distinct designs, then label rows -- and a stage drawn
# under another asserts that the second is a subset of the first. Two of those
# assertions are false. Distinct designs are counted over the whole scrape,
# because deduplication runs there rather than on the classified subset, so the
# design count exceeds the classified listing count above it. Label rows are
# rows, not designs, so 2,468 exceeds the 2,403 designs they cover.
#
# Drawn side by side, each track nests within itself and the relations between
# them are stated rather than implied.
TRACKS: list[tuple[str, list[tuple[str, int, str]]]] = [
    ("Listings", [
        ("Scraped", 3906, "Greetings Island 2,033\nRedbubble 1,873"),
        ("With a birthday subtype", 3491, "415 unclassified"),
    ]),
    ("Distinct designs", [
        ("Over the whole scrape", 2795, "1,111 redundant\ncolourways collapsed"),
        ("Carrying a subtype", 2416, "2,470 summed per subtype:\n54 clusters span two"),
    ]),
    ("Label rows", [
        ("Judged", 2468, "covering 2,403 designs\n(2,419 summed per subtype)"),
        ("Predictor training split", 1727, "plus 370 validation,\n371 test"),
    ]),
]

FOOTNOTE = ("Deduplication runs over the whole scrape, not the classified "
            "subset, so the tracks do not nest into one another.")


def fig_corpus_funnel() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    centres = [0.17, 0.5, 0.83]
    half, box_h = 0.125, 0.19
    top_y, bot_y = 0.60, 0.10          # box bottoms

    for x, (title, stages) in zip(centres, TRACKS):
        ax.text(x, 0.99, title, ha="center", va="center",
                fontsize=9.5, weight="bold", color="0.15")
        for j, ((label, n, note), y) in enumerate(zip(stages, (top_y, bot_y))):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - half, y), 2 * half, box_h,
                boxstyle="round,pad=0.006", linewidth=0,
                facecolor=BLUE, alpha=0.34 + 0.16 * j))
            ax.text(x, y + 0.132, f"{n:,}", ha="center", va="center",
                    fontsize=11, weight="bold")
            ax.text(x, y + 0.048, label, ha="center", va="center",
                    fontsize=6.9, color="0.2")
            # The note sits directly under its own box. Anything else puts it
            # in the path of the arrow to the box below.
            ax.text(x, y - 0.018, note, ha="center", va="top",
                    fontsize=6.3, color="0.42", linespacing=1.35)
        # Within a track, each stage is a subset of the one above it. The arrow
        # runs down the side so it never crosses the note.
        ax.annotate("", xy=(x + half - 0.012, bot_y + box_h + 0.005),
                    xytext=(x + half - 0.012, top_y - 0.005),
                    arrowprops=dict(arrowstyle="-|>", color="0.6", lw=0.9))

    # Between tracks the relation is a change of unit, not a subset. Drawn
    # above the boxes, where there is room for the arrow and its label.
    for x0, x1, text in ((0.17, 0.5, "deduplicate"), (0.5, 0.83, "label")):
        mid = (x0 + x1) / 2
        ax.annotate("", xy=(x1 - half - 0.008, 0.855),
                    xytext=(x0 + half + 0.008, 0.855),
                    arrowprops=dict(arrowstyle="-|>", color="0.6", lw=0.9,
                                    linestyle=(0, (3, 2))))
        ax.text(mid, 0.878, text, ha="center", va="bottom",
                fontsize=6.8, color="0.42", style="italic")

    ax.text(0.5, -0.10, FOOTNOTE, ha="center", va="center",
            fontsize=6.6, color="0.42")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1.06)
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
