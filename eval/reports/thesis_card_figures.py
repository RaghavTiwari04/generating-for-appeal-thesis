"""Export the card images the thesis figures include.

Runs on the cluster, where Postgres and the image store are reachable. The
generated covers are not in the repo and are not recoverable from
raw_ratings.csv, which carries only card_key.

Produces, into report/figures/:

  cards_conditions.pdf  4 rows (A/B/C/D) x 3 columns, one occasion per column,
                        so a column compares the same brief across conditions.
  cards_failures.pdf    the two failure cases discussed in the results chapter.

Cards are chosen by card_key from raw_ratings.csv, so every card shown is one
that was actually scored, and the scores quoted in the captions are the scores
it received. Selection is deterministic: within a (condition, occasion) cell the
card whose purchase intent is closest to that cell's median is used, which shows
typical output rather than the best available.

Usage, on a compute node with the services up:

    python -m eval.reports.thesis_card_figures
    python -m eval.reports.thesis_card_figures --run-tag <tag> --out report/figures
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import typer
from PIL import Image

from common.db import engine
from common.logging import get_logger
from common.storage import get_object

log = get_logger(__name__)

RATINGS = Path("artifacts/llm_system_eval/raw_ratings.csv")
OUT_DEFAULT = Path("report/figures")

# Matches the ordering used everywhere else in the writeup.
ROW_ORDER = [
    ("A_naive_ai", "A: naive prompt"),
    ("B_pipeline_no_rerank", "B: pipeline"),
    ("C_pipeline_rerank", "C: pipeline + rerank"),
    ("D_human_reference", "D: human reference"),
]
COL_OCCASIONS = ["birthday/general", "birthday/kids", "birthday/relationship"]

# The two cards named in the failure analysis, identified by the text the model
# rendered rather than by key, so the figure survives a regenerated run.
FAILURE_HEADLINES = ["HAPTHCAY MUPPET", "ALL ABARDS"]

_COVERS_SQL = """
SELECT gc.card_id::text AS card_key,
       gc.condition_tag,
       gc.cover_path,
       gc.headline_text,
       COALESCE(gc.brief->'request'->>'occasion', gc.brief->>'occasion') AS occasion
FROM generated_cards gc
WHERE (%(run_tag)s::text IS NULL
       OR gc.brief->'request'->>'eval_run' = %(run_tag)s::text)
"""


def _load_covers(run_tag: str | None) -> pd.DataFrame:
    """card_key -> cover_path for every card in the run, generated and human."""
    gen = pd.read_sql(_COVERS_SQL, engine(), params={"run_tag": run_tag})

    # Condition D rows live in generated_cards too, written by the eval, so one
    # query covers all four conditions. If a run predates that, fall back to the
    # evaluation's own loader rather than reimplementing its sampling.
    if "D_human_reference" not in set(gen["condition_tag"]):
        from eval.llm_system_eval import _load_human_reference

        human = _load_human_reference(COL_OCCASIONS, per_occasion=10)
        gen = pd.concat([gen, human], ignore_index=True)
    return gen


def _fetch(cover_path: str) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(get_object(cover_path))).convert("RGB")
    except Exception as e:
        log.warning(f"could not load {cover_path}: {e}")
        return None


def _median_card(cell: pd.DataFrame) -> pd.Series | None:
    """The card closest to the cell's median purchase intent."""
    if cell.empty:
        return None
    target = cell["purchase_intent"].median()
    return cell.loc[(cell["purchase_intent"] - target).abs().idxmin()]


def _grid(
    panels: list[list[tuple[Image.Image | None, str]]],
    row_labels: list[str],
    col_labels: list[str],
    out_path: Path,
    panel_w: float = 1.6,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(panels), len(panels[0])
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols + 1.1, panel_w * 1.4 * n_rows + 0.5),
        squeeze=False,
    )
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            img, caption = panels[r][c]
            if img is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center",
                        fontsize=7, color="0.5", transform=ax.transAxes)
                for s in ax.spines.values():
                    s.set_color("0.8")
            else:
                ax.imshow(img)
                for s in ax.spines.values():
                    s.set_color("0.3")
            if caption:
                ax.set_xlabel(caption, fontsize=6.5, labelpad=2)
            if r == 0 and c < len(col_labels):
                ax.set_title(col_labels[c], fontsize=8)
            if c == 0 and r < len(row_labels):
                ax.set_ylabel(row_labels[r], fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out_path}")


def _conditions_figure(df: pd.DataFrame, out: Path) -> None:
    panels, missing = [], 0
    for tag, _label in ROW_ORDER:
        row = []
        for occ in COL_OCCASIONS:
            pick = _median_card(df[(df.condition == tag) & (df.occasion == occ)])
            if pick is None or not pick.get("cover_path"):
                row.append((None, ""))
                missing += 1
                continue
            img = _fetch(pick["cover_path"])
            missing += img is None
            row.append((img, f"PI {pick['purchase_intent']:.2f}"))
        panels.append(row)

    occ_label = {"birthday/general": "General", "birthday/kids": "Kids",
                 "birthday/relationship": "Relationship"}
    _grid(panels, [lbl for _, lbl in ROW_ORDER],
          [occ_label[o] for o in COL_OCCASIONS], out / "cards_conditions.pdf")
    if missing:
        log.warning(f"{missing} of {4 * len(COL_OCCASIONS)} panels are empty")


def _failures_figure(df: pd.DataFrame, out: Path) -> None:
    hits = []
    for needle in FAILURE_HEADLINES:
        match = df[df["headline_text"].fillna("").str.contains(needle, case=False, na=False)]
        if match.empty:
            log.warning(f"no card found with headline containing {needle!r}")
            hits.append((None, ""))
            continue
        pick = match.iloc[0]
        img = _fetch(pick["cover_path"]) if pick.get("cover_path") else None
        pi = pick.get("purchase_intent")
        ae = pick.get("aesthetic")
        caption = f'"{pick["headline_text"]}"'
        if pd.notna(pi) and pd.notna(ae):
            caption += f"\nPI {pi:.2f}, aesthetic {ae:.2f}"
        hits.append((img, caption))

    _grid([hits], [], [], out / "cards_failures.pdf", panel_w=2.1)


def run(
    run_tag: str | None = typer.Option(None, help="eval_run tag; defaults to the latest"),
    ratings: Path = typer.Option(RATINGS, help="raw_ratings.csv from the scored run"),
    out: Path = typer.Option(OUT_DEFAULT, help="directory to write the PDFs into"),
) -> None:
    if not ratings.exists():
        raise SystemExit(f"ratings not found: {ratings}")

    if run_tag is None:
        from eval.llm_system_eval import _latest_run_tag

        run_tag = _latest_run_tag()
    log.info(f"run_tag={run_tag or '(all runs)'}")

    scores = pd.read_csv(ratings)
    # The superseded condition A rows are still in the file; the analysis keeps
    # the last 40, and the figure has to show the same cards it reports.
    a = scores[scores.condition == "A_naive_ai"]
    if len(a) > 40:
        scores = pd.concat(
            [scores[scores.condition != "A_naive_ai"], a.tail(40)], ignore_index=True
        )

    covers = _load_covers(run_tag)
    df = scores.merge(covers, on="card_key", how="left", suffixes=("", "_db"))
    matched = df["cover_path"].notna().sum()
    log.info(f"{matched} of {len(df)} scored cards resolved to a cover image")
    if matched == 0:
        raise SystemExit(
            "no scored card matched a cover_path; check --run-tag against the "
            "run that produced this ratings file"
        )

    _conditions_figure(df, out)
    _failures_figure(df, out)


if __name__ == "__main__":
    typer.run(run)
