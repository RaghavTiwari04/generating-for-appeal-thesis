"""Export the card images the thesis figures include.

Two sources, same output:

  --gallery DIR   read an exported eval gallery (the folder written by
                  eval.export_gallery: one directory per condition plus
                  gallery.html). Needs no database and no image store, so this
                  runs anywhere the gallery has been copied to.

  (default)       query Postgres and the image store directly. Runs on the
                  cluster only.

Prefer the gallery when you have one. It carries the scores alongside the
images, so the figures and their captions come from a single artefact that was
written by the evaluation itself.

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
import re
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

# The two cards named in the failure analysis, as (headline substring,
# condition). Matched on the headline the brief REQUESTED, not on what the model
# rendered: the mangled strings ("HAPTHCAY MUPPET", "ALL ABARDS") exist only as
# pixels, while headline_text holds the request. Matching on the render found
# nothing, which is the whole point of the failure.
#
# The condition is part of the key because "Muppet" also matches a condition C
# card that came out clean, at 0.78 aesthetic against the B card's 0.22. The
# results chapter describes the B card, so picking by substring alone would
# illustrate the failure with a card that did not fail.
# A "file:" key matches the cover filename instead of the headline. Condition A
# needs it: every naive card requested the same "Happy Birthday", so the headline
# cannot distinguish them and only the rendered pixels differ.
FAILURE_CARDS = [
    ("Muppet", "B_pipeline_no_rerank"),                       # mangled headline + IP
    ("Nice Work", "B_pipeline_no_rerank"),                    # mangled second line
    ("file:birthday_milestone_4dfc9d14", "A_naive_ai"),       # "Happy Birthday Dairy"
]

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


# Galleries exported before condition D was renamed carry the old tag.
_TAG_ALIASES = {"D_human_bestseller": "D_human_reference"}

_GALLERY_CARD = re.compile(
    r'<img src="(?P<path>[^"]+\.png)"[^>]*>'
    r'<br><small[^>]*>(?P<occasion>[^<]*)<br>(?P<headline>[^<]*)</small>'
    r'(?P<scores>.*?)(?=<div style="text-align:center;border:|$)',
    re.S,
)
_GALLERY_DIM = re.compile(
    r'<span style="width:20px">(\w+)</span>.*?'
    r'<span style="width:28px;text-align:right">([\d.]+)</span>',
    re.S,
)
_DIM_NAMES = {
    "PI": "purchase_intent", "OF": "occasion_fit", "AE": "aesthetic",
    "ER": "emotional_resonance", "DI": "distinctiveness",
}


def load_gallery(gallery: Path) -> pd.DataFrame:
    """Parse an exported gallery into the frame the figure builders expect.

    The gallery is self-contained: gallery.html carries each card's condition,
    occasion, headline and five scores, and the PNGs sit beside it. Nothing is
    recomputed here, so the numbers in the captions are the ones the evaluation
    recorded.
    """
    html_path = gallery / "gallery.html"
    if not html_path.exists():
        raise SystemExit(f"no gallery.html in {gallery}")
    html = html_path.read_text(encoding="utf-8", errors="replace")

    rows = []
    for m in _GALLERY_CARD.finditer(html):
        rel = m["path"]
        tag = rel.split("/")[0]
        rec = {
            "cover_path": str(gallery / rel),
            "condition": _TAG_ALIASES.get(tag, tag),
            "occasion": m["occasion"].strip(),
            "headline_text": m["headline"].strip(),
        }
        for short, val in _GALLERY_DIM.findall(m["scores"]):
            if short in _DIM_NAMES:
                rec[_DIM_NAMES[short]] = float(val)
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"parsed no cards from {html_path}")
    log.info(f"parsed {len(df)} cards from {html_path}")
    for cond, n in df.groupby("condition").size().items():
        log.info(f"  {cond}: {n}")
    return df


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
        # Gallery mode hands over ordinary filesystem paths; the object store is
        # only consulted for the keys that are not already local files.
        if Path(cover_path).exists():
            return Image.open(cover_path).convert("RGB")
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
            cell = df[(df.condition == tag) & (df.occasion == occ)]
            # Say which cell failed and at which step, otherwise a blank panel
            # is indistinguishable from a card that genuinely scored nothing.
            if cell.empty:
                log.warning(f"empty cell {tag} / {occ}: no scored cards")
                row.append((None, ""))
                missing += 1
                continue
            with_cover = cell[cell["cover_path"].notna()]
            if with_cover.empty:
                log.warning(
                    f"empty cell {tag} / {occ}: {len(cell)} scored cards, "
                    f"none resolved to a cover_path"
                )
                row.append((None, ""))
                missing += 1
                continue
            # Prefer a card whose image actually loads over the strict median,
            # so one unreadable blob does not blank an otherwise full cell.
            pick = _median_card(with_cover)
            img = _fetch(pick["cover_path"])
            if img is None:
                for _, alt in with_cover.iterrows():
                    if alt["cover_path"] != pick["cover_path"]:
                        img = _fetch(alt["cover_path"])
                        if img is not None:
                            pick = alt
                            break
            if img is None:
                log.warning(
                    f"empty cell {tag} / {occ}: {len(with_cover)} cover paths, "
                    f"none could be fetched"
                )
                missing += 1
            row.append((img, f"PI {pick['purchase_intent']:.2f}" if img is not None else ""))
        panels.append(row)

    occ_label = {"birthday/general": "General", "birthday/kids": "Kids",
                 "birthday/relationship": "Relationship"}
    _grid(panels, [lbl for _, lbl in ROW_ORDER],
          [occ_label[o] for o in COL_OCCASIONS], out / "cards_conditions.pdf")
    if missing:
        log.warning(f"{missing} of {4 * len(COL_OCCASIONS)} panels are empty")


def _failures_figure(df: pd.DataFrame, out: Path) -> None:
    hits = []
    for needle, cond in FAILURE_CARDS:
        in_cond = df[df.condition == cond]
        if needle.startswith("file:"):
            stem = needle[len("file:"):]
            match = in_cond[in_cond["cover_path"].fillna("").str.contains(stem, regex=False)]
        else:
            match = in_cond[
                in_cond["headline_text"].fillna("").str.contains(needle, case=False, na=False)
            ]
        if match.empty:
            log.warning(f"no {cond} card matched {needle!r}")
            hits.append((None, ""))
            continue
        # Lowest aesthetic among the matches: broken lettering is what that head
        # penalises, so this resolves ties toward the card being illustrated.
        pick = match.sort_values("aesthetic").iloc[0]
        log.info(
            f"failure card {needle!r} -> {pick['headline_text']!r} "
            f"(PI {pick['purchase_intent']:.2f}, aesthetic {pick['aesthetic']:.2f})"
        )
        img = _fetch(pick["cover_path"]) if pick.get("cover_path") else None
        pi = pick.get("purchase_intent")
        ae = pick.get("aesthetic")
        caption = f'"{pick["headline_text"]}"'
        if pd.notna(pi) and pd.notna(ae):
            caption += f"\nPI {pi:.2f}, aesthetic {ae:.2f}"
        hits.append((img, caption))

    _grid([hits], [], [], out / "cards_failures.pdf", panel_w=2.1)


def run(
    gallery: Path | None = typer.Option(
        None, help="exported eval gallery directory; skips the database entirely"
    ),
    run_tag: str | None = typer.Option(None, help="eval_run tag; defaults to the latest"),
    ratings: Path = typer.Option(RATINGS, help="raw_ratings.csv from the scored run"),
    out: Path = typer.Option(OUT_DEFAULT, help="directory to write the PDFs into"),
) -> None:
    if gallery is not None:
        df = load_gallery(gallery)
        _conditions_figure(df, out)
        _failures_figure(df, out)
        return

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

    # Per-condition, so a shortfall is attributable rather than just a total.
    # An entire condition missing points at the run tag or at condition D rows
    # living outside generated_cards; a scattering points at individual blobs.
    for tag, _ in ROW_ORDER:
        sub = df[df.condition == tag]
        if len(sub) and sub["cover_path"].notna().sum() < len(sub):
            log.warning(
                f"  {tag}: {sub['cover_path'].notna().sum()}/{len(sub)} resolved"
            )

    if matched == 0:
        raise SystemExit(
            "no scored card matched a cover_path; check --run-tag against the "
            "run that produced this ratings file"
        )

    _conditions_figure(df, out)
    _failures_figure(df, out)


if __name__ == "__main__":
    typer.run(run)
