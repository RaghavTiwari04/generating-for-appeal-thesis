"""Compare two label sources card by card, to choose a judge model.

Summary means and standard deviations say whether a judge spreads its scores.
They do not say whether two judges rank the same cards the same way, nor why
they disagree. Three things here that the `vlm_labels stats` block cannot show:

  rank agreement — Spearman rho per dimension. Two judges can differ in level
      while ranking cards identically (a calibration difference, harmless: the
      predictor learns a monotone target and the reranker only needs order) or
      agree on level while ranking differently (a substantive disagreement).
      The means alone cannot tell these apart.

  persona boilerplate — SSR's premise is that a synthetic consumer responds to
      the stimulus. If a model's replies are near-identical across different
      cards, the embedding step has nothing to separate and purchase_intent
      collapses toward mid-scale no matter how good the embedder is. Measured
      as word-level Jaccard between replies to *different* cards, against the
      same measure within a card as a baseline.

  the disagreements themselves — the persona replies and judge explanations
      for the cards the two sources score most differently, so the numbers can
      be checked against what each model actually said.

Read-only. Run on a compute node with services up:

    python -m scripts.compare_label_sources \\
        --sources llm_ssr_rubric_v2,llm_ssr_rubric_v2_gpt4o
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import typer

from common.db import engine
from scoring import DIMS

_SQL = """
SELECT sl.listing_id::text AS listing_id,
       sl.label_source,
       sl.score,
       sl.raw,
       l.title,
       lf.occasion
FROM saleability_labels sl
JOIN listings l USING (listing_id)
LEFT JOIN listing_features lf ON lf.listing_id = sl.listing_id
WHERE sl.label_source = ANY(%(sources)s);
"""


def _dim_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (listing, source) with the dimensions unpacked from raw."""
    rows = []
    for r in df.itertuples():
        raw = r.raw if isinstance(r.raw, dict) else {}
        row = {
            "listing_id": r.listing_id,
            "label_source": r.label_source,
            "composite": r.score,
            "title": r.title,
            "occasion": r.occasion,
        }
        row.update({d: raw.get(d) for d in DIMS})
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(wide: pd.DataFrame, sources: list[str]) -> None:
    print(f"\n{'=' * 78}\nPer-dimension summary\n")
    print(f"{'dimension':22s} {'source':28s} {'mean':>6s} {'sd':>6s} {'min':>6s} {'max':>6s}")
    for dim in ("composite", *DIMS):
        for src in sources:
            vals = wide[wide["label_source"] == src][dim].dropna().astype(float)
            if vals.empty:
                continue
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(
                f"{dim:22s} {src[:28]:28s} {vals.mean():6.3f} {sd:6.3f} "
                f"{vals.min():6.3f} {vals.max():6.3f}"
            )


def _rank_agreement(wide: pd.DataFrame, a: str, b: str) -> None:
    """Spearman rho between two sources over the cards both scored."""
    print(f"\n{'=' * 78}\nRank agreement over shared cards — {a} vs {b}\n")
    left = wide[wide["label_source"] == a].set_index("listing_id")
    right = wide[wide["label_source"] == b].set_index("listing_id")
    shared = left.index.intersection(right.index)
    print(f"{len(shared)} cards scored by both\n")
    if len(shared) < 3:
        print("  too few shared cards for a meaningful correlation")
        return

    for dim in ("composite", *DIMS):
        x = pd.to_numeric(left.loc[shared, dim], errors="coerce")
        y = pd.to_numeric(right.loc[shared, dim], errors="coerce")
        ok = x.notna() & y.notna()
        # Spearman is undefined when either side is constant — a judge that
        # gave every card the same score has no ranking to agree with.
        if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
            print(f"  {dim:22s}  n/a (constant or too few values)")
            continue
        rho = x[ok].corr(y[ok], method="spearman")
        print(f"  {dim:22s}  rho={rho:+.3f}   n={int(ok.sum())}")

    print(
        "\n  High rho with different means is a calibration gap, which the "
        "predictor\n  and the reranker both absorb. Low rho is a real "
        "disagreement about\n  which cards are better."
    )


def _jaccard(a: str, b: str) -> float:
    wa, wb = set((a or "").lower().split()), set((b or "").lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _persona_variation(df: pd.DataFrame, sources: list[str], max_pairs: int) -> None:
    """Do the personas respond to the card, or write the same reply every time?"""
    print(f"\n{'=' * 78}\nSSR persona variation (word-level Jaccard)\n")
    rng = np.random.default_rng(0)

    for src in sources:
        sub = df[df["label_source"] == src]
        by_card = {}
        for r in sub.itertuples():
            raw = r.raw if isinstance(r.raw, dict) else {}
            replies = [t for t in (raw.get("ssr_responses") or []) if t]
            if replies:
                by_card[r.listing_id] = replies
        if len(by_card) < 2:
            print(f"  {src}: no SSR replies stored")
            continue

        within = [
            _jaccard(x, y)
            for replies in by_card.values()
            for x, y in combinations(replies, 2)
        ]

        cards = list(by_card)
        cross_pairs = [
            (ca, cb) for ca, cb in combinations(cards, 2)
        ]
        if len(cross_pairs) > max_pairs:
            idx = rng.choice(len(cross_pairs), size=max_pairs, replace=False)
            cross_pairs = [cross_pairs[i] for i in idx]
        across = [
            _jaccard(by_card[ca][0], by_card[cb][0]) for ca, cb in cross_pairs
        ]

        w = float(np.mean(within)) if within else float("nan")
        a = float(np.mean(across)) if across else float("nan")
        print(
            f"  {src[:34]:34s} within-card={w:.3f}  across-card={a:.3f}  "
            f"ratio={a / w if w else float('nan'):.2f}"
        )

    print(
        "\n  across-card is the number that matters: it is how much two replies\n"
        "  about *different* cards still look alike. A value approaching the\n"
        "  within-card baseline means the personas are writing boilerplate and\n"
        "  the embedding step has little card-specific signal to separate."
    )


def _disagreements(
    df: pd.DataFrame, wide: pd.DataFrame, a: str, b: str, top: int, chars: int
) -> None:
    left = wide[wide["label_source"] == a].set_index("listing_id")
    right = wide[wide["label_source"] == b].set_index("listing_id")
    shared = left.index.intersection(right.index)
    if shared.empty:
        return

    gap = (
        pd.to_numeric(left.loc[shared, "composite"], errors="coerce")
        - pd.to_numeric(right.loc[shared, "composite"], errors="coerce")
    ).abs().sort_values(ascending=False)

    raw_by = {
        (r.listing_id, r.label_source): (r.raw if isinstance(r.raw, dict) else {})
        for r in df.itertuples()
    }

    print(f"\n{'=' * 78}\nLargest composite disagreements\n")
    for listing_id in gap.head(top).index:
        meta = left.loc[listing_id]
        print(f"\n{'-' * 78}")
        print(f"{str(meta['title'] or '')[:70]}   [{meta['occasion']}]")
        print(f"listing {listing_id[:8]}   composite gap {gap[listing_id]:.3f}\n")

        hdr = f"  {'source':30s} " + " ".join(f"{d[:9]:>9s}" for d in ("composite", *DIMS))
        print(hdr)
        for src, frame in ((a, left), (b, right)):
            vals = " ".join(
                f"{v:9.3f}" if pd.notna(v := pd.to_numeric(frame.loc[listing_id, d], errors="coerce")) else f"{'—':>9s}"
                for d in ("composite", *DIMS)
            )
            print(f"  {src[:30]:30s} {vals}")

        # Which dimension drives the gap — that is the one worth reading.
        diffs = {}
        for d in DIMS:
            x = pd.to_numeric(left.loc[listing_id, d], errors="coerce")
            y = pd.to_numeric(right.loc[listing_id, d], errors="coerce")
            if pd.notna(x) and pd.notna(y):
                diffs[d] = abs(x - y)
        worst = max(diffs, key=diffs.get) if diffs else None

        for src in (a, b):
            raw = raw_by.get((listing_id, src), {})
            replies = [t for t in (raw.get("ssr_responses") or []) if t]
            print(f"\n  [{src}]")
            for t in replies[:2]:
                print(f"    persona: {t[:chars].strip()}")
            if worst:
                expl = (raw.get("explanations") or {}).get(worst, "")
                if expl:
                    print(f"    judge/{worst}: {expl[:chars].strip()}")


def main(
    sources: str = typer.Option(
        ..., help="Comma-separated label sources, first is the reference"
    ),
    top: int = typer.Option(5, help="Cards to show in full, by disagreement size"),
    chars: int = typer.Option(300, help="Characters of each reply to print"),
    max_pairs: int = typer.Option(5000, help="Cross-card pairs sampled for Jaccard"),
) -> None:
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    if len(src_list) < 2:
        raise typer.BadParameter("give at least two label sources")

    df = pd.read_sql(_SQL, engine(), params={"sources": src_list})
    if df.empty:
        print(f"No labels found for {src_list}.")
        return

    found = set(df["label_source"])
    for s in src_list:
        n = int((df["label_source"] == s).sum())
        print(f"{s}: {n} cards" + ("" if s in found else "  (MISSING)"))

    wide = _dim_frame(df)
    _summary(wide, src_list)
    _persona_variation(df, src_list, max_pairs)
    for other in src_list[1:]:
        _rank_agreement(wide, src_list[0], other)
        _disagreements(df, wide, src_list[0], other, top, chars)


if __name__ == "__main__":
    typer.run(main)
