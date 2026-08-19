"""Measure how often the evaluated cards repeat a headline, by condition.

Inspecting the gallery showed condition C returning the same headline several
times over -- one phrase six times in forty cards -- where the single-candidate
condition repeated twice and the human reference not at all. Best-of-$N$
selection amplifies whatever the scorer prefers, so a reranked set can be
narrower than the generator that produced it. That is a cost the purchase
intent comparison cannot see, because a set of forty near-identical cards
scores exactly as well as forty varied ones.

Two things this measures and one it cannot:

  repetition per condition -- distinct headlines against cards, exactly and
      after normalising case, punctuation and spacing, since "Born To Party"
      and "Born to party!" are the same card to a customer.

  selection against generation -- the candidate-spread probe
      (`eval.reports.candidate_spread`) persisted every candidate it generated
      rather than only the winner, tagged `probe_candidate_spread`. Those are
      unselected, so their repetition rate is the generator's own. If the
      reranked condition repeats more than they do, selection is doing it; if
      the rates match, the generator was already narrow and reranking merely
      inherits it. The probe's occasion mix is reported alongside, because it
      need not match the evaluated set's.

  what it cannot measure: `headline_text` records the headline the brief
      requested, not the string the diffusion model actually rendered. Two
      cards sharing a requested headline may still differ on the page, and a
      card whose lettering came out malformed still counts here as its
      requested text. Rendered-text repetition would need OCR, which
      Section~4.3 records as unreliable on this artwork.

Read-only. Needs the database:

    source cluster/jobs/_start_services.sh
    python -m scripts.headline_diversity
    python -m scripts.headline_diversity --run-tag 2026-08-06-final
"""

from __future__ import annotations

import collections
import re
import string

import pandas as pd
import typer

from common.db import engine

CONDITIONS = [
    "A_naive_ai",
    "B_pipeline_no_rerank",
    "C_pipeline_rerank",
    "D_human_reference",
]
PROBE = "probe_candidate_spread"

CARDS_SQL = """
SELECT gc.condition_tag,
       gc.headline_text,
       COALESCE(gc.brief->'request'->>'occasion', gc.brief->>'occasion') AS occasion,
       gc.brief->'request'->>'eval_run' AS run_tag
FROM generated_cards gc
WHERE gc.condition_tag = ANY(%(conditions)s)
"""

LATEST_RUN_SQL = """
SELECT gc.brief->'request'->>'eval_run' AS run_tag
FROM generated_cards gc
WHERE gc.brief->'request'->>'eval_run' IS NOT NULL
ORDER BY gc.generated_at DESC
LIMIT 1
"""

# The human reference condition takes its text from the marketplace listing, so
# its headlines are titles rather than generated phrases. Reported for contrast,
# not as a like-for-like arm.
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Case, punctuation and spacing removed: what a customer would call the same."""
    return re.sub(r"\s+", " ", (text or "").lower().translate(_PUNCT)).strip()


def summarise(frame: pd.DataFrame, label: str) -> dict:
    heads = [h for h in frame["headline_text"].fillna("") if h.strip()]
    norm = [normalise(h) for h in heads]
    exact = collections.Counter(heads)
    fuzzy = collections.Counter(norm)
    top = fuzzy.most_common(1)[0] if fuzzy else ("", 0)
    return {
        "label": label,
        "cards": len(heads),
        "distinct_exact": len(exact),
        "distinct_norm": len(fuzzy),
        "redundant": len(heads) - len(fuzzy),
        "share": (len(heads) - len(fuzzy)) / len(heads) if heads else 0.0,
        "largest": top[1],
        "largest_text": top[0],
        "repeats": {h: c for h, c in fuzzy.items() if c > 1},
    }


def main(run_tag: str = typer.Option("", help="Evaluation run; default is the latest.")) -> None:
    eng = engine()
    if run_tag:
        tag = run_tag
    else:
        latest = pd.read_sql(LATEST_RUN_SQL, eng)
        tag = None if latest.empty else latest.iloc[0]["run_tag"]
    if tag is None:
        raise SystemExit(
            "No tagged generation run found. This would otherwise pool every "
            "card ever generated across incompatible development runs; see "
            "Section 3.6.3."
        )
    print(f"run_tag = {tag}\n")

    df = pd.read_sql(CARDS_SQL, eng, params={"conditions": CONDITIONS + [PROBE]})
    evaluated = df[(df.condition_tag != PROBE) & (df.run_tag == tag)]
    probe = df[df.condition_tag == PROBE]

    rows = []
    for cond in CONDITIONS:
        sub = evaluated[evaluated.condition_tag == cond]
        if len(sub):
            rows.append(summarise(sub, cond))
    if len(probe):
        rows.append(summarise(probe, f"{PROBE} (unselected)"))

    print(f"{'condition':32s} {'cards':>6s} {'distinct':>9s} {'redundant':>10s} "
          f"{'share':>7s} {'largest':>8s}")
    for r in rows:
        print(f"{r['label']:32s} {r['cards']:6d} {r['distinct_norm']:9d} "
              f"{r['redundant']:10d} {r['share']:6.0%} {r['largest']:8d}")

    for r in rows:
        if r["repeats"]:
            print(f"\n{r['label']} repeats:")
            for h, c in sorted(r["repeats"].items(), key=lambda kv: -kv[1]):
                print(f"    {c}x  {h!r}")

    # Selection against generation. The probe is unselected, so its rate is the
    # generator's; C's excess over it is what reranking added.
    by = {r["label"]: r for r in rows}
    c, p = by.get("C_pipeline_rerank"), by.get(f"{PROBE} (unselected)")
    b = by.get("B_pipeline_no_rerank")
    if c and p:
        print(f"\nreranked {c['share']:.0%} against unselected candidates "
              f"{p['share']:.0%}: "
              + ("selection is narrowing the set" if c["share"] > p["share"] + 0.05
                 else "the generator is already this narrow"))
        print("probe occasion mix: "
              + ", ".join(f"{k}={v}" for k, v in
                          probe["occasion"].value_counts().items()))
    elif c:
        print("\nNo probe cards found: run eval.reports.candidate_spread to get "
              "the unselected control, without which selection and generation "
              "cannot be told apart.")
    if c and b:
        print(f"reranked {c['share']:.0%} against single-candidate "
              f"{b['share']:.0%}")

    print("\n=== per subtype ===")
    for cond in ("B_pipeline_no_rerank", "C_pipeline_rerank"):
        sub = evaluated[evaluated.condition_tag == cond]
        if not len(sub):
            continue
        print(f"  {cond}")
        for occ, grp in sub.groupby("occasion"):
            s = summarise(grp, occ)
            print(f"    {occ:24s} {s['cards']:3d} cards, "
                  f"{s['distinct_norm']:3d} distinct, largest repeat {s['largest']}")

    print("\n=== LaTeX ===")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Condition & Cards & Distinct headlines & Repeated \\")
    print(r"\midrule")
    for r in rows:
        name = r["label"].replace("_", r"\_")
        print(f"\\texttt{{{name}}} & {r['cards']} & {r['distinct_norm']} "
              f"& {r['redundant']} ({r['share']:.0%}) " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    typer.run(main)
