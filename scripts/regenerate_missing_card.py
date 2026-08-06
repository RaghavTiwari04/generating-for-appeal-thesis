"""Regenerate the one condition B card lost to a generation failure.

The evaluation design is 10 cards per condition per subtype, 160 in total. One
condition B generation raised and was logged and skipped, so
\\texttt{birthday/kids} carries 9 and the reported set is 159.

This recovers that design point rather than adding a new one. The seed is
determined by arithmetic, not chosen: `generate_eval_set` assigns

    seed = seed_base + occasion_index * 1000 + condition_index * 100 + k

so the nine surviving cards fix the sequence and the tenth seed is whichever
value is absent from it. The card is generated through the same call the
original run used, with the same seed, subject index and run tag.

Two properties make this a recovery rather than a post-hoc addition, and both
matter if anyone asks:

  * the seed is derived from the existing cards, so there is no choice in which
    card gets made;
  * it is generated once and kept. There is no second attempt and no selection
    among attempts. If the regenerated card scores badly it still goes in.

Scoring is deliberately not done here. The card has to be judged by the same
instrument as the rest of the reported set (the Gemini judge, see
cluster/jobs/06_system_eval.sh) and, for Section 4.6, by the two robustness
judges as well. Run the evaluation afterwards; its ratings cache is keyed on
card_key, so it scores only the new card.

    source cluster/jobs/_start_services.sh
    python -m scripts.regenerate_missing_card --run-tag run_20260805_0236
    python -m scripts.regenerate_missing_card --run-tag ... --apply
"""

from __future__ import annotations

import pandas as pd
import typer

from common.db import engine
from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS

log = get_logger(__name__)

CONDITIONS = ("A", "B", "C", "D")
N_PER = 10

_EXISTING_SQL = """
SELECT gc.card_id::text AS card_id,
       -- The seed lives in its own column. Conditions B and C persist through
       -- the orchestrator, which puts the seed there and not in brief->request;
       -- only A and D route through _persist_eval_card, which does both.
       gc.seed::bigint AS seed,
       COALESCE(gc.brief->'request'->>'occasion', gc.brief->>'occasion') AS occasion,
       gc.condition_tag
FROM generated_cards gc
WHERE gc.brief->'request'->>'eval_run' = %(run_tag)s
  AND gc.condition_tag = %(condition_tag)s
ORDER BY seed
"""


def _missing_seed(seeds: list[int], occasion: str, expect: int) -> int | None:
    """The one seed absent from an arithmetic run of `expect` consecutive values."""
    if not seeds:
        log.error(f"no cards at all for {occasion}; cannot infer the seed base")
        return None
    if len(seeds) == expect:
        log.info(f"{occasion}: already {expect} cards, nothing to recover")
        return None
    if len(seeds) != expect - 1:
        log.error(f"{occasion}: {len(seeds)} cards, expected {expect - 1}; "
                  f"more than one is missing and the seed is ambiguous")
        return None
    base = min(seeds)
    full = set(range(base, base + expect))
    gap = sorted(full - set(seeds))
    # The gap is only unambiguous when the missing seed is interior. If the
    # absent card were the last of the run, min() would still be the base but
    # base + expect - 1 would look like a gap that is really the end.
    if len(gap) != 1:
        log.error(f"{occasion}: gaps {gap}; cannot identify a single missing seed")
        return None
    log.info(f"{occasion}: seeds {min(seeds)}..{max(seeds)}, missing {gap[0]}")
    return gap[0]


def run(
    run_tag: str = typer.Option(..., help="eval_run tag of the reported run"),
    condition: str = typer.Option("B", help="condition letter to repair"),
    apply: bool = typer.Option(False, help="actually generate; otherwise dry run"),
    scorer: str = typer.Option("ridge"),
) -> None:
    from generation.brief.market_signals import subject_pool_size
    from pipeline.conditions import (
        CONDITION_TAGS,
        _generate_pipeline_no_rerank,
        _generate_pipeline_rerank,
    )

    tag = CONDITION_TAGS[condition]
    df = pd.read_sql(_EXISTING_SQL, engine(),
                     params={"run_tag": run_tag, "condition_tag": tag})
    if df.empty:
        raise SystemExit(f"no {tag} cards for run_tag={run_tag}; check the tag")

    log.info(f"{tag} under {run_tag}: {len(df)} cards")
    counts = df.groupby("occasion").size()
    log.info(f"per occasion: {counts.to_dict()}")

    todo: list[tuple[str, int, int]] = []
    for occ_i, occasion in enumerate(ACTIVE_OCCASIONS):
        seeds = sorted(df[df.occasion == occasion].seed.dropna().astype(int))
        seed = _missing_seed(seeds, occasion, N_PER)
        if seed is None:
            continue
        cond_j = CONDITIONS.index(condition)
        k = seed - (min(seeds) if seeds else seed)
        expected_base = seed - occ_i * 1000 - cond_j * 100 - k
        log.info(f"  -> regenerate {occasion} seed={seed} "
                 f"(implies seed_base={expected_base}, k={k})")
        todo.append((occasion, seed, k))

    if not todo:
        log.info("nothing to do")
        return
    if not apply:
        log.info("dry run; pass --apply to generate")
        return

    gen = {"B": _generate_pipeline_no_rerank, "C": _generate_pipeline_rerank}[condition]
    for occasion, seed, k in todo:
        subject = str((k % subject_pool_size(occasion)) + 1)
        log.info(f"generating {occasion} seed={seed} subject={subject}")
        card = gen(occasion, seed, scorer=scorer, subject=subject, run_tag=run_tag)
        if card is None or not card.card_id:
            log.error(f"generation failed again for {occasion} seed={seed}")
            continue
        log.info(f"generated card_id={card.card_id}; headline={card.headline!r}")

    log.info("Now re-run the evaluation to score the new card, then regenerate "
             "figures. The ratings cache is keyed on card_key, so only the new "
             "card is scored.")


if __name__ == "__main__":
    typer.run(run)
