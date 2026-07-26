"""Occasion classification from listing titles, via NLI zero-shot entailment.

Every title is scored against one hypothesis per birthday subtype plus a
"not a birthday card" option, using a natural language inference model
(Yin et al., 2019). No training data, no API, runs locally on the cluster GPU.

This replaces the keyword rules as the labelling method. Rules could only fire
on explicit tokens, so semantic titles ("Fabulous at Fifty", "To my one and
only") defaulted to general, while substring matching mislabelled adult cards
as kids. A supervised classifier was not an option: the only labels available
were the rules' own output, so it could at best imitate them.

An explicit, decisive age overrides the model: a title saying "3rd Birthday"
or "50 years old" is settled by parsing the number, because NLI scored
"14 Year Old Birthday Gift" as kids at 0.99. Ages that decide nothing under
the taxonomy (14, 16, 29) fall through to NLI.

Anything failing to clear `threshold` falls back to birthday/general; a
confident "not a birthday card" clears the occasion so the listing is excluded
from training.

Writes listing_features.occasion with feature_version 'nli-v2'.

    python -m data.features.occasion_nli --dry-run --limit 200
    python -m data.features.occasion_nli
"""

from __future__ import annotations

import json

import typer

from common.db import connection
from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS as OCCASIONS
from common.occasions import ages_rule_out_kids, occasion_from_age

log = get_logger(__name__)

# bart-large-mnli (2019) was the classic zero-shot NLI checkpoint but is dated
# and over-fired on the relationship hypothesis — "Happy Birthday Old Chap!"
# and a wall-art poster both scored as romantic. deberta-v3-large-zeroshot-v2.0
# is trained specifically for zero-shot classification and handles that nuance
# better.
DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# Pass --revision <commit-sha> to pin the checkpoint. Worth doing before the
# labels are used in reported results: an upstream update silently changes
# them, and nothing else in the pipeline would reveal it.
DEFAULT_REVISION: str | None = None

# Phrased as sentence completions for the hypothesis template below.
NOT_BIRTHDAY = "__not_birthday__"

_HYPOTHESES: dict[str, str] = {
    "birthday/kids": "a birthday card for a young child",
    "birthday/milestone": "a birthday card for a landmark age such as 18th, 30th or 50th",
    "birthday/relationship": "a birthday card for a romantic partner, husband, wife or lover",
    "birthday/general": "an ordinary birthday card for anyone",
    # Lets NLI reject junk, so it can also be run on listings the rules could
    # not label at all without dragging non-birthday cards into the dataset.
    NOT_BIRTHDAY: "a greeting card for something other than a birthday",
}
_HYPOTHESIS_TEMPLATE = "This is {}."

_SELECT = """
SELECT l.listing_id, COALESCE(l.title, '') AS title, lf.occasion
FROM listings l
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
WHERE COALESCE(l.title, '') <> ''
ORDER BY l.listing_id
LIMIT %(limit)s;
"""

_UPSERT = """
INSERT INTO listing_features (listing_id, occasion, occasion_confidence, occasion_multilabel, feature_version)
VALUES (%(listing_id)s, %(occasion)s, %(confidence)s, %(multilabel)s, 'nli-v2')
ON CONFLICT (listing_id) DO UPDATE
SET occasion = EXCLUDED.occasion,
    occasion_confidence = EXCLUDED.occasion_confidence,
    occasion_multilabel = EXCLUDED.occasion_multilabel,
    computed_at = NOW();
"""


def classify(
    limit: int = 100000,
    model_id: str = DEFAULT_MODEL,
    revision: str | None = DEFAULT_REVISION,
    threshold: float = 0.55,
    batch_size: int = 32,
    dry_run: bool = False,
) -> None:
    # Query and report before the heavy imports. torch and transformers take
    # minutes to load cold from the NFS venv, and importing them first made a
    # working run indistinguishable from a hang: no output, idle GPU.
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT, {"limit": limit})
        rows = cur.fetchall()

    todo = rows
    log.info(f"Classifying {len(todo)} titles with NLI")
    if not todo:
        return

    log.info("Importing torch + transformers (slow on first use, NFS venv)...")
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    log.info(f"Loading {model_id}@{revision or 'latest'} on {'gpu' if device == 0 else 'cpu'}")
    clf = pipeline(
        "zero-shot-classification",
        model=model_id,
        revision=revision,
        device=device,
        batch_size=batch_size,
    )

    candidates = list(_HYPOTHESES.values())
    label_of = {v: k for k, v in _HYPOTHESES.items()}

    titles = [r["title"] for r in todo]
    results = clf(titles, candidates, hypothesis_template=_HYPOTHESIS_TEMPLATE)
    if isinstance(results, dict):
        results = [results]

    changed = 0
    dist: dict[str, int] = {}
    to_write = []
    for row, res in zip(todo, results):
        top_label = label_of[res["labels"][0]]
        top_score = float(res["scores"][0])

        # An explicit, decisive age beats the model. NLI scored "14 Year Old
        # Birthday Gift" as kids at 0.99; a parsed number is not a judgement
        # call. Ages that decide nothing (14, 16, 29) fall through to NLI.
        by_age = occasion_from_age(row["title"])
        if by_age is not None:
            occasion = by_age
            top_score = 1.0
        elif top_label == NOT_BIRTHDAY:
            occasion = None if top_score >= threshold else "birthday/general"
        else:
            occasion = top_label if top_score >= threshold else "birthday/general"

        # "16th Birthday Girl" and "14 Year Old" scored kids: those ages are not
        # decisive enough for occasion_from_age to claim them, but they still
        # rule out a card for a young child.
        if occasion == "birthday/kids" and ages_rule_out_kids(row["title"]):
            occasion = "birthday/general"
        dist[occasion or "(none)"] = dist.get(occasion or "(none)", 0) + 1
        if occasion != row["occasion"]:
            changed += 1
        multilabel = {
            label_of[lbl]: float(sc)
            for lbl, sc in zip(res["labels"], res["scores"])
            if label_of[lbl] != NOT_BIRTHDAY
        }
        to_write.append(
            {
                "listing_id": row["listing_id"],
                "occasion": occasion,
                "confidence": top_score,
                "multilabel": json.dumps(
                    {o: multilabel.get(o, 0.0) for o in OCCASIONS}
                ),
            }
        )

    log.info(f"NLI assigned: " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    log.info(f"{changed} of {len(todo)} would change from their current label")

    if dry_run:
        log.info("Dry run — nothing written")
        for row, rec in list(zip(todo, to_write))[:25]:
            print(f"  {str(rec['occasion']):24s} {rec['confidence']:.2f}  {row['title'][:64]}")
        return

    with connection() as conn, conn.cursor() as cur:
        for rec in to_write:
            cur.execute(_UPSERT, rec)
    log.info(f"Wrote {len(to_write)} NLI labels")


def main(
    limit: int = 100000,
    model_id: str = DEFAULT_MODEL,
    revision: str | None = typer.Option(DEFAULT_REVISION, help="Pin the model checkpoint to a commit sha"),
    threshold: float = 0.55,
    batch_size: int = 32,
    dry_run: bool = typer.Option(False, help="Report what would change without writing"),
) -> None:
    classify(
        limit=limit,
        model_id=model_id,
        revision=revision,
        threshold=threshold,
        batch_size=batch_size,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    typer.run(main)
