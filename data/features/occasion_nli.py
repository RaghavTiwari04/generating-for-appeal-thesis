"""Zero-shot birthday subtype classification from titles, via NLI entailment.

Keyword rules only fire on explicit evidence. Titles like "Fabulous at Fifty"
or "My little monster's big day" carry the subtype semantically with no
matchable token, and land in birthday/general by default.

This scores each title against one hypothesis per subtype with a natural
language inference model (Yin et al., 2019) — no training data, no API, runs
locally on the cluster GPU.

Deliberately a hybrid, not a replacement:

  1. Not a birthday card by the rules  -> left alone
  2. Explicit age evidence in the title -> rules win. "30th Birthday" is
     milestone regardless of what an entailment model thinks; parsed ages are
     the higher-precision signal.
  3. Everything else                    -> NLI decides, falling back to
     general when no hypothesis clears the threshold.

Writes listing_features.occasion with feature_version 'nli-v1', so a run can
be told apart from the keyword pass.

    python -m data.features.occasion_nli run
    python -m data.features.occasion_nli run --dry-run --limit 200
"""

from __future__ import annotations

import json

import typer

from common.db import connection
from common.logging import get_logger
from data.features.occasion_classifier import (
    _ages_in,
    pick_best_occasion,
    weak_label,
)
from common.occasions import ACTIVE_OCCASIONS as OCCASIONS

log = get_logger(__name__)

DEFAULT_MODEL = "facebook/bart-large-mnli"

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
VALUES (%(listing_id)s, %(occasion)s, %(confidence)s, %(multilabel)s, 'nli-v1')
ON CONFLICT (listing_id) DO UPDATE
SET occasion = EXCLUDED.occasion,
    occasion_confidence = EXCLUDED.occasion_confidence,
    occasion_multilabel = EXCLUDED.occasion_multilabel,
    computed_at = NOW();
"""


def _needs_nli(title: str) -> bool:
    """True when the rules have no explicit evidence to go on.

    Includes titles the rules could not label at all: "Fabulous at Fifty" is a
    milestone card that never says "birthday", so restricting NLI to
    rule-confirmed birthday cards would exclude the very cases it exists for.
    """
    labels = weak_label(title)
    if _ages_in(title.lower()):
        return False                      # parsed age beats an entailment score
    subtype = pick_best_occasion(labels)
    if subtype is None:
        return True                       # unlabelled — let NLI try
    if not subtype.startswith("birthday/"):
        return False                      # rules found a non-birthday occasion
    return subtype == "birthday/general"  # rules found nothing more specific


def classify(
    limit: int = 100000,
    model_id: str = DEFAULT_MODEL,
    threshold: float = 0.55,
    batch_size: int = 32,
    dry_run: bool = False,
) -> None:
    import torch
    from transformers import pipeline

    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT, {"limit": limit})
        rows = cur.fetchall()

    todo = [r for r in rows if _needs_nli(r["title"])]
    log.info(
        f"{len(rows)} listings; {len(todo)} need NLI "
        f"(rest have explicit evidence or are not birthday cards)"
    )
    if not todo:
        return

    device = 0 if torch.cuda.is_available() else -1
    log.info(f"Loading {model_id} on {'gpu' if device == 0 else 'cpu'}")
    clf = pipeline(
        "zero-shot-classification",
        model=model_id,
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
        if top_label == NOT_BIRTHDAY:
            occasion = None if top_score >= threshold else "birthday/general"
        else:
            occasion = top_label if top_score >= threshold else "birthday/general"
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
    threshold: float = 0.55,
    batch_size: int = 32,
    dry_run: bool = typer.Option(False, help="Report what would change without writing"),
) -> None:
    classify(
        limit=limit,
        model_id=model_id,
        threshold=threshold,
        batch_size=batch_size,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    typer.run(main)
