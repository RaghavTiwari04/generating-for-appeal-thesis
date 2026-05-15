"""Qualitative failure analysis.

Pull the worst-rated 20 cards from condition C, plus their per-rater
comments, and emit:
- a CSV of card_id, mean rating, free-text comments, predicted scores
- per-card image URLs (so the human coder can flip through them)
- a starter taxonomy template the coder fills in

Failure categories:
    occasion misfit, text/image incoherence, tonal mismatch, typography
    failures, uncanny faces, weird hands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.db import engine
from common.logging import get_logger

log = get_logger(__name__)


_QUERY = """
WITH ratings AS (
    SELECT sr.generated_card_id,
           AVG(sr.purchase_intent) AS mean_pi,
           COUNT(*)                AS n_raters,
           STRING_AGG(sr.free_text, ' || ' ORDER BY sr.rated_at)
               FILTER (WHERE sr.free_text IS NOT NULL AND length(sr.free_text) > 0) AS comments
    FROM survey_ratings sr
    WHERE sr.generated_card_id IS NOT NULL
      AND sr.study_id = %(study_id)s
    GROUP BY sr.generated_card_id
    HAVING COUNT(*) >= 5
)
SELECT gc.card_id,
       gc.condition_tag,
       gc.headline_text,
       gc.inside_message,
       gc.cover_path,
       gc.predicted_scores,
       (gc.brief->'request'->>'occasion') AS occasion,
       r.mean_pi,
       r.n_raters,
       r.comments
FROM generated_cards gc
JOIN ratings r ON r.generated_card_id = gc.card_id
WHERE gc.condition_tag = %(condition)s
ORDER BY r.mean_pi ASC
LIMIT %(top_n)s;
"""


CATEGORY_TEMPLATE = [
    "occasion_misfit",
    "text_image_incoherence",
    "tonal_mismatch",
    "typography_failure",
    "uncanny_face",
    "weird_hands",
    "other",
]


def run(study_id: str, condition: str = "C_pipeline_rerank", top_n: int = 20, out_dir: str = "./artifacts/failure_analysis") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_sql(
        _QUERY,
        engine(),
        params={"study_id": study_id, "condition": condition, "top_n": top_n},
    )

    coding = pd.DataFrame(
        {
            "card_id": df["card_id"],
            "mean_pi": df["mean_pi"],
            "n_raters": df["n_raters"],
            "occasion": df["occasion"],
            "headline_text": df["headline_text"],
            "inside_message": df["inside_message"],
            "cover_path": df["cover_path"],
            "comments": df["comments"],
            "predicted_scores": df["predicted_scores"].apply(lambda x: json.dumps(x) if x else ""),
        }
    )
    for cat in CATEGORY_TEMPLATE:
        coding[f"cat_{cat}"] = ""

    out_path = out / f"{condition}_bottom{top_n}.csv"
    coding.to_csv(out_path, index=False)
    log.info(f"Wrote {out_path} ({len(coding)} cards)")
    return out_path


if __name__ == "__main__":
    import typer

    typer.run(run)
