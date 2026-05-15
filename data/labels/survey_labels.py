"""Convert Prolific survey ratings into per-head training labels.

After a Prolific study run, `survey_ratings` rows exist with 7-point Likert
scores. This module:

1. Aggregates per-(card, dimension) across raters (mean + count).
2. Normalises 1–7 → 0–1.
3. Writes to `saleability_labels` for the saleability head (purchase_intent).
4. Returns a DataFrame suitable for `PredictorDataset` to join on listing_id.

The `raw` JSONB column on `saleability_labels` carries all five sub-scores
so the multi-head predictor can read them in one query.

Usage:
    python -m data.labels.survey_labels --study-id main_v1
"""

from __future__ import annotations

import json

from psycopg.types.json import Jsonb
from dataclasses import dataclass

import pandas as pd
import typer

from common.db import connection, engine
from common.logging import get_logger
from survey.analysis.survey_loader import (
    LIKERT_DIMENSIONS,
    aggregate_ratings,
    load_ratings,
    response_time_filter,
)

log = get_logger(__name__)

HEAD_SURVEY_MAP = {
    "saleability":   "purchase_intent",
    "occasion_fit":  "occasion_fit",
    "aesthetic":     "aesthetic",
    "emotional":     "emotional_resonance",
    "distinctiveness": "distinctiveness",
}


def _likert_to_01(val: float) -> float:
    return max(0.0, min(1.0, (val - 1.0) / 6.0))


_UPSERT = """
INSERT INTO saleability_labels (listing_id, label_source, score, raw)
VALUES (%(listing_id)s, %(label_source)s, %(score)s, %(raw)s)
ON CONFLICT (listing_id, label_source) DO UPDATE
SET score = EXCLUDED.score,
    raw   = EXCLUDED.raw,
    created_at = NOW();
"""


@dataclass
class LabelStats:
    study_id: str
    n_cards_labelled: int
    n_skipped_low_raters: int
    mean_purchase_intent: float


def build_and_persist(
    study_id: str,
    *,
    min_raters: int = 5,
    exclude_failed_attention: bool = True,
) -> LabelStats:
    df = load_ratings(study_id, exclude_failed_attention=exclude_failed_attention)
    df = response_time_filter(df)

    if df.empty:
        raise ValueError(f"No ratings for study_id={study_id!r}")

    agg = aggregate_ratings(df)
    per_card = agg.per_card
    label_source = f"survey_{study_id}"

    rows_written = 0
    rows_skipped = 0
    pi_vals = []

    with connection() as conn, conn.cursor() as cur:
        for card_key, row in per_card.iterrows():
            # Require minimum raters on purchase_intent
            n = int(row.get("purchase_intent_count", 0))
            if n < min_raters:
                rows_skipped += 1
                continue

            pi_mean = float(row["purchase_intent_mean"])
            pi_vals.append(pi_mean)

            raw_dict = {}
            for head_name, survey_dim in HEAD_SURVEY_MAP.items():
                col = f"{survey_dim}_mean"
                if col in row and pd.notna(row[col]):
                    raw_dict[head_name] = _likert_to_01(float(row[col]))

            cur.execute(
                _UPSERT,
                {
                    "listing_id": card_key,
                    "label_source": label_source,
                    "score": _likert_to_01(pi_mean),
                    "raw": Jsonb(raw_dict),
                },
            )
            rows_written += 1

    mean_pi = sum(pi_vals) / len(pi_vals) if pi_vals else 0.0
    log.info(
        f"study={study_id} written={rows_written} skipped={rows_skipped} "
        f"mean_purchase_intent={mean_pi:.2f}"
    )
    return LabelStats(
        study_id=study_id,
        n_cards_labelled=rows_written,
        n_skipped_low_raters=rows_skipped,
        mean_purchase_intent=mean_pi,
    )


def run(study_id: str = "main_v1") -> None:
    stats = build_and_persist(study_id)
    print(
        f"Labelled {stats.n_cards_labelled} cards "
        f"(skipped {stats.n_skipped_low_raters} with < 5 raters). "
        f"Mean PI = {stats.mean_purchase_intent:.2f}/7"
    )


if __name__ == "__main__":
    typer.run(run)
