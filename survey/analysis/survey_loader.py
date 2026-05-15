"""Load survey ratings from Postgres and compute per-card aggregates.

Provides two main outputs:
1. `load_ratings(study_id)` — raw long-format DataFrame
2. `aggregate_ratings(df)` — per-(listing_id, dimension) mean + n_raters
3. `to_saleability_labels(agg_df)` — persist purchase_intent mean as
   `saleability_labels` with label_source='survey_<study_id>'
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from common.db import connection, engine
from common.logging import get_logger

log = get_logger(__name__)


LIKERT_DIMENSIONS = [
    "purchase_intent",
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
]


_RATINGS_SQL = """
SELECT
    sr.rating_id,
    sr.participant_id,
    sr.study_id,
    COALESCE(sr.listing_id::text, sr.generated_card_id::text) AS card_key,
    sr.listing_id,
    sr.generated_card_id,
    sr.occasion_shown,
    sr.purchase_intent,
    sr.occasion_fit,
    sr.aesthetic,
    sr.emotional_resonance,
    sr.distinctiveness,
    sr.max_price_gbp,
    sr.free_text,
    sr.rated_at,
    sr.response_time_ms,
    sr.attention_check_pass
FROM survey_ratings sr
WHERE sr.study_id = %(study_id)s;
"""


def load_ratings(study_id: str, *, exclude_failed_attention: bool = True) -> pd.DataFrame:
    df = pd.read_sql(_RATINGS_SQL, engine(), params={"study_id": study_id})
    if exclude_failed_attention:
        df = df[df["attention_check_pass"].fillna(True)]
    return df.reset_index(drop=True)


@dataclass
class AggregatedRatings:
    per_card: pd.DataFrame       # index=card_key, cols=dimension_mean + dimension_n
    n_raters_total: int
    n_items: int


def aggregate_ratings(df: pd.DataFrame) -> AggregatedRatings:
    agg = (
        df.groupby("card_key")[LIKERT_DIMENSIONS]
        .agg(["mean", "count"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return AggregatedRatings(
        per_card=agg,
        n_raters_total=df["participant_id"].nunique(),
        n_items=df["card_key"].nunique(),
    )


_UPSERT_LABEL = """
INSERT INTO saleability_labels (listing_id, label_source, score, raw)
VALUES (%(listing_id)s, %(label_source)s, %(score)s, %(raw)s)
ON CONFLICT (listing_id, label_source) DO UPDATE
SET score = EXCLUDED.score,
    raw   = EXCLUDED.raw,
    created_at = NOW();
"""


def to_saleability_labels(agg: AggregatedRatings, *, study_id: str) -> int:
    """Write purchase_intent_mean as saleability label for each rated listing."""
    label_source = f"survey_{study_id}"
    rows = []
    for card_key, row in agg.per_card.iterrows():
        # Only write for marketplace listings (have a listing_id)
        if row.get("purchase_intent_count", 0) < 3:
            continue
        rows.append(
            {
                "listing_id": card_key,
                "label_source": label_source,
                "score": float((row["purchase_intent_mean"] - 1) / 6),  # 1-7 -> 0-1
                "raw": json.dumps(
                    {dim: float(row[f"{dim}_mean"]) for dim in LIKERT_DIMENSIONS if f"{dim}_mean" in row}
                ),
            }
        )
    with connection() as conn, conn.cursor() as cur:
        cur.executemany(_UPSERT_LABEL, rows)
    return len(rows)


def response_time_filter(df: pd.DataFrame, min_ms: int = 3000) -> pd.DataFrame:
    """Remove suspiciously fast responses."""
    mask = df["response_time_ms"].isna() | (df["response_time_ms"] >= min_ms)
    n_removed = (~mask).sum()
    if n_removed:
        log.info(f"Removed {n_removed} responses with response_time_ms < {min_ms}")
    return df[mask].reset_index(drop=True)
