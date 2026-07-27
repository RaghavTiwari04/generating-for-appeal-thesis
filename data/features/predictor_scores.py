"""Persist per-head predictor scores to listing_features.predictor_scores.

Runs the trained saleability predictor over all scraped listings that have
CLIP embeddings, stores the 5 head scores as JSONB. This makes the per-head
signals available for the pricing model and for analysis without re-running
inference at every use site.

Must be run AFTER `models.predictor.train` has produced best.ckpt.

Usage:
    python -m data.features.predictor_scores
    python -m data.features.predictor_scores --limit 5000
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from common.db import connection, engine
from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS as OCCASIONS

log = get_logger(__name__)

DEFAULT_CKPT = Path("./artifacts/predictor/best.ckpt")
DEFAULT_CALIB = Path("./artifacts/predictor/isotonic.joblib")

_SELECT_SQL = """
SELECT lf.listing_id,
       lf.clip_embedding,
       lf.extracted_text,
       lf.occasion,
       l.currency
FROM listing_features lf
JOIN listings l USING (listing_id)
WHERE lf.clip_embedding IS NOT NULL
  AND lf.occasion IS NOT NULL
  AND (lf.predictor_scores IS NULL OR lf.predictor_scores = 'null'::jsonb)
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_UPSERT_SQL = """
INSERT INTO listing_features (listing_id, predictor_scores, feature_version)
VALUES (%(listing_id)s, %(predictor_scores)s, 'predictor-scores-v1')
ON CONFLICT (listing_id) DO UPDATE
SET predictor_scores = EXCLUDED.predictor_scores,
    computed_at      = NOW();
"""

_ADD_COLUMN_SQL = """
ALTER TABLE listing_features
ADD COLUMN IF NOT EXISTS predictor_scores JSONB;
"""

OCCASION_TO_IDX = {o: i for i, o in enumerate(OCCASIONS)}


def run(
    limit: int = 10_000,
    ckpt: Path = DEFAULT_CKPT,
    calib: Path | None = DEFAULT_CALIB,
) -> int:
    if not ckpt.exists():
        raise SystemExit(
            f"Predictor checkpoint not found: {ckpt}. "
            "Run `make train-predictor` first."
        )

    # Ensure column exists (safe to run multiple times)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_ADD_COLUMN_SQL)

    from data.features.clip_embed import CLIPEmbedder
    from models.predictor.infer import CardFeatures, PredictorRunner

    predictor = PredictorRunner(ckpt, calib if (calib and calib.exists()) else None)
    embedder = CLIPEmbedder()

    df = pd.read_sql(_SELECT_SQL, engine(), params={"limit": limit})
    if df.empty:
        log.info("No listings missing predictor scores.")
        return 0

    log.info(f"Scoring {len(df)} listings with predictor...")

    def _parse_emb(val):
        return json.loads(val) if isinstance(val, str) else val
    df["clip_embedding"] = df["clip_embedding"].apply(_parse_emb)
    valid_mask = df["clip_embedding"].apply(lambda v: v is not None and len(v) > 0)
    if not valid_mask.all():
        n_bad = (~valid_mask).sum()
        log.warning(f"Dropping {n_bad} rows with missing clip_embedding")
        df = df[valid_mask].reset_index(drop=True)
    if df.empty:
        log.info("No valid embeddings to score.")
        return 0
    image_embs = np.stack(df["clip_embedding"].apply(np.asarray).to_list()).astype(np.float32)
    texts = df["extracted_text"].fillna("").tolist()
    text_embs = embedder.embed_texts(texts)

    features = [
        CardFeatures(
            image_emb=image_embs[i],
            text_emb=text_embs[i],
            occasion=str(df.iloc[i]["occasion"]),
        )
        for i in range(len(df))
    ]

    batch_size = 256
    all_scores: list[dict] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        all_scores.extend(predictor.score(batch))

    rows = [
        {
            "listing_id": str(df.iloc[i]["listing_id"]),
            "predictor_scores": Jsonb(all_scores[i]),
        }
        for i in range(len(df))
    ]

    with connection() as conn, conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, rows)

    log.info(f"Persisted predictor scores for {len(rows)} listings.")
    return len(rows)


if __name__ == "__main__":
    import typer

    typer.run(run)
