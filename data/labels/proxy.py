"""Compute marketplace-derived saleability proxy labels.

proxy_score = w1*favourite_velocity + w2*review_velocity
            + w3*is_bestseller     + w4*log_review_count

Velocities = change-per-week from `listing_snapshots`. Normalised within each
occasion (z-score then min-max into [0,1]) so that niche-occasion bestsellers
aren't punished for absolute volume.

Output → `saleability_labels` table, label_source='proxy_v1'.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.db import connection, engine
from common.logging import get_logger

log = get_logger(__name__)


@dataclass
class ProxyWeights:
    favourite_velocity: float = 0.35
    review_velocity: float = 0.35
    is_bestseller: float = 0.15
    log_review_count: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return {
            "favourite_velocity": self.favourite_velocity,
            "review_velocity": self.review_velocity,
            "is_bestseller": self.is_bestseller,
            "log_review_count": self.log_review_count,
        }


_SNAPSHOTS_SQL = """
SELECT l.listing_id,
       lf.occasion,
       l.is_bestseller,
       l.review_count,
       l.favourite_count,
       ls.snapshot_at,
       ls.review_count    AS snap_review_count,
       ls.favourite_count AS snap_favourite_count
FROM listings l
JOIN listing_features lf USING (listing_id)
LEFT JOIN listing_snapshots ls USING (listing_id)
WHERE lf.occasion IS NOT NULL
ORDER BY l.listing_id, ls.snapshot_at;
"""


def _velocity(snapshots: pd.DataFrame, col: str) -> float:
    """Per-week slope via simple linear fit. Returns 0 if <2 distinct snapshots."""
    df = snapshots.dropna(subset=[col, "snapshot_at"]).drop_duplicates("snapshot_at")
    if len(df) < 2:
        return 0.0
    ts = (df["snapshot_at"] - df["snapshot_at"].min()).dt.total_seconds().to_numpy() / (
        7 * 24 * 3600.0
    )
    y = df[col].astype(float).to_numpy()
    if ts.max() == 0:
        return 0.0
    slope = np.polyfit(ts, y, 1)[0]
    return max(0.0, float(slope))


def compute_proxy_scores(weights: ProxyWeights | None = None) -> pd.DataFrame:
    """Return one row per listing with features + final proxy score."""
    weights = weights or ProxyWeights()
    raw = pd.read_sql(_SNAPSHOTS_SQL, engine())
    if raw.empty:
        return raw

    base = raw.drop_duplicates("listing_id")[
        ["listing_id", "occasion", "is_bestseller", "review_count", "favourite_count"]
    ].set_index("listing_id")

    velocities = []
    for listing_id, group in raw.groupby("listing_id"):
        velocities.append(
            {
                "listing_id": listing_id,
                "favourite_velocity": _velocity(group, "snap_favourite_count"),
                "review_velocity": _velocity(group, "snap_review_count"),
            }
        )
    vel = pd.DataFrame(velocities).set_index("listing_id")
    df = base.join(vel, how="left").fillna({"favourite_velocity": 0.0, "review_velocity": 0.0})

    df["log_review_count"] = np.log1p(df["review_count"].fillna(0).astype(float))
    df["is_bestseller_f"] = df["is_bestseller"].fillna(False).astype(float)

    # Normalise per-occasion: z-score → clipped → min-max to [0,1]
    feat_cols = ["favourite_velocity", "review_velocity", "log_review_count"]
    for col in feat_cols:
        df[col + "_norm"] = (
            df.groupby("occasion")[col]
            .transform(lambda s: _zscore_clip_minmax(s))
            .fillna(0.0)
        )

    df["proxy_score"] = (
        weights.favourite_velocity * df["favourite_velocity_norm"]
        + weights.review_velocity * df["review_velocity_norm"]
        + weights.is_bestseller * df["is_bestseller_f"]
        + weights.log_review_count * df["log_review_count_norm"]
    ).clip(0.0, 1.0)

    return df.reset_index()


def _zscore_clip_minmax(s: pd.Series, *, clip: float = 3.0) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not math.isfinite(sd) or sd == 0:
        return pd.Series(0.5, index=s.index)
    z = ((s - mu) / sd).clip(-clip, clip)
    return (z + clip) / (2 * clip)


_UPSERT = """
INSERT INTO saleability_labels (listing_id, label_source, score, raw)
VALUES (%(listing_id)s, %(label_source)s, %(score)s, %(raw)s)
ON CONFLICT (listing_id, label_source) DO UPDATE
SET score = EXCLUDED.score,
    raw   = EXCLUDED.raw,
    created_at = NOW();
"""


def persist_proxy_scores(
    df: pd.DataFrame, *, label_source: str = "proxy_v1", weights: ProxyWeights | None = None
) -> int:
    weights = weights or ProxyWeights()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "listing_id": r["listing_id"],
                "label_source": label_source,
                "score": float(r["proxy_score"]),
                "raw": json.dumps(
                    {
                        "favourite_velocity": float(r["favourite_velocity"]),
                        "review_velocity": float(r["review_velocity"]),
                        "log_review_count": float(r["log_review_count"]),
                        "is_bestseller": bool(r["is_bestseller"]),
                        "weights": weights.as_dict(),
                    }
                ),
            }
        )
    with connection() as conn, conn.cursor() as cur:
        cur.executemany(_UPSERT, rows)
    return len(rows)


def run() -> int:
    df = compute_proxy_scores()
    n = persist_proxy_scores(df)
    log.info(f"Persisted proxy labels for {n} listings")
    return n


if __name__ == "__main__":
    import typer

    typer.run(run)
