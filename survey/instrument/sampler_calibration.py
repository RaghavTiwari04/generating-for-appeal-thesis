"""Pair sampler for the VLM-calibration Prolific study.

Selects ~200 card pairs stratified by VLM score terciles so the calibration
covers the full quality spectrum. Uses image URLs directly from listings.raw_metadata
(no S3/MinIO dependency — images served from original marketplace CDNs).

Design:
  - Pool: all cards with VLM labels in saleability_labels.
  - Stratified: tercile-balanced (low/mid/high VLM saleability score).
  - Cross-tercile and within-tercile pairs ensure BT can recover rankings
    across the full range.
  - Trapdoor attention checks (~5% of pairs): one card is a very-low-scoring
    card paired with a very-high-scoring card (expect clear winner).
  - Each card appears 3-5 times for BT identifiability.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import pandas as pd

from common.db import engine
from common.logging import get_logger

log = get_logger(__name__)

CALIBRATION_DIMS = (
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
)


@dataclass
class CalibrationCard:
    listing_id: str
    title: str | None
    image_url: str
    source: str
    vlm_saleability: float  # VLM composite score for stratification


@dataclass
class CalibrationPair:
    left: CalibrationCard
    right: CalibrationCard
    contrast_tag: str  # 'cross_tercile' | 'within_tercile' | 'trapdoor'


_POOL_SQL = """
SELECT l.listing_id::text AS listing_id,
       l.title,
       l.source,
       l.raw_metadata->'image_urls'->>0 AS image_url,
       sl.score AS vlm_score
FROM listings l
JOIN saleability_labels sl ON sl.listing_id = l.listing_id
WHERE sl.label_source = %(label_source)s
  AND l.raw_metadata->'image_urls' IS NOT NULL
  AND jsonb_array_length(l.raw_metadata->'image_urls') > 0
ORDER BY sl.score
"""


def _stable_seed(participant_id: str, study_id: str) -> int:
    h = hashlib.sha256(f"{study_id}:{participant_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def load_calibration_pool(
    label_source: str = "llm_ssr_rubric_v2",
) -> list[CalibrationCard]:
    """Load all VLM-labelled cards sorted by score."""
    df = pd.read_sql(_POOL_SQL, engine(), params={"label_source": label_source})
    return [
        CalibrationCard(
            listing_id=row["listing_id"],
            title=row.get("title"),
            image_url=row["image_url"],
            source=row.get("source", "unknown"),
            vlm_saleability=float(row["vlm_score"]),
        )
        for _, row in df.iterrows()
        if row.get("image_url")
    ]


def sample_pairs_calibration(
    participant_id: str,
    study_id: str = "calibration_v1",
    n_pairs: int = 25,
    n_trapdoors: int = 2,
    label_source: str = "llm_ssr_rubric_v2",
) -> list[CalibrationPair]:
    """Build per-participant pair list for the calibration study.

    Strategy:
      - Split pool into terciles by VLM score (low / mid / high).
      - 40% cross-tercile (high vs low) — maximises BT information.
      - 40% adjacent-tercile (high vs mid, mid vs low) — tests mid range.
      - 20% within-tercile — baseline variance.
      - Plus trapdoor pairs (obvious mismatch for attention check).

    Args:
        participant_id: Prolific PID for reproducible seeding.
        study_id: Study identifier.
        n_pairs: Number of real comparison pairs.
        n_trapdoors: Number of attention-check pairs.
        label_source: Which VLM label source to stratify by.

    Returns:
        List of CalibrationPair ready for rendering.
    """
    rng = random.Random(_stable_seed(participant_id, study_id))
    pool = load_calibration_pool(label_source)

    if len(pool) < 20:
        log.warning(f"Calibration pool too small ({len(pool)}). Need VLM labels first.")
        return []

    # Sort by score and split into terciles
    pool.sort(key=lambda c: c.vlm_saleability)
    n = len(pool)
    t1 = n // 3
    t2 = 2 * n // 3
    low = pool[:t1]
    mid = pool[t1:t2]
    high = pool[t2:]

    log.info(
        f"Calibration pool: {n} cards | "
        f"Low [{low[0].vlm_saleability:.2f}-{low[-1].vlm_saleability:.2f}] n={len(low)} | "
        f"Mid [{mid[0].vlm_saleability:.2f}-{mid[-1].vlm_saleability:.2f}] n={len(mid)} | "
        f"High [{high[0].vlm_saleability:.2f}-{high[-1].vlm_saleability:.2f}] n={len(high)}"
    )

    n_cross = int(n_pairs * 0.40)      # high vs low
    n_adjacent = int(n_pairs * 0.40)    # high vs mid, mid vs low
    n_within = n_pairs - n_cross - n_adjacent  # within same tercile

    pairs: list[CalibrationPair] = []
    used_pairs: set[tuple[str, str]] = set()

    def _make_pair(
        pool_a: list[CalibrationCard],
        pool_b: list[CalibrationCard],
        tag: str,
    ) -> CalibrationPair | None:
        for _ in range(50):
            a = rng.choice(pool_a)
            b = rng.choice(pool_b)
            if a.listing_id == b.listing_id:
                continue
            key = tuple(sorted([a.listing_id, b.listing_id]))
            if key in used_pairs:
                continue
            used_pairs.add(key)
            # Randomise L/R
            if rng.random() < 0.5:
                a, b = b, a
            return CalibrationPair(left=a, right=b, contrast_tag=tag)
        return None

    # Cross-tercile: high vs low (most informative)
    for _ in range(n_cross):
        p = _make_pair(high, low, "cross_tercile")
        if p:
            pairs.append(p)

    # Adjacent-tercile: alternate high-vs-mid and mid-vs-low
    for i in range(n_adjacent):
        if i % 2 == 0:
            p = _make_pair(high, mid, "adjacent_tercile")
        else:
            p = _make_pair(mid, low, "adjacent_tercile")
        if p:
            pairs.append(p)

    # Within-tercile
    terciles = [low, mid, high]
    for i in range(n_within):
        t = terciles[i % 3]
        p = _make_pair(t, t, "within_tercile")
        if p:
            pairs.append(p)

    # Trapdoor pairs: top-5% vs bottom-5% (should be obvious)
    top5 = pool[int(n * 0.95):]
    bot5 = pool[:int(n * 0.05)]
    for _ in range(n_trapdoors):
        p = _make_pair(top5, bot5, "trapdoor")
        if p:
            pairs.append(p)

    rng.shuffle(pairs)
    log.info(
        f"Sampled {len(pairs)} pairs: "
        f"{sum(1 for p in pairs if p.contrast_tag == 'cross_tercile')} cross, "
        f"{sum(1 for p in pairs if p.contrast_tag == 'adjacent_tercile')} adjacent, "
        f"{sum(1 for p in pairs if p.contrast_tag == 'within_tercile')} within, "
        f"{sum(1 for p in pairs if p.contrast_tag == 'trapdoor')} trapdoor"
    )
    return pairs


def study_design_summary(
    n_participants: int = 40,
    n_pairs_per_participant: int = 25,
    label_source: str = "llm_ssr_rubric_v2",
) -> dict:
    """Compute study design statistics for preregistration."""
    pool = load_calibration_pool(label_source)
    n_cards = len(pool)
    total_pairs = n_participants * n_pairs_per_participant
    total_judgments = total_pairs * len(CALIBRATION_DIMS)  # 5 dims per pair

    # Each card appears ~(2 * total_pairs / n_cards) times
    avg_appearances = (2 * total_pairs) / max(n_cards, 1)

    # Cost estimate: ~2 min per participant @ £9/hr = £0.30 each
    cost_per_participant = 0.30
    total_cost = n_participants * cost_per_participant

    return {
        "n_cards_in_pool": n_cards,
        "n_participants": n_participants,
        "pairs_per_participant": n_pairs_per_participant,
        "total_pairs": total_pairs,
        "total_judgments_5dim": total_judgments,
        "avg_card_appearances": round(avg_appearances, 1),
        "estimated_time_per_participant_min": 3,
        "estimated_cost_gbp": round(total_cost, 2),
        "dims": list(CALIBRATION_DIMS),
    }
