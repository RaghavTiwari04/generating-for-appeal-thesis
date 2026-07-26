"""Pair sampler for the primary purchase_intent Prolific 2AFC study.

This is the MAIN human labeling study — not a calibration exercise.
It produces Bradley-Terry purchase_intent scores that become the 5th
predictor head's training signal.

Pool: stratified subsample of ~500 cards from the 2,377 scraped listings,
ensuring coverage across VLM quality terciles, sources, and birthday
sub-occasions.

Design targets:
  - 500 cards, each appearing ~8 times → ~2,000 pairs
  - 30 pairs per participant → ~70 participants
  - 1 question per pair: "Which card would you be more likely to buy?"
  - BT scaling → [0,1] purchase_intent score per card
  - ~5 min per participant @ £6/hr → £0.50 each → ~£35 total

Also includes 3 trapdoor pairs per participant (obvious quality mismatch).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.db import connection, engine
from common.logging import get_logger

log = get_logger(__name__)

STUDY_ID_DEFAULT = "purchase_intent_v1"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PurchaseCard:
    listing_id: str
    title: str | None
    image_url: str
    source: str
    vlm_composite: float
    tercile: str  # 'low' | 'mid' | 'high'


@dataclass
class PurchasePair:
    left: PurchaseCard
    right: PurchaseCard
    contrast_tag: str  # 'cross' | 'adjacent' | 'within' | 'trapdoor'


# ---------------------------------------------------------------------------
# Pool loading + subsample selection
# ---------------------------------------------------------------------------

_FULL_POOL_SQL = """
SELECT l.listing_id::text AS listing_id,
       l.title,
       l.source,
       l.raw_metadata->'image_urls'->>0 AS image_url,
       sl.score AS vlm_composite
FROM listings l
JOIN saleability_labels sl ON sl.listing_id = l.listing_id
WHERE sl.label_source = %(label_source)s
  AND l.raw_metadata->'image_urls' IS NOT NULL
  AND jsonb_array_length(l.raw_metadata->'image_urls') > 0
  AND (l.raw_metadata->'image_urls'->>0) IS NOT NULL
ORDER BY sl.score
"""


def select_study_pool(
    n_cards: int = 500,
    label_source: str = "llm_ssr_rubric_v1",
    seed: int = 42,
) -> list[PurchaseCard]:
    """Select a stratified subsample of cards for the purchase_intent study.

    Strategy:
      - Split full pool into terciles by VLM composite score.
      - Sample proportionally: 150 low, 150 mid, 150 high = 450.
      - Oversample extremes: 25 from bottom 5%, 25 from top 5% = 50.
      - Total: 500 cards with guaranteed quality-range coverage.
      - Within each stratum, balance across sources where possible.

    Returns:
        Sorted list of PurchaseCard (by vlm_composite).
    """
    rng = random.Random(seed)
    df = pd.read_sql(_FULL_POOL_SQL, engine(), params={"label_source": label_source})

    if df.empty:
        log.error("No VLM-labelled cards found. Run vlm_labels first.")
        return []

    n_total = len(df)
    log.info(f"Full VLM pool: {n_total} cards")

    if n_total <= n_cards:
        log.info(f"Pool ({n_total}) <= target ({n_cards}), using all cards")
        df["tercile"] = pd.qcut(df["vlm_composite"], 3, labels=["low", "mid", "high"])
        return [_row_to_card(row) for _, row in df.iterrows()]

    # Tercile boundaries
    df = df.sort_values("vlm_composite").reset_index(drop=True)
    t1 = n_total // 3
    t2 = 2 * n_total // 3

    low_df = df.iloc[:t1].copy()
    mid_df = df.iloc[t1:t2].copy()
    high_df = df.iloc[t2:].copy()
    low_df["tercile"] = "low"
    mid_df["tercile"] = "mid"
    high_df["tercile"] = "high"

    # Extreme tails
    bot5 = df.iloc[:max(1, int(n_total * 0.05))].copy()
    top5 = df.iloc[max(0, int(n_total * 0.95)):].copy()
    bot5["tercile"] = "low"
    top5["tercile"] = "high"

    n_per_tercile = (n_cards - 50) // 3  # 150 each
    n_extreme = 25

    selected_ids: set[str] = set()

    def _sample_from(df_sub: pd.DataFrame, n: int) -> pd.DataFrame:
        """Source-balanced sampling within a stratum."""
        available = df_sub[~df_sub["listing_id"].isin(selected_ids)]
        if len(available) <= n:
            selected_ids.update(available["listing_id"])
            return available

        # Try to balance across sources
        sources = available["source"].unique()
        per_source = max(1, n // len(sources))
        picked = []
        for src in sources:
            src_df = available[available["source"] == src]
            take = min(per_source, len(src_df))
            sampled = src_df.sample(take, random_state=rng.randint(0, 10**9))
            picked.append(sampled)

        result = pd.concat(picked).drop_duplicates("listing_id")
        # Fill remaining budget randomly
        remaining = n - len(result)
        if remaining > 0:
            extra_pool = available[~available["listing_id"].isin(result["listing_id"])]
            if len(extra_pool) > 0:
                extra = extra_pool.sample(
                    min(remaining, len(extra_pool)),
                    random_state=rng.randint(0, 10**9),
                )
                result = pd.concat([result, extra])

        result = result.head(n)
        selected_ids.update(result["listing_id"])
        return result

    # Sample extremes first (they're rare, don't want to miss them)
    extreme_low = _sample_from(bot5, n_extreme)
    extreme_high = _sample_from(top5, n_extreme)

    # Then terciles (excluding already-selected extremes)
    sampled_low = _sample_from(low_df, n_per_tercile)
    sampled_mid = _sample_from(mid_df, n_per_tercile)
    sampled_high = _sample_from(high_df, n_per_tercile)

    all_sampled = pd.concat([
        extreme_low, sampled_low, sampled_mid, sampled_high, extreme_high
    ]).drop_duplicates("listing_id")

    cards = [_row_to_card(row) for _, row in all_sampled.iterrows()]
    cards.sort(key=lambda c: c.vlm_composite)

    # Log distribution
    by_tercile = {}
    for c in cards:
        by_tercile[c.tercile] = by_tercile.get(c.tercile, 0) + 1
    by_source = {}
    for c in cards:
        by_source[c.source] = by_source.get(c.source, 0) + 1

    log.info(
        f"Selected {len(cards)} cards | "
        f"Terciles: {by_tercile} | "
        f"Sources: {by_source}"
    )
    return cards


def _row_to_card(row: pd.Series) -> PurchaseCard:
    return PurchaseCard(
        listing_id=str(row["listing_id"]),
        title=row.get("title"),
        image_url=str(row["image_url"]),
        source=str(row.get("source", "unknown")),
        vlm_composite=float(row.get("vlm_composite", 0.5)),
        tercile=str(row.get("tercile", "mid")),
    )


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------

def _stable_seed(participant_id: str, study_id: str) -> int:
    h = hashlib.sha256(f"{study_id}:{participant_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


_PAIR_COUNTS_SQL = """
SELECT card_key, COUNT(*)::int AS n
FROM (
    SELECT left_listing_id::text AS card_key
      FROM survey_pairs WHERE study_id = %(study_id)s
    UNION ALL
    SELECT right_listing_id::text AS card_key
      FROM survey_pairs WHERE study_id = %(study_id)s
) sub
WHERE card_key IS NOT NULL
GROUP BY card_key;
"""


def _current_appearances(study_id: str) -> dict[str, int]:
    try:
        df = pd.read_sql(_PAIR_COUNTS_SQL, engine(), params={"study_id": study_id})
        return dict(zip(df["card_key"], df["n"], strict=False))
    except Exception:
        return {}


def sample_pairs_purchase(
    participant_id: str,
    study_id: str = STUDY_ID_DEFAULT,
    n_pairs: int = 30,
    n_trapdoors: int = 3,
    max_appearances: int = 12,
    label_source: str = "llm_ssr_rubric_v1",
) -> list[PurchasePair]:
    """Build per-participant pair list for the purchase_intent study.

    Sampling strategy:
      - Deficit-weighted: cards with fewer appearances get priority
        (keeps BT comparison graph balanced).
      - 25% FULLY RANDOM pairs (VLM-agnostic, for unbiased BT baseline).
      - 30% cross-tercile (high vs low) for max BT information.
      - 25% adjacent-tercile (high vs mid, mid vs low).
      - 20% within-tercile.
      - Trapdoor: top-5% vs bottom-5% for attention checks.

    The random component ensures BT estimates are not biased by VLM
    score stratification (see §4.4 limitations discussion).

    Args:
        participant_id: Prolific PID for reproducible seeding.
        study_id: Study identifier.
        n_pairs: Real comparison pairs per participant.
        n_trapdoors: Attention-check pairs.
        max_appearances: Target max appearances per card.
        label_source: VLM label source for pool loading.

    Returns:
        List of PurchasePair ready for rendering.
    """
    rng = random.Random(_stable_seed(participant_id, study_id))
    np_rng = np.random.default_rng(_stable_seed(participant_id, study_id))

    pool = select_study_pool(label_source=label_source)
    if len(pool) < 20:
        return []

    # Compute deficit weights
    appearances = _current_appearances(study_id)
    for card in pool:
        card._appearances = appearances.get(card.listing_id, 0)
        card._deficit = max(0, max_appearances - card._appearances)

    # Split by tercile
    by_tercile: dict[str, list[PurchaseCard]] = {"low": [], "mid": [], "high": []}
    for card in pool:
        by_tercile[card.tercile].append(card)

    # Pair budget: 25% random, 30% cross, 25% adjacent, 20% within
    n_random = int(n_pairs * 0.25)
    n_cross = int(n_pairs * 0.30)
    n_adjacent = int(n_pairs * 0.25)
    n_within = n_pairs - n_random - n_cross - n_adjacent

    pairs: list[PurchasePair] = []
    used_keys: set[tuple[str, str]] = set()

    def _draw_pair(
        pool_a: list[PurchaseCard],
        pool_b: list[PurchaseCard],
        tag: str,
        use_deficit: bool = True,
    ) -> PurchasePair | None:
        for _ in range(100):
            if use_deficit:
                weights_a = np.array([max(c._deficit, 0.1) for c in pool_a])
                weights_b = np.array([max(c._deficit, 0.1) for c in pool_b])
                weights_a /= weights_a.sum()
                weights_b /= weights_b.sum()
                i = np_rng.choice(len(pool_a), p=weights_a)
                j = np_rng.choice(len(pool_b), p=weights_b)
            else:
                i = rng.randrange(len(pool_a))
                j = rng.randrange(len(pool_b))

            a, b = pool_a[i], pool_b[j]
            if a.listing_id == b.listing_id:
                continue
            key = tuple(sorted([a.listing_id, b.listing_id]))
            if key in used_keys:
                continue
            used_keys.add(key)
            if rng.random() < 0.5:
                a, b = b, a
            return PurchasePair(left=a, right=b, contrast_tag=tag)
        return None

    # 1) Fully random pairs — VLM-agnostic, unbiased BT baseline
    for _ in range(n_random):
        p = _draw_pair(pool, pool, "random", use_deficit=True)
        if p:
            pairs.append(p)

    # 2) Cross-tercile: high vs low (most BT information)
    for _ in range(n_cross):
        p = _draw_pair(by_tercile["high"], by_tercile["low"], "cross")
        if p:
            pairs.append(p)

    # 3) Adjacent: high-vs-mid and mid-vs-low alternating
    for i in range(n_adjacent):
        if i % 2 == 0:
            p = _draw_pair(by_tercile["high"], by_tercile["mid"], "adjacent")
        else:
            p = _draw_pair(by_tercile["mid"], by_tercile["low"], "adjacent")
        if p:
            pairs.append(p)

    # 4) Within-tercile (fine-grained discrimination)
    tercile_keys = ["low", "mid", "high"]
    for i in range(n_within):
        t = tercile_keys[i % 3]
        p = _draw_pair(by_tercile[t], by_tercile[t], "within")
        if p:
            pairs.append(p)

    # Trapdoor pairs
    n_pool = len(pool)
    top5 = [c for c in pool if c.vlm_composite >= pool[int(n_pool * 0.95)].vlm_composite]
    bot5 = [c for c in pool if c.vlm_composite <= pool[max(0, int(n_pool * 0.05))].vlm_composite]
    if top5 and bot5:
        for _ in range(n_trapdoors):
            p = _draw_pair(top5, bot5, "trapdoor", use_deficit=False)
            if p:
                pairs.append(p)

    rng.shuffle(pairs)

    log.info(
        f"Sampled {len(pairs)} pairs for {participant_id}: "
        f"{sum(1 for p in pairs if p.contrast_tag == 'random')} random, "
        f"{sum(1 for p in pairs if p.contrast_tag == 'cross')} cross, "
        f"{sum(1 for p in pairs if p.contrast_tag == 'adjacent')} adjacent, "
        f"{sum(1 for p in pairs if p.contrast_tag == 'within')} within, "
        f"{sum(1 for p in pairs if p.contrast_tag == 'trapdoor')} trapdoor"
    )
    return pairs


# ---------------------------------------------------------------------------
# Persistence of the selected study pool (for reproducibility)
# ---------------------------------------------------------------------------

_POOL_PERSIST_SQL = """
INSERT INTO scrape_jobs (source, job_type, seed, status, counts)
VALUES ('purchase_intent_study', 'pool_selection', %(seed)s, 'completed',
        %(counts)s)
ON CONFLICT DO NOTHING;
"""


def persist_study_pool(cards: list[PurchaseCard], seed: int = 42) -> None:
    """Save the selected pool to DB for reproducibility."""
    from psycopg.types.json import Jsonb

    counts = {
        "n_cards": len(cards),
        "by_tercile": {},
        "by_source": {},
        "listing_ids": [c.listing_id for c in cards],
    }
    for c in cards:
        counts["by_tercile"][c.tercile] = counts["by_tercile"].get(c.tercile, 0) + 1
        counts["by_source"][c.source] = counts["by_source"].get(c.source, 0) + 1

    with connection() as conn, conn.cursor() as cur:
        cur.execute(_POOL_PERSIST_SQL, {"seed": str(seed), "counts": Jsonb(counts)})
    log.info(f"Persisted study pool: {len(cards)} cards")


def study_design_summary(
    n_cards: int = 500,
    n_participants: int = 70,
    n_pairs_per_participant: int = 30,
    label_source: str = "llm_ssr_rubric_v1",
) -> dict:
    """Compute study design stats for preregistration / protocol."""
    total_pairs = n_participants * n_pairs_per_participant
    avg_appearances = (2 * total_pairs) / max(n_cards, 1)
    cost_per_participant_gbp = 0.50
    total_cost_gbp = n_participants * cost_per_participant_gbp

    return {
        "n_cards_in_pool": n_cards,
        "n_participants": n_participants,
        "pairs_per_participant": n_pairs_per_participant,
        "total_pairs": total_pairs,
        "avg_card_appearances": round(avg_appearances, 1),
        "estimated_time_per_participant_min": 5,
        "estimated_cost_gbp": round(total_cost_gbp, 2),
        "payment_per_participant_gbp": cost_per_participant_gbp,
        "question": "Which card would you be more likely to buy for this birthday?",
        "analysis": "Bradley-Terry MLE → [0,1] purchase_intent score per card",
    }
