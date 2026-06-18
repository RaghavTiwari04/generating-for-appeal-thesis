"""Card and pair samplers for survey sessions.

Card samplers (Likert legacy) build per-participant card lists that are:
- Occasion-balanced (≥3 cards from each top-5 occasion per participant)
- Occasion-balanced across active occasions
- Reproducible given a participant_id seed
- Tracked so no card is assigned to a participant twice

For the system-eval study: ensures each participant sees exactly 8 cards
per condition (A/B/C/D) balanced across 8 occasions.

Pair samplers (v2 pairwise) build per-participant pair lists for the 2AFC
instrument. Two strategies:
- `sample_pairs_main`: birthday-only pool; active-learning queue prioritising
  pairs with high BT-score variance, plus a uniform-random fraction for graph
  connectivity. Each card appears 4–8× across the whole study.
- `sample_pairs_system_eval`: cross-condition pairs targeting the three
  pre-registered contrasts (C vs A, C vs B, C vs D). Matched-brief design
  where possible.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from common.db import engine
from common.occasions import ACTIVE_OCCASIONS, is_valid_occasion

StudyType = Literal["main", "system_eval"]

TOP_OCCASIONS = list(ACTIVE_OCCASIONS)

SYSTEM_EVAL_CONDITIONS = [
    "A_naive_ai",
    "B_pipeline_no_rerank",
    "C_pipeline_rerank",
    "D_human_bestseller",
]


@dataclass
class CardAssignment:
    card_key: str           # listing_id or card_id (str)
    is_generated: bool
    condition_tag: str | None
    occasion: str
    cover_path: str
    headline: str | None
    inside_message: str | None


@dataclass
class PairAssignment:
    """One side-by-side pair to present to a participant."""
    left:  CardAssignment
    right: CardAssignment
    occasion: str          # shared (forced match on occasion when possible)
    contrast_tag: str      # 'within' | 'C_vs_A' | 'C_vs_B' | 'C_vs_D' | 'trapdoor' | 'anchor'


def _stable_seed(participant_id: str, study_id: str) -> int:
    h = hashlib.sha256(f"{study_id}:{participant_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _load_main_pool() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT l.listing_id::text AS card_key,
               FALSE AS is_generated,
               NULL AS condition_tag,
               lf.occasion,
               li.storage_path AS cover_path,
               l.title AS headline,
               NULL AS inside_message,
               COALESCE(l.review_count, 0) + COALESCE(l.favourite_count, 0) AS engagement
        FROM listings l
        JOIN listing_features lf USING (listing_id)
        JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
        WHERE li.storage_path IS NOT NULL
        """,
        engine(),
    )


def _load_system_eval_pool() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT gc.card_id::text AS card_key,
               TRUE AS is_generated,
               gc.condition_tag,
               (gc.brief->'request'->>'occasion') AS occasion,
               gc.cover_path,
               gc.headline_text AS headline,
               gc.inside_message
        FROM generated_cards gc
        WHERE gc.condition_tag = ANY(%(conds)s)
          AND gc.cover_path IS NOT NULL
        """,
        engine(),
        params={"conds": SYSTEM_EVAL_CONDITIONS},
    )


def sample_main(
    participant_id: str,
    study_id: str = "main_v1",
    n_cards: int = 30,
    n_per_top_occasion: int = 3,
) -> list[CardAssignment]:
    rng = random.Random(_stable_seed(participant_id, study_id))
    pool = _load_main_pool()
    selected: list[str] = []
    # Guarantee ≥ n_per_top_occasion per top occasion
    for occ in TOP_OCCASIONS:
        sub = pool[pool["occasion"] == occ]
        n = min(n_per_top_occasion, len(sub))
        selected.extend(rng.sample(list(sub["card_key"]), n))

    remaining_budget = n_cards - len(selected)
    rest = pool[~pool["card_key"].isin(selected)]
    if remaining_budget > 0 and len(rest):
        extra = rng.sample(list(rest["card_key"]), min(remaining_budget, len(rest)))
        selected.extend(extra)

    rng.shuffle(selected)
    row_map = pool.set_index("card_key")
    return [
        CardAssignment(
            card_key=k,
            is_generated=bool(row_map.at[k, "is_generated"]),
            condition_tag=row_map.at[k, "condition_tag"],
            occasion=str(row_map.at[k, "occasion"] or ""),
            cover_path=str(row_map.at[k, "cover_path"]),
            headline=row_map.at[k, "headline"],
            inside_message=row_map.at[k, "inside_message"],
        )
        for k in selected
        if k in row_map.index
    ]


# ---------------------------------------------------------------------------
# Pairwise samplers (v2)
# ---------------------------------------------------------------------------

_PAIR_COUNTS_SQL = """
SELECT card_key, COUNT(*)::int AS n
FROM (
    SELECT COALESCE(left_listing_id::text, left_generated_id::text) AS card_key
      FROM survey_pairs WHERE study_id = %(study_id)s
    UNION ALL
    SELECT COALESCE(right_listing_id::text, right_generated_id::text) AS card_key
      FROM survey_pairs WHERE study_id = %(study_id)s
) sub
GROUP BY card_key;
"""


def _current_appearances(study_id: str) -> dict[str, int]:
    """Per-card count of how many times card has appeared in any pair so far."""
    try:
        df = pd.read_sql(_PAIR_COUNTS_SQL, engine(), params={"study_id": study_id})
        return dict(zip(df["card_key"], df["n"], strict=False))
    except Exception:
        # DB might not be reachable in dev; treat as empty.
        return {}


def _row_to_card(row: pd.Series) -> CardAssignment:
    return CardAssignment(
        card_key=str(row["card_key"]),
        is_generated=bool(row.get("is_generated", False)),
        condition_tag=(None if pd.isna(row.get("condition_tag")) else str(row["condition_tag"])),
        occasion=str(row.get("occasion") or ""),
        cover_path=str(row.get("cover_path") or ""),
        headline=(None if pd.isna(row.get("headline")) else str(row["headline"])),
        inside_message=(None if pd.isna(row.get("inside_message")) else str(row["inside_message"])),
    )


def sample_pairs_main(
    participant_id: str,
    study_id: str = "main_v2",
    n_pairs: int = 60,
    max_appearances_per_card: int = 8,
    random_anchor_frac: float = 0.20,
    n_trapdoors: int = 3,
) -> list[PairAssignment]:
    """Build a per-participant pair list for the main pairwise study.

    Active-learning logic (simple variant — full Hessian-variance scoring lives
    in eval/sims/bt_power.py / future BT online-update job):

    1. Load birthday-only card pool. Filter to is_valid_occasion(occ).
    2. Compute per-card *deficit* = max_appearances_per_card - appearances_so_far.
       Cards with high deficit are prioritised so the comparison graph stays
       balanced and BT scores stay identifiable.
    3. Sample `(1 - random_anchor_frac) * n_pairs` high-deficit pairs:
       weight each card by max(0, deficit) and draw pairs with replacement,
       rejecting same-card and cross-occasion pairs.
    4. Sample `random_anchor_frac * n_pairs` uniformly-random pairs for
       graph connectivity guarantees.
    5. Insert trapdoor pairs (broken-variant pairings) at random positions.

    Reproducible per (participant_id, study_id).
    """
    rng = random.Random(_stable_seed(participant_id, study_id))
    np_rng = np.random.default_rng(_stable_seed(participant_id, study_id))

    pool = _load_main_pool()
    pool = pool[pool["occasion"].apply(is_valid_occasion)].reset_index(drop=True)
    if pool.empty:
        return []

    appearances = _current_appearances(study_id)
    pool["appearances"] = pool["card_key"].map(appearances).fillna(0).astype(int)
    pool["deficit"] = (max_appearances_per_card - pool["appearances"]).clip(lower=0)

    n_random = round(n_pairs * random_anchor_frac)
    n_active = n_pairs - n_random - n_trapdoors
    n_active = max(0, n_active)

    pairs: list[PairAssignment] = []
    used_card_pair_keys: set[tuple[str, str]] = set()

    by_occ: dict[str, pd.DataFrame] = {
        occ: pool[pool["occasion"] == occ].reset_index(drop=True)
        for occ in pool["occasion"].unique()
    }

    def _draw_pair(weights_col: str | None) -> PairAssignment | None:
        occ = rng.choice(list(by_occ.keys()))
        sub = by_occ[occ]
        if len(sub) < 2:
            return None
        if weights_col is not None and sub[weights_col].sum() > 0:
            probs = sub[weights_col].astype(float).values
            probs = probs / probs.sum()
            i, j = np_rng.choice(len(sub), size=2, replace=False, p=probs)
        else:
            i, j = np_rng.choice(len(sub), size=2, replace=False)
        a, b = sub.iloc[i], sub.iloc[j]
        key = tuple(sorted([str(a["card_key"]), str(b["card_key"])]))
        if key in used_card_pair_keys:
            return None
        used_card_pair_keys.add(key)
        # Randomise L/R
        if rng.random() < 0.5:
            a, b = b, a
        return PairAssignment(
            left=_row_to_card(a),
            right=_row_to_card(b),
            occasion=occ,
            contrast_tag="active",
        )

    # 1) active-learning draws (weighted by deficit)
    attempts = 0
    while len([p for p in pairs if p.contrast_tag == "active"]) < n_active and attempts < n_active * 20:
        attempts += 1
        p = _draw_pair("deficit")
        if p is not None:
            pairs.append(p)

    # 2) uniform-random anchor draws
    attempts = 0
    while len([p for p in pairs if p.contrast_tag == "anchor"]) < n_random and attempts < n_random * 20:
        attempts += 1
        p = _draw_pair(None)
        if p is not None:
            p.contrast_tag = "anchor"
            pairs.append(p)

    # 3) trapdoor pairs — paired with a known-broken variant if a `_broken`
    #    condition_tag is present in the generated_cards pool; otherwise a
    #    synthetic placeholder. The instrument layer is responsible for
    #    ensuring trapdoor pairs aren't persisted as real comparisons.
    for _ in range(n_trapdoors):
        anchor = pool.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        broken = pd.Series(
            {
                "card_key": f"trapdoor_{anchor['card_key']}",
                "is_generated": True,
                "condition_tag": "trapdoor",
                "occasion": anchor["occasion"],
                "cover_path": "trapdoor://broken",
                "headline": "(blank)",
                "inside_message": "(this card has no message)",
            }
        )
        pair = PairAssignment(
            left=_row_to_card(anchor),
            right=_row_to_card(broken),
            occasion=str(anchor["occasion"]),
            contrast_tag="trapdoor",
        )
        if rng.random() < 0.5:
            pair.left, pair.right = pair.right, pair.left
        pairs.append(pair)

    rng.shuffle(pairs)
    return pairs


def sample_pairs_system_eval(
    participant_id: str,
    study_id: str = "system_eval_v2",
    n_pairs: int = 50,
    n_trapdoors: int = 3,
) -> list[PairAssignment]:
    """Pair list for the four-condition pairwise system eval.

    Sampling budget across the whole study (per-participant subsample drawn
    from this distribution; per-participant seed makes it reproducible):

    - 60% decision-critical (C vs A, C vs B, C vs D) — 20% each
    - 30% within-condition anchors (~7.5% per condition)
    - 10% trapdoors

    Matched-brief: where possible, C-vs-A and C-vs-B pairs use cards generated
    from the *same* brief; C-vs-D matches on occasion + sub-occasion.
    """
    rng = random.Random(_stable_seed(participant_id, study_id))
    np_rng = np.random.default_rng(_stable_seed(participant_id, study_id))

    pool = _load_system_eval_pool()
    pool = pool[pool["occasion"].apply(is_valid_occasion)].reset_index(drop=True)
    if pool.empty:
        return []

    by_cond: dict[str, pd.DataFrame] = {
        c: pool[pool["condition_tag"] == c].reset_index(drop=True)
        for c in SYSTEM_EVAL_CONDITIONS
    }

    n_decision = round(n_pairs * 0.60)
    n_within = n_pairs - n_decision - n_trapdoors
    n_within = max(0, n_within)

    contrasts = [
        ("C_pipeline_rerank", "A_naive_ai",            "C_vs_A"),
        ("C_pipeline_rerank", "B_pipeline_no_rerank",  "C_vs_B"),
        ("C_pipeline_rerank", "D_human_bestseller",    "C_vs_D"),
    ]
    per_contrast = n_decision // 3

    pairs: list[PairAssignment] = []

    def _pick(df: pd.DataFrame) -> pd.Series | None:
        if df.empty:
            return None
        return df.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]

    def _match_on_occasion(left_row: pd.Series, right_df: pd.DataFrame) -> pd.Series | None:
        same = right_df[right_df["occasion"] == left_row["occasion"]]
        return _pick(same) if not same.empty else _pick(right_df)

    for left_cond, right_cond, tag in contrasts:
        left_df = by_cond.get(left_cond, pd.DataFrame())
        right_df = by_cond.get(right_cond, pd.DataFrame())
        for _ in range(per_contrast):
            left_row = _pick(left_df)
            if left_row is None:
                continue
            right_row = _match_on_occasion(left_row, right_df)
            if right_row is None:
                continue
            lc, rc = _row_to_card(left_row), _row_to_card(right_row)
            if rng.random() < 0.5:
                lc, rc = rc, lc
            pairs.append(PairAssignment(left=lc, right=rc,
                                        occasion=str(left_row["occasion"]),
                                        contrast_tag=tag))

    # Within-condition anchors (split across 4 conds, occasion-matched)
    per_cond = n_within // 4
    for cond in SYSTEM_EVAL_CONDITIONS:
        sub = by_cond.get(cond, pd.DataFrame())
        if len(sub) < 2:
            continue
        # Group by occasion so within-condition pairs share the occasion
        by_occ_within = {
            occ: sub[sub["occasion"] == occ].reset_index(drop=True)
            for occ in sub["occasion"].unique()
        }
        eligible_occs = [o for o, df_ in by_occ_within.items() if len(df_) >= 2]
        if not eligible_occs:
            continue
        for _ in range(per_cond):
            occ = rng.choice(eligible_occs)
            df_occ = by_occ_within[occ]
            i, j = np_rng.choice(len(df_occ), size=2, replace=False)
            lc, rc = _row_to_card(df_occ.iloc[i]), _row_to_card(df_occ.iloc[j])
            if rng.random() < 0.5:
                lc, rc = rc, lc
            pairs.append(PairAssignment(
                left=lc, right=rc, occasion=occ,
                contrast_tag=f"within_{cond}"))

    # Trapdoors
    for _ in range(n_trapdoors):
        anchor_row = pool.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        broken = pd.Series({
            "card_key": f"trapdoor_{anchor_row['card_key']}",
            "is_generated": True, "condition_tag": "trapdoor",
            "occasion": anchor_row["occasion"], "cover_path": "trapdoor://broken",
            "headline": "(blank)", "inside_message": "(this card has no message)",
        })
        lc = _row_to_card(anchor_row)
        rc = _row_to_card(broken)
        if rng.random() < 0.5:
            lc, rc = rc, lc
        pairs.append(PairAssignment(
            left=lc, right=rc, occasion=str(anchor_row["occasion"]),
            contrast_tag="trapdoor"))

    rng.shuffle(pairs)
    return pairs


def sample_system_eval(
    participant_id: str,
    study_id: str = "system_eval_v1",
    cards_per_condition: int = 8,
    n_occasions: int = 8,
) -> list[CardAssignment]:
    rng = random.Random(_stable_seed(participant_id, study_id))
    pool = _load_system_eval_pool()

    selected: list[str] = []
    occasions = TOP_OCCASIONS[:n_occasions]
    for cond in SYSTEM_EVAL_CONDITIONS:
        sub_cond = pool[pool["condition_tag"] == cond]
        per_occ = cards_per_condition // len(occasions)
        for occ in occasions:
            sub = sub_cond[sub_cond["occasion"] == occ]
            n = min(per_occ, len(sub))
            selected.extend(rng.sample(list(sub["card_key"]), n))

    rng.shuffle(selected)
    row_map = pool.set_index("card_key")
    return [
        CardAssignment(
            card_key=k,
            is_generated=True,
            condition_tag=str(row_map.at[k, "condition_tag"]),
            occasion=str(row_map.at[k, "occasion"] or ""),
            cover_path=str(row_map.at[k, "cover_path"]),
            headline=row_map.at[k, "headline"],
            inside_message=row_map.at[k, "inside_message"],
        )
        for k in selected
        if k in row_map.index
    ]
