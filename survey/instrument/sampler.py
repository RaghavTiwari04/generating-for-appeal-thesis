"""Card sampler for survey sessions.

Builds per-participant card lists that are:
- Occasion-balanced (≥3 cards from each top-5 occasion per participant)
- Proxy-score stratified (low / mid / high thirds equally represented)
- Reproducible given a participant_id seed
- Tracked so no card is assigned to a participant twice

For the system-eval study: ensures each participant sees exactly 8 cards
per condition (A/B/C/D) balanced across 8 occasions.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from common.db import engine


StudyType = Literal["main", "system_eval"]

TOP_OCCASIONS = [
    "birthday/general",
    "christmas/general",
    "mothers_day",
    "valentines_day",
    "sympathy/bereavement",
    "thank_you",
    "graduation",
    "anniversary/general",
]

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
               sl.score AS proxy_score
        FROM listings l
        JOIN listing_features lf USING (listing_id)
        JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
        LEFT JOIN saleability_labels sl
               ON sl.listing_id = l.listing_id AND sl.label_source = 'proxy_v1'
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
    pool["tier"] = pd.qcut(
        pool["proxy_score"].fillna(0.5), q=3, labels=["low", "mid", "high"], duplicates="drop"
    )

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
