"""Cached-feature Dataset for predictor training.

Pulls a flat training table from Postgres + cached CLIP embeddings, then
serves (image_emb, text_emb, occasion_idx, price_rel, targets) tuples.

Splits are by `seller_id` to avoid style leakage (§4.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.db import engine
from common.occasions import OCCASIONS
from models.predictor.architecture import HEAD_NAMES


OCCASION_TO_IDX: dict[str, int] = {occ: i for i, occ in enumerate(OCCASIONS)}


_TRAIN_SQL = """
SELECT
    l.listing_id,
    l.seller_id,
    l.price_minor_units,
    l.currency,
    lf.occasion,
    lf.clip_embedding,
    lf.extracted_text,
    sl_proxy.score AS proxy_score,
    sl_survey.score AS survey_score,
    sl_survey.raw   AS survey_raw
FROM listings l
JOIN listing_features lf USING (listing_id)
LEFT JOIN saleability_labels sl_proxy
       ON sl_proxy.listing_id = l.listing_id AND sl_proxy.label_source = 'proxy_v1'
LEFT JOIN saleability_labels sl_survey
       ON sl_survey.listing_id = l.listing_id AND sl_survey.label_source LIKE 'survey_%%'
WHERE lf.clip_embedding IS NOT NULL
  AND lf.occasion IS NOT NULL;
"""


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    seed: int = 42


def load_training_frame() -> pd.DataFrame:
    df = pd.read_sql(_TRAIN_SQL, engine())
    if df.empty:
        return df

    # Price-relative-to-occasion-median (log scale)
    df["log_price"] = np.log1p(df["price_minor_units"].fillna(0).astype(float) / 100.0)
    medians = df.groupby("occasion")["log_price"].transform("median")
    df["price_rel"] = (df["log_price"] - medians).fillna(0.0)

    df["occasion_idx"] = df["occasion"].map(OCCASION_TO_IDX).fillna(0).astype(int)
    return df


def split_by_seller(df: pd.DataFrame, cfg: SplitConfig | None = None) -> dict[str, pd.DataFrame]:
    """Partition listings by `seller_id` to avoid style leakage."""
    cfg = cfg or SplitConfig()
    rng = np.random.default_rng(cfg.seed)
    sellers = df["seller_id"].dropna().unique().tolist()
    rng.shuffle(sellers)
    n = len(sellers)
    n_train = math.floor(n * cfg.train_frac)
    n_val = math.floor(n * cfg.val_frac)
    train_sellers = set(sellers[:n_train])
    val_sellers = set(sellers[n_train : n_train + n_val])
    test_sellers = set(sellers[n_train + n_val :])

    def take(rows: pd.DataFrame, sellers: set[str]) -> pd.DataFrame:
        return rows[rows["seller_id"].isin(sellers)].reset_index(drop=True)

    return {
        "train": take(df, train_sellers),
        "val": take(df, val_sellers),
        "test": take(df, test_sellers),
    }


class PredictorDataset(Dataset):
    """Serves (features, targets, mask) tensors.

    Targets are NaN where ground truth is missing for that head; the loss
    masks them out.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        text_embedder=None,
        text_emb_dim: int = 768,
    ):
        self.df = df.reset_index(drop=True)
        self.text_emb_dim = text_emb_dim
        self.text_embedder = text_embedder  # optional callable: list[str] -> np.ndarray
        self._text_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _text_emb(self, idx: int, text: str | None) -> np.ndarray:
        if idx in self._text_cache:
            return self._text_cache[idx]
        if self.text_embedder is None or not text:
            emb = np.zeros(self.text_emb_dim, dtype=np.float32)
        else:
            emb = self.text_embedder([text])[0].astype(np.float32)
        self._text_cache[idx] = emb
        return emb

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_emb = np.asarray(row["clip_embedding"], dtype=np.float32)
        text_emb = self._text_emb(idx, row.get("extracted_text"))
        occasion_idx = int(row["occasion_idx"])
        price_rel = float(row["price_rel"])

        targets, mask = _build_targets(row)

        return {
            "image_emb": torch.from_numpy(image_emb),
            "text_emb": torch.from_numpy(text_emb),
            "occasion_idx": torch.tensor(occasion_idx, dtype=torch.long),
            "price_rel": torch.tensor([price_rel], dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }


def _build_targets(row: pd.Series) -> tuple[list[float], list[float]]:
    """Map row → (targets, mask) aligned with HEAD_NAMES order.

    Targets are the per-head supervision signal in [0, 1]. Mask = 1 where a
    target exists, 0 otherwise.
    """
    targets = [0.0] * len(HEAD_NAMES)
    mask = [0.0] * len(HEAD_NAMES)

    survey_raw = row.get("survey_raw") or {}
    if isinstance(survey_raw, str):
        import json

        try:
            survey_raw = json.loads(survey_raw)
        except Exception:
            survey_raw = {}

    survey_to_head = {
        "occasion_fit": "occasion_fit",
        "aesthetic": "aesthetic",
        "emotional_resonance": "emotional",
        "distinctiveness": "distinctiveness",
        "purchase_intent": "saleability",
    }
    for survey_key, head_name in survey_to_head.items():
        if survey_key in survey_raw:
            head_i = HEAD_NAMES.index(head_name)
            val = float(survey_raw[survey_key])
            # 1..7 Likert → 0..1
            targets[head_i] = max(0.0, min(1.0, (val - 1.0) / 6.0))
            mask[head_i] = 1.0

    # Proxy-only supervision falls onto saleability head if survey absent
    if not mask[HEAD_NAMES.index("saleability")] and pd.notna(row.get("proxy_score")):
        head_i = HEAD_NAMES.index("saleability")
        targets[head_i] = float(row["proxy_score"])
        mask[head_i] = 1.0

    return targets, mask


def make_occasion_sampler(df: pd.DataFrame) -> torch.utils.data.WeightedRandomSampler:
    """Weighted sampler so each batch has balanced occasion representation."""
    counts = df["occasion"].value_counts()
    weights = df["occasion"].map(lambda o: 1.0 / max(1, counts.get(o, 1))).to_numpy()
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(df),
        replacement=True,
    )
