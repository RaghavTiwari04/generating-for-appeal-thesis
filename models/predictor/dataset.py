"""Cached-feature Dataset for predictor training.

Pulls a flat training table from Postgres + cached CLIP embeddings, then
serves (image_emb, text_emb, occasion_idx, targets, mask) tuples.

Label sources:
  - Heads 1-4 (LLM): saleability_labels.label_source = 'llm_ssr_rubric_v2'
    → rubric-guided judge scores for occasion_fit, aesthetic,
      emotional_resonance, distinctiveness.
  - Head 5 (human): saleability_labels.label_source LIKE 'survey_%_bt_purchase_intent'
    → ~500-card subsample with Bradley-Terry purchase_intent scores
      from Prolific 2AFC study.

Masked multi-task loss: mask[i]=1 only where label exists for that head.
Head 5 mask is 0 on ~80% of cards → loss only backprops purchase_intent
on the labelled subsample (Ruder 2017).

Splits are by `seller_id` to avoid style leakage (§4.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.db import engine
from common.occasions import ACTIVE_OCCASIONS as OCCASIONS
from models.predictor.architecture import HEAD_NAMES, VLM_HEADS

OCCASION_TO_IDX: dict[str, int] = {occ: i for i, occ in enumerate(OCCASIONS)}


_TRAIN_SQL = """
SELECT
    l.listing_id,
    l.seller_id,
    lf.occasion,
    lf.clip_embedding,
    lf.extracted_text,
    -- LLM labels: rubric judge for the quality dims, SSR for purchase intent
    -- `raw` carries every dimension; `score` is only the sortable summary and
    -- duplicates one of them, so the heads read `raw`.
    sl_vlm.raw AS vlm_raw,
    -- Human BT purchase_intent (available for ~500-card subsample)
    sl_bt_pi.score AS bt_purchase_intent
FROM listings l
JOIN listing_features lf USING (listing_id)
LEFT JOIN saleability_labels sl_vlm
       ON sl_vlm.listing_id = l.listing_id
      AND sl_vlm.label_source = 'llm_ssr_rubric_v2'
LEFT JOIN saleability_labels sl_bt_pi
       ON sl_bt_pi.listing_id = l.listing_id
      AND sl_bt_pi.label_source LIKE 'survey_%%_bt_purchase_intent'
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

    import json as _json
    def _parse_embedding(val):
        if isinstance(val, str):
            return _json.loads(val)
        return val
    df["clip_embedding"] = df["clip_embedding"].apply(_parse_embedding)

    df["occasion_idx"] = df["occasion"].map(OCCASION_TO_IDX).fillna(0).astype(int)
    return df


def split_by_seller(df: pd.DataFrame, cfg: SplitConfig | None = None) -> dict[str, pd.DataFrame]:
    """Partition listings by `seller_id` to avoid style leakage."""
    cfg = cfg or SplitConfig()
    rng = np.random.default_rng(cfg.seed)
    # Give NULL-seller rows a unique synthetic seller each so they enter splits
    null_mask = df["seller_id"].isna()
    df = df.copy()
    df.loc[null_mask, "seller_id"] = [
        f"__null_{i}" for i in range(null_mask.sum())
    ]
    sellers = df["seller_id"].unique().tolist()
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
        # Embedded once up front, in one batch. Embedding per __getitem__ ran a
        # transformer forward pass per row per epoch, which dominated a training
        # step whose actual model is a small MLP over cached features.
        self._text_embs = self._embed_all(text_embedder)

    def _embed_all(self, text_embedder) -> np.ndarray:
        texts = self.df.get("extracted_text")
        n = len(self.df)
        embs = np.zeros((n, self.text_emb_dim), dtype=np.float32)
        if text_embedder is None or texts is None:
            return embs
        present = [(i, t) for i, t in enumerate(texts.tolist()) if t]
        if not present:
            return embs
        vectors = text_embedder([t for _, t in present])
        for (i, _), v in zip(present, vectors, strict=True):
            embs[i] = np.asarray(v, dtype=np.float32)
        return embs

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_emb = np.asarray(row["clip_embedding"], dtype=np.float32)
        text_emb = self._text_embs[idx]
        occasion_idx = int(row["occasion_idx"])

        targets, mask = _build_targets(row)

        return {
            "image_emb": torch.from_numpy(image_emb),
            "text_emb": torch.from_numpy(text_emb),
            "occasion_idx": torch.tensor(occasion_idx, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }


def _parse_raw(val) -> dict:
    if not val:
        return {}
    if isinstance(val, str):
        import json
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val


def _build_targets(row: pd.Series) -> tuple[list[float], list[float]]:
    """Map row → (targets, mask) aligned with HEAD_NAMES order.

    Label priority per head:
      heads 1-4: rubric judge scores
      head 5 (purchase_intent): human Bradley-Terry > SSR

    Masked multi-task loss: mask[i]=0 where no label → head gets zero
    gradient for that sample (Ruder 2017).
    """
    targets = [0.0] * len(HEAD_NAMES)
    mask = [0.0] * len(HEAD_NAMES)

    def _set(head_name: str, val: float) -> None:
        i = HEAD_NAMES.index(head_name)
        targets[i] = max(0.0, min(1.0, val))
        mask[i] = 1.0

    merged_vlm = _parse_raw(row.get("vlm_raw"))

    for head_name in VLM_HEADS:
        if head_name in merged_vlm:
            _set(head_name, float(merged_vlm[head_name]))

    # purchase_intent: human Bradley-Terry > SSR
    if pd.notna(row.get("bt_purchase_intent")):
        _set("purchase_intent", float(row["bt_purchase_intent"]))
    elif "purchase_intent" in merged_vlm:
        _set("purchase_intent", float(merged_vlm["purchase_intent"]))

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
