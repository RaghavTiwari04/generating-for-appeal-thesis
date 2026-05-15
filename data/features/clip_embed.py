"""CLIP / SigLIP embeddings for listing images and headline text.

We use a single frozen vision-language backbone for both:
- image embeddings → stored in `listing_features.clip_embedding` (pgvector)
- joint image+text embeddings → consumed by predictor + dedup

Backbone choice (SigLIP-base or CLIP-ViT-L) configurable via env var. SigLIP
default — better text-image alignment per evals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from common.logging import get_logger

log = get_logger(__name__)


DEFAULT_MODEL_ID = os.environ.get("CLIP_MODEL_ID", "google/siglip-base-patch16-224")
EMBED_DIM = 768  # matches migrations/0001_init.sql VECTOR(768)


@dataclass
class EmbedderConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    batch_size: int = 32


class CLIPEmbedder:
    def __init__(self, cfg: EmbedderConfig | None = None):
        self.cfg = cfg or EmbedderConfig()
        log.info(f"Loading embedder: {self.cfg.model_id} on {self.cfg.device}")
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self.model = (
            AutoModel.from_pretrained(self.cfg.model_id, torch_dtype=self.cfg.dtype)
            .to(self.cfg.device)
            .eval()
        )

    @torch.inference_mode()
    def embed_images(self, images: Iterable[Image.Image | Path]) -> np.ndarray:
        imgs = [Image.open(i).convert("RGB") if isinstance(i, (str, Path)) else i for i in images]
        if not imgs:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        out = []
        for i in range(0, len(imgs), self.cfg.batch_size):
            batch = imgs[i : i + self.cfg.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.cfg.device)
            feats = self.model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i : i + self.cfg.batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.cfg.device)
            feats = self.model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Bulk job: compute + persist embeddings for listings missing them
# ---------------------------------------------------------------------------
_SELECT_MISSING = """
SELECT l.listing_id, li.storage_path
FROM listings l
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
WHERE lf.clip_embedding IS NULL
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_UPSERT_FEATURE = """
INSERT INTO listing_features (listing_id, clip_embedding, feature_version)
VALUES (%(listing_id)s, %(embedding)s, %(version)s)
ON CONFLICT (listing_id) DO UPDATE
SET clip_embedding = EXCLUDED.clip_embedding,
    feature_version = EXCLUDED.feature_version,
    computed_at = NOW();
"""


def run_embed_missing(limit: int = 1000, feature_version: str = "siglip-base-v1") -> int:
    """Embed images for listings without a CLIP feature row. Returns count processed."""
    from common.db import connection
    from common.storage import get_object

    embedder = CLIPEmbedder()
    processed = 0
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_MISSING, {"limit": limit})
        rows = cur.fetchall()

        for row in rows:
            try:
                data = get_object(row["storage_path"])
                import io as _io
                img = Image.open(_io.BytesIO(data))
                img = img.convert("RGB")
                emb = embedder.embed_images([img])[0]
                cur.execute(
                    _UPSERT_FEATURE,
                    {
                        "listing_id": row["listing_id"],
                        "embedding": emb.tolist(),
                        "version": feature_version,
                    },
                )
                processed += 1
            except Exception as e:
                log.warning(f"Embed failed for listing {row['listing_id']}: {e}")
    return processed


if __name__ == "__main__":
    import typer

    typer.run(run_embed_missing)
