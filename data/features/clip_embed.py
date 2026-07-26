"""CLIP / SigLIP embeddings for listing images and headline text.

We use a single frozen vision-language backbone for both:
- image embeddings → stored in `listing_features.clip_embedding` (pgvector)
- joint image+text embeddings → consumed by predictor + dedup

Backbone is set by CLIP_MODEL_ID. Note the pooling below reads
`vision_model(...).pooler_output`, which is correct for SigLIP (no projection
head) but not for CLIP, where the projected `get_image_features()` is wanted —
so swapping to a CLIP checkpoint needs a code change, not just the env var.

The processor resizes to a square, so portrait cards are distorted or cropped.
That applies to every CLIP-style backbone and is a limitation of the approach
rather than of this checkpoint.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from common.logging import get_logger

log = get_logger(__name__)


# 384px rather than 224px: cards carry fine typography and illustration detail,
# and the predictor is asked to judge aesthetic and distinctiveness, which live
# in exactly that detail. Same architecture and still 768-d, so it drops into
# the VECTOR(768) column and the predictor's image_dim without a migration —
# roughly 2.9x the pixels for ~3x the embedding compute, once.
#
# siglip-so400m-patch14-384 is stronger again but emits 1152-d, which needs a
# schema change and a predictor config change.
DEFAULT_MODEL_ID = os.environ.get("CLIP_MODEL_ID", "google/siglip-base-patch16-384")
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
            vision_out = self.model.vision_model(**inputs)
            feats = vision_out.pooler_output
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
            text_out = self.model.text_model(**inputs)
            feats = text_out.pooler_output
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


# Rows written per transaction, so an interrupted run keeps what it finished.
COMMIT_EVERY = 200


def run_embed_missing(
    limit: int = 100_000,
    feature_version: str = "siglip-base-384-v1",
    batch_size: int = 32,
) -> int:
    """Embed cover images for listings with no CLIP feature. Returns count written.

    The default limit covers the whole catalogue: the job script invokes this
    with no arguments, and a small default silently embedded a fraction of the
    listings while still reporting success.
    """
    import io as _io

    from common.db import connection
    from common.storage import get_object

    embedder = CLIPEmbedder()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_MISSING, {"limit": limit})
        rows = cur.fetchall()

    log.info(f"Embedding {len(rows)} listings ({batch_size} per GPU batch)")
    if not rows:
        return 0

    processed = failed = 0
    for start in range(0, len(rows), COMMIT_EVERY):
        chunk = rows[start : start + COMMIT_EVERY]

        # Load first, then embed as batches — the previous version called
        # embed_images([img]) per listing, one GPU round trip per image.
        ids, imgs = [], []
        for row in chunk:
            try:
                data = get_object(row["storage_path"])
                imgs.append(Image.open(_io.BytesIO(data)).convert("RGB"))
                ids.append(row["listing_id"])
            except Exception as e:
                failed += 1
                log.warning(f"Load failed for listing {row['listing_id']}: {e}")

        if not imgs:
            continue

        embedder.cfg.batch_size = batch_size
        embeddings = embedder.embed_images(imgs)
        if embeddings.shape[1] != EMBED_DIM:
            raise RuntimeError(
                f"{embedder.cfg.model_id} produced {embeddings.shape[1]}-d embeddings, "
                f"but listing_features.clip_embedding is VECTOR({EMBED_DIM})"
            )

        with connection() as conn, conn.cursor() as cur:
            for listing_id, emb in zip(ids, embeddings):
                cur.execute(
                    _UPSERT_FEATURE,
                    {
                        "listing_id": listing_id,
                        "embedding": emb.tolist(),
                        "version": feature_version,
                    },
                )
        processed += len(ids)
        log.info(f"  {min(start + COMMIT_EVERY, len(rows))}/{len(rows)} — {processed} embedded, {failed} failed")

    log.info(f"Embeddings written: {processed} ({failed} failed)")
    return processed


if __name__ == "__main__":
    import typer

    typer.run(run_embed_missing)
