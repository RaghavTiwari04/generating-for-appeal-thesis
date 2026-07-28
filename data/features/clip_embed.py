"""CLIP / SigLIP embeddings for listing images and headline text.

We use a single frozen vision-language backbone for both:
- image embeddings → stored in `listing_features.clip_embedding` (pgvector)
- joint image+text embeddings → consumed by predictor + dedup

Backbone is set by CLIP_MODEL_ID. Note the pooling below reads
`vision_model(...).pooler_output`, which is correct for SigLIP (no projection
head) but not for CLIP, where the projected `get_image_features()` is wanted —
so swapping to a CLIP checkpoint needs a code change, not just the env var.

Two knobs address what the backbone would otherwise lose on card images:

  CLIP_PAD_SQUARE=1  pad to square before the processor rather than letting it
      squash a portrait card into 384x384. Padding uses the border's median
      colour, which for card art is usually the background, so the model is not
      shown hard letterbox edges as a feature.

  CLIP_CROPS=5  embed the whole image plus four quadrants and average. The
      judge sees these cards at ~1030px while the backbone sees a 384px
      downsample, so fine typography and print texture — exactly what the
      aesthetic and distinctiveness dimensions turn on — never reach it.
      Averaging keeps the result 768-d, so no migration.

Both must match between training and rerank time: pipeline/rerank.py embeds
candidates live, and a mismatch would silently score them against a different
feature distribution.
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
    pad_to_square: bool = os.environ.get("CLIP_PAD_SQUARE", "0") == "1"
    # 1 = whole image only. 5 = whole image plus four quadrants, averaged.
    crops: int = int(os.environ.get("CLIP_CROPS", "1"))


def _pad_to_square(img: Image.Image) -> Image.Image:
    """Letterbox onto a square using the border's median colour.

    The processor resizes to a square regardless, so a portrait card is
    otherwise squashed — every proportion the model sees is wrong. Padding with
    the median border colour keeps the geometry without teaching the model a
    hard edge, since for card art that colour is usually the background.
    """
    w, h = img.size
    if w == h:
        return img
    arr = np.asarray(img)
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    bg = tuple(int(v) for v in np.median(edges, axis=0))
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), bg)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def _quadrants(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    return [
        img.crop((0, 0, w // 2, h // 2)),
        img.crop((w // 2, 0, w, h // 2)),
        img.crop((0, h // 2, w // 2, h)),
        img.crop((w // 2, h // 2, w, h)),
    ]


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

    def _views(self, img: Image.Image) -> list[Image.Image]:
        """The views of one card that get embedded and averaged."""
        views = [img] + (_quadrants(img) if self.cfg.crops >= 5 else [])
        return [_pad_to_square(v) for v in views] if self.cfg.pad_to_square else views

    @torch.inference_mode()
    def _encode(self, imgs: list[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), self.cfg.batch_size):
            batch = imgs[i : i + self.cfg.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.cfg.device)
            feats = self.model.vision_model(**inputs).pooler_output
            out.append(torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def embed_images(self, images: Iterable[Image.Image | Path]) -> np.ndarray:
        imgs = [Image.open(i).convert("RGB") if isinstance(i, (str, Path)) else i for i in images]
        if not imgs:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        views = [self._views(img) for img in imgs]
        n_views = len(views[0])
        feats = self._encode([v for per_image in views for v in per_image])
        if n_views == 1:
            return feats
        # Average the views back to one vector per card, then renormalise so the
        # result stays a unit vector like the single-view case — dedup and the
        # predictor both assume that.
        pooled = feats.reshape(len(imgs), n_views, -1).mean(axis=1)
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

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
            for listing_id, emb in zip(ids, embeddings, strict=True):
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
