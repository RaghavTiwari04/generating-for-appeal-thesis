"""CLIP / SigLIP embeddings for listing images and headline text.

We use a single frozen vision-language backbone for both:
- image embeddings → stored in `listing_features.clip_embedding` (pgvector)
- joint image+text embeddings → consumed by predictor + dedup

Backbone is set by CLIP_MODEL_ID. Note the pooling below reads
`vision_model(...).pooler_output`, which is correct for SigLIP (no projection
head) but not for CLIP, where the projected `get_image_features()` is wanted —
so swapping to a CLIP checkpoint needs a code change, not just the env var.

Three knobs exist for what the backbone might otherwise lose on card images.
The first two were measured and made things worse; they stay switchable because
the ablation is worth reporting, but both default off.

  CLIP_PAD_SQUARE=1  pad to square before the processor rather than letting it
      squash a portrait card into 384x384. Padding uses the border's median
      colour, which for card art is usually the background, so the model is not
      shown hard letterbox edges as a feature.

  CLIP_CROPS=5  embed the whole image plus four quadrants and average. The
      judge sees these cards at ~1030px while the backbone sees a 384px
      downsample, so fine typography and print texture — exactly what the
      aesthetic and distinctiveness dimensions turn on — never reach it.
      Averaging keeps the result 768-d, so no migration.

      Measured on the fixed split, both together lose on every head against the
      plain single view and cost four extra forward passes per image. See
      EmbedderConfig below for the numbers.

  CLIP_MODEL_ID=a,b  embed with each backbone and concatenate. A second
      encoder with a different inductive bias — DINOv2 alongside SigLIP —
      carries texture and visual-similarity structure a single tower does not.
      The first entry supplies the text tower, since DINOv2 has none.

Every variant lands in `listing_features.image_features` (REAL[], any width, no
index), and readers prefer that column when it is populated. `clip_embedding`
is VECTOR(768) behind the HNSW index dedup searches, and it is written once and
then left alone: the duplicate clusters the labelled pool was built from came
out of those vectors, so re-embedding must not move them.

All of these must match between training and rerank time: pipeline/rerank.py
embeds candidates live, and a mismatch would silently score them against a
different feature distribution.
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
# The width `clip_embedding` can hold; wider features go to image_features.
VECTOR_COLUMN_DIM = 768
EMBED_DIM = VECTOR_COLUMN_DIM  # kept for callers that assume the default stack


@dataclass
class EmbedderConfig:
    # Comma-separated to concatenate backbones; the first supplies the text
    # tower, since a vision-only encoder such as DINOv2 has none.
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    batch_size: int = 32
    # Plain single-view embedding, and it has to be a default rather than an env
    # var: pipeline/rerank.py builds a CLIPEmbedder() with no environment set,
    # so a candidate would otherwise be embedded differently from the corpus the
    # predictor trained on, and the only symptom would be poor reranking.
    #
    # Padding and multi-crop were briefly the default on an ablation measuring
    # 0.641 for base+pad+crops against 0.601 for base. That ablation ran while
    # the train/test split still depended on query row order, so it compared
    # different test sets. Re-measured on the fixed split, ridge on held-out
    # cards, base wins every head: purchase intent 0.624 against 0.619,
    # aesthetic 0.844 against 0.830, best-of-8 recovery 75.5% against 73.6%,
    # and the MLP gains more still (purchase intent 0.621 against 0.582). So
    # the four extra forward passes per image bought a regression.
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
        self.model_ids = [m.strip() for m in self.cfg.model_id.split(",") if m.strip()]
        log.info(f"Loading embedder(s): {', '.join(self.model_ids)} on {self.cfg.device}")
        self.processors, self.models = [], []
        for model_id in self.model_ids:
            self.processors.append(AutoProcessor.from_pretrained(model_id))
            self.models.append(
                AutoModel.from_pretrained(model_id, torch_dtype=self.cfg.dtype)
                .to(self.cfg.device)
                .eval()
            )
        # The text tower is the first backbone's; vision-only encoders have none.
        self.processor = self.processors[0]
        self.model = self.models[0]

    @property
    def source(self) -> str:
        """Identifies the encoder stack that produced a set of features."""
        parts = list(self.model_ids)
        if self.cfg.crops > 1:
            parts.append(f"crops={self.cfg.crops}")
        if self.cfg.pad_to_square:
            parts.append("pad")
        return "|".join(parts)

    def _views(self, img: Image.Image) -> list[Image.Image]:
        """The views of one card that get embedded and averaged."""
        views = [img] + (_quadrants(img) if self.cfg.crops >= 5 else [])
        return [_pad_to_square(v) for v in views] if self.cfg.pad_to_square else views

    @torch.inference_mode()
    def _encode_one(self, model, processor, imgs: list[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), self.cfg.batch_size):
            batch = imgs[i : i + self.cfg.batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(self.cfg.device)
            # SigLIP exposes vision_model; a vision-only encoder is called
            # directly. Both expose pooler_output.
            tower = getattr(model, "vision_model", model)
            feats = tower(**inputs).pooler_output
            out.append(torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def _encode(self, imgs: list[Image.Image]) -> np.ndarray:
        # Each backbone's block is unit-normalised before concatenation, so no
        # one encoder dominates the distance purely by having a larger scale.
        return np.hstack(
            [
                self._encode_one(m, p, imgs)
                for m, p in zip(self.models, self.processors, strict=True)
            ]
        )

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
# Selection is by encoder stack, not by NULL. Gating on `clip_embedding IS
# NULL` meant a catalogue that had been embedded once could never be re-embedded
# with a different configuration: the padding and multi-crop settings would be
# read from the environment, the job would report success, and nothing would
# change. `image_feature_source` records the stack that produced each row, so a
# row whose features came from a different one is the definition of stale.
_SELECT_STALE = """
SELECT l.listing_id, li.storage_path
FROM listings l
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
WHERE lf.image_features IS NULL
   OR lf.image_feature_source IS DISTINCT FROM %(source)s
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

# image_features always takes the current stack, at whatever width.
#
# clip_embedding is filled only when the row has none — COALESCE keeps whatever
# is already there. It is VECTOR(768) behind the HNSW index dedup searches, and
# the duplicate clusters the labelled pool was built from came out of those
# exact vectors. Overwriting them with a re-embed would move cluster boundaries
# underneath 2,468 existing labels. A fresh catalogue still gets its canonical
# vectors from the first run, so dedup is never left with nothing.
_UPSERT_FEATURE = """
INSERT INTO listing_features
    (listing_id, clip_embedding, image_features, image_feature_source, feature_version)
VALUES (%(listing_id)s, %(vector)s, %(features)s, %(source)s, %(version)s)
ON CONFLICT (listing_id) DO UPDATE
SET clip_embedding       = COALESCE(listing_features.clip_embedding, EXCLUDED.clip_embedding),
    image_features       = EXCLUDED.image_features,
    image_feature_source = EXCLUDED.image_feature_source,
    feature_version      = EXCLUDED.feature_version,
    computed_at          = NOW();
"""


# Rows written per transaction, so an interrupted run keeps what it finished.
COMMIT_EVERY = 200


def run_embed_missing(
    limit: int = 100_000,
    feature_version: str = "siglip-base-384-v1",
    batch_size: int = 32,
) -> int:
    """Embed cover images whose features are missing or from another encoder
    stack. Returns count written.

    Re-running after changing the backbone, padding or crop settings picks up
    every row again, because selection compares `image_feature_source` against
    the current stack rather than testing for NULL.

    The default limit covers the whole catalogue: the job script invokes this
    with no arguments, and a small default silently embedded a fraction of the
    listings while still reporting success.
    """
    import io as _io

    from common.db import connection
    from common.storage import get_object

    embedder = CLIPEmbedder()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_STALE, {"source": embedder.source, "limit": limit})
        rows = cur.fetchall()

    log.info(f"Feature source: {embedder.source}")
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
        # Wider than the vector column means it can only go to image_features;
        # NULL leaves any existing clip_embedding alone either way.
        fits_vector_column = embeddings.shape[1] == VECTOR_COLUMN_DIM

        with connection() as conn, conn.cursor() as cur:
            for listing_id, emb in zip(ids, embeddings, strict=True):
                features = emb.tolist()
                cur.execute(
                    _UPSERT_FEATURE,
                    {
                        "listing_id": listing_id,
                        "vector": features if fits_vector_column else None,
                        "features": features,
                        "source": embedder.source,
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
