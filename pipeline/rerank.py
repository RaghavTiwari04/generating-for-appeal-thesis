"""Best-of-N reranking.

Given N candidate composed cards, score each via the predictor and return
sorted by calibrated saleability score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from data.features.clip_embed import CLIPEmbedder
from models.predictor.infer import CardFeatures, PredictorRunner


@dataclass
class Candidate:
    image: Image.Image
    headline: str
    inside_message: str
    brief: dict
    occasion: str
    seed: int | None = None
    scores: dict[str, float] | None = None


def rerank(
    candidates: list[Candidate],
    *,
    predictor: PredictorRunner,
    embedder: CLIPEmbedder,
    top_k: int | None = None,
) -> list[Candidate]:
    if not candidates:
        return []

    images = [c.image for c in candidates]
    texts = [f"{c.headline}\n{c.inside_message}" for c in candidates]
    image_embs = embedder.embed_images(images)
    text_embs = embedder.embed_texts(texts)

    feats = [
        CardFeatures(
            image_emb=image_embs[i],
            text_emb=text_embs[i],
            occasion=candidates[i].occasion,
            price_rel=0.0,
        )
        for i in range(len(candidates))
    ]
    all_scores = predictor.score(feats)
    for cand, scores in zip(candidates, all_scores):
        cand.scores = scores

    candidates.sort(
        key=lambda c: c.scores.get("saleability_calibrated", c.scores.get("saleability", 0.0)),
        reverse=True,
    )
    return candidates[:top_k] if top_k else candidates
