"""Best-of-N reranking.

Given N candidate composed cards, score each via the predictor (or LLM for
testing) and return sorted by composite saleability = 0.7 * purchase_intent
+ 0.3 * distinctiveness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

from common.logging import get_logger

if TYPE_CHECKING:
    from typing import Protocol

    from data.features.clip_embed import CLIPEmbedder

    class Predictor(Protocol):
        """Either scoring model. The caller picks; rerank only needs scores."""

        def score(self, features: list) -> list[dict[str, float]]: ...

log = get_logger(__name__)

PI_WEIGHT = 0.7
DIST_WEIGHT = 0.3


@dataclass
class Candidate:
    image: Image.Image
    headline: str
    inside_message: str
    brief: dict
    occasion: str
    seed: int | None = None
    scores: dict[str, float] | None = None
    card_id: str | None = None


def _compute_saleability(scores: dict[str, float]) -> float:
    pi = scores.get("purchase_intent_calibrated", scores.get("purchase_intent", 0.0))
    dist = scores.get("distinctiveness", 0.0)
    return PI_WEIGHT * pi + DIST_WEIGHT * dist


def rerank(
    candidates: list[Candidate],
    *,
    predictor: Predictor,
    embedder: CLIPEmbedder,
    top_k: int | None = None,
) -> list[Candidate]:
    """Rerank via the trained predictor on cached CLIP features."""
    from models.predictor.infer import CardFeatures

    if not candidates:
        return []

    images = [c.image for c in candidates]
    # Headline only. The predictor's text tower is trained on `extracted_text`,
    # which is OCR of the card's front, so the inside message is content it has
    # never seen in that channel — appending it here fed the model a different
    # kind of string at the moment it is used to rank.
    texts = [c.headline for c in candidates]
    image_embs = embedder.embed_images(images)
    text_embs = embedder.embed_texts(texts)

    feats = [
        CardFeatures(
            image_emb=image_embs[i],
            text_emb=text_embs[i],
            occasion=candidates[i].occasion,
        )
        for i in range(len(candidates))
    ]
    all_scores = predictor.score(feats)
    for cand, scores in zip(candidates, all_scores, strict=True):
        scores["saleability_calibrated"] = _compute_saleability(scores)
        cand.scores = scores

    candidates.sort(
        key=lambda c: c.scores.get("saleability_calibrated", 0.0),
        reverse=True,
    )
    return candidates[:top_k] if top_k else candidates


def rerank_llm(
    candidates: list[Candidate],
    *,
    top_k: int | None = None,
    provider: str = "anthropic",
    model: str | None = None,
) -> list[Candidate]:
    """Rerank via LLM vision API — no trained model needed."""
    from scoring import CardScorer

    if not candidates:
        return []

    scorer = CardScorer(provider=provider, model=model)
    log.info(f"LLM reranking {len(candidates)} candidates via {scorer.provider}/{scorer.model}")

    for i, cand in enumerate(candidates):
        scores = scorer.score(cand.image, occasion=cand.occasion)
        scores["saleability_calibrated"] = _compute_saleability(scores)
        cand.scores = scores
        log.info(
            f"  candidate {i+1}/{len(candidates)}: "
            f"saleability={scores['saleability_calibrated']:.3f}"
        )

    candidates.sort(
        key=lambda c: c.scores.get("saleability_calibrated", 0.0),
        reverse=True,
    )
    return candidates[:top_k] if top_k else candidates
