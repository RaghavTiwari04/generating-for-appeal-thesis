"""Best-of-N reranking.

Given N candidate composed cards, score each via the predictor (or LLM for
testing) and return them sorted by predicted purchase intent — the same
quantity condition D's human cards are selected on, so the two conditions are
ranked by one objective. See DIST_WEIGHT below for what changed and why.
"""

from __future__ import annotations

import os
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

# What reranking maximises: purchase intent alone.
#
# It was 0.7 purchase intent + 0.3 distinctiveness. Condition D's cards are
# drawn on `saleability_labels.score`, which is purchase intent by itself, so
# any distinctiveness weight had C and D optimising different objectives — C
# spending part of its selection budget on a dimension D is not chosen for, and
# part of any C-versus-D gap reflecting that mismatch rather than card quality.
# The predictor is also strongest on purchase intent, so weight spent elsewhere
# is spent on its weaker heads.
#
# RERANK_DIST_WEIGHT > 0 restores the blend. The argument for it is that
# distinctiveness stops best-of-N converging on the safest design, which is a
# real failure mode for a generator asked to produce a catalogue rather than a
# card — but that is a property of the output set, not of the comparison this
# study makes, and it is measurable from the distinctiveness head afterwards
# rather than something selection has to enforce.
DIST_WEIGHT = float(os.environ.get("RERANK_DIST_WEIGHT", "0"))
PI_WEIGHT = 1.0 - DIST_WEIGHT

# Read once at import, from the environment, and not otherwise recorded: a
# stale export or a .env left over from an experiment would silently change
# what reranking optimises, and every artefact of the run would look identical
# to one made under the reported objective. Anything other than the reported
# objective is therefore announced loudly rather than assumed deliberate.
if DIST_WEIGHT:
    log.warning(
        f"RERANK_DIST_WEIGHT={DIST_WEIGHT} is set: reranking is maximising "
        f"{PI_WEIGHT:.2f} purchase intent + {DIST_WEIGHT:.2f} distinctiveness, "
        "NOT the objective reported in the thesis. Results from this run are "
        "not comparable with the reported evaluation."
    )


def objective_description() -> str:
    """What reranking maximised, for recording alongside a run's output."""
    if not DIST_WEIGHT:
        return "purchase_intent"
    return f"{PI_WEIGHT:.2f}*purchase_intent+{DIST_WEIGHT:.2f}*distinctiveness"


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
    # Whether Flux lettered the headline itself, and how much of it OCR read
    # back. Kept off `scores`, which the reranker overwrites wholesale.
    text_in_image: bool | None = None
    headline_match: float | None = None


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
