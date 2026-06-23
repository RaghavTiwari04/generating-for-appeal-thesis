"""Semantic Similarity Rating scorer for greeting card evaluation.

Implements the SSR methodology from:
    Maier et al. (2025) "LLMs Reproduce Human Purchase Intent via Semantic
    Similarity Elicitation of Likert Ratings" (arXiv: 2510.08338)

Key insight: direct numerical ratings from LLMs produce unrealistic
centre-clustered distributions (KS=0.26). SSR elicits free-text responses
and maps them to Likert distributions via embedding similarity (KS=0.88).

Steps:
    1. Prime LLM as synthetic consumer with demographic profile
    2. Ask open-ended question about the card (no numerical scale)
    3. Collect free-text response
    4. Embed response with sentence-transformers
    5. Compute cosine similarity to reference anchor statements
    6. Convert similarities to PMF over 1-5 Likert scale
    7. Average PMFs across multiple reference statement sets
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)

DIMS = (
    "purchase_intent",
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
)

# ---------------------------------------------------------------------------
# Reference anchor statement sets (6 sets × 5 Likert points per dimension)
# Short, generic, domain-independent as recommended by paper.
# ---------------------------------------------------------------------------
REFERENCE_SETS: dict[str, list[list[str]]] = {
    "purchase_intent": [
        [
            "I would never buy this card",
            "I probably would not buy this card",
            "I am unsure whether I would buy this card",
            "I would likely buy this card",
            "I would definitely buy this card",
        ],
        [
            "This card has no commercial appeal",
            "This card has weak commercial appeal",
            "This card has moderate commercial appeal",
            "This card has strong commercial appeal",
            "This card has excellent commercial appeal",
        ],
        [
            "No one would want to purchase this",
            "Few people would consider purchasing this",
            "Some people might consider purchasing this",
            "Many people would consider purchasing this",
            "Most people would want to purchase this",
        ],
    ],
    "occasion_fit": [
        [
            "This card does not match the occasion at all",
            "This card weakly matches the occasion",
            "This card somewhat matches the occasion",
            "This card matches the occasion well",
            "This card perfectly matches the occasion",
        ],
        [
            "The design is completely wrong for this occasion",
            "The design is a poor fit for this occasion",
            "The design is an acceptable fit for this occasion",
            "The design is a good fit for this occasion",
            "The design is an ideal fit for this occasion",
        ],
        [
            "Someone receiving this would be confused by the occasion",
            "Someone receiving this might question the occasion choice",
            "Someone receiving this would understand the occasion",
            "Someone receiving this would feel it suits the occasion",
            "Someone receiving this would feel it captures the occasion perfectly",
        ],
    ],
    "aesthetic": [
        [
            "This card looks unprofessional and poorly designed",
            "This card looks below average in design quality",
            "This card looks acceptable in design quality",
            "This card looks well designed and polished",
            "This card looks exceptionally beautiful and professional",
        ],
        [
            "The visual quality is very poor",
            "The visual quality is below average",
            "The visual quality is average",
            "The visual quality is above average",
            "The visual quality is outstanding",
        ],
        [
            "The colours and composition are jarring and unappealing",
            "The colours and composition need significant improvement",
            "The colours and composition are adequate",
            "The colours and composition are harmonious and appealing",
            "The colours and composition are stunning and masterful",
        ],
    ],
    "emotional_resonance": [
        [
            "This card evokes no emotional response",
            "This card evokes a weak emotional response",
            "This card evokes a moderate emotional response",
            "This card evokes a strong emotional response",
            "This card evokes a powerful emotional response",
        ],
        [
            "The recipient would feel nothing upon seeing this card",
            "The recipient would barely notice this card",
            "The recipient would appreciate this card",
            "The recipient would be touched by this card",
            "The recipient would be deeply moved by this card",
        ],
        [
            "There is no warmth or sentiment in this card",
            "There is little warmth or sentiment in this card",
            "There is some warmth and sentiment in this card",
            "There is clear warmth and sentiment in this card",
            "This card radiates warmth and heartfelt sentiment",
        ],
    ],
    "distinctiveness": [
        [
            "This card is completely generic and unoriginal",
            "This card is mostly generic with little originality",
            "This card has some original elements",
            "This card has a distinctive artistic voice",
            "This card is truly unique and creatively original",
        ],
        [
            "I have seen countless cards exactly like this",
            "I have seen many cards similar to this",
            "This card is somewhat different from typical cards",
            "This card stands out from typical cards",
            "This card is unlike anything I have seen before",
        ],
        [
            "This is a cookie-cutter stock design",
            "This design feels mass-produced",
            "This design has a few unique touches",
            "This design shows clear creative thought",
            "This design is boldly original and inventive",
        ],
    ],
}

# ---------------------------------------------------------------------------
# Demographic profiles for synthetic consumers (paper recommends varying age/income)
# ---------------------------------------------------------------------------
CONSUMER_PROFILES = [
    {"age": 28, "gender": "female", "income": "moderate", "region": "urban UK"},
    {"age": 45, "gender": "male", "income": "above average", "region": "suburban UK"},
    {"age": 62, "gender": "female", "income": "comfortable", "region": "rural UK"},
]

# ---------------------------------------------------------------------------
# Embedding + SSR computation
# ---------------------------------------------------------------------------
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Loaded sentence-transformers: all-MiniLM-L6-v2")
    return _embedder


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_embedder()
    return model.encode(texts, normalize_embeddings=True)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (1 + np.dot(a, b.T)) / 2


def similarities_to_pmf(
    sims: np.ndarray, temperature: float = 1.0, epsilon: float = 0.0,
) -> np.ndarray:
    """Convert cosine similarities to PMF over Likert scale (paper Eq. 8-9)."""
    s = sims.copy()
    s_min = s.min()
    s = s - s_min

    if epsilon > 0:
        min_idx = np.argmin(sims)
        s[min_idx] += epsilon

    if s.sum() == 0:
        return np.ones(len(s)) / len(s)

    pmf = s / s.sum()

    if temperature != 1.0 and temperature > 0:
        pmf = pmf ** (1.0 / temperature)
        pmf = pmf / pmf.sum()

    return pmf


def compute_ssr_scores(
    response_text: str,
    dimension: str,
    temperature: float = 1.0,
    epsilon: float = 0.0,
) -> dict:
    """Compute SSR PMF for one response across all reference sets for a dimension."""
    ref_sets = REFERENCE_SETS[dimension]
    response_emb = _embed([response_text])[0]

    pmfs = []
    for ref_set in ref_sets:
        ref_embs = _embed(ref_set)
        sims = _cosine_similarity(response_emb.reshape(1, -1), ref_embs)[0]
        pmf = similarities_to_pmf(sims, temperature=temperature, epsilon=epsilon)
        pmfs.append(pmf)

    avg_pmf = np.mean(pmfs, axis=0)
    avg_pmf = avg_pmf / avg_pmf.sum()

    expected = np.dot(avg_pmf, np.array([1, 2, 3, 4, 5]))
    normalised = (expected - 1) / 4  # map 1-5 to 0-1

    return {
        "pmf": avg_pmf.tolist(),
        "expected_likert": float(expected),
        "score_0_1": float(normalised),
    }


# ---------------------------------------------------------------------------
# LLM free-text elicitation (SSR step 1-3)
# ---------------------------------------------------------------------------
def _build_system_prompt(profile: dict) -> str:
    return (
        f"You are a {profile['age']}-year-old {profile['gender']} living in "
        f"{profile['region']} with {profile['income']} income. You regularly "
        f"buy greeting cards for friends and family. "
        f"Reply briefly and naturally to any questions posed to you. "
        f"Do not use numerical ratings or scales — just share your honest "
        f"reaction in your own words."
    )


def _build_questions(headline: str, inside_message: str, occasion: str) -> dict[str, str]:
    ctx = f"a greeting card for {occasion}"
    if headline:
        ctx += f' with the headline "{headline}"'

    return {
        "purchase_intent": (
            f"You see {ctx} in a shop. "
            f"Would you buy it? How appealing is it as a purchase?"
        ),
        "occasion_fit": (
            f"You see {ctx}. "
            f"How well does the design and message match the occasion?"
        ),
        "aesthetic": (
            f"You see {ctx}. "
            f"What do you think of the visual design quality and artistic merit?"
        ),
        "emotional_resonance": (
            f"You see {ctx}. "
            f"What emotional impact does this card have? Would the recipient feel something?"
        ),
        "distinctiveness": (
            f"You see {ctx}. "
            f"How original and unique is this card compared to typical greeting cards?"
        ),
    }


def _image_to_b64(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/png"


def _call_llm_freetext(
    image_b64: str,
    media_type: str,
    system_prompt: str,
    question: str,
    model: str | None = None,
) -> str:
    """Get free-text response from LLM (no numerical rating)."""
    import anthropic

    model = model or settings.llm_model
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.8,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
    except Exception as e:
        log.warning(f"LLM API error: {e}")
        return "I am unsure about this card."

    blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return blocks[0] if blocks else "I am unsure about this card."


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------
@dataclass
class SSRScorer:
    """Score greeting cards using Semantic Similarity Rating methodology."""

    model: str | None = None
    temperature: float = 1.0
    epsilon: float = 0.0
    profiles: list[dict] = field(default_factory=lambda: list(CONSUMER_PROFILES))

    def score_one(
        self,
        image: Image.Image,
        headline: str,
        inside_message: str,
        occasion: str,
    ) -> dict[str, float]:
        b64, media_type = _image_to_b64(image)
        questions = _build_questions(headline, inside_message, occasion)

        dim_scores: dict[str, list[float]] = {d: [] for d in DIMS}
        dim_pmfs: dict[str, list[list[float]]] = {d: [] for d in DIMS}
        all_responses: list[dict] = []

        for profile in self.profiles:
            system_prompt = _build_system_prompt(profile)

            for dim in DIMS:
                response = _call_llm_freetext(
                    b64, media_type, system_prompt, questions[dim], self.model,
                )
                ssr_result = compute_ssr_scores(
                    response, dim,
                    temperature=self.temperature,
                    epsilon=self.epsilon,
                )
                dim_scores[dim].append(ssr_result["score_0_1"])
                dim_pmfs[dim].append(ssr_result["pmf"])
                all_responses.append({
                    "profile_age": profile["age"],
                    "dimension": dim,
                    "response": response,
                    "score_0_1": ssr_result["score_0_1"],
                    "pmf": ssr_result["pmf"],
                })

                time.sleep(0.2)

        scores = {}
        for dim in DIMS:
            scores[dim] = float(np.mean(dim_scores[dim]))
            scores[f"{dim}_std"] = float(np.std(dim_scores[dim]))
            scores[f"{dim}_pmf"] = np.mean(dim_pmfs[dim], axis=0).tolist()

        scores["purchase_intent_calibrated"] = scores["purchase_intent"]
        scores["_responses"] = all_responses

        return scores
