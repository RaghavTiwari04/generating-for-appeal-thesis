"""Hybrid evaluation scorer for greeting card system evaluation.

Two methodologies, each backed by peer-reviewed research:

1. **Purchase intent — SSR** (Maier et al. 2025, arXiv:2510.08338):
   Semantic Similarity Rating. Elicits free-text responses from LLM
   synthetic consumers, maps to Likert via embedding cosine similarity
   against reference anchor statements. KS>0.85 vs human distributions.

2. **Aesthetic, occasion_fit, emotional_resonance, distinctiveness —
   Rubric-guided LLM judge** (Zheng et al. 2023, "Judging LLM-as-a-Judge
   with MT-Bench and Chatbot Arena", NeurIPS 2023):
   Explanation-first chain-of-thought scoring on 1-10 scale with detailed
   per-dimension rubrics. Temperature=0 for deterministic judgments.
   Score extracted via regex. >80% agreement with human annotators.
"""

from __future__ import annotations

import base64
import io
import re
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

SSR_DIMS = ("purchase_intent",)
RUBRIC_DIMS = ("occasion_fit", "aesthetic", "emotional_resonance", "distinctiveness")

# ---------------------------------------------------------------------------
# SSR: Reference anchor statements for purchase intent (Maier et al. 2025)
# Multiple sets averaged for robustness as recommended by paper.
# ---------------------------------------------------------------------------
SSR_REFERENCE_SETS: list[list[str]] = [
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
]

# ---------------------------------------------------------------------------
# Rubric judge: per-dimension rubrics (Zheng et al. 2023 methodology)
# Explanation-first, 1-10 scale, detailed criteria per score band.
# ---------------------------------------------------------------------------
RUBRIC_PROMPTS: dict[str, str] = {
    "occasion_fit": """\
You are an impartial expert judge evaluating how well a greeting card \
matches its stated occasion. You will see one card image plus metadata.

Evaluate OCCASION FIT: How well does the card's imagery, text, colour \
palette, and overall theme match the stated occasion?

Scoring rubric (1-10):
  1-2: Completely wrong occasion. A sympathy card that looks celebratory, \
or a birthday card with no birthday-relevant imagery whatsoever.
  3-4: Weak connection. Generic design that could apply to any occasion. \
The occasion is not clearly communicated by the visual or text.
  5-6: Acceptable match. The card communicates the correct occasion but \
through generic or overused means. A reasonable but uninspired choice.
  7-8: Strong match. The imagery, text, and design clearly and effectively \
communicate the occasion. The recipient would immediately understand \
what the card is for and feel it was well-chosen.
  9-10: Perfect match. The card captures the spirit of the occasion in a \
way that feels both specific and thoughtful. Every element reinforces \
the occasion naturally.

First provide a brief explanation of your assessment (2-3 sentences). \
Then output your score strictly in this format: [[rating]]
Example: Rating: [[7]]""",

    "aesthetic": """\
You are an impartial expert judge evaluating the visual design quality \
of a greeting card. You will see one card image plus metadata.

Evaluate AESTHETIC QUALITY: Consider composition, colour harmony, \
typography, illustration quality, whitespace usage, and overall \
professionalism.

Scoring rubric (1-10):
  1-2: Unprofessional. Poor resolution, clashing colours, illegible text, \
clip-art quality, broken layout. Would not pass for a commercial product.
  3-4: Below average. Identifiable quality issues — awkward composition, \
jarring colour palette, amateurish illustration, or typography problems.
  5-6: Average commercial quality. Competent design that meets basic \
standards but lacks polish or distinction. Acceptable for a budget range.
  7-8: High quality. Well-composed, harmonious colours, clean typography, \
polished illustration. Looks professional and gift-worthy. Standard to \
premium card shop quality.
  9-10: Exceptional. Museum-quality illustration, masterful composition, \
stunning colour work. Could anchor a premium card collection or win \
design awards.

First provide a brief explanation of your assessment (2-3 sentences). \
Then output your score strictly in this format: [[rating]]
Example: Rating: [[7]]""",

    "emotional_resonance": """\
You are an impartial expert judge evaluating the emotional impact of a \
greeting card. You will see one card image plus metadata.

Evaluate EMOTIONAL RESONANCE: Would this card make the recipient feel \
something? Consider warmth, humour, sentiment, joy, tenderness, or any \
genuine emotional response the card evokes.

Scoring rubric (1-10):
  1-2: No emotional impact. Cold, corporate, or confusing. The recipient \
would feel nothing or feel the sender put in no effort.
  3-4: Minimal emotional impact. Generic sentiment without genuine feeling. \
The recipient might glance at it and set it aside.
  5-6: Moderate emotional impact. The card communicates care adequately. \
The recipient would appreciate the gesture but not be particularly moved.
  7-8: Strong emotional impact. The card evokes genuine warmth, laughter, \
or sentiment. The recipient would likely keep it displayed for a while \
or share it with someone.
  9-10: Deeply moving. The card captures something truly heartfelt or \
genuinely funny. The recipient might tear up, laugh out loud, or keep \
it for years. The kind of card people photograph and share.

First provide a brief explanation of your assessment (2-3 sentences). \
Then output your score strictly in this format: [[rating]]
Example: Rating: [[7]]""",

    "distinctiveness": """\
You are an impartial expert judge evaluating the originality of a \
greeting card. You will see one card image plus metadata.

Evaluate DISTINCTIVENESS: How original and unique is this card compared \
to typical mass-market greeting cards? Does it have a distinctive \
artistic voice or creative concept?

Scoring rubric (1-10):
  1-2: Completely generic. Indistinguishable from thousands of stock \
greeting cards. Cookie-cutter design with no creative thought.
  3-4: Mostly generic. One or two mildly unusual elements but the overall \
concept and execution are familiar and predictable.
  5-6: Somewhat distinctive. Has identifiable creative choices that \
separate it from pure stock designs, but still within well-trodden \
territory. Would not stand out on a shelf of cards.
  7-8: Distinctive. Clear artistic voice or creative concept that would \
catch a shopper's eye. Memorable design that stands apart from typical \
offerings. The kind of card someone picks up and shows to a friend.
  9-10: Highly original. Boldly creative concept or artistic approach \
unlike typical greeting cards. Would stand out immediately in any \
card shop. The kind of design that might start a trend.

First provide a brief explanation of your assessment (2-3 sentences). \
Then output your score strictly in this format: [[rating]]
Example: Rating: [[7]]""",
}

JUDGE_SYSTEM_PROMPT = (
    "You are an expert greeting card designer and market analyst serving "
    "as an impartial judge. Be objective and calibrated. Use the full 1-10 "
    "range — most cards should score 4-7. Do not let verbosity or "
    "elaborateness bias your judgment."
)

# ---------------------------------------------------------------------------
# Demographic profiles for SSR synthetic consumers
# Paper recommends varying age and income (most impactful demographics).
# ---------------------------------------------------------------------------
CONSUMER_PROFILES = [
    {"age": 28, "gender": "female", "income": "moderate", "region": "urban UK"},
    {"age": 45, "gender": "male", "income": "above average", "region": "suburban UK"},
    {"age": 62, "gender": "female", "income": "comfortable", "region": "rural UK"},
]

# ---------------------------------------------------------------------------
# Embedding model for SSR (paper default: all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------
_embedder = None
_ref_embedding_cache: dict[str, np.ndarray] = {}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Loaded sentence-transformers: all-MiniLM-L6-v2")
    return _embedder


def _embed(texts: list[str]) -> np.ndarray:
    return _get_embedder().encode(texts, normalize_embeddings=True)


def _embed_cached(texts: list[str]) -> np.ndarray:
    key = "\0".join(texts)
    if key not in _ref_embedding_cache:
        _ref_embedding_cache[key] = _embed(texts)
    return _ref_embedding_cache[key]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (1 + np.dot(a, b.T)) / 2


def similarities_to_pmf(
    sims: np.ndarray, epsilon: float = 0.0,
) -> np.ndarray:
    """Convert cosine similarities to PMF over Likert scale (paper Eq. 8).

    Temperature scaling is applied separately after averaging across
    reference sets, matching the reference implementation.
    """
    s = sims.copy()
    s_min = s.min()
    s = s - s_min

    if epsilon > 0:
        min_idx = np.argmin(sims)
        s[min_idx] += epsilon

    if s.sum() == 0:
        return np.ones(len(s)) / len(s)

    return s / s.sum()


def scale_pmf(pmf: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale a PMF (paper Eq. 9). Matches reference compute.scale_pmf."""
    if temperature == 1.0:
        return pmf
    if temperature == 0.0:
        out = np.zeros_like(pmf)
        if np.all(pmf == pmf[0]):
            return pmf.copy()
        out[np.argmax(pmf)] = 1.0
        return out
    scaled = pmf ** (1.0 / temperature)
    return scaled / scaled.sum()


def compute_ssr_score(
    response_text: str,
    temperature: float = 1.0,
    epsilon: float = 0.0,
) -> dict:
    """Compute SSR PMF for purchase intent across all reference sets."""
    response_emb = _embed([response_text])[0]

    pmfs = []
    for ref_set in SSR_REFERENCE_SETS:
        ref_embs = _embed_cached(ref_set)
        sims = _cosine_similarity(response_emb.reshape(1, -1), ref_embs)[0]
        pmf = similarities_to_pmf(sims, epsilon=epsilon)
        pmfs.append(pmf)

    avg_pmf = np.mean(pmfs, axis=0)

    # Temperature scaling applied AFTER averaging across reference sets
    # (Maier et al. 2025, response_rater.py lines 436-439)
    if temperature != 1.0:
        avg_pmf = scale_pmf(avg_pmf, temperature)

    expected = np.dot(avg_pmf, np.array([1, 2, 3, 4, 5]))
    normalised = (expected - 1) / 4

    return {
        "pmf": avg_pmf.tolist(),
        "expected_likert": float(expected),
        "score_0_1": float(normalised),
    }


# ---------------------------------------------------------------------------
# LLM API calls
# ---------------------------------------------------------------------------
def _image_to_b64(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/png"


def _call_anthropic(
    image_b64: str,
    media_type: str,
    system_prompt: str,
    user_text: str,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 200,
    api_retries: int = 5,
) -> str:
    import anthropic

    model = model or "claude-sonnet-4-6"
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    for attempt in range(api_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
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
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
            blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            return blocks[0] if blocks else ""
        except Exception as e:
            wait = min(2 ** attempt, 30)
            log.warning(f"Anthropic API error (attempt {attempt + 1}/{api_retries}): {e}")
            if attempt < api_retries - 1:
                time.sleep(wait)

    return ""


def _call_openai(
    image_b64: str,
    media_type: str,
    system_prompt: str,
    user_text: str,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 200,
    api_retries: int = 5,
) -> str:
    from openai import OpenAI

    model = model or "gpt-4.1-mini"
    client = OpenAI(api_key=settings.openai_api_key)

    for attempt in range(api_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                    "detail": "low",
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
            )
            choice = resp.choices[0] if resp.choices else None
            if choice and choice.message.content:
                return choice.message.content
            return ""
        except Exception as e:
            wait = min(2 ** attempt, 30)
            log.warning(f"OpenAI API error (attempt {attempt + 1}/{api_retries}): {e}")
            if attempt < api_retries - 1:
                time.sleep(wait)

    return ""


def _call_vlm(
    image_b64: str,
    media_type: str,
    system_prompt: str,
    user_text: str,
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 200,
) -> str:
    if provider == "openai":
        return _call_openai(
            image_b64, media_type, system_prompt, user_text,
            model=model, temperature=temperature, max_tokens=max_tokens,
        )
    return _call_anthropic(
        image_b64, media_type, system_prompt, user_text,
        model=model, temperature=temperature, max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# SSR: free-text elicitation for purchase intent
# ---------------------------------------------------------------------------
def _ssr_system_prompt(profile: dict) -> str:
    return (
        f"You are a {profile['age']}-year-old {profile['gender']} living in "
        f"{profile['region']} with {profile['income']} income. You regularly "
        f"buy greeting cards for friends and family. "
        f"Reply briefly and naturally to any questions posed to you. "
        f"Do not use numerical ratings or scales — just share your honest "
        f"reaction in your own words."
    )


def _score_purchase_intent_ssr(
    image_b64: str,
    media_type: str,
    headline: str,
    occasion: str,
    profiles: list[dict],
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 1.0,
    epsilon: float = 0.0,
) -> dict:
    """Score purchase intent using SSR methodology."""
    ctx = f"a greeting card for {occasion}"
    if headline:
        ctx += f' with the headline "{headline}"'
    question = f"You see {ctx} in a shop. Would you buy it? How appealing is it as a purchase?"

    scores = []
    pmfs = []
    responses = []

    for profile in profiles:
        sys_prompt = _ssr_system_prompt(profile)
        response = _call_vlm(
            image_b64, media_type, sys_prompt, question,
            provider=provider, model=model, temperature=0.8, max_tokens=200,
        )
        if not response:
            response = "I am unsure about this card."

        ssr = compute_ssr_score(response, temperature=temperature, epsilon=epsilon)
        scores.append(ssr["score_0_1"])
        pmfs.append(ssr["pmf"])
        responses.append({
            "profile_age": profile["age"],
            "dimension": "purchase_intent",
            "response": response,
            "score_0_1": ssr["score_0_1"],
            "pmf": ssr["pmf"],
        })
        time.sleep(0.2)

    return {
        "score": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "pmf": np.mean(pmfs, axis=0).tolist(),
        "responses": responses,
    }


# ---------------------------------------------------------------------------
# Rubric judge: explanation-first scoring (Zheng et al. 2023)
# ---------------------------------------------------------------------------
_SCORE_PATTERN = re.compile(r"\[\[(\d+\.?\d*)\]\]")
_SCORE_FALLBACK = re.compile(r"\[(\d+\.?\d*)\]")


def _extract_score(text: str) -> float | None:
    """Extract [[rating]] from judge response (FastChat pattern)."""
    m = _SCORE_PATTERN.search(text)
    if not m:
        m = _SCORE_FALLBACK.search(text)
    if m:
        val = float(m.group(1))
        return max(1.0, min(10.0, val))
    return None


def _score_rubric_dim(
    image_b64: str,
    media_type: str,
    dim: str,
    headline: str,
    inside_message: str,
    occasion: str,
    provider: str = "anthropic",
    model: str | None = None,
) -> dict:
    """Score one dimension using rubric-guided LLM judge (temp=0)."""
    rubric = RUBRIC_PROMPTS[dim]

    user_text = (
        f"Occasion: {occasion}\n"
        f"Headline: {headline}\n"
        f"Inside message: {inside_message}\n\n"
        f"{rubric}"
    )

    scores = []
    explanations = []

    for attempt in range(3):
        response = _call_vlm(
            image_b64, media_type, JUDGE_SYSTEM_PROMPT, user_text,
            provider=provider, model=model, temperature=0.0, max_tokens=1024,
        )
        score = _extract_score(response)
        if score is not None:
            scores.append(score)
            explanations.append(response)
            break
        log.warning(f"Rubric judge attempt {attempt + 1} for {dim}: no score extracted")
        time.sleep(0.5)

    if not scores:
        return None

    raw = scores[0]
    return {
        "score": raw,
        "score_0_1": (raw - 1) / 9,  # map 1-10 to 0-1
        "explanation": explanations[0],
    }


# ---------------------------------------------------------------------------
# Main hybrid scorer
# ---------------------------------------------------------------------------
@dataclass
class SSRScorer:
    """Hybrid scorer: SSR for purchase intent, rubric judge for other dims.

    Purchase intent: SSR (Maier et al. 2025) — free-text → embedding
    similarity → Likert PMF. Validated KS>0.85 against human data.

    Occasion fit, aesthetic, emotional resonance, distinctiveness:
    Rubric-guided LLM judge (Zheng et al. 2023) — explanation-first
    chain-of-thought, 1-10 scale, temperature=0. >80% human agreement.
    """

    provider: str = "openai"
    model: str | None = None
    ssr_temperature: float = 1.0
    ssr_epsilon: float = 0.0
    profiles: list[dict] = field(default_factory=lambda: list(CONSUMER_PROFILES))

    def __post_init__(self):
        if not self.model:
            if self.provider == "openai":
                self.model = "gpt-4.1-mini"
            else:
                self.model = "claude-sonnet-4-6"

    def score_one(
        self,
        image: Image.Image,
        headline: str,
        inside_message: str,
        occasion: str,
    ) -> dict[str, float]:
        b64, media_type = _image_to_b64(image)
        all_responses: list[dict] = []

        # --- Purchase intent via SSR (Maier et al. 2025) ---
        pi_result = _score_purchase_intent_ssr(
            b64, media_type, headline, occasion,
            profiles=self.profiles,
            provider=self.provider,
            model=self.model,
            temperature=self.ssr_temperature,
            epsilon=self.ssr_epsilon,
        )
        all_responses.extend(pi_result["responses"])

        scores: dict[str, float] = {
            "purchase_intent": pi_result["score"],
            "purchase_intent_std": pi_result["std"],
            "purchase_intent_pmf": pi_result["pmf"],
            "purchase_intent_calibrated": pi_result["score"],
            "purchase_intent_method": "ssr",
        }

        # --- Other dims via rubric judge (Zheng et al. 2023) ---
        for dim in RUBRIC_DIMS:
            result = _score_rubric_dim(
                b64, media_type, dim, headline, inside_message, occasion,
                provider=self.provider, model=self.model,
            )
            if result is None:
                log.warning(f"Rubric judge failed for {dim} after 3 retries, excluding")
                continue
            scores[dim] = result["score_0_1"]
            scores[f"{dim}_raw_1_10"] = result["score"]
            scores[f"{dim}_method"] = "rubric_judge"
            all_responses.append({
                "dimension": dim,
                "score_0_1": result["score_0_1"],
                "score_raw": result["score"],
                "explanation": result["explanation"],
            })

        scores["_responses"] = all_responses
        return scores
