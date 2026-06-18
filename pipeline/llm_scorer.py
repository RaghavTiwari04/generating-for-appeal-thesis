"""LLM-based saleability scorer — drop-in replacement for PredictorRunner.

Sends each candidate's composed image to a VLM (Claude / GPT-4o) and asks
for structured scores on all five predictor dimensions.  Used for testing
the full pipeline end-to-end without a trained PyTorch checkpoint.

Unlike VLM labeling (data/labels/vlm_labels.py) which scores existing scraped
listings, this module scores freshly *generated* PIL images in-memory.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass

from PIL import Image

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)

DIMS = (
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
    "purchase_intent",
)

SYSTEM_PROMPT = """\
You are an expert greeting-card designer and market analyst evaluating \
card designs for commercial potential.

You will see one greeting card image plus its headline and inside message. \
Score the card on five dimensions, each on a 0.0-1.0 continuous scale \
(two decimal places).

Dimensions:
  occasion_fit (0-1): How well the card matches the stated occasion. \
Consider imagery, text, and overall theme.
  aesthetic (0-1): Visual quality — composition, colour harmony, \
typography, professionalism. High = polished design. Low = clip-art, \
poor layout.
  emotional_resonance (0-1): Emotional impact — does it evoke warmth, \
joy, humour, or sentiment? Would the recipient feel something?
  distinctiveness (0-1): How original vs generic. High = unique artistic \
voice. Low = cookie-cutter stock design.
  purchase_intent (0-1): How likely is a typical buyer to purchase this \
card? Consider overall appeal, quality, and market fit. 0.1 = would not \
buy, 0.9 = would definitely buy.

Guidelines:
- Score independently per dimension.
- Use the full 0-1 range. Average cards ~0.5.
- Be calibrated: most cards should cluster 0.3-0.7.

Reply with ONLY a JSON object:
{"occasion_fit": 0.XX, "aesthetic": 0.XX, "emotional_resonance": 0.XX, \
"distinctiveness": 0.XX, "purchase_intent": 0.XX, \
"reasoning": "<one sentence>"}"""


def _image_to_b64(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/png"


def _parse_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        obj = json.loads(text)
        for d in DIMS:
            if d not in obj:
                return None
            obj[d] = max(0.0, min(1.0, float(obj[d])))
        return obj
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _call_anthropic(
    image_b64: str, media_type: str, user_text: str, model: str
) -> dict | None:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
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
    except Exception as e:
        log.warning(f"Anthropic API error: {e}")
        return None

    blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return _parse_response(blocks[0]) if blocks else None


def _call_openai(
    image_b64: str, media_type: str, user_text: str, model: str
) -> dict | None:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
    except Exception as e:
        log.warning(f"OpenAI API error: {e}")
        return None

    choice = resp.choices[0] if resp.choices else None
    if not choice or not choice.message.content:
        return None
    return _parse_response(choice.message.content)


@dataclass
class LLMScorer:
    """Score generated card images via VLM — same output format as PredictorRunner."""

    provider: str = settings.llm_provider
    model: str | None = None

    def __post_init__(self):
        if not self.model:
            if self.provider == "openai":
                self.model = "gpt-4o"
            else:
                self.model = settings.llm_model

    def score_one(
        self, image: Image.Image, headline: str, inside_message: str, occasion: str
    ) -> dict[str, float]:
        b64, media_type = _image_to_b64(image)
        user_text = (
            f"Occasion: {occasion}\n"
            f"Headline: {headline}\n"
            f"Inside message: {inside_message}\n\n"
            f"Score this card."
        )

        for attempt in range(3):
            if self.provider == "openai":
                result = _call_openai(b64, media_type, user_text, self.model)
            else:
                result = _call_anthropic(b64, media_type, user_text, self.model)

            if result:
                scores = {d: result[d] for d in DIMS}
                scores["purchase_intent_calibrated"] = scores["purchase_intent"]
                return scores

            log.warning(f"LLM scoring attempt {attempt + 1} failed, retrying...")
            time.sleep(1)

        log.error("LLM scoring failed after 3 attempts, returning defaults")
        return {d: 0.5 for d in DIMS} | {"purchase_intent_calibrated": 0.5}
