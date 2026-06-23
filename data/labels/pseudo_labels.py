"""LLM pseudo-labels for the predictor sub-score heads.

The v2 survey instrument collects only two human dimensions (purchase_intent,
aesthetic) via 2AFC. The other three predictor heads — occasion_fit,
emotional_resonance, distinctiveness — are trained on LLM pseudo-labels
validated against a small human-rated subset.

Pipeline
--------
1. For each card in the survey pool, render a compact prompt with the cover
   image (as base64) + headline + inside_message + occasion.
2. Call the configured LLM provider (Claude or GPT-4V) with a structured
   JSON-output schema asking for the three sub-dimension scores on 0..1.
3. Persist as `saleability_labels` rows with `label_source='llm_pseudo_v1'`,
   sub-scores in the `raw` JSON column.
4. Validate on a calibration set: humans rate the same ~40 cards on the same
   three dimensions; compute Spearman ρ(LLM, human) per dim. If ρ ≥ 0.5 the
   pseudo-labels are accepted for predictor training; otherwise fall back to
   proxy-only training for that head and document in the thesis.

Cost
----
~150 cards × ~1 LLM call each. Claude Sonnet 4.6 vision ≈ £0.01/call →
total ~£1.50. Cheap relative to the human survey.

Run:
    python -m data.labels.pseudo_labels --label-source llm_pseudo_v1
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import pandas as pd
import typer
from psycopg.types.json import Jsonb

from common.config import settings
from common.db import connection, engine
from common.logging import get_logger
from common.occasions import is_valid_occasion
from common.storage import get_object

log = get_logger(__name__)

PSEUDO_DIMS = ("occasion_fit", "emotional_resonance", "distinctiveness")

# Strict JSON-output prompt. Keep the schema small to minimise token cost.
RATING_SYSTEM = """You are an expert greeting-card buyer evaluating cards for the UK market.

You will see one greeting card image plus its headline and inside message and
the intended occasion. Score the card on three dimensions, each on a 0–1
continuous scale (two decimal places).

  - occasion_fit: How well the card matches the stated occasion.
  - emotional_resonance: How well the card captures the right feeling for the occasion.
  - distinctiveness: How original the card is relative to typical cards for this occasion.

Reply with ONLY a JSON object of the form:
{"occasion_fit": <float>, "emotional_resonance": <float>, "distinctiveness": <float>, "reasoning": "<<=200 chars>"}
"""


@dataclass
class PseudoLabelRow:
    listing_id: str
    occasion_fit: float
    emotional_resonance: float
    distinctiveness: float
    reasoning: str


_POOL_SQL = """
SELECT l.listing_id::text AS listing_id,
       lf.occasion,
       l.title AS headline,
       NULL    AS inside_message,
       li.storage_path AS cover_path
FROM listings l
JOIN listing_features lf USING (listing_id)
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
WHERE li.storage_path IS NOT NULL
  AND lf.occasion IS NOT NULL
"""


def _load_pool(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_sql(_POOL_SQL, engine())
    df = df[df["occasion"].apply(is_valid_occasion)].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df


def _fetch_image_b64(storage_path: str) -> str | None:
    if not storage_path or not storage_path.startswith("s3://"):
        return None
    try:
        data = get_object(storage_path)
        return base64.standard_b64encode(data).decode("ascii")
    except Exception as e:
        log.warning(f"Failed to fetch {storage_path}: {e}")
        return None


def _user_prompt(row: pd.Series) -> str:
    return (
        f"Occasion: {row['occasion']}\n"
        f"Headline: {row.get('headline') or '(none)'}\n"
        f"Inside message: {row.get('inside_message') or '(none)'}\n"
        f"\nNow score the card."
    )


def _parse_response(text: str) -> dict | None:
    """Tolerant JSON parse — strip code fences if model wraps output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # remove possible 'json' language hint
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    try:
        obj = json.loads(text)
        if all(d in obj for d in PSEUDO_DIMS):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _call_anthropic(image_b64: str, user_text: str, model: str) -> dict | None:
    """Call Anthropic Messages API with a vision-capable model."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=400,
        system=RATING_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )
    blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not blocks:
        return None
    return _parse_response(blocks[0])


def _call_openai(image_b64: str, user_text: str, model: str) -> dict | None:
    """Call OpenAI vision API."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": RATING_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
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


_UPSERT = """
INSERT INTO saleability_labels (listing_id, label_source, score, raw)
VALUES (%(listing_id)s, %(label_source)s, %(score)s, %(raw)s)
ON CONFLICT (listing_id, label_source) DO UPDATE
SET score = EXCLUDED.score,
    raw   = EXCLUDED.raw,
    created_at = NOW();
"""


def _persist(rows: list[PseudoLabelRow], label_source: str) -> int:
    if not rows:
        return 0
    with connection() as conn, conn.cursor() as cur:
        for r in rows:
            # Composite "score" = mean of three sub-dims; the predictor uses raw[*] directly.
            score = (r.occasion_fit + r.emotional_resonance + r.distinctiveness) / 3.0
            cur.execute(
                _UPSERT,
                {
                    "listing_id": r.listing_id,
                    "label_source": label_source,
                    "score": float(score),
                    "raw": Jsonb(
                        {
                            "occasion_fit": r.occasion_fit,
                            "emotional_resonance": r.emotional_resonance,
                            "distinctiveness": r.distinctiveness,
                            "reasoning": r.reasoning,
                        }
                    ),
                },
            )
    return len(rows)


def build_and_persist(
    label_source: str = "llm_pseudo_v1",
    *,
    limit: int | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> int:
    pool = _load_pool(limit=limit)
    if pool.empty:
        log.warning("No cards in pool to pseudo-label.")
        return 0

    provider = provider or settings.llm_provider
    if not model:
        if provider == "openai":
            model = "gpt-4.1-mini"
        else:
            model = "claude-sonnet-4-6"

    caller = _call_openai if provider == "openai" else _call_anthropic

    rows: list[PseudoLabelRow] = []
    for _, row in pool.iterrows():
        b64 = _fetch_image_b64(row["cover_path"])
        if not b64:
            continue
        resp = caller(b64, _user_prompt(row), model)
        if not resp:
            log.warning(f"No usable JSON for listing_id={row['listing_id']}")
            continue
        rows.append(
            PseudoLabelRow(
                listing_id=row["listing_id"],
                occasion_fit=float(resp.get("occasion_fit", 0.0)),
                emotional_resonance=float(resp.get("emotional_resonance", 0.0)),
                distinctiveness=float(resp.get("distinctiveness", 0.0)),
                reasoning=str(resp.get("reasoning", ""))[:200],
            )
        )

    written = _persist(rows, label_source=label_source)
    log.info(f"Pseudo-labelled {written} listings as label_source={label_source!r}")
    return written


def run(
    label_source: str = "llm_pseudo_v1",
    limit: int = 0,
    provider: str = "",
    model: str = "",
) -> None:
    written = build_and_persist(
        label_source=label_source,
        limit=(limit or None),
        provider=(provider or None),
        model=(model or None),
    )
    print(f"Wrote {written} pseudo-label rows with label_source={label_source!r}")


if __name__ == "__main__":
    typer.run(run)
