"""VLM-based 4-head perceptual quality labels for the multi-headed predictor.

Sends each scraped card image to a vision-language model (Claude / GPT-4o)
and asks for structured scores on four perceptual quality dimensions:

    occasion_fit | aesthetic | emotional_resonance | distinctiveness

The fifth predictor head — purchase_intent — is derived from human pairwise
preferences via Prolific 2AFC + Bradley-Terry scaling (see survey/ package).
VLMs cannot reliably assess commercial appeal; only humans know what they'd buy.

Labels stored in ``saleability_labels`` with ``label_source='vlm_4head_v1'``.


Run:
    python -m data.labels.vlm_labels                     # label all unlabelled
    python -m data.labels.vlm_labels --limit 50          # test run
    python -m data.labels.vlm_labels --provider openai   # use GPT-4o
    python -m data.labels.vlm_labels --dual              # both providers → inter-model agreement
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx
import typer
from psycopg.types.json import Jsonb

from common.config import settings
from common.db import connection, engine
from common.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_SOURCE = "vlm_4head_v1"
LABEL_SOURCE_5HEAD = "vlm_5head_v1"
VLM_DIMS = ("occasion_fit", "aesthetic", "emotional_resonance", "distinctiveness")
VLM_DIMS_5HEAD = ("occasion_fit", "aesthetic", "emotional_resonance", "distinctiveness", "purchase_intent")

# Greedy decoding. Neither provider defaults to 0, so the same card scored
# twice returned different numbers: labels were not reproducible, and the
# sampling noise landed straight in the predictor's training targets.
SCORING_TEMPERATURE = 0.0

# Max concurrent VLM calls — respect rate limits
MAX_CONCURRENCY = 5
# Max concurrent image downloads
DL_CONCURRENCY = 20
# Timeout per image download
DL_TIMEOUT = 15.0
# Retry count for failed VLM calls
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Prompt — structured JSON output for all 5 heads
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_4HEAD = """\
You are an expert greeting-card designer and visual critic evaluating \
card designs on perceptual quality dimensions.

You will see one greeting card image. All cards are birthday cards. \
Score the card on four dimensions, each on a 0.0–1.0 continuous scale \
(two decimal places).

Dimensions:
  occasion_fit (0-1): How well the card matches a birthday occasion. \
Consider imagery, text, and overall theme. A generic landscape with no \
birthday elements = 0.1. Clear birthday cake/candles/age text = 0.9.
  aesthetic (0-1): Visual quality — composition, colour harmony, \
typography, professionalism, print-readiness. High = gallery-worthy, \
polished design. Low = clip-art, poor layout, amateur typography.
  emotional_resonance (0-1): Emotional impact — does it evoke warmth, \
joy, humour, or sentiment? Would the recipient feel something? \
A blank template = 0.1. A card that makes you smile or feel warm = 0.9.
  distinctiveness (0-1): How original vs generic template. High = unique \
artistic voice, creative concept. Low = cookie-cutter stock design \
seen a thousand times.

Guidelines:
- Score independently per dimension — a funny but ugly card can score \
high on emotional_resonance but low on aesthetic.
- Use the full 0-1 range. Average cards ~0.5. Truly exceptional = 0.9+. \
Poor quality = 0.1-0.2.
- Be calibrated: most cards should cluster 0.3-0.7, with tails.
- Do NOT assess saleability or commercial viability — that is a separate \
human judgment.

Reply with ONLY a JSON object:
{"occasion_fit": 0.XX, "aesthetic": 0.XX, "emotional_resonance": 0.XX, \
"distinctiveness": 0.XX, \
"reasoning": "<one sentence, max 150 chars>"}"""

SYSTEM_PROMPT_5HEAD = """\
You are an expert greeting-card designer and market analyst evaluating \
card designs for quality and commercial potential.

You will see one greeting card image. Score the card on five dimensions, \
each on a 0.0–1.0 continuous scale (two decimal places).

Dimensions:
  occasion_fit (0-1): How well the card matches the occasion. \
Consider imagery, text, and overall theme.
  aesthetic (0-1): Visual quality — composition, colour harmony, \
typography, professionalism. High = polished design. Low = clip-art.
  emotional_resonance (0-1): Emotional impact — does it evoke warmth, \
joy, humour, or sentiment? Would the recipient feel something?
  distinctiveness (0-1): How original vs generic. High = unique artistic \
voice. Low = cookie-cutter stock design.
  purchase_intent (0-1): How likely is a typical buyer to purchase this \
card? Consider overall appeal, quality, shelf presence, and market fit. \
0.1 = would not buy. 0.9 = would definitely buy.

Guidelines:
- Score independently per dimension.
- Use the full 0-1 range. Average cards ~0.5. Truly exceptional = 0.9+. \
Poor quality = 0.1-0.2.
- Be calibrated: most cards should cluster 0.3-0.7, with tails.

Reply with ONLY a JSON object:
{"occasion_fit": 0.XX, "aesthetic": 0.XX, "emotional_resonance": 0.XX, \
"distinctiveness": 0.XX, "purchase_intent": 0.XX, \
"reasoning": "<one sentence, max 150 chars>"}"""

SYSTEM_PROMPT = SYSTEM_PROMPT_4HEAD


def _user_prompt(title: str | None, description: str | None) -> str:
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if description:
        desc = description[:300]  # truncate long descriptions
        parts.append(f"Description: {desc}")
    parts.append("\nScore this card.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CardRecord:
    listing_id: str
    title: str | None
    description: str | None
    image_url: str
    source: str


@dataclass
class VLMResult:
    listing_id: str
    scores: dict[str, float]
    reasoning: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


@dataclass
class RunStats:
    total: int = 0
    labelled: int = 0
    skipped: int = 0
    failed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> str:
        return (
            f"Done: {self.labelled}/{self.total} labelled, "
            f"{self.skipped} skipped (already done), {self.failed} failed | "
            f"tokens: {self.tokens_in:,}in + {self.tokens_out:,}out | "
            f"est. cost: ${self.cost_usd:.2f} | "
            f"time: {self.elapsed_s:.0f}s"
        )


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

# One listing per duplicate cluster, and only cards with an occasion.
#
# Print-on-demand catalogues carry the same design in several colourways as
# separate listings, and scoring each costs a vision call for a card the model
# has already judged. Unclustered listings group by their own id, so they are
# all kept.
#
# Listings with a NULL occasion are skipped because nothing downstream can use
# them: the predictor requires occasion IS NOT NULL, and the LoRA and
# condition-D selections all filter by occasion. This does couple the run to
# the occasion labels — if those are revised, re-run and _already_labelled will
# skip everything already scored.
_POOL_SQL = """
SELECT listing_id, title, description, source, image_url
FROM (
    SELECT DISTINCT ON (COALESCE(lf.duplicate_cluster_id, l.listing_id::text))
           l.listing_id::text AS listing_id,
           l.title,
           l.description,
           l.source,
           l.raw_metadata->'image_urls'->>0 AS image_url,
           l.last_seen_at
    FROM listings l
    LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
    WHERE l.raw_metadata->'image_urls' IS NOT NULL
      AND jsonb_array_length(l.raw_metadata->'image_urls') > 0
      AND (l.raw_metadata->'image_urls'->>0) IS NOT NULL
      AND lf.occasion IS NOT NULL
    ORDER BY COALESCE(lf.duplicate_cluster_id, l.listing_id::text), l.listing_id
) one_per_design
ORDER BY source, last_seen_at DESC
"""

_ALREADY_LABELLED_SQL = """
SELECT listing_id::text
FROM saleability_labels
WHERE label_source = %(label_source)s
"""


def _load_pool(limit: int | None = None) -> list[CardRecord]:
    """Load all cards with image URLs from DB."""
    import pandas as pd
    df = pd.read_sql(_POOL_SQL, engine())
    if df.empty:
        return []
    records = [
        CardRecord(
            listing_id=row["listing_id"],
            title=row.get("title"),
            description=row.get("description"),
            image_url=row["image_url"],
            source=row.get("source", "unknown"),
        )
        for _, row in df.iterrows()
        if row.get("image_url")
    ]
    if limit:
        records = records[:limit]
    return records


def _already_labelled(label_source: str) -> set[str]:
    """Return set of listing_ids that already have labels."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_ALREADY_LABELLED_SQL, {"label_source": label_source})
        return {row["listing_id"] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

async def _download_image_b64(
    client: httpx.AsyncClient, url: str
) -> str | None:
    """Download image from URL → base64 string."""
    try:
        resp = await client.get(url, timeout=DL_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        data = resp.content
        if len(data) < 1000:  # too small = probably error page
            return None
        if len(data) > 20_000_000:  # >20MB = skip
            return None
        return base64.standard_b64encode(data).decode("ascii")
    except Exception as e:
        log.debug(f"Image download failed {url[:80]}: {e}")
        return None


def _guess_media_type(url: str) -> str:
    url_lower = url.lower()
    if ".png" in url_lower:
        return "image/png"
    if ".webp" in url_lower:
        return "image/webp"
    if ".gif" in url_lower:
        return "image/gif"
    return "image/jpeg"


# ---------------------------------------------------------------------------
# VLM providers
# ---------------------------------------------------------------------------

async def _call_anthropic(
    image_b64: str,
    media_type: str,
    user_text: str,
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
    dims: tuple[str, ...] = VLM_DIMS,
) -> VLMResult | None:
    """Call Claude vision API (sync wrapper — anthropic SDK is sync)."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    t0 = time.time()
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=SCORING_TEMPERATURE,
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
    except Exception as e:
        log.warning(f"Anthropic API error: {e}")
        return None

    latency = int((time.time() - t0) * 1000)
    blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not blocks:
        return None

    parsed = _parse_response(blocks[0], dims=dims)
    if not parsed:
        return None

    return VLMResult(
        listing_id="",  # filled by caller
        scores={d: float(parsed.get(d, 0.0)) for d in dims},
        reasoning=str(parsed.get("reasoning", ""))[:150],
        provider="anthropic",
        model=model,
        tokens_in=getattr(msg.usage, "input_tokens", 0),
        tokens_out=getattr(msg.usage, "output_tokens", 0),
        latency_ms=latency,
    )


async def _call_openai(
    image_b64: str,
    media_type: str,
    user_text: str,
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
    dims: tuple[str, ...] = VLM_DIMS,
) -> VLMResult | None:
    """Call OpenAI GPT-4o vision API."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=300,
            temperature=SCORING_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                                "detail": "low",  # cheaper, sufficient for card eval
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

    latency = int((time.time() - t0) * 1000)
    choice = resp.choices[0] if resp.choices else None
    if not choice or not choice.message.content:
        return None

    parsed = _parse_response(choice.message.content, dims=dims)
    if not parsed:
        return None

    usage = resp.usage
    return VLMResult(
        listing_id="",
        scores={d: float(parsed.get(d, 0.0)) for d in dims},
        reasoning=str(parsed.get("reasoning", ""))[:150],
        provider="openai",
        model=model,
        tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
        tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(text: str, dims: tuple[str, ...] = VLM_DIMS) -> dict | None:
    """Tolerant JSON parse — strip code fences if model wraps output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        obj = json.loads(text)
        # Validate all dims present and in range
        for d in dims:
            if d not in obj:
                return None
            val = float(obj[d])
            if not (0.0 <= val <= 1.0):
                obj[d] = max(0.0, min(1.0, val))  # clamp
        return obj
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Approximate per-token costs (USD) as of 2025
_COST_TABLE: dict[str, tuple[float, float]] = {
    # (input $/1M tokens, output $/1M tokens)
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-3-5-20241022": (0.80, 4.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Rough cost estimate in USD."""
    # Find best matching model key
    costs = None
    for key, val in _COST_TABLE.items():
        if key in model or model in key:
            costs = val
            break
    if not costs:
        costs = (3.0, 15.0)  # default to Sonnet pricing
    return (tokens_in * costs[0] + tokens_out * costs[1]) / 1_000_000


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_UPSERT_LABEL = """
INSERT INTO saleability_labels (listing_id, label_source, score, raw)
VALUES (%(listing_id)s, %(label_source)s, %(score)s, %(raw)s)
ON CONFLICT (listing_id, label_source) DO UPDATE
SET score = EXCLUDED.score,
    raw   = EXCLUDED.raw,
    created_at = NOW();
"""


def _persist_results(results: list[VLMResult], label_source: str) -> int:
    """Write VLM results to saleability_labels table."""
    if not results:
        return 0
    with connection() as conn, conn.cursor() as cur:
        for r in results:
            # Composite score = unweighted mean of 4 perceptual quality dims
            composite = sum(
                r.scores.get(d, 0.0) for d in VLM_DIMS
            ) / len(VLM_DIMS)

            cur.execute(
                _UPSERT_LABEL,
                {
                    "listing_id": r.listing_id,
                    "label_source": label_source,
                    "score": float(composite),
                    "raw": Jsonb(
                        {
                            **r.scores,
                            "reasoning": r.reasoning,
                            "provider": r.provider,
                            "model": r.model,
                            "tokens_in": r.tokens_in,
                            "tokens_out": r.tokens_out,
                            "latency_ms": r.latency_ms,
                        }
                    ),
                },
            )
    return len(results)


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------

async def _label_one(
    card: CardRecord,
    http_client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    provider: Literal["anthropic", "openai"],
    model: str,
    stats: RunStats,
    system_prompt: str = SYSTEM_PROMPT,
    dims: tuple[str, ...] = VLM_DIMS,
) -> VLMResult | None:
    """Download image + call VLM for a single card."""
    # Download image
    b64 = await _download_image_b64(http_client, card.image_url)
    if not b64:
        stats.failed += 1
        return None

    media_type = _guess_media_type(card.image_url)
    user_text = _user_prompt(card.title, card.description)

    # Rate-limited VLM call
    async with sem:
        for attempt in range(MAX_RETRIES + 1):
            if provider == "anthropic":
                result = await _call_anthropic(
                    b64, media_type, user_text, model,
                    system_prompt=system_prompt, dims=dims,
                )
            else:
                result = await _call_openai(
                    b64, media_type, user_text, model,
                    system_prompt=system_prompt, dims=dims,
                )

            if result:
                result.listing_id = card.listing_id
                stats.labelled += 1
                stats.tokens_in += result.tokens_in
                stats.tokens_out += result.tokens_out
                stats.cost_usd += _estimate_cost(model, result.tokens_in, result.tokens_out)
                return result

            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)  # exponential backoff

    log.warning(f"Failed after {MAX_RETRIES + 1} attempts: {card.listing_id}")
    stats.failed += 1
    return None


async def _run_pipeline(
    cards: list[CardRecord],
    provider: Literal["anthropic", "openai"],
    model: str,
    label_source: str,
    batch_size: int = 50,
    system_prompt: str = SYSTEM_PROMPT,
    dims: tuple[str, ...] = VLM_DIMS,
) -> RunStats:
    """Run VLM labeling pipeline with async concurrency."""
    stats = RunStats(total=len(cards))
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    log.info(
        f"VLM labeling: {len(cards)} cards, provider={provider}, model={model}"
    )

    async with httpx.AsyncClient(
        headers={"User-Agent": settings.scraper_user_agent},
        follow_redirects=True,
        timeout=30.0,
    ) as http_client:
        # Process in batches for periodic persistence
        for batch_start in range(0, len(cards), batch_size):
            batch = cards[batch_start : batch_start + batch_size]
            tasks = [
                _label_one(
                    card, http_client, sem, provider, model, stats,
                    system_prompt=system_prompt, dims=dims,
                )
                for card in batch
            ]
            results = await asyncio.gather(*tasks)
            valid = [r for r in results if r is not None]

            if valid:
                written = _persist_results(valid, label_source)
                log.info(
                    f"Batch {batch_start // batch_size + 1}: "
                    f"{written} written | running: {stats.summary()}"
                )

    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_and_persist(
    *,
    provider: Literal["anthropic", "openai"] = "anthropic",
    model: str | None = None,
    label_source: str = LABEL_SOURCE,
    limit: int | None = None,
    force: bool = False,
    five_heads: bool = False,
) -> RunStats:
    """Label all unlabelled cards with VLM scores.

    Args:
        provider: LLM provider (anthropic or openai).
        model: Model name override. Defaults to claude-sonnet-4-20250514 / gpt-4o.
        label_source: Label source tag for DB.
        limit: Max cards to label (None = all).
        force: Re-label even if already done.
        five_heads: Score purchase_intent too (for training without survey data).
    """
    if five_heads:
        dims = VLM_DIMS_5HEAD
        system_prompt = SYSTEM_PROMPT_5HEAD
        if label_source == LABEL_SOURCE:
            label_source = LABEL_SOURCE_5HEAD
    else:
        dims = VLM_DIMS
        system_prompt = SYSTEM_PROMPT_4HEAD

    # Resolve model
    if not model:
        if provider == "anthropic":
            model = settings.llm_model
        else:
            model = "gpt-4.1-mini"

    # Validate API key
    if provider == "anthropic" and not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
    if provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    # Load cards
    pool = _load_pool(limit=limit)
    if not pool:
        log.warning("No cards with image URLs found in DB.")
        return RunStats()

    # Filter already-labelled (unless force)
    if not force:
        done = _already_labelled(label_source)
        before = len(pool)
        pool = [c for c in pool if c.listing_id not in done]
        skipped = before - len(pool)
        log.info(f"Pool: {before} total, {skipped} already labelled, {len(pool)} to do")
    else:
        skipped = 0

    if not pool:
        log.info("All cards already labelled. Use --force to re-label.")
        stats = RunStats()
        stats.skipped = skipped
        return stats

    # Run async pipeline
    stats = asyncio.run(_run_pipeline(
        pool, provider, model, label_source,
        system_prompt=system_prompt, dims=dims,
    ))
    stats.skipped = skipped
    return stats


def build_dual(
    *,
    limit: int | None = None,
    force: bool = False,
) -> tuple[RunStats, RunStats]:
    """Run both Anthropic + OpenAI and store as separate label sources.

    Inter-model agreement (Spearman rho) computed after both finish.
    """
    stats_a = build_and_persist(
        provider="anthropic",
        label_source="vlm_4head_claude",
        limit=limit,
        force=force,
    )
    stats_o = build_and_persist(
        provider="openai",
        label_source="vlm_4head_openai",
        limit=limit,
        force=force,
    )
    _report_agreement("vlm_4head_claude", "vlm_4head_openai")
    return stats_a, stats_o


def _report_agreement(source_a: str, source_b: str) -> None:
    """Compute and log Spearman rho between two label sources per dimension."""
    import pandas as pd
    from scipy import stats as sp_stats

    sql = """
    SELECT a.listing_id::text,
           a.raw AS raw_a,
           b.raw AS raw_b
    FROM saleability_labels a
    JOIN saleability_labels b ON a.listing_id = b.listing_id
    WHERE a.label_source = %(source_a)s
      AND b.label_source = %(source_b)s
    """
    df = pd.read_sql(sql, engine(), params={"source_a": source_a, "source_b": source_b})
    if df.empty or len(df) < 10:
        log.warning(f"Not enough paired labels ({len(df)}) for agreement analysis")
        return

    log.info(f"\n=== Inter-model agreement ({source_a} vs {source_b}), N={len(df)} ===")
    for dim in VLM_DIMS:
        try:
            vals_a = [row.get(dim, 0.0) for row in df["raw_a"]]
            vals_b = [row.get(dim, 0.0) for row in df["raw_b"]]
            rho, p = sp_stats.spearmanr(vals_a, vals_b)
            log.info(f"  {dim:25s}: rho={rho:.3f}  p={p:.4f}")
        except Exception:
            log.warning(f"  {dim:25s}: computation failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="VLM 5-head saleability labeling pipeline")


@app.command()
def label(
    provider: str = typer.Option("anthropic", help="LLM provider: anthropic or openai"),
    model: str = typer.Option("", help="Model override"),
    label_source: str = typer.Option(LABEL_SOURCE, help="Label source tag"),
    limit: int = typer.Option(0, help="Max cards (0=all)"),
    force: bool = typer.Option(False, help="Re-label already-done cards"),
    dual: bool = typer.Option(False, help="Run both providers for inter-model agreement"),
    five_heads: bool = typer.Option(False, "--five-heads", help="Include purchase_intent (all-LLM training)"),
) -> None:
    """Score card images with VLM on saleability dimensions."""
    if dual:
        stats_a, stats_o = build_dual(limit=limit or None, force=force)
        print(f"\nClaude:  {stats_a.summary()}")
        print(f"GPT-4o:  {stats_o.summary()}")
    else:
        stats = build_and_persist(
            provider=provider,  # type: ignore[arg-type]
            model=model or None,
            label_source=label_source,
            limit=limit or None,
            force=force,
            five_heads=five_heads,
        )
        print(f"\n{stats.summary()}")


@app.command()
def stats() -> None:
    """Show labeling statistics from DB."""
    import pandas as pd

    sql = """
    SELECT label_source,
           COUNT(*) AS n_labels,
           ROUND(AVG(score)::numeric, 3) AS avg_score,
           ROUND(STDDEV(score)::numeric, 3) AS std_score
    FROM saleability_labels
    GROUP BY label_source
    ORDER BY label_source
    """
    df = pd.read_sql(sql, engine())
    if df.empty:
        print("No labels in DB yet.")
        return

    print("\n=== Label Sources ===")
    print(df.to_string(index=False))

    # Per-dimension stats for VLM labels
    for src in df["label_source"]:
        if "vlm" not in src:
            continue
        dim_sql = """
        SELECT COUNT(*) AS n,
               ROUND(AVG((raw->>%(dim)s)::numeric), 3) AS mean,
               ROUND(STDDEV((raw->>%(dim)s)::numeric), 3) AS std,
               ROUND(MIN((raw->>%(dim)s)::numeric), 3) AS min,
               ROUND(MAX((raw->>%(dim)s)::numeric), 3) AS max
        FROM saleability_labels
        WHERE label_source = %(src)s
          AND raw->>%(dim)s IS NOT NULL
        """
        print(f"\n=== {src} — per dimension ===")
        for dim in VLM_DIMS:
            row = pd.read_sql(dim_sql, engine(), params={"src": src, "dim": dim})
            if not row.empty:
                r = row.iloc[0]
                print(
                    f"  {dim:25s}: "
                    f"mean={r['mean']}  std={r['std']}  "
                    f"range=[{r['min']}, {r['max']}]"
                )


@app.command()
def agreement() -> None:
    """Compute inter-model agreement between Claude and GPT-4o labels."""
    _report_agreement("vlm_4head_claude", "vlm_4head_openai")


if __name__ == "__main__":
    app()
