"""Score scraped listings with the LLM card scorer, for predictor training.

Uses scoring.CardScorer — the same instrument that reranks generated candidates
and runs the system evaluation. SSR (Maier et al. 2025) for purchase intent,
rubric-guided judge (Zheng et al. 2023) for the four quality dimensions.

This module previously had its own direct-rating prompt, so the predictor was
trained on targets measured differently from how its output is finally judged.

The pool is one listing per duplicate cluster, birthday cards only: colourways
of one design would otherwise each cost a full scoring pass, and listings with
no occasion are unusable downstream.

Labels land in `saleability_labels` with `score` = mean of the four quality
dimensions and the full per-dimension detail in `raw`.

    python -m data.labels.vlm_labels label
    python -m data.labels.vlm_labels label --limit 20      # smoke test
    python -m data.labels.vlm_labels label --provider openai
    python -m data.labels.vlm_labels stats
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass

import httpx
import typer
from PIL import Image
from psycopg.types.json import Jsonb

from common.config import settings
from common.db import connection, engine
from common.logging import get_logger
from scoring import CardScorer, DIMS, quality_composite

log = get_logger(__name__)

LABEL_SOURCE = "llm_ssr_rubric_v1"

# Cards scored concurrently. Each card is 7 VLM calls, so this is the real
# multiplier against the provider's rate limit.
MAX_CONCURRENCY = 4
COMMIT_EVERY = 25

# One listing per duplicate cluster, and only cards with an occasion.
#
# Print-on-demand catalogues carry the same design in several colourways as
# separate listings; scoring each costs a full pass for a card already judged.
# Unclustered listings group by their own id, so they are all kept.
#
# Listings with a NULL occasion are skipped because nothing downstream can use
# them: the predictor requires occasion IS NOT NULL, and the LoRA and
# condition-D selections filter by occasion.
_POOL_SQL = """
SELECT listing_id, title, occasion, image_url
FROM (
    SELECT DISTINCT ON (COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text))
           l.listing_id::text AS listing_id,
           l.title,
           lf.occasion,
           l.raw_metadata->'image_urls'->>0 AS image_url,
           l.last_seen_at
    FROM listings l
    LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
    WHERE l.raw_metadata->'image_urls' IS NOT NULL
      AND jsonb_array_length(l.raw_metadata->'image_urls') > 0
      AND (l.raw_metadata->'image_urls'->>0) IS NOT NULL
      AND lf.occasion IS NOT NULL
    ORDER BY COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text), l.listing_id
) one_per_design
ORDER BY last_seen_at DESC
"""

_ALREADY_LABELLED_SQL = """
SELECT listing_id::text FROM saleability_labels WHERE label_source = %(label_source)s
"""

_UPSERT_LABEL = """
INSERT INTO saleability_labels (listing_id, label_source, score, raw)
VALUES (%(listing_id)s, %(label_source)s, %(score)s, %(raw)s)
ON CONFLICT (listing_id, label_source) DO UPDATE
SET score = EXCLUDED.score,
    raw   = EXCLUDED.raw,
    created_at = NOW();
"""


@dataclass
class Card:
    listing_id: str
    title: str | None
    occasion: str | None
    image_url: str


def _load_pool(limit: int | None) -> list[Card]:
    import pandas as pd

    df = pd.read_sql(_POOL_SQL, engine())
    cards = [
        Card(
            listing_id=r["listing_id"],
            title=r.get("title"),
            occasion=r.get("occasion"),
            image_url=r["image_url"],
        )
        for _, r in df.iterrows()
        if r.get("image_url")
    ]
    return cards[:limit] if limit else cards


def _already_labelled(label_source: str) -> set[str]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_ALREADY_LABELLED_SQL, {"label_source": label_source})
        return {r["listing_id"] for r in cur.fetchall()}


def _persist(rows: list[dict], label_source: str) -> int:
    if not rows:
        return 0
    with connection() as conn, conn.cursor() as cur:
        for row in rows:
            cur.execute(
                _UPSERT_LABEL,
                {
                    "listing_id": row["listing_id"],
                    "label_source": label_source,
                    "score": row["score"],
                    "raw": Jsonb(row["raw"]),
                },
            )
    return len(rows)


async def _fetch_image(client: httpx.AsyncClient, url: str) -> Image.Image | None:
    try:
        resp = await client.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) < 1000:      # an error page, not an image
            return None
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        log.debug(f"Image fetch failed {url[:70]}: {e}")
        return None


async def _score_card(
    card: Card,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    scorer: CardScorer,
) -> dict | None:
    image = await _fetch_image(client, card.image_url)
    if image is None:
        return None
    async with sem:
        # CardScorer is synchronous; running it in a thread keeps the event
        # loop free so other cards' image fetches proceed during the API waits.
        scores = await asyncio.to_thread(
            scorer.score,
            image,
            occasion=card.occasion or "",
            headline=card.title or "",
        )
    if not any(d in scores for d in DIMS):
        return None
    return {
        "listing_id": card.listing_id,
        "score": quality_composite(scores),
        "raw": scores,
    }


async def _run(cards: list[Card], scorer: CardScorer, label_source: str) -> int:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    written = failed = 0
    async with httpx.AsyncClient(
        headers={"User-Agent": settings.scraper_user_agent}, follow_redirects=True
    ) as client:
        for start in range(0, len(cards), COMMIT_EVERY):
            chunk = cards[start : start + COMMIT_EVERY]
            results = await asyncio.gather(
                *(_score_card(c, client, sem, scorer) for c in chunk)
            )
            good = [r for r in results if r is not None]
            failed += len(results) - len(good)
            written += _persist(good, label_source)
            log.info(
                f"  {min(start + COMMIT_EVERY, len(cards))}/{len(cards)} — "
                f"{written} scored, {failed} failed"
            )
    return written


app = typer.Typer(help="LLM saleability labelling (SSR + rubric judge)")


@app.command()
def label(
    provider: str = typer.Option("anthropic", help="anthropic | openai"),
    model: str | None = typer.Option(None, help="Model override"),
    label_source: str = typer.Option(LABEL_SOURCE, help="Label source tag"),
    limit: int | None = typer.Option(None, help="Score only the first N cards"),
    force: bool = typer.Option(False, help="Re-score cards that already have labels"),
) -> None:
    """Score listings on all five dimensions and persist the results."""
    cards = _load_pool(limit)
    if force:
        log.info(f"Pool: {len(cards)} cards (--force, re-scoring all)")
    else:
        done = _already_labelled(label_source)
        before = len(cards)
        cards = [c for c in cards if c.listing_id not in done]
        log.info(
            f"Pool: {before} total, {before - len(cards)} already scored, "
            f"{len(cards)} to do"
        )

    if not cards:
        log.info("Nothing to do.")
        return

    log.info(
        f"{len(cards)} cards x 7 calls = {len(cards) * 7} VLM calls via {provider}"
    )
    written = asyncio.run(
        _run(cards, CardScorer(provider=provider, model=model), label_source)
    )
    print(f"Scored {written} cards")


@app.command()
def stats(label_source: str = typer.Option(LABEL_SOURCE)) -> None:
    """Report label coverage and per-dimension means."""
    import pandas as pd

    df = pd.read_sql(
        "SELECT score, raw FROM saleability_labels WHERE label_source = %(s)s",
        engine(),
        params={"s": label_source},
    )
    if df.empty:
        print(f"No labels for {label_source}.")
        return
    import numpy as np

    print(
        f"{len(df)} cards scored, composite mean {df['score'].mean():.3f} "
        f"sd {df['score'].std():.3f}"
    )
    # Spread matters more than the mean: a dimension every card scores the same
    # on carries no signal for the predictor to learn, however sensible its
    # average looks.
    print(f"  {'dimension':22s} {'mean':>6s} {'sd':>6s} {'min':>6s} {'max':>6s}    n")
    for dim in DIMS:
        vals = [
            r.get(dim)
            for r in df["raw"]
            if isinstance(r, dict) and r.get(dim) is not None
        ]
        if vals:
            a = np.asarray(vals, dtype=float)
            print(
                f"  {dim:22s} {a.mean():6.3f} {a.std(ddof=1) if len(a) > 1 else 0.0:6.3f} "
                f"{a.min():6.3f} {a.max():6.3f} {len(a):4d}"
            )


if __name__ == "__main__":
    app()
