"""Weekly re-snapshot job for engagement metrics.

Fetches current review_count, favourite_count, price for a rolling subset
of ~5k listings and inserts into `listing_snapshots`. Run weekly via cron
or a scheduled task. Starts week 2 and runs for ≥ 12 consecutive weeks so
velocity features have meaningful slope data.

Usage:
    python -m data.scrapers.snapshot_job --limit 5000
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import httpx
import typer

from common.db import connection
from common.logging import get_logger
from data.scrapers.base import AsyncRateLimiter, ParsedListing
from common.config import settings

log = get_logger(__name__)

_SELECT_LISTINGS = """
SELECT listing_id, source, source_listing_id, url
FROM listings
WHERE last_seen_at < NOW() - INTERVAL '23 hours'
ORDER BY last_seen_at ASC
LIMIT %(limit)s;
"""

_UPSERT_SNAPSHOT = """
INSERT INTO listing_snapshots (listing_id, snapshot_at, review_count, favourite_count, price_minor_units)
VALUES (%(listing_id)s, %(ts)s, %(review_count)s, %(favourite_count)s, %(price_minor_units)s)
ON CONFLICT DO NOTHING;
"""

_UPDATE_LAST_SEEN = """
UPDATE listings
SET last_seen_at = %(ts)s,
    review_count = COALESCE(%(review_count)s, review_count),
    favourite_count = COALESCE(%(favourite_count)s, favourite_count),
    price_minor_units = COALESCE(%(price_minor_units)s, price_minor_units)
WHERE listing_id = %(listing_id)s;
"""


async def _refetch_one(
    client: httpx.AsyncClient,
    limiter: AsyncRateLimiter,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    """Lightweight re-fetch: parse only the metrics we need."""
    await limiter.acquire()
    try:
        resp = await client.get(row["url"], timeout=30.0)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        log.debug(f"Re-fetch failed {row['url']}: {e}")
        return None

    # Dispatch to per-source minimal parser
    source = row["source"]
    try:
        if source == "etsy":
            from data.scrapers.etsy import EtsyScraper
            parsed = EtsyScraper().parse(html, row["url"])
        else:
            return None
    except Exception as e:
        log.debug(f"Re-parse failed {row['url']}: {e}")
        return None

    return {
        "listing_id": row["listing_id"],
        "review_count": parsed.review_count,
        "favourite_count": parsed.favourite_count,
        "price_minor_units": parsed.price_minor_units,
    }


async def snapshot_batch(limit: int = 5000) -> int:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_LISTINGS, {"limit": limit})
        rows = cur.fetchall()

    log.info(f"Snapshotting {len(rows)} listings")
    limiter = AsyncRateLimiter(settings.scraper_rate_limit_per_sec)
    ua = settings.scraper_user_agent

    async with httpx.AsyncClient(
        headers={"User-Agent": ua}, follow_redirects=True
    ) as client:
        tasks = [_refetch_one(client, limiter, row) for row in rows]
        results = await asyncio.gather(*tasks)

    ts = datetime.now(tz=timezone.utc)
    updated = 0
    with connection() as conn, conn.cursor() as cur:
        for r in results:
            if r is None:
                continue
            cur.execute(_UPSERT_SNAPSHOT, {**r, "ts": ts})
            cur.execute(_UPDATE_LAST_SEEN, {**r, "ts": ts})
            updated += 1

    log.info(f"Snapshots written: {updated}")
    return updated


def run(limit: int = 5000) -> None:
    count = asyncio.run(snapshot_batch(limit))
    print(f"Done: {count} snapshots")


if __name__ == "__main__":
    typer.run(run)
