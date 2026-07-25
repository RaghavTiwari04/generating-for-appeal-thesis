"""Re-parse cached Redbubble pages so image_urls point at flat artwork.

`listings.raw_metadata.image_urls` was written by the old `_flat_image_url`,
which produced a transform Redbubble does not serve and skipped .webp URLs
entirely — so the stored primary image is the tilted card mockup. Re-running
the image downloader alone would re-fetch those same mockups; the pages have
to be parsed again with the current code first.

Raw HTML is cached for 30 days, so this normally re-parses from disk without
touching the network. Pages missing from the cache are re-fetched, subject to
the usual rate limit.

    python -m scripts.reparse_redbubble
"""

from __future__ import annotations

import asyncio

from common.db import connection
from common.logging import get_logger
from data.scrapers.redbubble import RedbubbleScraper

log = get_logger(__name__)

_URLS_SQL = "SELECT url FROM listings WHERE source = 'redbubble' ORDER BY listing_id;"


async def _run(limit: int | None) -> tuple[int, int]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_URLS_SQL)
        urls = [r["url"] for r in cur.fetchall()]
    if limit:
        urls = urls[:limit]

    log.info(f"Re-parsing {len(urls)} Redbubble listings")
    ok = failed = 0
    scraper = RedbubbleScraper()
    async with scraper:
        for i, url in enumerate(urls, 1):
            parsed = await scraper.fetch_and_store(url, use_cache=True)
            if parsed is None:
                failed += 1
            else:
                ok += 1
            if i % 100 == 0:
                log.info(f"  {i}/{len(urls)} ({ok} ok, {failed} failed)")
    return ok, failed


def main(limit: int | None = None) -> None:
    ok, failed = asyncio.run(_run(limit))
    print(f"Re-parsed {ok} listings ({failed} failed)")


if __name__ == "__main__":
    import typer

    typer.run(main)
