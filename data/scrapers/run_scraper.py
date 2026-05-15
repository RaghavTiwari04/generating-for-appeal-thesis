"""CLI driver: run a single scraper for a list of occasion queries.

Usage:
    python -m data.scrapers.run_scraper --source etsy --limit 500
    python -m data.scrapers.run_scraper --source redbubble --queries "birthday card,christmas card" --limit 200

Occasions from the canonical taxonomy are used as search queries by default.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import typer

from common.logging import get_logger
from common.occasions import OCCASIONS

log = get_logger(__name__)
app = typer.Typer()


def _make_scraper(source: str):
    if source == "etsy":
        from data.scrapers.etsy import EtsyScraper
        return EtsyScraper()
    elif source == "redbubble":
        from data.scrapers.redbubble import RedbubbleScraper
        return RedbubbleScraper()
    elif source == "zazzle":
        from data.scrapers.zazzle import ZazzleScraper
        return ZazzleScraper()
    elif source == "greetings_island":
        from data.scrapers.greetings_island import GreetingsIslandScraper
        return GreetingsIslandScraper()
    else:
        raise ValueError(f"Unknown source: {source!r}. Choose from etsy, redbubble, zazzle, greetings_island")


def _occasion_to_query(occasion: str) -> str:
    return occasion.replace("/", " ").replace("_", " ") + " greeting card"


async def _run(
    source: str,
    queries: list[str],
    limit_per_query: int,
    use_cache: bool,
) -> int:
    scraper = _make_scraper(source)
    total = 0
    async with scraper:
        for query in queries:
            log.info(f"[{source}] query={query!r} limit={limit_per_query}")
            count = 0
            async for url in scraper.discover(query=query, max_results=limit_per_query):
                parsed = await scraper.fetch_and_store(url, use_cache=use_cache)
                if parsed is not None:
                    count += 1
                    total += 1
            log.info(f"[{source}] query={query!r} → {count} listings")
    return total


@app.command()
def run(
    source: str = typer.Option(..., help="etsy | redbubble | zazzle | greetings_island"),
    queries: Optional[str] = typer.Option(
        None,
        help="Comma-separated search queries. Defaults to all canonical occasions.",
    ),
    occasions: Optional[str] = typer.Option(
        None,
        help="Comma-separated subset of OCCASIONS taxonomy (alternative to --queries).",
    ),
    limit: int = typer.Option(100, help="Max listings per query."),
    no_cache: bool = typer.Option(False, help="Bypass the raw HTML cache."),
) -> None:
    if queries:
        query_list = [q.strip() for q in queries.split(",") if q.strip()]
    elif occasions:
        query_list = [_occasion_to_query(o.strip()) for o in occasions.split(",") if o.strip()]
    else:
        query_list = [_occasion_to_query(o) for o in OCCASIONS]

    log.info(f"Starting {source} scraper: {len(query_list)} queries, limit={limit} each")
    t0 = time.monotonic()
    total = asyncio.run(_run(source, query_list, limit, use_cache=not no_cache))
    elapsed = time.monotonic() - t0
    log.info(f"Done: {total} listings in {elapsed:.1f}s")


if __name__ == "__main__":
    app()
