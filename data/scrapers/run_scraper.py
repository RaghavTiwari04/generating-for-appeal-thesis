"""CLI driver: run a single scraper for a list of occasion queries.

Usage:
    python -m data.scrapers.run_scraper --source etsy --limit 500
    python -m data.scrapers.run_scraper --source redbubble --queries "birthday card,christmas card" --limit 200

Occasions from the canonical taxonomy are used as search queries by default.
"""

from __future__ import annotations

import asyncio
import time

import typer

from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS as OCCASIONS

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


# Some taxonomy names are not phrases anyone titles a card with. Searching
# the literal name returns almost nothing — "birthday relationship greeting
# card" yielded 6 listings against 629 for birthday/general — so expand those
# to the terms sellers actually use. One occasion may map to several queries.
_OCCASION_QUERIES: dict[str, list[str]] = {
    "birthday/relationship": [
        "husband birthday card",
        "wife birthday card",
        "boyfriend birthday card",
        "girlfriend birthday card",
        "partner birthday card",
        "birthday card for him",
        "birthday card for her",
        "fiance birthday card",
    ],
    "birthday/kids": [
        "kids birthday card",
        "childrens birthday card",
        "birthday card for son",
        "birthday card for daughter",
        "1st birthday card",
        "birthday card boy",
        "birthday card girl",
    ],
    "birthday/milestone": [
        "18th birthday card",
        "21st birthday card",
        "30th birthday card",
        "40th birthday card",
        "50th birthday card",
        "60th birthday card",
        "70th birthday card",
        "milestone birthday card",
    ],
}


def _occasion_to_query(occasion: str) -> str:
    return occasion.replace("/", " ").replace("_", " ") + " greeting card"


def _occasion_to_queries(occasion: str) -> list[str]:
    """Search terms for an occasion — real seller phrasing where we have it."""
    return _OCCASION_QUERIES.get(occasion, [_occasion_to_query(occasion)])


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
            log.info(f"[{source}] query={query!r} -> {count} listings")
    return total


@app.command()
def run(
    source: str = typer.Option(..., help="etsy | redbubble | zazzle | greetings_island"),
    queries: str | None = typer.Option(
        None,
        help="Comma-separated search queries. Defaults to all canonical occasions.",
    ),
    occasions: str | None = typer.Option(
        None,
        help="Comma-separated subset of OCCASIONS taxonomy (alternative to --queries).",
    ),
    limit: int = typer.Option(100, help="Max listings per query."),
    no_cache: bool = typer.Option(False, help="Bypass the raw HTML cache."),
) -> None:
    if queries:
        query_list = [q.strip() for q in queries.split(",") if q.strip()]
    elif occasions:
        query_list = [
            q
            for o in occasions.split(",")
            if o.strip()
            for q in _occasion_to_queries(o.strip())
        ]
    else:
        query_list = [q for o in OCCASIONS for q in _occasion_to_queries(o)]

    log.info(f"Starting {source} scraper: {len(query_list)} queries, limit={limit} each")
    t0 = time.monotonic()
    total = asyncio.run(_run(source, query_list, limit, use_cache=not no_cache))
    elapsed = time.monotonic() - t0
    log.info(f"Done: {total} listings in {elapsed:.1f}s")


if __name__ == "__main__":
    app()
