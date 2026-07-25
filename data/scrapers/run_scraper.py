"""CLI driver: sweep a site's birthday catalogue into `listings`.

Scrape broadly first, then let `data.features.occasion_nli` split the
result into subtypes — the search term must not decide the label, or the
subtype distribution just reflects how many queries we wrote per subtype.

Usage:
    python -m data.scrapers.run_scraper --source redbubble --limit 1000
    python -m data.scrapers.run_scraper --source redbubble --queries "birthday card" --limit 200
"""

from __future__ import annotations

import asyncio
import time

import typer

from common.logging import get_logger

log = get_logger(__name__)
app = typer.Typer()


def _make_scraper(source: str):
    if source == "redbubble":
        from data.scrapers.redbubble import RedbubbleScraper
        return RedbubbleScraper()
    elif source == "greetings_island":
        from data.scrapers.greetings_island import GreetingsIslandScraper
        return GreetingsIslandScraper()
    else:
        raise ValueError(f"Unknown source: {source!r}. Choose from redbubble, greetings_island")


# Sweep the whole birthday catalogue, then let the occasion classifier split
# it into subtypes.
#
# Querying per subcategory ("husband birthday card" -> birthday/relationship)
# would make the search term decide the label, so the subtype distribution
# would just mirror how many queries we wrote per subtype rather than what
# the market actually stocks. These are broad and tone-based instead, chosen
# to widen catalogue coverage without encoding a subtype.
BIRTHDAY_QUERIES: list[str] = [
    "birthday card",
    "happy birthday card",
    "birthday greeting card",
    "funny birthday card",
    "cute birthday card",
    "birthday wishes card",
]


async def _run(
    source: str,
    queries: list[str],
    limit_per_query: int,
    use_cache: bool,
) -> int:
    scraper = _make_scraper(source)
    total = 0
    seen: set[str] = set()

    # Some scrapers enumerate category pages and ignore the query entirely
    # (Greetings Island). Running them once per query re-fetched identical
    # pages N times and deduped only at the DB.
    if getattr(scraper, "ignores_query", False):
        queries = queries[:1]
        limit_per_query = limit_per_query * max(1, len(BIRTHDAY_QUERIES))
        log.info(f"[{source}] enumerates categories; one pass, limit={limit_per_query}")

    async with scraper:
        for query in queries:
            log.info(f"[{source}] query={query!r} limit={limit_per_query}")
            count = 0
            async for url in scraper.discover(query=query, max_results=limit_per_query):
                if url in seen:
                    continue
                seen.add(url)
                parsed = await scraper.fetch_and_store(url, use_cache=use_cache)
                if parsed is not None:
                    count += 1
                    total += 1
            log.info(f"[{source}] query={query!r} -> {count} new listings")
    return total


@app.command()
def run(
    source: str = typer.Option(..., help="redbubble | greetings_island"),
    queries: str | None = typer.Option(
        None,
        help="Comma-separated search queries. Defaults to a broad birthday sweep.",
    ),
    limit: int = typer.Option(100, help="Max listings per query."),
    no_cache: bool = typer.Option(False, help="Bypass the raw HTML cache."),
) -> None:
    if queries:
        query_list = [q.strip() for q in queries.split(",") if q.strip()]
    else:
        query_list = list(BIRTHDAY_QUERIES)

    log.info(f"Starting {source} scraper: {len(query_list)} queries, limit={limit} each")
    t0 = time.monotonic()
    total = asyncio.run(_run(source, query_list, limit, use_cache=not no_cache))
    elapsed = time.monotonic() - t0
    log.info(f"Done: {total} listings in {elapsed:.1f}s")


if __name__ == "__main__":
    app()
