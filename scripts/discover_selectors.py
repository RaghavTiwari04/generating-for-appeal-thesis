"""Selector discovery tool.

Uses Playwright (real browser, bypasses bot detection) to fetch a live
listing from each scraper source, runs the parse() function, and reports
what was extracted vs what was None.

Run once after TOS review to verify selectors before a production scrape:

    pip install playwright
    playwright install chromium
    python scripts/discover_selectors.py

Or target one source:
    python scripts/discover_selectors.py --source etsy
"""

from __future__ import annotations

import asyncio
from typing import Any

import typer

TEST_URLS = {
    "etsy": "https://www.etsy.com/uk/listing/1234567890/",   # replace with a real listing
    "redbubble": "https://www.redbubble.com/i/greeting-card/Birthday-by-test/1234567890.5MT14",
    "zazzle": "https://www.zazzle.com/happy_birthday_card-137590148547323136",
    "greetings_island": "https://www.greetingsisland.com/cards/birthday/general/1",
}

app = typer.Typer()


async def _fetch_with_playwright(url: str) -> str:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(2000)  # let JS hydrate
        html = await page.content()
        await browser.close()
        return html


def _report(source: str, listing) -> None:
    fields = {
        "title":          listing.title,
        "description":    (listing.description or "")[:80] if listing.description else None,
        "seller_id":      listing.seller_id,
        "price_minor":    listing.price_minor_units,
        "currency":       listing.currency,
        "review_count":   listing.review_count,
        "review_avg":     listing.review_avg,
        "favourite_count":listing.favourite_count,
        "is_bestseller":  listing.is_bestseller,
        "image_urls":     f"{len(listing.image_urls)} URLs" if listing.image_urls else None,
    }
    print(f"\n{'='*50}")
    print(f"  {source.upper()} parse results")
    print(f"{'='*50}")
    ok, missing = 0, 0
    for field, val in fields.items():
        status = "✓" if val is not None else "✗"
        print(f"  {status}  {field:<20} {val!r}")
        if val is not None:
            ok += 1
        else:
            missing += 1
    print(f"\n  {ok} fields populated, {missing} missing")


@app.command()
def run(
    source: str = typer.Option("all", help="all | etsy | redbubble | zazzle | greetings_island"),
    url: str | None = typer.Option(None, help="Override URL for testing"),
) -> None:
    from data.scrapers.etsy import EtsyScraper
    from data.scrapers.redbubble import RedbubbleScraper
    from data.scrapers.zazzle import ZazzleScraper
    from data.scrapers.greetings_island import GreetingsIslandScraper

    scraper_map = {
        "etsy":              EtsyScraper(),
        "redbubble":         RedbubbleScraper(),
        "zazzle":            ZazzleScraper(),
        "greetings_island":  GreetingsIslandScraper(),
    }
    sources = list(scraper_map.keys()) if source == "all" else [source]

    for src in sources:
        test_url = url or TEST_URLS.get(src, "")
        if not test_url:
            print(f"No test URL for {src} — skipping")
            continue

        print(f"\nFetching {src}: {test_url}")
        try:
            html = asyncio.run(_fetch_with_playwright(test_url))
        except Exception as e:
            print(f"  Playwright fetch failed: {e}")
            print("  Install: pip install playwright && playwright install chromium")
            continue

        scraper = scraper_map[src]
        try:
            listing = scraper.parse(html, test_url)
            _report(src, listing)
        except Exception as e:
            print(f"  Parse error: {e}")


if __name__ == "__main__":
    app()
