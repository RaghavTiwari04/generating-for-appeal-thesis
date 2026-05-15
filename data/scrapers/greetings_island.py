"""Greetings Island scraper (~5k cards). Free ecards + print. TOS review required. Stub."""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import quote_plus

from selectolax.parser import HTMLParser

from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.etsy import _parse_price


class GreetingsIslandScraper(Scraper):
    source = "greetings_island"
    BASE = "https://www.greetingsisland.com"

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        assert self._client is not None
        page, emitted = 1, 0
        while emitted < max_results:
            url = f"{self.BASE}/search?q={quote_plus(query)}&page={page}"
            await self.rate_limiter.acquire()
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
            except Exception:
                return
            tree = HTMLParser(resp.text)
            links = {
                a.attributes.get("href", "")
                for a in tree.css("a[href*='/cards/']")
                if a.attributes.get("href")
            }
            if not links:
                return
            for link in links:
                if emitted >= max_results:
                    return
                full = link if link.startswith("http") else self.BASE + link
                emitted += 1
                yield full
            page += 1

    def parse(self, html: str, url: str) -> ParsedListing:
        tree = HTMLParser(html)
        id_match = re.search(r"/cards/([^/?#]+)", url)
        source_id = id_match.group(1) if id_match else url

        title = _text(tree.css_first("h1"))
        price_text = _text(tree.css_first(".price"))
        price_minor, currency = _parse_price(price_text)
        image_urls = [
            img.attributes.get("src", "")
            for img in tree.css("img.card-img")
            if img.attributes.get("src")
        ]

        return ParsedListing(
            source_listing_id=source_id,
            url=url,
            title=title,
            price_minor_units=price_minor,
            currency=currency,
            image_urls=image_urls,
            raw_metadata={"selectors_version": "v1-stub"},
        )


def _text(node) -> str | None:
    if node is None:
        return None
    t = node.text(strip=True)
    return t or None
