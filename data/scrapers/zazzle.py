"""Zazzle scraper (~5k cards). TOS review required. Stub selectors."""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import quote_plus

from selectolax.parser import HTMLParser

from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.etsy import _parse_price, _parse_int


class ZazzleScraper(Scraper):
    source = "zazzle"
    BASE = "https://www.zazzle.com"

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        assert self._client is not None
        page, emitted = 1, 0
        while emitted < max_results:
            url = f"{self.BASE}/s/{quote_plus(query)}+greeting+card?pg={page}"
            await self.rate_limiter.acquire()
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
            except Exception:
                return
            tree = HTMLParser(resp.text)
            links = {
                a.attributes.get("href", "")
                for a in tree.css("a[href*='-greeting_card-']")
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
        id_match = re.search(r"-(\d{16,})$", url.rstrip("/"))
        source_id = id_match.group(1) if id_match else url

        title = _text(tree.css_first("h1"))
        seller_node = tree.css_first("a[href*='/store/']")
        seller_id = None
        if seller_node:
            m = re.search(r"/store/([^/?#]+)", seller_node.attributes.get("href", ""))
            seller_id = m.group(1) if m else None

        price_text = _text(tree.css_first("[data-test='price']"))
        price_minor, currency = _parse_price(price_text)

        review_text = _text(tree.css_first("[data-test='review-count']"))
        review_count = _parse_int(review_text)

        return ParsedListing(
            source_listing_id=source_id,
            url=url,
            title=title,
            seller_id=seller_id,
            price_minor_units=price_minor,
            currency=currency,
            review_count=review_count,
            raw_metadata={"selectors_version": "v1-stub"},
        )


def _text(node) -> str | None:
    if node is None:
        return None
    t = node.text(strip=True)
    return t or None
