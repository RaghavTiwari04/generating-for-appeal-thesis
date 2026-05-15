"""Redbubble scraper (~10k cards, art-style diversity).

STATUS: stub. TOS review required before production run. Selectors are
placeholders — fill in after inspecting live HTML.
"""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import quote_plus

from selectolax.parser import HTMLParser

from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.etsy import _parse_float, _parse_int, _parse_price


class RedbubbleScraper(Scraper):
    source = "redbubble"

    BASE = "https://www.redbubble.com"

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        assert self._client is not None
        page, emitted = 1, 0
        while emitted < max_results:
            search_url = (
                f"{self.BASE}/shop?query={quote_plus(query)}"
                f"&page={page}"
            )
            await self.rate_limiter.acquire()
            try:
                resp = await self._client.get(search_url)
                resp.raise_for_status()
            except Exception:
                return
            tree = HTMLParser(resp.text)
            links = {
                a.attributes.get("href", "")
                for a in tree.css("a[href*='/people/'][href*='/works/']")
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

        listing_id_match = re.search(r"/works/(\d+)", url)
        source_listing_id = listing_id_match.group(1) if listing_id_match else url

        # Placeholder selectors — update after inspecting live page
        title = _text(tree.css_first("h1"))
        description = _text(tree.css_first("[data-testid='work-description']"))
        seller_id_node = tree.css_first("a[href*='/people/']")
        seller_id = None
        if seller_id_node:
            m = re.search(r"/people/([^/?#]+)", seller_id_node.attributes.get("href", ""))
            seller_id = m.group(1) if m else None

        price_text = _text(tree.css_first("[data-testid='product-price']"))
        price_minor, currency = _parse_price(price_text)
        review_count = None
        review_avg = None
        favourite_count = None
        is_bestseller = False
        image_urls: list[str] = [
            img.attributes.get("src", "")
            for img in tree.css("img[src*='ih1.redbubble.net']")
            if img.attributes.get("src")
        ]

        return ParsedListing(
            source_listing_id=source_listing_id,
            url=url,
            title=title,
            description=description,
            seller_id=seller_id,
            price_minor_units=price_minor,
            currency=currency,
            review_count=review_count,
            review_avg=review_avg,
            favourite_count=favourite_count,
            is_bestseller=is_bestseller,
            image_urls=image_urls,
            raw_metadata={"selectors_version": "v1-stub"},
        )


def _text(node) -> str | None:
    if node is None:
        return None
    t = node.text(strip=True)
    return t or None
