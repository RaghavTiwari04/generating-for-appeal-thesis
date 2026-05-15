"""Etsy scraper.

STATUS: stub. Etsy's terms of service restrict automated scraping; before any
production run, complete the per-source TOS review documented in the thesis
ethics chapter and (where required) use the official Etsy Open API instead.

Once authorised, fill in the selectors below. The structure is in place so the
rest of the pipeline (storage, dedup, features) can be wired and tested with
fixture HTML.
"""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import quote_plus, urljoin

from selectolax.parser import HTMLParser

from data.scrapers.base import ParsedListing, Scraper


class EtsyScraper(Scraper):
    source = "etsy"

    BASE = "https://www.etsy.com"

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        """Yield listing URLs for a search query. Paginates until max_results."""
        assert self._client is not None
        per_page = 48
        page = 1
        emitted = 0
        while emitted < max_results:
            search_url = (
                f"{self.BASE}/search?q={quote_plus(query)}"
                f"&category=greeting-cards&page={page}"
            )
            await self.rate_limiter.acquire()
            try:
                resp = await self._client.get(search_url)
                resp.raise_for_status()
            except Exception:
                return
            tree = HTMLParser(resp.text)
            links = {
                urljoin(self.BASE, a.attributes.get("href", ""))
                for a in tree.css("a[href*='/listing/']")
                if a.attributes.get("href")
            }
            if not links:
                return
            for link in links:
                if emitted >= max_results:
                    return
                emitted += 1
                yield link
            page += 1
            if emitted >= per_page * 20:  # safety cap
                return

    def parse(self, html: str, url: str) -> ParsedListing:
        tree = HTMLParser(html)

        listing_id_match = re.search(r"/listing/(\d+)", url)
        source_listing_id = listing_id_match.group(1) if listing_id_match else url

        title = _text(tree.css_first("h1[data-buy-box-listing-title]"))
        description = _text(tree.css_first("[data-product-details-description-text]"))
        seller_id = _attr(tree.css_first("a[href*='/shop/']"), "href")
        if seller_id:
            m = re.search(r"/shop/([^/?#]+)", seller_id)
            seller_id = m.group(1) if m else None

        price_text = _text(tree.css_first("[data-buy-box-region='price']"))
        price_minor, currency = _parse_price(price_text)

        review_count = _parse_int(_text(tree.css_first("[data-review-count]")))
        review_avg = _parse_float(_text(tree.css_first("[data-review-rating]")))
        favourite_count = _parse_int(_text(tree.css_first("[data-favorites-count]")))

        is_bestseller = bool(tree.css_first("[data-bestseller-badge]"))

        image_urls = [
            img.attributes.get("src") or img.attributes.get("data-src", "")
            for img in tree.css("img.wt-max-width-full")
            if img.attributes
        ]
        image_urls = [u for u in image_urls if u]

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
            raw_metadata={"selectors_version": "v1"},
        )


def _text(node) -> str | None:
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


def _attr(node, key: str) -> str | None:
    if node is None:
        return None
    return node.attributes.get(key)


_PRICE_RE = re.compile(r"([£$€])\s*([0-9]+(?:[.,][0-9]{1,2})?)")
_CURRENCY_MAP = {"£": "GBP", "$": "USD", "€": "EUR"}


def _parse_price(text: str | None) -> tuple[int | None, str | None]:
    if not text:
        return None, None
    m = _PRICE_RE.search(text)
    if not m:
        return None, None
    symbol, amount = m.group(1), m.group(2).replace(",", ".")
    try:
        minor = int(round(float(amount) * 100))
    except ValueError:
        return None, None
    return minor, _CURRENCY_MAP.get(symbol)


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def _parse_float(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
    return float(m.group(0)) if m else None
