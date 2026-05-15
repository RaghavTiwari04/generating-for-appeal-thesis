"""Redbubble scraper (~10k cards).

Primary: JSON-LD Product schema (present on all Redbubble product pages).
Fallback: CSS selectors.

TOS review required before production run.
"""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import quote_plus, urljoin

from selectolax.parser import HTMLParser

from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.etsy import (
    _attr, _extract_jsonld, _parse_price, _text, _to_float, _to_int,
)


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
                f"&iaCode=u-greeting-cards&page={page}"
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
                for a in tree.css("a[href*='/works/']")
                if a.attributes.get("href")
            }
            if not links:
                return
            for link in links:
                if emitted >= max_results:
                    return
                full = link if link.startswith("http") else self.BASE + link
                emitted += 1
                yield re.sub(r"\?.*", "", full)
            page += 1

    def parse(self, html: str, url: str) -> ParsedListing:
        tree = HTMLParser(html)
        id_match = re.search(r"/works/(\d+)", url)
        source_listing_id = id_match.group(1) if id_match else url

        # Primary: JSON-LD
        ld = _extract_jsonld(tree, "Product")
        title = description = seller_id = None
        price_minor: int | None = None
        currency: str | None = None
        review_count: int | None = None
        review_avg: float | None = None
        image_urls: list[str] = []

        if ld:
            title = ld.get("name")
            description = ld.get("description")

            brand = ld.get("brand") or ld.get("seller") or {}
            if isinstance(brand, dict):
                seller_id = brand.get("name")

            offers = ld.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if isinstance(offers, dict):
                try:
                    price_minor = int(round(float(str(offers.get("price", ""))) * 100))
                except (ValueError, TypeError):
                    pass
                currency = str(offers.get("priceCurrency", "")).upper() or None

            agg = ld.get("aggregateRating") or {}
            if isinstance(agg, dict):
                review_count = _to_int(agg.get("reviewCount") or agg.get("ratingCount"))
                review_avg = _to_float(agg.get("ratingValue"))

            imgs = ld.get("image") or []
            if isinstance(imgs, str):
                imgs = [imgs]
            image_urls = [i for i in imgs if isinstance(i, str) and i.startswith("http")]

        # Fallbacks
        if not title:
            title = _text(tree.css_first("h1"))
        if not description:
            description = _text(tree.css_first(
                "[data-testid='work-description'], "
                "div.work-description, "
                "p.work-description-text"
            ))
        if not seller_id:
            node = tree.css_first("a[href*='/people/']")
            href = _attr(node, "href") or ""
            m = re.search(r"/people/([^/?#]+)", href)
            seller_id = m.group(1) if m else None
        if price_minor is None:
            price_text = _text(tree.css_first(
                "[data-testid='product-price'], "
                "span.price, "
                "div.price"
            ))
            price_minor, currency = _parse_price(price_text)
        if not image_urls:
            image_urls = [
                img.attributes.get("src", "")
                for img in tree.css("img[src*='ih1.redbubble.net'], img[src*='rdbl.co']")
                if img.attributes
            ]
            image_urls = [u for u in image_urls if u.startswith("http")]

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
            favourite_count=None,
            is_bestseller=False,
            image_urls=image_urls,
            raw_metadata={"parse_strategy": "jsonld+css", "selectors_version": "v2"},
        )
