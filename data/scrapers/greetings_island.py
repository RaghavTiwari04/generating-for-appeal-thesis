"""Greetings Island scraper (~5k cards). TOS review required.

Primary: JSON-LD. Secondary: meta tags. Fallback: CSS.
Greetings Island is a free ecards + print site — pricing may not always
be present (free tier cards have no price).
"""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import quote_plus, urljoin

from selectolax.parser import HTMLParser

from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.etsy import (
    _extract_jsonld, _parse_price, _text, _to_float, _to_int,
)


class GreetingsIslandScraper(Scraper):
    source = "greetings_island"
    BASE = "https://www.greetingsisland.com"

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        assert self._client is not None
        page, emitted = 1, 0
        while emitted < max_results:
            url = (
                f"{self.BASE}/cards/all?"
                f"search={quote_plus(query)}&page={page}"
            )
            await self.rate_limiter.acquire()
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
            except Exception:
                return
            tree = HTMLParser(resp.text)
            links = {
                urljoin(self.BASE, a.attributes.get("href", ""))
                for a in tree.css("a[href*='/cards/']")
                if a.attributes.get("href")
                and "/cards/" in a.attributes.get("href", "")
                and "search" not in a.attributes.get("href", "")
            }
            if not links:
                return
            for link in links:
                if emitted >= max_results:
                    return
                emitted += 1
                yield link
            page += 1

    def parse(self, html: str, url: str) -> ParsedListing:
        tree = HTMLParser(html)
        id_match = re.search(r"/cards/[^/]+/([^/?#]+)", url)
        source_listing_id = id_match.group(1) if id_match else url

        # Primary: JSON-LD
        ld = _extract_jsonld(tree, "Product")
        title = description = None
        price_minor: int | None = None
        currency: str | None = None
        review_count: int | None = None
        review_avg: float | None = None
        image_urls: list[str] = []

        if ld:
            title = ld.get("name")
            description = ld.get("description")
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

        # Meta tag fallback (OG tags are very common)
        if not title:
            title = _meta(tree, "og:title") or _text(tree.css_first("h1"))
        if not description:
            description = _meta(tree, "og:description") or _meta(tree, "description")
        if not image_urls:
            og_img = _meta(tree, "og:image")
            if og_img:
                image_urls = [og_img]
        if not image_urls:
            image_urls = [
                img.attributes.get("src", "")
                for img in tree.css(
                    "img.card-img, img.card-image, "
                    "img[class*='card'], img[class*='preview']"
                )
                if img.attributes and img.attributes.get("src", "").startswith("http")
            ]

        # CSS price fallback
        if price_minor is None:
            price_text = _text(tree.css_first(
                ".price, [class*='price'], [itemprop='price'], "
                "span.cost, div.card-price"
            ))
            price_minor, currency = _parse_price(price_text)

        return ParsedListing(
            source_listing_id=source_listing_id,
            url=url,
            title=title,
            description=description,
            seller_id=None,
            price_minor_units=price_minor,
            currency=currency,
            review_count=review_count,
            review_avg=review_avg,
            favourite_count=None,
            is_bestseller=False,
            image_urls=image_urls,
            raw_metadata={"parse_strategy": "jsonld+meta+css", "selectors_version": "v2"},
        )


def _meta(tree: HTMLParser, name: str) -> str | None:
    node = (
        tree.css_first(f"meta[property='{name}']")
        or tree.css_first(f"meta[name='{name}']")
    )
    if node is None:
        return None
    return node.attributes.get("content") or None
