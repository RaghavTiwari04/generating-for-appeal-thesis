"""Greetings Island scraper (~5k cards). TOS review required.

Greetings Island is a free ecards + print site. Category pages are
server-rendered and contain /preview/cards/ links. Individual card
pages have JSON-LD and meta tags. Plain httpx works (no JS needed).

Note: many cards are free — pricing may be absent or zero.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from urllib.parse import urljoin

from selectolax.parser import HTMLParser

from common.logging import get_logger
from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.etsy import (
    _extract_jsonld,
    _parse_price,
    _text,
    _to_float,
    _to_int,
)

log = get_logger(__name__)

# Birthday subcategories to crawl for broad coverage
_BIRTHDAY_PATHS = [
    "/cards/birthday",
    "/cards/birthday/kids",
    "/cards/birthday/funny",
    "/cards/birthday/milestone",
    "/cards/birthday/family",
    "/cards/birthday/for-her",
    "/cards/birthday/for-him",
]


class GreetingsIslandScraper(Scraper):
    source = "greetings_island"
    use_playwright = False  # Static HTML works
    BASE = "https://www.greetingsisland.com"

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        """Crawl birthday category pages for /preview/cards/ links."""
        emitted = 0
        seen: set[str] = set()

        # Use category browsing instead of search (more reliable)
        paths = list(_BIRTHDAY_PATHS)
        # Also try search as fallback
        from urllib.parse import quote_plus
        search_url = f"{self.BASE}/cards/all?search={quote_plus(query)}"
        paths.append(search_url)

        for path in paths:
            if emitted >= max_results:
                return
            url = path if path.startswith("http") else self.BASE + path
            page_num = 1
            while emitted < max_results:
                page_url = f"{url}/{page_num}" if page_num > 1 else url
                try:
                    html = await self._fetch(page_url)
                except Exception as e:
                    log.debug(f"GI page failed: {page_url}: {e}")
                    break
                tree = HTMLParser(html)
                links = tree.css("a[href*='/preview/cards/']")
                if not links:
                    break
                new_count = 0
                for a in links:
                    href = a.attributes.get("href", "")
                    if not href:
                        continue
                    full = urljoin(self.BASE, href)
                    full = re.sub(r"\?.*", "", full)
                    if full in seen:
                        continue
                    seen.add(full)
                    new_count += 1
                    emitted += 1
                    yield full
                    if emitted >= max_results:
                        return
                if new_count == 0:
                    break
                page_num += 1
                if page_num > 10:
                    break

    def parse(self, html: str, url: str) -> ParsedListing:
        tree = HTMLParser(html)
        # URL pattern: /preview/cards/{slug}/{category}-{id}
        id_match = re.search(r"/(\d+[-\d]*)$", url.rstrip("/"))
        if not id_match:
            id_match = re.search(r"/([^/]+)$", url.rstrip("/"))
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
                    price_minor = round(float(str(offers.get("price", ""))) * 100)
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

        # Meta tag fallback (OG tags)
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
            seller_id=None,  # GI doesn't have individual sellers
            price_minor_units=price_minor,
            currency=currency,
            review_count=review_count,
            review_avg=review_avg,
            favourite_count=None,
            is_bestseller=False,
            image_urls=image_urls,
            raw_metadata={"parse_strategy": "jsonld+meta+css", "selectors_version": "v3"},
        )


def _meta(tree: HTMLParser, name: str) -> str | None:
    node = (
        tree.css_first(f"meta[property='{name}']")
        or tree.css_first(f"meta[name='{name}']")
    )
    if node is None:
        return None
    return node.attributes.get("content") or None
