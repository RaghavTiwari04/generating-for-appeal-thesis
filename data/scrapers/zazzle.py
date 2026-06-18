"""Zazzle scraper — Playwright-based (~5k cards). TOS review required.

Zazzle uses React SSR with client-side hydration. Search result pages
render product links via JS, so discovery requires Playwright.
Individual product pages have JSON-LD and __NEXT_DATA__, parseable
from rendered HTML.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from urllib.parse import quote_plus

from selectolax.parser import HTMLParser

from common.logging import get_logger
from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.etsy import (
    _attr,
    _extract_jsonld,
    _parse_price,
    _text,
    _to_float,
    _to_int,
)

log = get_logger(__name__)


class ZazzleScraper(Scraper):
    source = "zazzle"
    use_playwright = True
    BASE = "https://www.zazzle.com"

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        page, emitted = 1, 0
        while emitted < max_results:
            url = (
                f"{self.BASE}/s/{quote_plus(query.replace(' ', '+'))}"
                f"+greeting+card?pg={page}&st=date_created"
            )
            try:
                html = await self._pw_fetch(
                    url,
                    # Zazzle product tiles use data-testid or anchor patterns
                    wait_selector="a[href*='greeting_card'], a[href*='/pd/']",
                    wait_ms=5000,
                )
            except Exception as e:
                log.warning(f"Zazzle discover page {page} failed: {e}")
                return
            tree = HTMLParser(html)
            # Zazzle product URLs contain long numeric IDs
            links = set()
            for a in tree.css("a[href]"):
                href = a.attributes.get("href", "")
                # Match product pages: contain long number and greeting_card or /pd/
                if re.search(r"(\d{12,})", href) and (
                    "greeting_card" in href or "/pd/" in href
                ):
                    full = href if href.startswith("http") else self.BASE + href
                    links.add(re.sub(r"\?.*", "", full))
            if not links:
                log.debug(f"Zazzle discover: no product links on page {page}")
                return
            for link in links:
                if emitted >= max_results:
                    return
                emitted += 1
                yield link
            page += 1
            if page > 20:
                return

    def parse(self, html: str, url: str) -> ParsedListing:
        tree = HTMLParser(html)
        id_match = re.search(r"-(\d{16,})", url.rstrip("/"))
        if not id_match:
            id_match = re.search(r"/(\d{12,})", url)
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
            brand = ld.get("brand") or {}
            if isinstance(brand, dict):
                seller_id = brand.get("name")
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

        # Secondary: __NEXT_DATA__ (Zazzle React SSR)
        if not title:
            next_data = _extract_next_data(tree)
            if next_data:
                try:
                    props = next_data.get("props", {}).get("pageProps", {})
                    product = props.get("product") or props.get("item") or {}
                    title = title or product.get("name") or product.get("title")
                    description = description or product.get("description")
                    if not price_minor:
                        raw_price = product.get("price") or product.get("basePrice")
                        if raw_price:
                            try:
                                price_minor = round(float(str(raw_price)) * 100)
                                currency = currency or "USD"
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    pass

        # CSS fallbacks
        if not title:
            title = _text(tree.css_first("h1, h1.product-title"))
        if not title:
            # Zazzle page <title> pattern: "Product Name | Zazzle"
            t = _text(tree.css_first("title"))
            if t and " | Zazzle" in t:
                title = t.split(" | Zazzle")[0].strip()
        if not seller_id:
            node = tree.css_first("a[href*='/store/']")
            href = _attr(node, "href") or ""
            m = re.search(r"/store/([^/?#]+)", href)
            seller_id = m.group(1) if m else None
        if price_minor is None:
            # Zazzle uses itemprop="price" with content attribute (no visible text)
            price_node = tree.css_first("[itemprop='price']")
            if price_node:
                content_val = price_node.attributes.get("content", "")
                if content_val:
                    try:
                        price_minor = round(float(content_val) * 100)
                    except (ValueError, TypeError):
                        pass
            # Currency from itemprop
            if price_minor is not None and currency is None:
                cur_node = tree.css_first("[itemprop='priceCurrency']")
                if cur_node:
                    currency = cur_node.attributes.get("content", "").upper() or None
                if not currency:
                    currency = "USD"  # Zazzle default
        if price_minor is None:
            price_text = _text(tree.css_first(
                "[data-test='price'], span.price"
            ))
            price_minor, currency = _parse_price(price_text)
        if review_count is None:
            review_count = _to_int(_text(tree.css_first(
                "[data-test='review-count'], [itemprop='reviewCount']"
            )))
        if not image_urls:
            # Prefer og:image (single product hero) over all page images
            og_img = tree.css_first("meta[property='og:image']")
            if og_img:
                c = og_img.attributes.get("content", "")
                if c and c.startswith("http"):
                    image_urls = [c]
            # itemprop image as secondary
            if not image_urls:
                ip_img = tree.css_first("[itemprop='image']")
                if ip_img:
                    c = ip_img.attributes.get("content", "") or ip_img.attributes.get("src", "")
                    if c and c.startswith("http"):
                        image_urls = [c]
            # Last resort: first rlv.zcache image only
            if not image_urls:
                first = tree.css_first("img[src*='rlv.zcache']")
                if first:
                    src = first.attributes.get("src", "")
                    if src.startswith("http"):
                        image_urls = [src]

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
            raw_metadata={"parse_strategy": "jsonld+nextdata+css", "selectors_version": "v3_pw"},
        )


def _extract_next_data(tree: HTMLParser) -> dict | None:
    node = tree.css_first("script#__NEXT_DATA__")
    if not node:
        return None
    try:
        return json.loads(node.text())
    except (json.JSONDecodeError, TypeError):
        return None
