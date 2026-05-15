"""Zazzle scraper (~5k cards). TOS review required.

Primary: JSON-LD Product schema.
Fallback: CSS selectors (Zazzle uses React; some data ends up in __NEXT_DATA__
or window.__PRELOADED_STATE__ — checked as secondary fallback).
"""

from __future__ import annotations

import json
import re
from typing import AsyncIterator
from urllib.parse import quote_plus

from selectolax.parser import HTMLParser

from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.etsy import (
    _attr, _extract_jsonld, _parse_price, _text, _to_float, _to_int,
)


class ZazzleScraper(Scraper):
    source = "zazzle"
    BASE = "https://www.zazzle.com"

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        assert self._client is not None
        page, emitted = 1, 0
        while emitted < max_results:
            url = (
                f"{self.BASE}/s/{quote_plus(query.replace(' ', '+'))}"
                f"+greeting+card?pg={page}&st=date_created"
            )
            await self.rate_limiter.acquire()
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
            except Exception:
                return
            tree = HTMLParser(resp.text)
            # Zazzle product links contain "-greeting_card-" or "-card-" in the slug
            links = {
                a.attributes.get("href", "")
                for a in tree.css("a[href*='.com/']")
                if a.attributes.get("href")
                and re.search(r"-(greeting_card|card)-", a.attributes.get("href", ""))
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
        id_match = re.search(r"-(\d{16,})", url.rstrip("/"))
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
                                price_minor = int(round(float(str(raw_price)) * 100))
                                currency = currency or "USD"
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    pass

        # CSS fallbacks
        if not title:
            title = _text(tree.css_first("h1, h1.product-title"))
        if not seller_id:
            node = tree.css_first("a[href*='/store/']")
            href = _attr(node, "href") or ""
            m = re.search(r"/store/([^/?#]+)", href)
            seller_id = m.group(1) if m else None
        if price_minor is None:
            price_text = _text(tree.css_first(
                "[data-test='price'], span.price, [itemprop='price']"
            ))
            price_minor, currency = _parse_price(price_text)
        if review_count is None:
            review_count = _to_int(_text(tree.css_first(
                "[data-test='review-count'], [itemprop='reviewCount']"
            )))
        if not image_urls:
            image_urls = [
                img.attributes.get("src", "")
                for img in tree.css("img[src*='rlv.zcache'], img[src*='zazzle']")
                if img.attributes and img.attributes.get("src", "").startswith("http")
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
            favourite_count=None,
            is_bestseller=False,
            image_urls=image_urls,
            raw_metadata={"parse_strategy": "jsonld+nextdata+css", "selectors_version": "v2"},
        )


def _extract_next_data(tree: HTMLParser) -> dict | None:
    node = tree.css_first("script#__NEXT_DATA__")
    if not node:
        return None
    try:
        return json.loads(node.text())
    except (json.JSONDecodeError, TypeError):
        return None
