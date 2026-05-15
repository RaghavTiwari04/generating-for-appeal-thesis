"""Etsy scraper.

Primary parse strategy: JSON-LD structured data (`application/ld+json`).
Etsy embeds Product schema on every listing page — this is far more stable
than CSS selectors which change with every deploy.

Fallback: CSS selectors for fields not covered by JSON-LD (bestseller badge,
favourite count, image carousel).

TOS: Etsy's Terms of Use restrict automated scraping. Before any production
run, complete the per-source TOS review documented in the thesis ethics chapter,
and consider using the Etsy Open API for review/favourite counts instead.
"""

from __future__ import annotations

import json
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
        assert self._client is not None
        page, emitted = 1, 0
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
            # Both href="/listing/…" and full URLs appear in Etsy search results
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
                # Strip query-string noise so cache keys are clean
                clean = re.sub(r"\?.*", "", link)
                emitted += 1
                yield clean
            page += 1
            if emitted >= 48 * 20:
                return

    def parse(self, html: str, url: str) -> ParsedListing:
        tree = HTMLParser(html)
        listing_id_match = re.search(r"/listing/(\d+)", url)
        source_listing_id = listing_id_match.group(1) if listing_id_match else url

        # ── Primary: JSON-LD ────────────────────────────────────────────────
        ld = _extract_jsonld(tree, typ="Product")
        title = description = seller_id = None
        price_minor: int | None = None
        currency: str | None = None
        review_count: int | None = None
        review_avg: float | None = None
        image_urls: list[str] = []

        if ld:
            title = ld.get("name")
            description = ld.get("description")

            # Seller: brand or seller field
            brand = ld.get("brand") or ld.get("seller") or {}
            if isinstance(brand, dict):
                seller_id = brand.get("name") or brand.get("url")
                if seller_id and "/" in seller_id:
                    m = re.search(r"/shop/([^/?#]+)", seller_id)
                    seller_id = m.group(1) if m else seller_id

            # Price: offers[0].price
            offers = ld.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if isinstance(offers, dict):
                price_str = str(offers.get("price", ""))
                cur_code = offers.get("priceCurrency", "")
                if price_str:
                    try:
                        price_minor = int(round(float(price_str) * 100))
                    except ValueError:
                        pass
                if cur_code:
                    currency = cur_code.upper()

            # Ratings
            agg = ld.get("aggregateRating") or {}
            if isinstance(agg, dict):
                review_count = _to_int(agg.get("reviewCount") or agg.get("ratingCount"))
                review_avg = _to_float(agg.get("ratingValue"))

            # Images
            imgs = ld.get("image") or []
            if isinstance(imgs, str):
                imgs = [imgs]
            image_urls = [i for i in imgs if isinstance(i, str) and i.startswith("http")]

        # ── Fallback: CSS selectors ──────────────────────────────────────────
        if not title:
            title = _text(tree.css_first(
                "h1[data-buy-box-listing-title], "
                "h1.wt-text-heading-01, "
                "h1"
            ))
        if not description:
            description = _text(tree.css_first(
                "[data-product-details-description-text-toggle-text], "
                "[data-product-details-description-text], "
                "p.wt-text-body-01"
            ))
        if not seller_id:
            node = tree.css_first("a[href*='/shop/']")
            href = _attr(node, "href") or ""
            m = re.search(r"/shop/([^/?#]+)", href)
            seller_id = m.group(1) if m else None

        if price_minor is None:
            price_text = _text(tree.css_first(
                "[data-buy-box-region='price'], "
                "[data-buy-box-region='price'] p, "
                "p.wt-text-heading-01"
            ))
            price_minor, currency = _parse_price(price_text)

        if review_count is None:
            review_count = _to_int(_text(tree.css_first(
                "[data-review-count], span[title*='review']"
            )))
        if review_avg is None:
            review_avg = _to_float(_text(tree.css_first(
                "[data-review-rating], input[name='rating']"
            )))

        # Favourite count not in JSON-LD — CSS only
        fav_text = _text(tree.css_first(
            "[data-favorites-count], button[data-wt-analytics-region='listing-page-heart-count'] span"
        ))
        favourite_count = _to_int(fav_text)

        # Bestseller badge
        is_bestseller = bool(tree.css_first(
            "[data-bestseller-badge], "
            "[data-first-bestseller-listing-v2], "
            "span.wt-badge[data-badge-type='bestseller']"
        ))

        # Extra images from carousel if JSON-LD gave none
        if not image_urls:
            image_urls = [
                img.attributes.get("src") or img.attributes.get("data-src", "")
                for img in tree.css(
                    "img[data-listing-image], "
                    "img.wt-max-width-full, "
                    "img[data-lazy-img-xl]"
                )
                if img.attributes
            ]
            image_urls = [u for u in image_urls if u and u.startswith("http")]

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
            raw_metadata={"parse_strategy": "jsonld+css", "selectors_version": "v2"},
        )


# ── Shared helpers ────────────────────────────────────────────────────────────

def _extract_jsonld(tree: HTMLParser, typ: str) -> dict | None:
    """Find first JSON-LD block with @type matching `typ`."""
    for node in tree.css("script[type='application/ld+json']"):
        try:
            data = json.loads(node.text())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") == typ:
                    return item
        elif isinstance(data, dict):
            if data.get("@type") == typ:
                return data
            # Some sites wrap in @graph
            for item in data.get("@graph", []):
                if isinstance(item, dict) and item.get("@type") == typ:
                    return item
    return None


def _text(node) -> str | None:
    if node is None:
        return None
    return node.text(strip=True) or None


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
        return int(round(float(amount) * 100)), _CURRENCY_MAP.get(symbol)
    except ValueError:
        return None, None


def _to_int(val) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    digits = re.sub(r"[^0-9]", "", str(val))
    return int(digits) if digits else None


def _to_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, float):
        return val
    m = re.search(r"[0-9]+(?:\.[0-9]+)?", str(val))
    return float(m.group(0)) if m else None
