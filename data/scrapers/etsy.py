"""Etsy scraper — dual strategy: API primary, Playwright fallback.

Strategy 1 (preferred): Etsy Open API v3
  - Requires ETSY_API_KEY in .env (register at developers.etsy.com)
  - Returns structured JSON: title, price, reviews, favorites, images, seller
  - Rate limit: 5000 requests/day (generous for research)

Strategy 2 (fallback): Playwright headless browser
  - Used only if no API key is configured
  - Etsy uses aggressive DataDome bot detection; success rate is low
  - Individual listing pages may still work if search discovery fails

TOS: Etsy Open API is the sanctioned access method for research.
Register a development application at developers.etsy.com.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from urllib.parse import quote_plus, urljoin

from selectolax.parser import HTMLParser

from common.config import settings
from common.logging import get_logger
from data.scrapers.base import ParsedListing, Scraper

log = get_logger(__name__)


class EtsyScraper(Scraper):
    source = "etsy"
    # Only use Playwright if no API key (fallback mode)
    use_playwright = False
    BASE = "https://www.etsy.com"
    API_BASE = "https://openapi.etsy.com/v3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._api_key = getattr(settings, "etsy_api_key", None) or None
        self._api_cache: dict[str, ParsedListing] = {}  # url -> parsed from API
        if self._api_key:
            log.info("Etsy: using Open API v3")
        else:
            log.warning(
                "Etsy: no ETSY_API_KEY set. Using Playwright fallback "
                "(low success rate due to bot detection). "
                "Register at developers.etsy.com for API access."
            )
            self.use_playwright = True

    async def discover(  # type: ignore[override]
        self, *, query: str, max_results: int = 100
    ) -> AsyncIterator[str]:
        if self._api_key:
            async for url in self._discover_api(query=query, max_results=max_results):
                yield url
        else:
            async for url in self._discover_playwright(query=query, max_results=max_results):
                yield url

    async def fetch_and_store(self, url: str, *, use_cache: bool = True,
                               wait_selector: str | None = None) -> ParsedListing | None:
        """Override: if API data cached, upsert directly (no HTTP fetch needed)."""
        if url in self._api_cache:
            parsed = self._api_cache.pop(url)
            from data.scrapers.base import upsert_listing
            upsert_listing(self.source, parsed)
            return parsed
        # Playwright fallback path
        return await super().fetch_and_store(url, use_cache=use_cache, wait_selector=wait_selector)

    # -- API-based discovery ---------------------------------------------------

    async def _discover_api(self, *, query: str, max_results: int) -> AsyncIterator[str]:
        """Search via Etsy Open API v3. Returns listing URLs."""
        assert self._client is not None
        emitted, offset = 0, 0
        limit_per_page = min(100, max_results)
        while emitted < max_results:
            api_url = (
                f"{self.API_BASE}/application/listings/active?"
                f"keywords={quote_plus(query)}"
                f"&taxonomy_id=1244"  # greeting cards taxonomy
                f"&sort_on=score"
                f"&limit={limit_per_page}&offset={offset}"
            )
            await self.rate_limiter.acquire()
            try:
                resp = await self._client.get(
                    api_url,
                    headers={"x-api-key": self._api_key},
                )
                resp.raise_for_status()
            except Exception as e:
                log.warning(f"Etsy API search failed: {e}")
                return
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return
            for item in results:
                if emitted >= max_results:
                    return
                listing_id = item.get("listing_id")
                if listing_id:
                    # Enrich: images + reviews (search endpoint lacks these)
                    item = await self._enrich_with_images(listing_id, item)
                    item = await self._enrich_with_reviews(listing_id, item)
                    parsed = self._parse_api(item)
                    if parsed:
                        self._api_cache[parsed.url] = parsed
                        emitted += 1
                        yield parsed.url
            offset += limit_per_page
            if offset >= data.get("count", 0):
                return

    async def _enrich_with_images(self, listing_id: int, item: dict) -> dict:
        """Fetch listing images via API (search results don't include them)."""
        assert self._client is not None
        try:
            await self.rate_limiter.acquire()
            resp = await self._client.get(
                f"{self.API_BASE}/application/listings/{listing_id}/images",
                headers={"x-api-key": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            item["images"] = data.get("results", [])
        except Exception as e:
            log.debug(f"Image fetch failed for listing {listing_id}: {e}")
        return item

    async def _enrich_with_reviews(self, listing_id: int, item: dict) -> dict:
        """Fetch review stats for a listing via API."""
        assert self._client is not None
        try:
            await self.rate_limiter.acquire()
            resp = await self._client.get(
                f"{self.API_BASE}/application/listings/{listing_id}/reviews",
                headers={"x-api-key": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            item["_review_count"] = data.get("count", 0)
            # Compute average from results
            reviews = data.get("results", [])
            if reviews:
                ratings = [r.get("rating", 0) for r in reviews if r.get("rating")]
                item["_review_avg"] = sum(ratings) / len(ratings) if ratings else None
        except Exception as e:
            log.debug(f"Review fetch failed for listing {listing_id}: {e}")
        return item

    def _parse_api(self, item: dict) -> ParsedListing | None:
        """Parse an Etsy API listing response into ParsedListing."""
        listing_id = item.get("listing_id")
        if not listing_id:
            return None

        # Price: Etsy API returns amount as dict {amount, divisor, currency_code}
        price = item.get("price", {})
        price_minor = None
        currency = None
        if isinstance(price, dict):
            amount = price.get("amount")
            divisor = price.get("divisor", 100)
            currency = price.get("currency_code")
            if amount is not None:
                # Convert to minor units (pence/cents)
                price_minor = round(amount * 100 / divisor)

        # Images from API
        images = item.get("images", [])
        image_urls = [
            img.get("url_fullxfull") or img.get("url_570xN", "")
            for img in images
            if isinstance(img, dict)
        ]
        image_urls = [u for u in image_urls if u]

        # Seller
        shop = item.get("shop", {}) or {}
        seller_id = shop.get("shop_name") or str(item.get("shop_id", ""))

        # Reviews from enrichment, or fallback to num_favorers
        review_count = item.get("_review_count") or None
        review_avg = item.get("_review_avg")
        favourite_count = item.get("num_favorers")

        return ParsedListing(
            source_listing_id=str(listing_id),
            url=f"{self.BASE}/listing/{listing_id}",
            title=item.get("title"),
            description=item.get("description"),
            seller_id=seller_id or None,
            price_minor_units=price_minor,
            currency=currency,
            review_count=review_count,
            review_avg=review_avg,
            favourite_count=favourite_count,
            is_bestseller=False,  # Not in search API
            image_urls=image_urls,
            raw_metadata={
                "parse_strategy": "api_v3",
                "views": item.get("views"),
                "tags": item.get("tags", []),
                "taxonomy_id": item.get("taxonomy_id"),
                "creation_tsz": item.get("creation_tsz"),
                "original_creation_tsz": item.get("original_creation_tsz"),
            },
        )

    # -- Playwright fallback ---------------------------------------------------

    async def _discover_playwright(self, *, query: str, max_results: int) -> AsyncIterator[str]:
        """Fallback: Playwright headless browser. Low success rate with DataDome."""
        page_num, emitted = 1, 0
        while emitted < max_results:
            search_url = (
                f"{self.BASE}/search?q={quote_plus(query)}"
                f"&category=greeting-cards&page={page_num}"
            )
            try:
                html = await self._pw_fetch(
                    search_url,
                    wait_selector="a[href*='/listing/']",
                    wait_ms=5000,
                )
            except Exception as e:
                log.warning(f"Etsy PW discover page {page_num} failed: {e}")
                return
            tree = HTMLParser(html)
            links = {
                urljoin(self.BASE, a.attributes.get("href", ""))
                for a in tree.css("a[href*='/listing/']")
                if a.attributes.get("href")
            }
            if not links:
                log.debug(f"Etsy PW: no listing links on page {page_num} (likely bot-blocked)")
                return
            for link in links:
                if emitted >= max_results:
                    return
                clean = re.sub(r"\?.*", "", link)
                emitted += 1
                yield clean
            page_num += 1
            if page_num > 20:
                return

    def parse(self, html: str, url: str) -> ParsedListing:
        """Parse listing page HTML. Used by fetch_and_store for PW-fetched pages."""
        tree = HTMLParser(html)
        listing_id_match = re.search(r"/listing/(\d+)", url)
        source_listing_id = listing_id_match.group(1) if listing_id_match else url

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
            brand = ld.get("brand") or ld.get("seller") or {}
            if isinstance(brand, dict):
                seller_id = brand.get("name") or brand.get("url")
                if seller_id and "/" in seller_id:
                    m = re.search(r"/shop/([^/?#]+)", seller_id)
                    seller_id = m.group(1) if m else seller_id
            offers = ld.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if isinstance(offers, dict):
                price_str = str(offers.get("price", ""))
                cur_code = offers.get("priceCurrency", "")
                if price_str:
                    try:
                        price_minor = round(float(price_str) * 100)
                    except ValueError:
                        pass
                if cur_code:
                    currency = cur_code.upper()
            agg = ld.get("aggregateRating") or {}
            if isinstance(agg, dict):
                review_count = _to_int(agg.get("reviewCount") or agg.get("ratingCount"))
                review_avg = _to_float(agg.get("ratingValue"))
            imgs = ld.get("image") or []
            if isinstance(imgs, str):
                imgs = [imgs]
            image_urls = [i for i in imgs if isinstance(i, str) and i.startswith("http")]

        if not title:
            title = _text(tree.css_first("h1[data-buy-box-listing-title], h1"))
        if not seller_id:
            node = tree.css_first("a[href*='/shop/']")
            href = _attr(node, "href") or ""
            m = re.search(r"/shop/([^/?#]+)", href)
            seller_id = m.group(1) if m else None
        if price_minor is None:
            price_text = _text(tree.css_first(
                "[data-buy-box-region='price'], p.wt-text-heading-01"
            ))
            price_minor, currency = _parse_price(price_text)

        fav_text = _text(tree.css_first("[data-favorites-count]"))
        favourite_count = _to_int(fav_text)
        is_bestseller = bool(tree.css_first("[data-bestseller-badge]"))

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
            raw_metadata={"parse_strategy": "jsonld+css", "selectors_version": "v3_pw"},
        )


# -- Shared helpers (imported by other scrapers) ------------------------------

def _extract_jsonld(tree: HTMLParser, typ: str) -> dict | None:
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
        return round(float(amount) * 100), _CURRENCY_MAP.get(symbol)
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
