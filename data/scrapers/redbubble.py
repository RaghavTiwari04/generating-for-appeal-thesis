"""Redbubble scraper (~10k cards).

Primary: JSON-LD Product schema (present on all Redbubble product pages).
Fallback: CSS selectors.

TOS review required before production run.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from urllib.parse import quote_plus

from selectolax.parser import HTMLParser

from common.logging import get_logger
from data.scrapers.base import ParsedListing, Scraper
from data.scrapers.parsing import (
    _attr,
    _extract_jsonld,
    _parse_price,
    _text,
    _to_float,
    _to_int,
)


log = get_logger(__name__)


# Redbubble serves one artwork under several transforms, e.g.
#   .../image.1072219682.1798/papergc,300x,...,f8f8f8.u1.jpg   <- tilted card
#   .../image.1072219682.1798/flat,750x,075,f-pad,...,f8f8f8.u1.webp  <- artwork
# Training on the tilted mockup teaches paper edges and perspective instead of
# card design, so always prefer the `flat` variant.
#
# Constructing a flat URL is unreliable — the exact transform string varies
# and a guessed one 404s. The page contains every variant, so collect them all
# and pick the best per artwork id instead.
_RB_IMAGE_URL_RE = re.compile(
    r"https://ih\d+\.redbubble\.net/image\.[^\"'\s\\]+?\.(?:jpg|jpeg|png|webp)",
    re.IGNORECASE,
)
_RB_IMAGE_ID_RE = re.compile(r"/image\.([0-9.]+)/")


def _image_id(url: str) -> str | None:
    m = _RB_IMAGE_ID_RE.search(url)
    return m.group(1) if m else None


def _variant_rank(url: str) -> tuple[int, int]:
    """Lower is better: flat artwork first, then jpg over webp."""
    lowered = url.lower()
    if "/flat," in lowered:
        kind = 0
    elif "/papergc," in lowered:
        kind = 2      # tilted card mockup — last resort
    else:
        kind = 1
    ext = 0 if lowered.endswith((".jpg", ".jpeg")) else 1
    return (kind, ext)


def _best_variants(html: str) -> dict[str, str]:
    """Map artwork id -> best available image URL found anywhere in the page."""
    best: dict[str, str] = {}
    for url in _RB_IMAGE_URL_RE.findall(html):
        img_id = _image_id(url)
        if not img_id:
            continue
        current = best.get(img_id)
        if current is None or _variant_rank(url) < _variant_rank(current):
            best[img_id] = url
    return best


def _flat_image_url(url: str, variants: dict[str, str] | None = None) -> str:
    """Swap a mockup URL for the flat variant of the same artwork, if present."""
    if not variants:
        return url
    img_id = _image_id(url)
    if not img_id:
        return url
    return variants.get(img_id, url)


def _cover_first(urls: list[str]) -> list[str]:
    """Order so the cover (index 0) is flat artwork wherever one exists.

    Only the first URL is downloaded, and mapping variants per artwork is not
    enough on its own — if the page lists a different artwork first, the cover
    could still end up a mockup. Sorting is stable, so ordering is otherwise
    preserved.
    """
    return sorted(dict.fromkeys(urls), key=_variant_rank)


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
            # Redbubble product links use /i/ pattern (e.g. /i/greeting-card/...)
            links = {
                a.attributes.get("href", "")
                for a in tree.css("a[href*='/i/greeting-card']")
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
        # Product URLs: /i/greeting-card/Title-by-Artist/12345678/qjsu
        id_match = re.search(r"/(\d{6,})", url)
        source_listing_id = id_match.group(1) if id_match else url

        # Every transform of every artwork on the page, best variant per id.
        variants = _best_variants(html)

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
            image_urls = [
                _flat_image_url(i, variants) for i in imgs
                if isinstance(i, str) and i.startswith("http")
            ]

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
            image_urls = [_flat_image_url(u, variants) for u in image_urls if u.startswith("http")]

        image_urls = _cover_first(image_urls)
        if image_urls and "/flat," not in image_urls[0].lower():
            log.debug(f"No flat artwork variant for {url}; cover is a mockup")

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
