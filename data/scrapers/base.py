"""Abstract scraper base class + shared rate-limiter, cache, and upsert helpers.

Each concrete scraper subclasses `Scraper` and implements:

- `source: str`               (class attribute, e.g. "redbubble")
- `discover(query, ...)`      -> iterable of listing URLs to fetch
- `parse(html, url)`          -> a `ParsedListing` dataclass

`fetch_and_store(url)` is implemented here and handles the polite-fetch
pipeline (rate limit, cache check, network, store raw HTML, upsert listing).

Both live scrapers (Redbubble, Greetings Island) serve static HTML, so this
uses plain httpx. The Playwright path was only ever needed by the Zazzle
scraper and went with it.
"""

from __future__ import annotations

import asyncio
import ssl
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from common.config import settings
from common.db import connection
from common.logging import get_logger
from common.storage import put_raw_html

log = get_logger(__name__)

def _is_birthday_title(title: str) -> bool:
    """True if the occasion classifier would file this title under birthday/*.

    Reuses the classifier's own rules rather than a second keyword list, so the
    scrape gate cannot drift from what the classifier will accept. A hand-rolled
    regex was both too strict ("Fabulous at 40th" matches the milestone rules
    but contains no "birthday") and too loose in the other direction.
    """
    from data.features.occasion_classifier import weak_label

    return any(lbl.startswith("birthday/") for lbl in weak_label(title))


@dataclass
class ParsedListing:
    """Normalised listing fields produced by `Scraper.parse`."""

    source_listing_id: str
    url: str
    title: str | None = None
    description: str | None = None
    seller_id: str | None = None
    price_minor_units: int | None = None
    currency: str | None = None
    review_count: int | None = None
    review_avg: float | None = None
    favourite_count: int | None = None
    is_bestseller: bool = False
    listing_created_at: datetime | None = None
    image_urls: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class AsyncRateLimiter:
    """Token-bucket-style limiter: at most `rate` calls per second, globally."""

    def __init__(self, rate_per_sec: float):
        self.rate = rate_per_sec
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            if wait:
                await asyncio.sleep(wait)
            self._next_at = max(now, self._next_at) + (1.0 / self.rate)


class Scraper(ABC):
    """Base class for marketplace scrapers.

    Subclasses must set the `source` class attribute and implement
    `discover` / `parse`. `fetch_and_store` is provided.
    """

    source: str = ""
    # True when discover() enumerates fixed category pages and ignores the
    # query, so the driver runs it once instead of once per query.
    ignores_query: bool = False
    # Drop listings whose title has no birthday marker. Marketplace search is
    # relevance-ranked, so deep pages drift off-topic ("Surgeon Greeting
    # Card"); without this they reach the DB and have to be cleaned up later.
    require_birthday: bool = False

    def __init__(
        self,
        *,
        rate_per_sec: float | None = None,
        user_agent: str | None = None,
        cache_dir: Path | None = None,
        cache_ttl_days: int | None = None,
    ) -> None:
        if not self.source:
            raise ValueError(f"{type(self).__name__} must set `source` class attribute")
        self.rate_limiter = AsyncRateLimiter(rate_per_sec or settings.scraper_rate_limit_per_sec)
        self.user_agent = user_agent or settings.scraper_user_agent
        self.cache_dir = (cache_dir or settings.scraper_cache_dir) / self.source
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(days=cache_ttl_days or settings.scraper_raw_html_ttl_days)
        self._client: httpx.AsyncClient | None = None
        self.skipped_off_topic = 0

    # -- to be implemented by subclasses --------------------------------------
    @abstractmethod
    async def discover(self, *, query: str, max_results: int = 100) -> Iterable[str]:
        """Yield listing URLs for a given query/occasion seed."""

    @abstractmethod
    def parse(self, html: str, url: str) -> ParsedListing:
        """Parse a single listing HTML page into a ParsedListing."""

    # -- shared machinery -----------------------------------------------------
    async def __aenter__(self) -> Scraper:
        ssl_ctx = ssl.create_default_context()
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            verify=ssl_ctx,
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _cache_path(self, url: str) -> Path:
        import hashlib

        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.html"

    def _cache_get(self, url: str) -> str | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if datetime.now(tz=UTC) - mtime > self.cache_ttl:
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    def _cache_put(self, url: str, html: str) -> None:
        self._cache_path(url).write_text(html, encoding="utf-8")

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError,)),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1.0, max=30.0),
        reraise=True,
    )
    async def _fetch(self, url: str) -> str:
        assert self._client is not None, "Use `async with scraper:` to manage the client"
        await self.rate_limiter.acquire()
        log.debug(f"GET {url}")
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp.text

    def should_store(self, parsed: ParsedListing) -> bool:
        """Gate a parsed listing before it reaches the database."""
        if self.require_birthday and not _is_birthday_title(parsed.title or ""):
            return False
        return True

    async def fetch_and_store(self, url: str, *, use_cache: bool = True) -> ParsedListing | None:
        """Fetch (cached or network), parse, persist raw HTML + upsert listing."""
        html: str | None = self._cache_get(url) if use_cache else None
        if html is None:
            try:
                html = await self._fetch(url)
            except Exception as e:
                log.warning(f"Fetch failed for {url}: {e}")
                return None
            self._cache_put(url, html)

        try:
            parsed = self.parse(html, url)
        except Exception as e:
            log.exception(f"Parse failed for {url}: {e}")
            return None

        if not self.should_store(parsed):
            self.skipped_off_topic += 1
            log.debug(f"Skipping non-birthday listing: {(parsed.title or url)[:80]}")
            return None

        # Persist raw HTML to MinIO for re-parsing later.
        try:
            put_raw_html(self.source, parsed.source_listing_id, html.encode("utf-8"))
        except Exception as e:
            log.warning(f"Raw HTML upload failed for {url}: {e}")

        upsert_listing(self.source, parsed)
        return parsed


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------
_UPSERT_SQL = """
INSERT INTO listings (
    source, source_listing_id, url, title, description, seller_id,
    price_minor_units, currency, review_count, review_avg,
    favourite_count, is_bestseller, listing_created_at, raw_metadata,
    first_seen_at, last_seen_at
) VALUES (
    %(source)s, %(source_listing_id)s, %(url)s, %(title)s, %(description)s, %(seller_id)s,
    %(price_minor_units)s, %(currency)s, %(review_count)s, %(review_avg)s,
    %(favourite_count)s, %(is_bestseller)s, %(listing_created_at)s, %(raw_metadata)s,
    NOW(), NOW()
)
ON CONFLICT (source, source_listing_id) DO UPDATE SET
    url               = EXCLUDED.url,
    title             = COALESCE(EXCLUDED.title, listings.title),
    description       = COALESCE(EXCLUDED.description, listings.description),
    seller_id         = COALESCE(EXCLUDED.seller_id, listings.seller_id),
    price_minor_units = EXCLUDED.price_minor_units,
    currency          = COALESCE(EXCLUDED.currency, listings.currency),
    review_count      = EXCLUDED.review_count,
    review_avg        = EXCLUDED.review_avg,
    favourite_count   = EXCLUDED.favourite_count,
    is_bestseller     = EXCLUDED.is_bestseller,
    raw_metadata      = COALESCE(EXCLUDED.raw_metadata, listings.raw_metadata),
    last_seen_at      = NOW()
RETURNING listing_id;
"""

_SNAPSHOT_SQL = """
INSERT INTO listing_snapshots (listing_id, snapshot_at, review_count, favourite_count, price_minor_units)
VALUES (%(listing_id)s, NOW(), %(review_count)s, %(favourite_count)s, %(price_minor_units)s)
ON CONFLICT DO NOTHING;
"""


def upsert_listing(source: str, parsed: ParsedListing) -> str:
    """Upsert into `listings`, append a snapshot, return listing_id."""
    from psycopg.types.json import Jsonb

    payload = {
        "source": source,
        "source_listing_id": parsed.source_listing_id,
        "url": parsed.url,
        "title": parsed.title,
        "description": parsed.description,
        "seller_id": parsed.seller_id,
        "price_minor_units": parsed.price_minor_units,
        "currency": parsed.currency,
        "review_count": parsed.review_count,
        "review_avg": parsed.review_avg,
        "favourite_count": parsed.favourite_count,
        "is_bestseller": parsed.is_bestseller,
        "listing_created_at": parsed.listing_created_at,
        "raw_metadata": Jsonb(
            {**(parsed.raw_metadata or {}), "image_urls": parsed.image_urls}
        ),
    }
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_UPSERT_SQL, payload)
        row = cur.fetchone()
        listing_id = row["listing_id"]
        cur.execute(
            _SNAPSHOT_SQL,
            {
                "listing_id": listing_id,
                "review_count": parsed.review_count,
                "favourite_count": parsed.favourite_count,
                "price_minor_units": parsed.price_minor_units,
            },
        )
    return str(listing_id)
