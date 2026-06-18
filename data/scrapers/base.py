"""Abstract scraper base class + shared rate-limiter, cache, and upsert helpers.

Each concrete scraper subclasses `Scraper` and implements:

- `source: str`               (class attribute, e.g. "etsy")
- `discover(query, ...)`      -> iterable of listing URLs to fetch
- `parse(html, url)`          -> a `ParsedListing` dataclass

`fetch_and_store(url)` is implemented here and handles the polite-fetch
pipeline (rate limit, cache check, network, store raw HTML, upsert listing).

For sites with bot-detection or JS-rendered content, subclasses can set
`use_playwright = True` to use a headless Chromium browser instead of httpx.
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

    Set `use_playwright = True` for sites that require JS rendering
    or have bot-detection blocking plain HTTP requests.
    """

    source: str = ""
    use_playwright: bool = False

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
        self._browser: Any = None  # playwright Browser
        self._pw: Any = None       # playwright async API

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
        if self.use_playwright:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            log.info(f"[{self.source}] Playwright browser launched")
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    async def _new_page(self):
        """Create a new Playwright page with stealth-ish settings."""
        assert self._browser is not None, "Playwright not initialised"
        ctx = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="en-GB",
        )
        page = await ctx.new_page()
        # Basic stealth: hide webdriver flag
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)
        return page

    async def _pw_fetch(self, url: str, *, wait_selector: str | None = None,
                         wait_ms: int = 3000) -> str:
        """Fetch a URL via Playwright, return rendered HTML.

        Args:
            url: Page to load.
            wait_selector: CSS selector to wait for before extracting HTML.
            wait_ms: Extra time (ms) to let JS finish after navigation.
        """
        await self.rate_limiter.acquire()
        page = await self._new_page()
        try:
            log.debug(f"PW GET {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if wait_selector:
                try:
                    await page.wait_for_selector(wait_selector, timeout=10000)
                except Exception:
                    pass  # best-effort; page may still have content
            # Extra settle time for lazy-loaded content
            await page.wait_for_timeout(wait_ms)
            return await page.content()
        finally:
            await page.context.close()

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

    async def _smart_fetch(self, url: str, *, wait_selector: str | None = None) -> str:
        """Fetch via Playwright if enabled, else httpx."""
        if self.use_playwright:
            return await self._pw_fetch(url, wait_selector=wait_selector)
        return await self._fetch(url)

    async def fetch_and_store(self, url: str, *, use_cache: bool = True,
                               wait_selector: str | None = None) -> ParsedListing | None:
        """Fetch (cached or network), parse, persist raw HTML + upsert listing."""
        html: str | None = self._cache_get(url) if use_cache else None
        if html is None:
            try:
                html = await self._smart_fetch(url, wait_selector=wait_selector)
            except Exception as e:
                log.warning(f"Fetch failed for {url}: {e}")
                return None
            self._cache_put(url, html)

        try:
            parsed = self.parse(html, url)
        except Exception as e:
            log.exception(f"Parse failed for {url}: {e}")
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
