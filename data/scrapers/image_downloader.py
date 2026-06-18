"""Download listing images from URLs into MinIO.

After a scraper run, `listing_images` rows exist with `storage_path = NULL`
(or not yet inserted). This module:

1. Reads all listings that have image_urls in their `raw_metadata` but no
   corresponding `listing_images` row with a storage_path.
2. Downloads each image (rate-limited per domain).
3. Computes SHA-256 + pHash.
4. Uploads to MinIO under `s3://greeting-cards/images/<sha256[:2]>/...`.
5. Inserts/upserts `listing_images` row.

Run after every scrape batch:
    python -m data.scrapers.image_downloader --limit 5000
"""

from __future__ import annotations

import asyncio
import io
from collections import defaultdict
from typing import Any

import httpx
import imagehash
import typer
from PIL import Image, UnidentifiedImageError

from common.config import settings
from common.db import connection
from common.logging import get_logger
from common.storage import put_image

log = get_logger(__name__)


_SELECT_PENDING = """
SELECT l.listing_id, l.source, l.raw_metadata
FROM listings l
WHERE l.raw_metadata IS NOT NULL
  AND l.raw_metadata ? 'image_urls'
  AND NOT EXISTS (
      SELECT 1 FROM listing_images li
      WHERE li.listing_id = l.listing_id AND li.storage_path IS NOT NULL
  )
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_UPSERT_IMAGE = """
INSERT INTO listing_images (listing_id, storage_path, is_primary, width, height, phash, sha256_hex)
VALUES (%(listing_id)s, %(storage_path)s, %(is_primary)s, %(width)s, %(height)s, %(phash)s, %(sha256_hex)s)
ON CONFLICT (listing_id, storage_path) DO UPDATE
SET width = EXCLUDED.width,
    height = EXCLUDED.height,
    phash  = EXCLUDED.phash,
    sha256_hex = EXCLUDED.sha256_hex;
"""


def _phash_bits(img: Image.Image) -> str:
    h = imagehash.phash(img)
    bits = bin(int(str(h), 16))[2:].zfill(64)
    return bits


async def _download_one(
    client: httpx.AsyncClient,
    limiters: dict[str, asyncio.Semaphore],
    listing_id: str,
    url: str,
    is_primary: bool,
) -> dict[str, Any] | None:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    limiter = limiters[domain]
    async with limiter:
        try:
            resp = await client.get(url, timeout=20.0)
            resp.raise_for_status()
            raw = resp.content
        except Exception as e:
            log.debug(f"Image fetch failed {url}: {e}")
            return None

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        log.debug(f"Image open failed {url}: {e}")
        return None

    sha256, storage_path = put_image(raw, content_type=resp.headers.get("content-type", "image/jpeg"))
    phash = _phash_bits(img)

    return {
        "listing_id": listing_id,
        "storage_path": storage_path,
        "is_primary": is_primary,
        "width": img.width,
        "height": img.height,
        "phash": phash,
        "sha256_hex": sha256,
    }


async def download_batch(limit: int = 5000, max_per_domain: int = 4) -> int:
    import json

    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_PENDING, {"limit": limit})
        rows = cur.fetchall()

    log.info(f"Image download: {len(rows)} listings pending")
    if not rows:
        return 0

    limiters: dict[str, asyncio.Semaphore] = defaultdict(
        lambda: asyncio.Semaphore(max_per_domain)
    )
    ua = settings.scraper_user_agent

    async with httpx.AsyncClient(
        headers={"User-Agent": ua}, follow_redirects=True
    ) as client:
        tasks = []
        for row in rows:
            meta = row["raw_metadata"] if isinstance(row["raw_metadata"], dict) else json.loads(row["raw_metadata"] or "{}")
            image_urls: list[str] = meta.get("image_urls", [])
            for i, url in enumerate(image_urls[:5]):  # max 5 images per listing
                tasks.append(
                    _download_one(client, limiters, str(row["listing_id"]), url, is_primary=(i == 0))
                )
        results = await asyncio.gather(*tasks)

    downloaded = 0
    with connection() as conn, conn.cursor() as cur:
        for r in results:
            if r is None:
                continue
            cur.execute(_UPSERT_IMAGE, r)
            downloaded += 1

    log.info(f"Images persisted: {downloaded}")
    return downloaded


def run(limit: int = 5000) -> None:
    n = asyncio.run(download_batch(limit))
    print(f"Done: {n} images downloaded")


if __name__ == "__main__":
    typer.run(run)
