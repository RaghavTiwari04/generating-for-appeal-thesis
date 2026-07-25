"""Download listing cover images into object storage.

For each listing that has `image_urls` in `raw_metadata` but no stored image,
fetch the cover, compute SHA-256 + pHash, store the blob, and upsert a
`listing_images` row.

Only the cover (first URL) is fetched. Every consumer in the pipeline joins
`listing_images` with `AND li.is_primary` — feature extraction, LoRA training,
condition D, the gallery and the eval — so the remaining images per listing
were downloaded and stored but never read.

Work is committed in batches so an interrupted run keeps what it finished; the
next run resumes because the query skips listings that already have an image.

    python -m data.scrapers.image_downloader --limit 20000
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any
from urllib.parse import urlparse

import httpx
import imagehash
import typer
from PIL import Image

from common.config import settings
from common.db import connection
from common.logging import get_logger
from common.storage import put_image

log = get_logger(__name__)

# Concurrent downloads per host. Kept low deliberately — these are the same
# hosts the scraper is politely rate-limiting.
CONCURRENCY_PER_HOST = 4
# Rows persisted per transaction, so an interrupted run does not lose the lot.
COMMIT_EVERY = 200

_SELECT_PENDING = """
SELECT l.listing_id, l.raw_metadata
FROM listings l
WHERE l.raw_metadata ? 'image_urls'
  AND NOT EXISTS (
      SELECT 1 FROM listing_images li
      WHERE li.listing_id = l.listing_id AND li.storage_path IS NOT NULL
  )
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_UPSERT_IMAGE = """
INSERT INTO listing_images (listing_id, storage_path, is_primary, width, height, phash, sha256_hex)
VALUES (%(listing_id)s, %(storage_path)s, TRUE, %(width)s, %(height)s, %(phash)s, %(sha256_hex)s)
ON CONFLICT (listing_id, storage_path) DO UPDATE
SET width = EXCLUDED.width,
    height = EXCLUDED.height,
    phash  = EXCLUDED.phash,
    sha256_hex = EXCLUDED.sha256_hex;
"""


def _phash_bits(img: Image.Image) -> str:
    return bin(int(str(imagehash.phash(img)), 16))[2:].zfill(64)


def _cover_url(raw_metadata: Any) -> str | None:
    meta = raw_metadata if isinstance(raw_metadata, dict) else json.loads(raw_metadata or "{}")
    urls = meta.get("image_urls") or []
    return urls[0] if urls else None


async def _download_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    listing_id: str,
    url: str,
) -> dict[str, Any] | None:
    async with sem:
        try:
            resp = await client.get(url, timeout=20.0)
            resp.raise_for_status()
            raw = resp.content
        except Exception as e:
            log.debug(f"Image fetch failed {url}: {e}")
            return None

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        log.debug(f"Image open failed {url}: {e}")
        return None

    sha256, storage_path = put_image(
        raw, content_type=resp.headers.get("content-type", "image/jpeg")
    )
    return {
        "listing_id": listing_id,
        "storage_path": storage_path,
        "width": img.width,
        "height": img.height,
        "phash": _phash_bits(img),
        "sha256_hex": sha256,
    }


def _persist(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    with connection() as conn, conn.cursor() as cur:
        for row in rows:
            cur.execute(_UPSERT_IMAGE, row)
    return len(rows)


async def download_batch(limit: int = 5000) -> int:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_PENDING, {"limit": limit})
        pending = cur.fetchall()

    jobs: list[tuple[str, str]] = []
    for row in pending:
        url = _cover_url(row["raw_metadata"])
        if url:
            jobs.append((str(row["listing_id"]), url))

    log.info(f"Image download: {len(jobs)} covers pending ({len(pending)} listings)")
    if not jobs:
        return 0

    # One semaphore per host rather than a global cap, so a slow host cannot
    # starve the other.
    sems: dict[str, asyncio.Semaphore] = {}

    def sem_for(url: str) -> asyncio.Semaphore:
        host = urlparse(url).netloc
        if host not in sems:
            sems[host] = asyncio.Semaphore(CONCURRENCY_PER_HOST)
        return sems[host]

    stored = failed = 0
    async with httpx.AsyncClient(
        headers={"User-Agent": settings.scraper_user_agent}, follow_redirects=True
    ) as client:
        for start in range(0, len(jobs), COMMIT_EVERY):
            chunk = jobs[start : start + COMMIT_EVERY]
            results = await asyncio.gather(
                *(_download_one(client, sem_for(url), lid, url) for lid, url in chunk)
            )
            good = [r for r in results if r is not None]
            failed += len(results) - len(good)
            stored += _persist(good)
            log.info(f"  {min(start + COMMIT_EVERY, len(jobs))}/{len(jobs)} — {stored} stored, {failed} failed")

    log.info(f"Images persisted: {stored} ({failed} failed)")
    return stored


def run(limit: int = 5000) -> None:
    n = asyncio.run(download_batch(limit))
    print(f"Done: {n} images downloaded")


if __name__ == "__main__":
    typer.run(run)
