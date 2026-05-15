"""Ingest manually-curated JSON files for Moonpig, Thortful, Papier, Scribbler.

These sources restrict or disallow automated scraping, so we collect a small
curated set (~2k total) by hand and store as JSON files matching this schema:

[
  {
    "source": "moonpig",
    "source_listing_id": "abc123",
    "url": "https://www.moonpig.com/uk/personalised-cards/...",
    "title": "...",
    "description": "...",
    "seller_id": null,
    "price_minor_units": 499,
    "currency": "GBP",
    "review_count": 1204,
    "review_avg": 4.8,
    "favourite_count": null,
    "is_bestseller": true,
    "image_urls": ["https://..."]
  }
]

Usage:
    python -m data.scrapers.manual_curation.ingest moonpig.json thortful.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from common.db import connection
from common.logging import get_logger
from data.scrapers.base import ParsedListing, upsert_listing

log = get_logger(__name__)


def ingest_file(path: Path) -> int:
    records = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for r in records:
        source = r.pop("source", path.stem)
        parsed = ParsedListing(
            source_listing_id=str(r.get("source_listing_id", "")),
            url=r.get("url", ""),
            title=r.get("title"),
            description=r.get("description"),
            seller_id=r.get("seller_id"),
            price_minor_units=r.get("price_minor_units"),
            currency=r.get("currency"),
            review_count=r.get("review_count"),
            review_avg=r.get("review_avg"),
            favourite_count=r.get("favourite_count"),
            is_bestseller=bool(r.get("is_bestseller", False)),
            image_urls=list(r.get("image_urls", [])),
            raw_metadata=r.get("raw_metadata", {}),
        )
        upsert_listing(source, parsed)
        count += 1
    return count


def run(files: list[Path]) -> None:
    total = 0
    for f in files:
        n = ingest_file(f)
        log.info(f"{f.name}: {n} records ingested")
        total += n
    log.info(f"Total: {total}")


if __name__ == "__main__":
    typer.run(run)
