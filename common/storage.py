"""S3/MinIO object-store helpers.

Stores blobs keyed by SHA-256 content hash, with a separate raw-HTML bucket
for scraper response caching.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from common.config import settings


def _client() -> Minio:
    endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")
    secure = settings.minio_endpoint.startswith("https://")
    return Minio(
        endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=secure,
    )


def ensure_buckets() -> None:
    client = _client()
    for bucket in (settings.minio_bucket, settings.minio_bucket_raw):
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
        except S3Error:
            pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_image(data: bytes, *, content_type: str = "image/jpeg") -> tuple[str, str]:
    """Upload an image keyed by content hash. Returns (sha256_hex, storage_path)."""
    digest = sha256_hex(data)
    key = f"images/{digest[:2]}/{digest[2:4]}/{digest}"
    client = _client()
    client.put_object(
        bucket_name=settings.minio_bucket,
        object_name=key,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return digest, f"s3://{settings.minio_bucket}/{key}"


def put_raw_html(source: str, source_listing_id: str, html: bytes) -> str:
    """Cache a scraped HTML response. Returns storage path."""
    digest = sha256_hex(html)[:16]
    key = f"{source}/{source_listing_id}/{digest}.html"
    client = _client()
    client.put_object(
        bucket_name=settings.minio_bucket_raw,
        object_name=key,
        data=io.BytesIO(html),
        length=len(html),
        content_type="text/html",
    )
    return f"s3://{settings.minio_bucket_raw}/{key}"


def get_object(storage_path: str) -> bytes:
    """Fetch an object by s3:// path (format: s3://bucket/key)."""
    if not storage_path.startswith("s3://"):
        raise ValueError(f"Not an s3 path: {storage_path}")
    rest = storage_path[5:]          # strip "s3://"
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Cannot parse bucket/key from: {storage_path}")
    client = _client()
    resp = client.get_object(bucket, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def get_image_to_path(storage_path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = get_object(storage_path)
    dest.write_bytes(data)
    return dest
