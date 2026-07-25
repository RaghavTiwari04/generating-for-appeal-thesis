"""S3/MinIO object-store helpers.

Stores blobs keyed by SHA-256 content hash, with a separate raw-HTML bucket
for scraper response caching.

Falls back to the local filesystem when MinIO rejects a write. /vol/bitbucket
is a shared volume that sits at ~100% use, and MinIO refuses writes once free
space drops below a safety margin computed from total capacity — on a 174T
volume that margin is terabytes, so no amount of local cleanup clears it. The
filesystem itself still accepts writes, and the payloads here are small (the
whole image bucket is well under 1GB), so writing directly is a workable
fallback rather than a silent data-loss path.

Set GC_BLOB_ROOT to override where fallback blobs are written.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)

LOCAL_SCHEME = "file://"

# Once MinIO has refused a write, stop retrying it for the rest of the process.
# minio-py retries several times per call, so without this a full backend costs
# tens of seconds per object across thousands of images. A fresh process
# re-probes, so recovery needs no code change.
_minio_unavailable = False


def _blob_root() -> Path:
    root = os.environ.get("GC_BLOB_ROOT")
    if root:
        return Path(root)
    work = Path(f"/vol/bitbucket/{os.environ.get('USER', 'unknown')}")
    if work.is_dir():
        return work / "blobstore"
    return Path("./artifacts/blobstore")


def _put_local(bucket: str, key: str, data: bytes) -> str:
    dest = _blob_root() / bucket / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return f"{LOCAL_SCHEME}{dest}"


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
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_image(data: bytes, *, content_type: str = "image/jpeg") -> tuple[str, str]:
    """Upload an image keyed by content hash. Returns (sha256_hex, storage_path)."""
    global _minio_unavailable
    digest = sha256_hex(data)
    key = f"images/{digest[:2]}/{digest[2:4]}/{digest}"

    if not _minio_unavailable:
        try:
            client = _client()
            client.put_object(
                bucket_name=settings.minio_bucket,
                object_name=key,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
            return digest, f"s3://{settings.minio_bucket}/{key}"
        except Exception as e:
            _minio_unavailable = True
            log.warning(f"MinIO put failed ({e}); writing blobs to {_blob_root()} for this run")

    return digest, _put_local(settings.minio_bucket, key, data)


def put_raw_html(source: str, source_listing_id: str, html: bytes) -> str:
    """Cache a scraped HTML response. Returns storage path.

    Raw HTML is a convenience for re-parsing without refetching; nothing in
    the pipeline reads it, and it is bulky. So this deliberately has no local
    fallback — it raises, and `Scraper.fetch_and_store` logs and continues.
    """
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
    """Fetch an object by s3:// path, file:// path, or plain filesystem path."""
    if storage_path.startswith(LOCAL_SCHEME):
        return Path(storage_path[len(LOCAL_SCHEME):]).read_bytes()
    if not storage_path.startswith("s3://"):
        # Plain path written by an earlier local-fallback path.
        p = Path(storage_path)
        if p.exists():
            return p.read_bytes()
        raise ValueError(f"Not an s3 path and not on disk: {storage_path}")
    rest = storage_path[5:]          # strip "s3://"
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Cannot parse bucket/key from: {storage_path}")
    try:
        client = _client()
        resp = client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()
    except S3Error:
        # May have been written by the local fallback before/after an outage.
        local = _blob_root() / bucket / key
        if local.exists():
            return local.read_bytes()
        raise


def get_image_to_path(storage_path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = get_object(storage_path)
    dest.write_bytes(data)
    return dest
