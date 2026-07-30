"""What the embed job selects, and what it is allowed to overwrite.

Both rules exist to protect the labelled pool. Selection has to notice a
changed encoder stack, or a catalogue embedded once can never be re-embedded
with different settings. Persistence has to leave `clip_embedding` alone once
set, because dedup's clusters — and therefore which cards got labelled — came
out of those exact vectors.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from data.features import clip_embed


class _Cursor:
    """Records every statement and its parameters."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.calls: list[tuple[str, dict]] = []

    def execute(self, sql: str, params: dict) -> None:
        self.calls.append((sql, params))

    def fetchall(self) -> list[dict]:
        return self._rows

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Conn:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _Cursor:
        return self._cursor

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 48), (200, 120, 90)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def run_job():
    """Run the embed job against a fake DB, returning the recorded statements."""

    def _run(width: int, n_rows: int = 2):
        rows = [
            {"listing_id": f"l{i}", "storage_path": f"covers/{i}.png"}
            for i in range(n_rows)
        ]
        cursor = _Cursor(rows)

        embedder = type("E", (), {})()
        embedder.cfg = type("C", (), {"batch_size": 1})()
        embedder.source = "google/siglip-base-patch16-384|crops=5|pad"
        embedder.embed_images = lambda imgs: np.ones((len(imgs), width), dtype=np.float32)

        with (
            patch("common.db.connection", return_value=_Conn(cursor)),
            patch("common.storage.get_object", return_value=_png()),
            patch.object(clip_embed, "CLIPEmbedder", return_value=embedder),
        ):
            written = clip_embed.run_embed_missing()
        return written, cursor.calls, embedder.source

    return _run


def test_selection_is_by_encoder_stack_not_null(run_job) -> None:
    """A row embedded by another stack is stale, even with features present."""
    _, calls, source = run_job(width=768)
    sql, params = calls[0]

    assert "image_feature_source IS DISTINCT FROM" in sql
    assert params["source"] == source
    # The old gate. Selecting on it meant an already-embedded catalogue always
    # returned zero rows, so changing the crop settings changed nothing.
    assert "clip_embedding IS NULL" not in sql


def test_existing_clip_embedding_is_never_overwritten(run_job) -> None:
    """Dedup's vectors survive a re-embed; only image_features is replaced."""
    written, calls, _ = run_job(width=768)
    upserts = [(sql, p) for sql, p in calls if "INSERT INTO listing_features" in sql]

    assert written == 2
    assert len(upserts) == 2
    sql, params = upserts[0]
    assert "clip_embedding       = COALESCE(listing_features.clip_embedding" in sql
    assert "image_features       = EXCLUDED.image_features" in sql
    # 768-d still seeds the column on a fresh row, so a first run leaves dedup
    # with something to cluster.
    assert params["vector"] == params["features"]


def test_wide_features_leave_the_vector_column_null(run_job) -> None:
    """1536-d cannot enter VECTOR(768); it goes to image_features alone."""
    written, calls, _ = run_job(width=1536)
    _, params = next((s, p) for s, p in calls if "INSERT INTO listing_features" in s)

    assert written == 2
    assert params["vector"] is None
    assert len(params["features"]) == 1536
