"""Postgres connection helpers.

Thin wrappers around psycopg3 for synchronous use; sqlalchemy engine factory
for ORM use where helpful. Most code should prefer raw SQL via `connection()`
to keep dependencies minimal.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from common.config import settings


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Yield a psycopg3 connection with pgvector adapters registered.

    Use as: `with connection() as conn: ...`. Auto-commits on success,
    rolls back on exception.
    """
    conn = psycopg.connect(settings.postgres_dsn, row_factory=dict_row)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_engine: Engine | None = None


def engine() -> Engine:
    """Lazily-built SQLAlchemy engine. Use for ORM / pandas IO."""
    global _engine
    if _engine is None:
        _engine = create_engine(settings.postgres_dsn, pool_pre_ping=True)
    return _engine
