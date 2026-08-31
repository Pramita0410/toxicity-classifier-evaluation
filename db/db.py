"""
db/db.py : SQLAlchemy engine factory.

Why SQLAlchemy?
  It lets us write Python instead of raw SQL for inserts/selects,
  and pandas integrates with it directly via DataFrame.to_sql().
  We never hardcode credentials here — they come from config.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@lru_cache(maxsize=1)
def get_engine(database_url: str) -> Engine:
    """
    Return a SQLAlchemy engine for the given database URL.
    Cached so we reuse the same connection pool across the pipeline.

    Args:
        database_url: Full PostgreSQL connection string from config.
                      e.g. "postgresql://user:pass@localhost:5432/trust_safety"
    """
    engine = create_engine(
        database_url,
        pool_pre_ping=True,   # checks connection health before using from pool
        pool_size=5,
        max_overflow=10,
    )
    return engine


def test_connection(database_url: str) -> bool:
    """
    Quick connectivity check. Returns True if DB is reachable, False otherwise.
    Useful to call at pipeline startup to give a clear error message.
    """
    try:
        engine = get_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[db] Connection failed: {exc}")
        return False
