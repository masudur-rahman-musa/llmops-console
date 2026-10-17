"""Pool timeouts: the wait cap and the statement timeout.

An earlier change adds three constants to `app.storage.db`: a pool wait timeout, a
connect timeout and a statement timeout so a slow or stuck database gives up in
seconds rather than holding a request forever.

Two cases need a real PostgreSQL (they check what the server actually enforces);
those carry `requires_postgres` and are skipped without
`OPSCONSOLE_TEST_DATABASE_URL`.
"""

from __future__ import annotations

import time

import psycopg
import pytest

from app.storage.db import (
    POOL_WAIT_TIMEOUT,
    build_pool,
)

from .conftest import DATABASE_URL, Pool, requires_postgres

# ---------------------------------------------------------------------------
# 1. The constant value
# ---------------------------------------------------------------------------


def test_the_pool_wait_timeout_is_exactly_2_seconds() -> None:
    """Catches an implementation that chose a different default — 30s or
    None — which would let a saturated pool hold a request for far longer
    than a caller's own timeout, masking the saturation as a slow response."""
    assert POOL_WAIT_TIMEOUT == 2.0


# ---------------------------------------------------------------------------
# 2. statement_timeout is live on pooled connections (real Postgres)
# ---------------------------------------------------------------------------


@requires_postgres
def test_the_statement_timeout_is_two_seconds(pool: Pool) -> None:
    """Catches an implementation that sets the constant but forgets to push it
    into the connection options, leaving the server default (0 = no limit) in
    place so a slow query still holds a connection for minutes."""
    assert DATABASE_URL is not None
    fresh = build_pool(DATABASE_URL)
    try:
        with fresh.connection() as conn:
            row = conn.execute("SHOW statement_timeout").fetchone()
        assert row is not None
        assert str(row[0]) == "2s"
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# 3. A statement over the timeout is cancelled (real Postgres)
# ---------------------------------------------------------------------------


@requires_postgres
def test_a_statement_past_the_timeout_is_cancelled() -> None:
    """Catches an implementation that sets `statement_timeout` but uses the
    wrong unit (e.g. seconds instead of milliseconds), leaving the server
    treating `2000` as 2000 seconds rather than 2 seconds."""
    assert DATABASE_URL is not None
    fresh = build_pool(DATABASE_URL)
    start = time.monotonic()
    try:
        with pytest.raises(psycopg.OperationalError), fresh.connection() as conn:
            conn.execute("SELECT pg_sleep(5)")
    finally:
        fresh.close()
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"query ran for {elapsed:.1f}s — statement_timeout is not taking effect"
