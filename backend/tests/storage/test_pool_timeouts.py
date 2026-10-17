"""Pool timeouts: the wait cap and the statement timeout.

An earlier change adds three constants to `app.storage.db`: a pool wait timeout, a
connect timeout and a statement timeout so a slow or stuck database gives up in
seconds rather than holding a request forever.

Two cases need a real PostgreSQL (they check what the server actually enforces);
those carry `requires_postgres` and are skipped without
`OPSCONSOLE_TEST_DATABASE_URL`. They read the session `pool`, which
`build_pool` builds, so they check the same settings a deploy runs on.
"""

from __future__ import annotations

import time

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from app.storage.db import POOL_WAIT_TIMEOUT, pool_conninfo

from .conftest import Pool, requires_postgres

# ---------------------------------------------------------------------------
# 1. The constant value
# ---------------------------------------------------------------------------


def test_the_pool_wait_timeout_is_exactly_2_seconds() -> None:
    """Catches an implementation that chose a different default — 30s or
    None — which would let a saturated pool hold a request for far longer
    than a caller's own timeout, masking the saturation as a slow response."""
    assert POOL_WAIT_TIMEOUT == 2.0


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://opsconsole:opsconsole@localhost:15432/opsconsole",
        "host=localhost port=15432 dbname=opsconsole user=opsconsole",
    ],
)
def test_the_pool_connection_string_parses_for_a_url_and_for_key_value_form(
    database_url: str,
) -> None:
    """Catches the timeouts glued onto the URL as plain text. libpq refuses
    `postgresql://...db connect_timeout=5`, so every connection failed and every
    caller saw only a pool timeout. The string handed to the pool must parse and
    still carry both timeouts and the database it was given."""
    parsed = conninfo_to_dict(pool_conninfo(database_url))
    assert parsed["dbname"] == "opsconsole"
    assert parsed["connect_timeout"] == "5"
    assert parsed["options"] == "-c statement_timeout=2000ms"


# ---------------------------------------------------------------------------
# 2. statement_timeout is live on pooled connections (real Postgres)
# ---------------------------------------------------------------------------


@requires_postgres
def test_the_statement_timeout_is_two_seconds(pool: Pool) -> None:
    """Catches an implementation that sets the constant but forgets to push it
    into the connection options, leaving the server default (0 = no limit) in
    place so a slow query still holds a connection for minutes."""
    with pool.connection() as conn:
        row = conn.execute("SHOW statement_timeout").fetchone()
    assert row is not None
    assert str(row[0]) == "2s"


# ---------------------------------------------------------------------------
# 3. A statement over the timeout is cancelled (real Postgres)
# ---------------------------------------------------------------------------


@requires_postgres
def test_a_statement_past_the_timeout_is_cancelled(pool: Pool) -> None:
    """Catches an implementation that sets `statement_timeout` but uses the
    wrong unit (e.g. seconds instead of milliseconds), leaving the server
    treating `2000` as 2000 seconds rather than 2 seconds."""
    start = time.monotonic()
    with pytest.raises(psycopg.OperationalError), pool.connection() as conn:
        conn.execute("SELECT pg_sleep(5)")
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"query ran for {elapsed:.1f}s — statement_timeout is not taking effect"
