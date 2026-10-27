"""`PostgresCostLedgerWriter` against a real server.

Every case here runs the SHIPPED migration into a throwaway schema and then
writes through the real writer, never against a hand-built table. That is the
whole point of the file: `002_cost_ledger.sql` and the writer's INSERT are
written in two different places and nothing type-checks between them, so a column
added to one and not the other, or a positional binding that swaps two `text`
columns, is invisible to PostgreSQL and to mypy both. The `numeric` scale is the
same kind of gap in the other direction - a column narrower than six places
rounds a cost the library computed exactly, and it shows up as a cent per million
requests rather than as a failure.

Skipped without `OPSCONSOLE_TEST_DATABASE_URL`. One
session pool, one throwaway schema per test - the fixtures in
tests/storage/conftest.py, unchanged.

The statement-level view of the same writer - that the pre-checks refuse before a
connection is borrowed at all - is in tests/storage/test_cost_ledger.py, where a
pool double can see it; a real server cannot tell a statement that was never
issued from one that was rejected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from psycopg import sql

from app.storage import CostLedgerRow, PostgresCostLedgerWriter
from app.storage.cost_ledger import COST_LEDGER

from .conftest import Pool, requires_postgres

# 22:45 UTC on purpose: a time of day far from both midnights.
WHEN = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)
NEXT_DAY = WHEN + timedelta(days=1)

# Every value distinct from every other, and none of them a substring of another,
# so a positional binding that swapped two of the six `text` columns - which
# PostgreSQL would accept silently - shows up as a wrong value rather than as a
# type error.
TENANT = "nordlicht"
ALIAS = "agent-default"
PROVIDER = "local"
MODEL = "qwen2.5-7b-instruct"
SESSION = "sess-01j9x8"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PRICE_VERSION = "local/qwen2.5-7b-instruct@2026-01-01"

COST = Decimal("1234.567891")
"""Six decimal places, and digits in every one of them: a column declared
`numeric(18,2)` rounds this to 1234.57 and PostgreSQL says nothing about it."""

LEDGER_COLUMNS = (
    "ts",
    "tenant",
    "alias",
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "cached",
    "cost_usd",
    "price_version",
    "estimated",
    "session",
    "trace_id",
)


def a_row(**overrides: Any) -> CostLedgerRow:  # noqa: ANN401
    """One well-formed row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "ts": WHEN,
        "tenant": TENANT,
        "alias": ALIAS,
        "provider": PROVIDER,
        "model": MODEL,
        "input_tokens": 1000,
        "output_tokens": 500,
        "cached": False,
        "cost_usd": COST,
        "price_version": PRICE_VERSION,
        "estimated": False,
        "session": SESSION,
        "trace_id": TRACE_ID,
    }
    fields.update(overrides)
    return CostLedgerRow(**fields)


def ledger_rows(pool: Pool, schema: str) -> list[tuple[object, ...]]:
    """Every `cost_ledger` row, in the column order the migration declares."""
    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL("select {} from {} order by ts").format(
                sql.SQL(", ").join(sql.Identifier(name) for name in LEDGER_COLUMNS),
                sql.Identifier(schema, COST_LEDGER),
            )
        ).fetchall()
    return [tuple(row) for row in rows]


@requires_postgres
def test_every_column_round_trips_through_the_shipped_migration(
    pool: Pool, migrated_schema: str
) -> None:
    """The plan's schema-drift risk in one case. Catches an INSERT whose column list
    disagrees with the SQL file: `tenant`, `alias`, `provider`, and `model` are
    all `text`, so two of them swapped is a ledger that attributes a tenant's spend to
    another tenant's model and PostgreSQL accepts it without a murmur. Catches a
    timestamp stored without its zone, and an `estimated` column defaulting to false -
    which would make every estimate look measured - because both flags are asserted at
    the value the row carried."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row(estimated=True))
    writer.flush()

    assert ledger_rows(pool, migrated_schema) == [
        (
            WHEN,
            TENANT,
            ALIAS,
            PROVIDER,
            MODEL,
            1000,
            500,
            False,
            COST,
            PRICE_VERSION,
            True,
            SESSION,
            TRACE_ID,
        )
    ]


@requires_postgres
def test_the_cost_reads_back_exactly_at_six_decimal_places(
    pool: Pool, migrated_schema: str
) -> None:
    """The criterion, and the plan's `numeric` scale risk. Catches a column declared with
    a scale narrower than six, which rounds the number the library computed exactly and
    then sums a month of rows into an invoice nobody can reconcile - and catches
    `double precision`, which round-trips this value to something that merely prints the
    same. The exponent is asserted because `Decimal("1234.5678910") == COST`, so equality
    alone cannot see a scale that changed; the type is asserted because a `float` coming
    back would compare equal to a `Decimal` for kinder numbers than this one."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row())
    writer.flush()

    stored = ledger_rows(pool, migrated_schema)[0][LEDGER_COLUMNS.index("cost_usd")]

    assert isinstance(stored, Decimal)
    assert stored == COST
    assert stored.as_tuple().exponent == -6


@requires_postgres
def test_a_null_session_and_a_cached_row_round_trip(pool: Pool, migrated_schema: str) -> None:
    """Catches a `session` column declared NOT NULL, which turns every request without a
    session header into a lost ledger row on the one path - the unauthenticated console
    probe, an eval run - where nobody is watching, and a writer that coerces `None` into
    an empty string on the way. `cached=True` is carried in the same row because the
    cost page counts cached requests separately, so a boolean bound positionally into the
    wrong column would make every cached request look uncached."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row(session=None, cached=True))
    writer.flush()

    row = ledger_rows(pool, migrated_schema)[0]

    assert row[LEDGER_COLUMNS.index("session")] is None
    assert row[LEDGER_COLUMNS.index("cached")] is True


@requires_postgres
@pytest.mark.db
def test_cache_token_columns_round_trip(pool: Pool, migrated_schema: str) -> None:
    """Catches a migration that never adds the columns, adds them nullable or without a
    default of 0 (an older INSERT would then store null), and an INSERT that binds the two
    counts into the wrong columns or into the input and output ones: the writer's row reads
    back 2048 and 512, and a raw INSERT that names neither reads back 0 and 0."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row(cache_read_tokens=2048, cache_write_tokens=512))
    writer.flush()
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, input_tokens, "
                "output_tokens, cached, cost_usd, price_version, estimated, session, trace_id) "
                "VALUES (%s, 'nordlicht', 'agent-default', 'anthropic', 'claude-haiku-5-5', "
                "1, 1, false, 0, 'v', false, null, %s)"
            ).format(sql.Identifier(migrated_schema, COST_LEDGER)),
            (NEXT_DAY, TRACE_ID),
        )
        found = conn.execute(
            sql.SQL(
                "select cache_read_tokens, cache_write_tokens, input_tokens, output_tokens "
                "from {} order by ts"
            ).format(sql.Identifier(migrated_schema, COST_LEDGER))
        ).fetchall()
        nullable = conn.execute(
            "select column_name, is_nullable, data_type from information_schema.columns "
            "where table_schema = %s and table_name = %s "
            "and column_name in ('cache_read_tokens', 'cache_write_tokens') order by column_name",
            (migrated_schema, COST_LEDGER),
        ).fetchall()

    assert [tuple(row) for row in found] == [(2048, 512, 1000, 500), (0, 0, 1, 1)]
    assert [tuple(row) for row in nullable] == [
        ("cache_read_tokens", "NO", "integer"),
        ("cache_write_tokens", "NO", "integer"),
    ]


USER_ID = "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c"


def stored_user_ids(pool: Pool, schema: str) -> list[object]:
    """The `user_id` column of every row, oldest first, read by name."""
    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL("select user_id from {} order by ts").format(
                sql.Identifier(schema, COST_LEDGER)
            )
        ).fetchall()
    return [row[0] for row in rows]


@pytest.mark.db
@requires_postgres
def test_the_user_id_column_is_written_and_nullable(pool: Pool, migrated_schema: str) -> None:
    """Catches an INSERT that leaves `user_id` out of its column list (every row reads back
    `None`), a column declared NOT NULL (every request without `X-User` becomes a lost
    ledger row), and a writer that coerces `None` into an empty string, which the top-users
    read would rank as one user. The id and a `None` are written in the same run."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row(ts=WHEN, user_id=USER_ID))
    writer.write(a_row(ts=NEXT_DAY, user_id=None))
    writer.flush()

    assert stored_user_ids(pool, migrated_schema) == [USER_ID, None]
