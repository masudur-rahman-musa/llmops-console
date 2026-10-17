"""`PostgresCostLedgerWriter` and the daily rollup against a real server.

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

The contract these cases are written against, before it exists:

* `cost_ledger_daily` has the columns `day, tenant, alias, provider, model,
  region, requests, cached_requests, input_tokens, output_tokens, cost_usd`, with
  `day` the UTC date of `ts`;
* `app.cli.ledger.refresh_daily_rollup(pool, *, schema=...) -> bool`
  refreshes it and answers whether it ran CONCURRENTLY - `False` the first time,
  because a never-populated materialised view cannot be refreshed concurrently,
  and `True` afterwards. It takes the POOL rather than a connection because
  `REFRESH MATERIALIZED VIEW CONCURRENTLY` cannot run inside a transaction block,
  so the code that borrows the connection is the code that has to set autocommit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from psycopg import sql

from app.storage import CostLedgerRow, PostgresCostLedgerWriter
from app.storage.cost_ledger import COST_LEDGER, COST_LEDGER_DAILY, refresh_daily_rollup

from .conftest import Pool, requires_postgres

# 22:45 UTC on purpose: a rollup that grouped by the SERVER's local day rather
# than the UTC one puts this row on 2026-09-08 for anywhere east of Greenwich, so
# the day assertions below are what pin the anchoring.
WHEN = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)
SAME_DAY = datetime(2026, 9, 7, 23, 12, 40, tzinfo=UTC)
NEXT_DAY = WHEN + timedelta(days=1)

# Every value distinct from every other, and none of them a substring of another,
# so a positional binding that swapped two of the six `text` columns - which
# PostgreSQL would accept silently - shows up as a wrong value rather than as a
# type error.
TENANT = "nordlicht"
ALIAS = "agent-default"
PROVIDER = "local"
MODEL = "qwen2.5-7b-instruct"
REGION = "self"
SESSION = "sess-01j9x8"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PRICE_VERSION = "local/qwen2.5-7b-instruct@2026-01-01"

COST = Decimal("1234.567891")
"""Six decimal places, and digits in every one of them: a column declared
`numeric(18,2)` rounds this to 1234.57 and PostgreSQL says nothing about it."""

VIEW_COLUMNS = (
    "day",
    "tenant",
    "alias",
    "provider",
    "model",
    "region",
    "requests",
    "cached_requests",
    "input_tokens",
    "output_tokens",
    "cost_usd",
)

LEDGER_COLUMNS = (
    "ts",
    "tenant",
    "alias",
    "provider",
    "model",
    "region",
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
        "region": REGION,
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


def rollup_rows(pool: Pool, schema: str) -> list[tuple[object, ...]]:
    """Every `cost_ledger_daily` row, ordered so two reads compare directly.

    Selecting the columns by NAME rather than `select *` is deliberate: it is what
    makes a view that dropped `cached_requests`, or renamed `day` to `bucket`, a
    failure here rather than a console page that renders nothing.
    """
    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL(
                "select {} from {} order by day, tenant, alias, provider, model, region"
            ).format(
                sql.SQL(", ").join(sql.Identifier(name) for name in VIEW_COLUMNS),
                sql.Identifier(schema, COST_LEDGER_DAILY),
            )
        ).fetchall()
    return [tuple(row) for row in rows]


@requires_postgres
def test_every_column_round_trips_through_the_shipped_migration(
    pool: Pool, migrated_schema: str
) -> None:
    """The plan's schema-drift risk in one case. Catches an INSERT whose column list
    disagrees with the SQL file: `tenant`, `alias`, `provider`, `model` and `region` are
    all `text`, so two of them swapped is a ledger that attributes a tenant's spend to
    another tenant's model and PostgreSQL accepts it without a murmur. Catches a
    timestamp stored without its zone, and an `estimated` column defaulting to false -
    which would make every estimate look measured - because both flags are asserted at
    the value the row carried."""
    PostgresCostLedgerWriter(pool, schema=migrated_schema).write(a_row(estimated=True))

    assert ledger_rows(pool, migrated_schema) == [
        (
            WHEN,
            TENANT,
            ALIAS,
            PROVIDER,
            MODEL,
            REGION,
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
    PostgresCostLedgerWriter(pool, schema=migrated_schema).write(a_row())

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
    rollup counts cached requests separately, so a boolean bound positionally into the
    wrong column would make every cached request look uncached."""
    PostgresCostLedgerWriter(pool, schema=migrated_schema).write(a_row(session=None, cached=True))

    row = ledger_rows(pool, migrated_schema)[0]

    assert row[LEDGER_COLUMNS.index("session")] is None
    assert row[LEDGER_COLUMNS.index("cached")] is True


@requires_postgres
def test_rows_sharing_the_whole_group_key_become_one_rollup_row(
    pool: Pool, migrated_schema: str
) -> None:
    """The rollup's reason to exist. Catches a view grouping by something the console
    does not ask for - `session`, `trace_id`, the raw `ts` - which produces one output
    row per request and makes the daily page slower than the table it replaced. The
    aggregates are asserted individually because they fail differently: `requests` counts,
    `cached_requests` counts a SUBSET (a view counting all rows there would report every
    request as a cache hit), and the three sums must sum rather than take the first row's
    value. The day is asserted as the UTC date of a 22:45 timestamp, which is what pins
    the grouping to UTC rather than to the server's local day."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row(ts=WHEN, cached=False, input_tokens=1000, output_tokens=500))
    writer.write(
        a_row(
            ts=SAME_DAY,
            cached=True,
            input_tokens=7,
            output_tokens=3,
            cost_usd=Decimal("0.000100"),
            session=None,
            trace_id="ffffffffffffffffffffffffffffffff",
        )
    )

    refresh_daily_rollup(pool, schema=migrated_schema)

    assert rollup_rows(pool, migrated_schema) == [
        (
            WHEN.date(),
            TENANT,
            ALIAS,
            PROVIDER,
            MODEL,
            REGION,
            2,
            1,
            1007,
            503,
            COST + Decimal("0.000100"),
        )
    ]


SPLITTING_ROWS: list[tuple[str, dict[str, Any]]] = [
    ("a different day", {"ts": NEXT_DAY}),
    ("a different tenant", {"tenant": "cuan"}),
    ("a different alias", {"alias": "agent-strong"}),
    ("a different provider", {"provider": "vllm"}),
    ("a different model", {"model": "bge-small-en-v1.5"}),
    ("a different region", {"region": "eu"}),
]


@requires_postgres
@pytest.mark.parametrize(
    ("case", "difference"), SPLITTING_ROWS, ids=[case for case, _ in SPLITTING_ROWS]
)
def test_rows_differing_in_any_group_column_are_kept_apart(
    pool: Pool, migrated_schema: str, case: str, difference: dict[str, Any]
) -> None:
    """One row per column of the group key, because a GROUP BY that dropped exactly one of
    them still passes the case above and every other row here. Dropping `region` merges a
    tenant's EU and US spend into a single line, which is the number the residency page
    exists to keep apart; dropping `provider` or `model` hides the failover that doubled
    the bill; dropping the day makes the daily page a lifetime total. Two rollup rows is
    the assertion, and the two ledger rows are identical in every other column, so the
    column under test is the only thing that can split them."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row())
    writer.write(a_row(**difference))

    refresh_daily_rollup(pool, schema=migrated_schema)

    assert len(rollup_rows(pool, migrated_schema)) == 2


@requires_postgres
def test_the_first_refresh_populates_and_the_second_runs_concurrently(
    pool: Pool, migrated_schema: str
) -> None:
    """The `make refresh-rollup` criterion, and the plan's unique-index risk. A
    materialised view that has never been refreshed cannot be refreshed CONCURRENTLY, so
    a command that always uses CONCURRENTLY fails on exactly the database where it is run
    for the first time - and a command that NEVER uses it takes an exclusive lock that
    blanks the console for the length of every later refresh. The second call is also the
    only thing that can see a UNIQUE index which does not cover the whole group key:
    PostgreSQL refuses a concurrent refresh at runtime and only at runtime. The rows are
    compared before and after so a refresh that emptied the view - or one that answered
    each row twice - cannot pass on the return values alone."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row())
    writer.write(a_row(ts=NEXT_DAY))

    first_was_concurrent = refresh_daily_rollup(pool, schema=migrated_schema)
    after_first = rollup_rows(pool, migrated_schema)
    second_was_concurrent = refresh_daily_rollup(pool, schema=migrated_schema)
    after_second = rollup_rows(pool, migrated_schema)

    assert first_was_concurrent is False
    assert second_was_concurrent is True
    assert len(after_first) == 2
    assert after_second == after_first
