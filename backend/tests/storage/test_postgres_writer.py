"""`PostgresResidencyLogWriter` against a real server, over the real migration.

Every case here runs the SHIPPED migration into a throwaway schema and then
writes through the real writer, never against a hand-built table. That is the
whole point of the file: the SQL file and the writer's INSERT are written in two
different places and nothing type-checks between them, so a column added to one
and not the other, or a positional binding that swaps two `text` columns, is
invisible to PostgreSQL and to mypy both.

Skipped without `OPSCONSOLE_TEST_DATABASE_URL`. One
session pool, one throwaway schema per test.

The statement-level view of the same writer - how many `CREATE`s it issues, and
in what order - is in tests/storage/test_residency_writer.py, where a recording
pool double can see it; a real server cannot tell a create that was skipped from
one that was issued and did nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from psycopg import sql

from app.model_routing.policy import DataClass
from app.storage import PostgresResidencyLogWriter, ResidencyLogRow
from app.storage.db import apply_migrations
from app.storage.residency_log import partition_name

from .conftest import Pool, new_schema_name, partitions, requires_postgres, table_names

DEFAULT_PARTITION = "residency_log_default"

WHEN = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)
NEXT_DAY = WHEN + timedelta(days=1)

# Every value distinct from every other, and none of them a substring of another,
# so a positional binding that swapped two of the four `text` columns - which
# PostgreSQL would accept silently - shows up as a wrong value rather than as a
# type error.
ROW = ResidencyLogRow(
    ts=WHEN,
    tenant="nordlicht",
    data_class=DataClass.CONFIDENTIAL,
    alias="agent-default",
    provider="local",
    region="self",
    allowed=True,
)


def count_in(pool: Pool, schema: str, relation: str) -> int:
    """How many rows one relation holds. `relation` is a literal in this module."""
    with pool.connection() as conn:
        row = conn.execute(
            sql.SQL("select count(*) from {}").format(sql.Identifier(schema, relation))
        ).fetchone()
    assert row is not None
    return int(row[0])


def only_row(pool: Pool, schema: str) -> tuple[object, ...]:
    """The one row in `residency_log`, in the column order the migration declares."""
    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL(
                "select ts, tenant, data_class, alias, provider, region, allowed from {}"
            ).format(sql.Identifier(schema, "residency_log"))
        ).fetchall()
    assert len(rows) == 1, rows
    return tuple(rows[0])


@requires_postgres
def test_a_write_creates_the_days_partition_and_the_row_lands_in_it(
    pool: Pool, migrated_schema: str
) -> None:
    """The criterion in one case. Catches a writer that never creates a partition - the
    row would still be readable from the parent, out of the DEFAULT partition, so reading
    the parent alone proves nothing - and one that creates the partition but leaves the
    row behind in the DEFAULT, which is what a create issued AFTER the insert does. Both
    counts are asserted, so exactly one of the two possible homes has it."""
    writer = PostgresResidencyLogWriter(pool, schema=migrated_schema)

    writer.write(ROW)

    day = partition_name(WHEN.date())
    assert day in partitions(pool, migrated_schema)
    assert count_in(pool, migrated_schema, day) == 1
    assert count_in(pool, migrated_schema, DEFAULT_PARTITION) == 0


@requires_postgres
def test_every_column_round_trips_through_the_shipped_migration(
    pool: Pool, migrated_schema: str
) -> None:
    """The plan's schema-drift risk. Catches an INSERT whose column list disagrees with
    the SQL file: `tenant`, `alias`, `provider` and `region` are all `text`, so two of
    them swapped is a residency log that names the wrong provider for a request and
    PostgreSQL accepts it without a murmur. Catches a `data_class` written as the enum's
    repr rather than its value, and a timestamp stored without its zone."""
    writer = PostgresResidencyLogWriter(pool, schema=migrated_schema)

    writer.write(ROW)

    ts, tenant, data_class, alias, provider, region, allowed = only_row(pool, migrated_schema)
    assert ts == WHEN
    assert tenant == ROW.tenant
    assert data_class == ROW.data_class.value
    assert alias == ROW.alias
    assert provider == ROW.provider
    assert region == ROW.region
    assert allowed is True


@requires_postgres
def test_rows_on_two_days_land_in_a_partition_each(pool: Pool, migrated_schema: str) -> None:
    """Catches a range bound wider than a day - a month, or `FROM (day) TO (infinity)` -
    which passes the single-day case and then swallows every later day into one partition,
    making the retention purge drop ninety days of evidence in one statement. Catches a
    partition created once per process regardless of the day (a project rule's reset,
    seen from the server's side)."""
    writer = PostgresResidencyLogWriter(pool, schema=migrated_schema)

    writer.write(ROW)
    writer.write(ROW.model_copy(update={"ts": NEXT_DAY}))

    today = partition_name(WHEN.date())
    tomorrow = partition_name(NEXT_DAY.date())
    assert {today, tomorrow} <= set(partitions(pool, migrated_schema))
    assert count_in(pool, migrated_schema, today) == 1
    assert count_in(pool, migrated_schema, tomorrow) == 1
    assert count_in(pool, migrated_schema, DEFAULT_PARTITION) == 0


@requires_postgres
def test_two_rows_on_the_same_day_both_land(pool: Pool, migrated_schema: str) -> None:
    """Catches a second write on a day whose partition already exists raising
    `DuplicateTable` - the failure mode of a create the cache is supposed to suppress but
    the DDL does not guard - which would make the SECOND request of every day the one
    that loses its row."""
    writer = PostgresResidencyLogWriter(pool, schema=migrated_schema)

    writer.write(ROW)
    writer.write(ROW.model_copy(update={"tenant": "cuan"}))

    assert count_in(pool, migrated_schema, partition_name(WHEN.date())) == 2


@requires_postgres
def test_a_row_whose_partition_cannot_be_created_lands_in_the_default(
    pool: Pool, migrated_schema: str
) -> None:
    """Decision B's backstop, end to end. The day's name is taken by an unrelated table
    here, so the `CREATE ... PARTITION OF` cannot succeed however it is written - which is
    what a read-only role or a revoked privilege does in production. Catches a writer
    whose create failure propagates (the row is lost) and one that skips the INSERT when
    the create failed (the row is lost more quietly). The DEFAULT partition holding it is
    the criterion's own words: a badly-placed row rather than a lost one."""
    blocked = partition_name(WHEN.date())
    with pool.connection() as conn:
        conn.execute(
            sql.SQL("create table {} (unrelated int)").format(
                sql.Identifier(migrated_schema, blocked)
            )
        )
    writer = PostgresResidencyLogWriter(pool, schema=migrated_schema)

    writer.write(ROW)

    assert count_in(pool, migrated_schema, "residency_log") == 1
    assert count_in(pool, migrated_schema, DEFAULT_PARTITION) == 1
    assert blocked not in partitions(pool, migrated_schema)


@requires_postgres
def test_the_writer_writes_into_the_schema_it_was_given(pool: Pool, migrated_schema: str) -> None:
    """Catches an unqualified INSERT that lands wherever `search_path` points. The
    neighbour schema is MIGRATED too, so it has a `residency_log` of its own that a
    misdirected write could land in - an empty count from a table that does not exist
    would be a query error, not a passing assertion."""
    neighbour = new_schema_name()
    with pool.connection() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(neighbour)))
    try:
        with pool.connection() as conn:
            apply_migrations(conn, schema=neighbour)
        assert "residency_log" in table_names(pool, neighbour)

        PostgresResidencyLogWriter(pool, schema=migrated_schema).write(ROW)

        assert count_in(pool, migrated_schema, "residency_log") == 1
        assert count_in(pool, neighbour, "residency_log") == 0
    finally:
        with pool.connection() as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(neighbour)))
