"""`make migrate`, `make purge-residency` and `make refresh-rollup`: the entry points
and the purge boundary.

The retention boundary is asserted EXACTLY: `today - 90` is
kept and `today - 91` is dropped, with the neighbouring days present in the same
schema so the case cannot pass by dropping everything or nothing. `today` is
injected rather than read from the clock, so the rows below mean the same thing
on every day of the year and at 23:59.

Skipped without `OPSCONSOLE_TEST_DATABASE_URL`, except
the three cases about an UNCONFIGURED database, which are the whole point of the
non-zero exit code and need no server at all. Those three set
`OPSCONSOLE_DATABASE_URL` through `monkeypatch` where they need it: the root
tests/conftest.py clears it for the session, so a developer with a local stack up
runs the same suite as one without.

The two success cases run against the DEFAULT schema of the test database rather
than a throwaway one, because the entry points take their configuration from
`Settings` and nothing on the command line points them elsewhere. Both are
idempotent, which is what makes that safe to repeat.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta

import pytest
from psycopg import sql

from app.ledger import RESIDENCY_LOG_RETENTION_DAYS, PostgresResidencyLogWriter
from app.ledger.cli import migrate, purge, purge_partitions, refresh_rollup
from app.ledger.residency import ResidencyLogRow, partition_name
from app.model_routing.policy import DataClass

from .conftest import DATABASE_URL, Pool, partitions, requires_postgres

DEFAULT_PARTITION = "residency_log_default"

TODAY = date(2026, 9, 7)
"""The purge's `today`, injected. Nothing here reads the real clock."""

DATABASE_URL_ENV = "OPSCONSOLE_DATABASE_URL"
"""What the entry points read through `Settings`. Cleared by tests/conftest.py."""

# Offsets in days before `TODAY`, and whether the purge must drop that day at the
# shipped retention. 90 and 91 are the boundary pair the plan names; 89 and 92
# are their neighbours, so an off-by-one in either direction has somewhere to
# show. Every offset is seeded into the SAME schema before the purge runs.
PURGE_ROWS: list[tuple[int, bool]] = [
    (0, False),
    (1, False),
    (89, False),
    (90, False),
    (91, True),
    (92, True),
    (365, True),
]

SEEDED_OFFSETS = tuple(offset for offset, _ in PURGE_ROWS)


def day_at(offset: int) -> date:
    """The day `offset` days before `TODAY`."""
    return TODAY - timedelta(days=offset)


def row_on(day: date) -> ResidencyLogRow:
    """One well-formed row stamped at midday on `day`, so no zone shifts it."""
    return ResidencyLogRow(
        ts=datetime.combine(day, time(12, 0), tzinfo=UTC),
        tenant="evals",
        data_class=DataClass.INTERNAL,
        alias="agent-default",
        provider="local",
        region="self",
        allowed=True,
    )


def seed(pool: Pool, schema: str, offsets: Iterable[int]) -> set[str]:
    """Write one row per offset, so each of those days has a partition of its own."""
    writer = PostgresResidencyLogWriter(pool, schema=schema)
    for offset in offsets:
        writer.write(row_on(day_at(offset)))
    return {partition_name(day_at(offset)) for offset in offsets}


def parent_count(pool: Pool, schema: str) -> int:
    """How many rows the whole partitioned table holds."""
    with pool.connection() as conn:
        row = conn.execute(
            sql.SQL("select count(*) from {}").format(sql.Identifier(schema, "residency_log"))
        ).fetchone()
    assert row is not None
    return int(row[0])


# --------------------------------------------------------------------------
# the boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "dropped"), PURGE_ROWS, ids=[f"today-{offset}" for offset, _ in PURGE_ROWS]
)
@requires_postgres
def test_the_purge_boundary_is_inclusive_keep(
    pool: Pool, migrated_schema: str, offset: int, dropped: bool
) -> None:
    """Catches every off-by-one the boundary has. `>=` where `>` belongs drops `today -
    90`, which is a day of evidence destroyed inside the retention the project promised;
    the other direction keeps `today - 91` forever and the table never stops growing.
    Every offset in the table is seeded into this schema first, so a purge that dropped
    all of them, or none, fails the rows it is not being asked about as well as this one."""
    seed(pool, migrated_schema, SEEDED_OFFSETS)

    with pool.connection() as conn:
        purge_partitions(conn, schema=migrated_schema, today=TODAY)

    remaining = set(partitions(pool, migrated_schema))
    assert (partition_name(day_at(offset)) in remaining) is not dropped


@requires_postgres
def test_the_purge_answers_exactly_the_partitions_it_dropped(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches a purge that answers every partition it CONSIDERED, or the names it kept:
    the CronJob that will call this logs the answer, so an operator asking what happened
    to last quarter's rows would be told something untrue. Sorted, so the answer does not
    depend on the catalogue's order."""
    seed(pool, migrated_schema, SEEDED_OFFSETS)

    with pool.connection() as conn:
        answered = purge_partitions(conn, schema=migrated_schema, today=TODAY)

    assert answered == tuple(
        sorted(partition_name(day_at(offset)) for offset, dropped in PURGE_ROWS if dropped)
    )


@requires_postgres
def test_the_default_partition_is_never_dropped(pool: Pool, migrated_schema: str) -> None:
    """Catches a purge that walks every child of the parent table instead of the day
    partitions it can date: the DEFAULT partition has no day in its name, so a purge that
    cannot parse one and drops it anyway removes decision B's backstop, and the next
    refused `CREATE` loses its row instead of misfiling it."""
    seed(pool, migrated_schema, SEEDED_OFFSETS)

    with pool.connection() as conn:
        answered = purge_partitions(conn, schema=migrated_schema, today=TODAY)

    assert DEFAULT_PARTITION in partitions(pool, migrated_schema)
    assert DEFAULT_PARTITION not in answered


@requires_postgres
def test_a_purge_with_nothing_old_enough_drops_nothing(pool: Pool, migrated_schema: str) -> None:
    """Catches a purge whose comparison is inverted or whose date parsing silently fails -
    both of which drop everything - and one that reports work it did not do. The days
    seeded here are all inside the retention, so the correct answer is the empty tuple and
    the partitions are still there afterwards."""
    young = (0, 1, 89, 90)
    expected = seed(pool, migrated_schema, young)

    with pool.connection() as conn:
        answered = purge_partitions(conn, schema=migrated_schema, today=TODAY)

    assert answered == ()
    assert expected <= set(partitions(pool, migrated_schema))
    assert parent_count(pool, migrated_schema) == len(young)


@requires_postgres
def test_the_retention_argument_moves_the_boundary(pool: Pool, migrated_schema: str) -> None:
    """Catches a purge that takes `retention_days` and then compares against the 90 in the
    module: the parameter is what lets an operator run a shorter retention deliberately,
    and a hardcoded comparison would keep this test's `today - 89` when it was asked to
    drop everything over a day old. The kept row is the same one the shipped retention
    would also keep, so the difference asserted is the ARGUMENT's."""
    seed(pool, migrated_schema, (0, 1, 89))

    with pool.connection() as conn:
        answered = purge_partitions(conn, schema=migrated_schema, today=TODAY, retention_days=1)

    assert answered == (partition_name(day_at(89)),)
    assert partition_name(day_at(0)) in partitions(pool, migrated_schema)
    assert partition_name(day_at(1)) in partitions(pool, migrated_schema)


@requires_postgres
def test_the_purge_leaves_the_rows_it_kept(pool: Pool, migrated_schema: str) -> None:
    """Catches a purge that empties the whole table and then re-creates the young
    partitions, which would satisfy every name-level assertion above while destroying
    exactly the evidence the retention exists to preserve. One row per seeded day, so the
    count after the purge is the number of days inside the retention."""
    seed(pool, migrated_schema, SEEDED_OFFSETS)
    kept = sum(1 for _, dropped in PURGE_ROWS if not dropped)

    with pool.connection() as conn:
        purge_partitions(conn, schema=migrated_schema, today=TODAY)

    assert parent_count(pool, migrated_schema) == kept


def test_the_shipped_retention_is_the_purges_default() -> None:
    """Catches a default that drifted from the published constant: `make purge-residency`
    passes no retention, so what that target enforces is whatever this default is, and a
    purge defaulting to 30 would quietly destroy sixty days the project promised to keep.
    The signature is read rather than the behaviour, because the behavioural cases above
    all pass `today` and would happily agree with a wrong default."""
    default = inspect.signature(purge_partitions).parameters["retention_days"].default

    assert default == RESIDENCY_LOG_RETENTION_DAYS
    assert RESIDENCY_LOG_RETENTION_DAYS == 90


# --------------------------------------------------------------------------
# the three entry points
# --------------------------------------------------------------------------


def test_migrate_refuses_without_a_database_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches an entry point that reports success having done nothing, which in a deploy
    pipeline is the worst possible answer: the next step starts the gateway against a
    database with no table in it. Catches a bare traceback too - the exit code is the
    contract, and a message an operator can read is the rest of it."""
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)

    code = migrate([])

    assert code != 0
    printed = capsys.readouterr()
    assert (printed.out + printed.err).strip() != ""


def test_purge_refuses_without_a_database_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same claim for the second entry point, which is a separate function and would
    otherwise be the one that runs unconfigured in a CronJob and reports success forever
    while nothing is ever purged."""
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)

    code = purge([])

    assert code != 0
    printed = capsys.readouterr()
    assert (printed.out + printed.err).strip() != ""


def test_refresh_rollup_refuses_without_a_database_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same claim for the third entry point. This one runs on a schedule rather than
    in a deploy, so an unconfigured refresh reporting success is a console cost page that
    silently stops moving while every CronJob run is recorded as a success - the failure
    mode nobody looks for, because there is nothing to look at."""
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)

    code = refresh_rollup([])

    assert code != 0
    printed = capsys.readouterr()
    assert (printed.out + printed.err).strip() != ""


@requires_postgres
def test_migrate_applies_the_migrations_and_answers_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches the entry point never reaching the runner - a `Makefile` target that exits
    0 having imported a module - and an entry point that raises instead of answering a
    code, which is the difference between a failed deploy step and a stack trace. Run
    twice, because `make migrate` on an already-migrated database is the ordinary case and
    it must also answer 0."""
    assert DATABASE_URL is not None
    monkeypatch.setenv(DATABASE_URL_ENV, DATABASE_URL)

    assert migrate([]) == 0
    assert migrate([]) == 0

    printed = capsys.readouterr()
    assert (printed.out + printed.err).strip() != ""


@requires_postgres
def test_purge_answers_zero_when_there_is_nothing_to_drop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a purge entry point that treats "no partition was old enough" as a failure,
    which is what a CronJob would page on every night for the first ninety days of the
    system's life. Migrating first is what makes this about the purge rather than about a
    missing table."""
    assert DATABASE_URL is not None
    monkeypatch.setenv(DATABASE_URL_ENV, DATABASE_URL)
    assert migrate([]) == 0

    assert purge([]) == 0

    printed = capsys.readouterr()
    assert (printed.out + printed.err).strip() != ""
