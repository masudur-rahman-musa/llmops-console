r"""The three commands that own the ledger's schema, its retention and its rollup.

`make migrate` applies the SQL files; `make purge-residency` drops the day
partitions that have aged out; `make refresh-rollup` recomputes the materialised
view the console's cost page reads. None of them runs on app start, deliberately: a
gateway that migrated its own database would race every other replica coming up
at the same time, and a deploy would have no step at which someone could look at
the DDL before it ran. `create_app` therefore assumes the table is already there,
and `/readyz`'s `postgres` probe is what says whether the server is reachable.

The purge is shipped here rather than left to the platform plan (an earlier change
decision G), and the refresh for the same reason: the CronJob that calls either
one belongs with the deployment, but the code that knows which partition is old,
or how a materialised view has to be refreshed the first time, belongs with the
file that defines the table.

All three entry points read `OPSCONSOLE_DATABASE_URL` through `Settings` and
answer a process exit code. An unconfigured database is a non-zero code with a message,
never a quiet success: in a deploy pipeline "migrated nothing, exit 0" is the
worst available answer, because the next step starts the gateway against a
database with no table in it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

from app.core.settings import Settings
from app.ledger.cost_ledger import COST_LEDGER_DAILY
from app.ledger.db import DEFAULT_SCHEMA, apply_migrations, build_pool, checked_schema
from app.ledger.residency import (
    PARTITION_DAY_FORMAT,
    RESIDENCY_LOG,
    RESIDENCY_LOG_RETENTION_DAYS,
)

NO_DATABASE_URL = 2
"""The exit code for "nothing is configured", distinct from 1 so a CronJob's
alerting can tell a misconfiguration from a failed statement. One case per entry
point, because each reads the setting for itself:
`test_migrate_refuses_without_a_database_url`,
`test_purge_refuses_without_a_database_url` and
`test_refresh_rollup_refuses_without_a_database_url`."""


def _day_of(partition: str) -> date | None:
    """The day a partition name encodes, or `None` if it does not encode one.

    `None` is what keeps `residency_log_default` alive: it is a child of the
    parent table with no day in its name, and a purge that dropped what it could
    not date would remove decision B's backstop, so the next refused `CREATE`
    would lose its row instead of misfiling it
    (`test_the_default_partition_is_never_dropped`).
    """
    prefix = f"{RESIDENCY_LOG}_"
    if not partition.startswith(prefix):
        return None
    try:
        return datetime.strptime(partition[len(prefix) :], PARTITION_DAY_FORMAT).date()
    except ValueError:
        return None


def _partition_names(conn: Connection[TupleRow], schema: str) -> list[str]:
    """Every child of `residency_log` in this schema, the DEFAULT one included."""
    rows = conn.execute(
        """
        select child.relname
        from pg_inherits i
        join pg_class parent on parent.oid = i.inhparent
        join pg_class child on child.oid = i.inhrelid
        join pg_namespace n on n.oid = parent.relnamespace
        where n.nspname = %s and parent.relname = %s
        """,
        (schema, RESIDENCY_LOG),
    ).fetchall()
    return [str(row[0]) for row in rows]


def purge_partitions(
    conn: Connection[TupleRow],
    *,
    schema: str = DEFAULT_SCHEMA,
    today: date,
    retention_days: int = RESIDENCY_LOG_RETENTION_DAYS,
) -> tuple[str, ...]:
    """Drop the day partitions older than the retention, and answer what was dropped.

    The boundary is inclusive-keep: `today - retention_days` STAYS and
    `today - retention_days - 1` goes, asserted pair by pair in
    `test_the_purge_boundary_is_inclusive_keep`. Both directions of an off-by-one
    here are real damage - one destroys a day of evidence inside the retention
    the project promised, the other keeps everything forever.

    `today` is a parameter rather than a clock read so a caller can say which day
    it means; `retention_days` is one so an operator can run a shorter retention
    deliberately, and `make purge-residency` passes neither, which is why the
    default is checked against the published constant
    (`test_the_shipped_retention_is_the_purges_default`).

    Answers only the partitions it actually dropped, sorted: the CronJob that
    will call this logs the answer, and an operator asking what happened to last
    quarter's rows has to be told the truth.

    DROP rather than DELETE, which is the reason the table is partitioned at all:
    dropping ninety days is a catalogue update, deleting the same rows is a table
    rewrite and a vacuum on the hot write path.
    """
    checked = checked_schema(schema)
    cutoff = today - timedelta(days=retention_days)
    dropped: list[str] = []
    for name in _partition_names(conn, checked):
        day = _day_of(name)
        if day is None or day >= cutoff:
            continue
        conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(checked, name)))
        dropped.append(name)
    return tuple(sorted(dropped))


def _parser(description: str) -> argparse.ArgumentParser:
    """The one option all three commands share."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help=f"the schema holding the ledger tables (default: {DEFAULT_SCHEMA})",
    )
    return parser


def _database_url() -> str | None:
    """Where Postgres is, or `None` when nothing configured one."""
    return Settings().database_url


def migrate(argv: Sequence[str] | None = None) -> int:
    """Apply every migration this database has not seen. `make migrate`.

    Safe to run twice: the second run applies nothing and leaves the data alone,
    which is what makes it a deploy step rather than a thing someone has to
    remember whether they already did.
    """
    options = _parser("Apply the ledger migrations.").parse_args(argv)
    url = _database_url()
    if url is None:
        print("OPSCONSOLE_DATABASE_URL is not set; nothing was migrated", file=sys.stderr)
        return NO_DATABASE_URL

    pool = build_pool(url)
    try:
        with pool.connection() as conn:
            applied = apply_migrations(conn, schema=options.schema)
    finally:
        pool.close()

    if applied:
        print(f"applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        print("already up to date; nothing was applied")
    return 0


def purge(argv: Sequence[str] | None = None) -> int:
    """Drop residency partitions past the retention. `make purge-residency`.

    Nothing to drop is a SUCCESS, not a failure: for the first ninety days of the
    system's life that is the only correct outcome, and a CronJob that treated it
    as an error would page someone every night until then
    (`test_purge_answers_zero_when_there_is_nothing_to_drop`).
    """
    parser = _parser(
        f"Drop residency log partitions older than {RESIDENCY_LOG_RETENTION_DAYS} days."
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=RESIDENCY_LOG_RETENTION_DAYS,
        help=f"how many days to keep (default: {RESIDENCY_LOG_RETENTION_DAYS})",
    )
    options = parser.parse_args(argv)
    url = _database_url()
    if url is None:
        print("OPSCONSOLE_DATABASE_URL is not set; nothing was purged", file=sys.stderr)
        return NO_DATABASE_URL

    pool = build_pool(url)
    try:
        with pool.connection() as conn:
            dropped = purge_partitions(
                conn,
                schema=options.schema,
                today=datetime.now(tz=UTC).date(),
                retention_days=options.retention_days,
            )
    finally:
        pool.close()

    if dropped:
        print(f"dropped {len(dropped)} partition(s): {', '.join(dropped)}")
    else:
        print("nothing was old enough to drop")
    return 0


def refresh_daily_rollup(
    pool: ConnectionPool[Connection[TupleRow]],
    *,
    schema: str = DEFAULT_SCHEMA,
) -> bool:
    """Refresh `cost_ledger_daily`, and answer whether it went CONCURRENTLY.

    Two shapes, and which one runs is not a preference. A materialised view that
    has never been populated CANNOT be refreshed concurrently, so a command that
    always used `CONCURRENTLY` would fail on exactly the database where it is run
    for the first time; one that never used it takes an `ACCESS EXCLUSIVE` lock
    for the length of every later refresh, and the console reads nothing while it
    is held. So the first refresh is the plain one and every later refresh is
    concurrent, decided from `pg_class.relispopulated` rather than from a flag
    someone has to set. Both are idempotent - the second answers the same rows as
    the first - and the pair is
    `test_the_first_refresh_populates_and_the_second_runs_concurrently`.

    Takes the POOL rather than a connection, unlike `purge_partitions`:
    `REFRESH MATERIALIZED VIEW CONCURRENTLY` cannot run inside a transaction
    block and psycopg opens one on the first statement, so the code that borrows
    the connection has to be the code that sets autocommit.

    The concurrent refresh is also what a UNIQUE index over the whole group key
    buys: without one PostgreSQL refuses it at runtime, and only at runtime.
    """
    checked = checked_schema(schema)
    with pool.connection() as conn:
        conn.autocommit = True
        row = conn.execute(
            """
            select c.relispopulated
            from pg_class c
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s and c.relname = %s
            """,
            (checked, COST_LEDGER_DAILY),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"{checked}.{COST_LEDGER_DAILY} does not exist; run `make migrate` first"
            )
        concurrently = bool(row[0])
        statement = (
            "REFRESH MATERIALIZED VIEW CONCURRENTLY {}"
            if concurrently
            else "REFRESH MATERIALIZED VIEW {}"
        )
        conn.execute(sql.SQL(statement).format(sql.Identifier(checked, COST_LEDGER_DAILY)))
    return concurrently


def refresh_rollup(argv: Sequence[str] | None = None) -> int:
    """Recompute the daily cost rollup. `make refresh-rollup`.

    Safe to run at any time and as often as an operator likes: the view is
    derived entirely from `cost_ledger`, so a refresh answers the same rows as
    the one before it unless a request was served in between. The CronJob that
    calls this on a schedule belongs to the platform plan.
    """
    options = _parser(f"Refresh the {COST_LEDGER_DAILY} materialised view.").parse_args(argv)
    url = _database_url()
    if url is None:
        print("OPSCONSOLE_DATABASE_URL is not set; nothing was refreshed", file=sys.stderr)
        return NO_DATABASE_URL

    pool = build_pool(url)
    try:
        concurrently = refresh_daily_rollup(pool, schema=options.schema)
    finally:
        pool.close()

    how = "concurrently" if concurrently else "for the first time, exclusively"
    print(f"refreshed {options.schema}.{COST_LEDGER_DAILY} {how}")
    return 0
