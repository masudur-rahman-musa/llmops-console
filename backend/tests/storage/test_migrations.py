r"""The migration runner: what it applies, what it refuses, and what the table becomes.

Most of this file needs a real PostgreSQL, because the claims are PostgreSQL's
own: a dropped table, a rebuilt rollup, and a second run that
changes nothing. Those cases carry `requires_postgres` from
tests/storage/conftest.py, which skips with `OPSCONSOLE_TEST_DATABASE_URL` named
in the reason. Per-test isolation is a throwaway
SCHEMA the runner migrates into; the pool is built once for
the session.

Three cases here need nothing at all and are deliberately not marked: the file
discovery, the published schema default, and the `SCHEMA_PATTERN` rejection
table. That last one is the sharpest case in the file - the schema name is
INTERPOLATED into DDL rather than bound as a parameter, because a `CREATE SCHEMA`
cannot take a placeholder - so it is checked against a connection double that
fails the test if a single statement reaches it (a project rule for the
trailing-newline rows).
"""

from __future__ import annotations

import threading
from typing import NoReturn, cast

import pytest
from psycopg import Connection, sql
from psycopg.rows import TupleRow

from app.storage.db import (
    DEFAULT_SCHEMA,
    MAX_POOL_SIZE,
    MAX_SCHEMA_LENGTH,
    MIGRATIONS,
    MIN_POOL_SIZE,
    POOL_WAIT_TIMEOUT,
    SCHEMA_PATTERN,
    apply_migrations,
    apply_migrations_locked,
    checked_schema,
    migrations,
)

from .conftest import Pool, new_schema_name, requires_postgres, table_names

FIRST_VERSION = "001_residency_log"
SECOND_VERSION = "002_cost_ledger"
THIRD_VERSION = "003_eval_runs"
FOURTH_VERSION = "004_console_events"
FIFTH_VERSION = "005_drop_residency"
SIXTH_VERSION = "006_ledger_prompt_version"
SEVENTH_VERSION = "007_drop_incidents"
EIGHTH_VERSION = "008_ledger_cache_tokens"
NINTH_VERSION = "009_ledger_status_latency"
TENTH_VERSION = "010_ledger_user_id"
ALL_VERSIONS = (
    FIRST_VERSION,
    SECOND_VERSION,
    THIRD_VERSION,
    FOURTH_VERSION,
    FIFTH_VERSION,
    SIXTH_VERSION,
    SEVENTH_VERSION,
    EIGHTH_VERSION,
    NINTH_VERSION,
    TENTH_VERSION,
)
"""Every migration that ships, in the order the runner applies them. A stem is
the version string, so a file renamed after it has been applied somewhere would
re-run as a new version."""

SCHEMA_MIGRATIONS = "schema_migrations"


class ForbiddenConnection:
    """A connection that fails the test if any statement reaches it."""

    def execute(self, query: object, params: object = None) -> NoReturn:
        raise AssertionError(f"a statement ran before the schema name was refused: {query!r}")


def as_connection(double: ForbiddenConnection) -> Connection[TupleRow]:
    """Hand a double to a runner typed for a real connection, at one seam."""
    return cast(Connection[TupleRow], double)


def applied_versions(pool: Pool, schema: str) -> list[str]:
    """Every version `schema_migrations` records, in the order it records them."""
    with pool.connection() as conn:
        rows = conn.execute(
            f'select version from "{schema}".{SCHEMA_MIGRATIONS} order by applied_at, version'
        ).fetchall()
    return [str(row[0]) for row in rows]


# --------------------------------------------------------------------------
# discovery and the published default (no server)
# --------------------------------------------------------------------------


def test_the_default_schema_is_the_published_one() -> None:
    """Catches a default renamed after data exists under the old one: a gateway writing
    to `opsconsole` and a purge reading somewhere else would leave the table growing
    forever with nothing reporting a problem."""
    assert DEFAULT_SCHEMA == "opsconsole"


def test_the_runner_applies_all_ten_sql_files_from_backend_migrations() -> None:
    """The runner reads `backend/migrations/` (outside the package, shipped by the
    image's COPY), found from the module position rather than the working directory,
    so `make migrate` works both at the repo root and inside the container. Reading
    a file back proves a readable file, not a `Path` that merely exists."""
    assert MIGRATIONS.is_dir()
    assert MIGRATIONS.name == "migrations"
    assert MIGRATIONS.parent.name == "backend"
    sql_files = sorted(path.name for path in MIGRATIONS.glob("*.sql"))
    assert sql_files == [
        "001_residency_log.sql",
        "002_cost_ledger.sql",
        "003_eval_runs.sql",
        "004_console_events.sql",
        "005_drop_residency.sql",
        "006_ledger_prompt_version.sql",
        "007_drop_incidents.sql",
        "008_ledger_cache_tokens.sql",
        "009_ledger_status_latency.sql",
        "010_ledger_user_id.sql",
    ]
    assert (MIGRATIONS / f"{FIRST_VERSION}.sql").read_text(encoding="utf-8").strip() != ""


def test_the_migrations_are_answered_in_file_name_order() -> None:
    """Catches discovery left in `Path.glob` order, which is the filesystem's order and
    not sorted: migration 002 would apply before 001 on some machines and after it on
    others, and the failure would look like a broken SQL file rather than like ordering.
    The version is the file STEM, which is what `schema_migrations` records."""
    found = migrations()

    versions = [migration.version for migration in found]
    assert versions == sorted(versions)
    assert versions[0] == FIRST_VERSION
    assert all(migration.sql.strip() != "" for migration in found)


def test_the_pool_is_bounded() -> None:
    """Catches a pool with `psycopg_pool`'s own defaults, which is four times this size.

    The bound is a promise made to the server, not a tuning knob: `max_connections`
    there is a global, and a gateway scaled to eight replicas holding a pool each is
    what starves the readiness probes - which open their own connection precisely so
    they report on the server rather than on the pool. Asserted exactly, because a
    range would admit the default this case exists to catch.

    An earlier change also adds `POOL_WAIT_TIMEOUT`: a caller that cannot get a connection
    in 2.0 seconds gets an error rather than waiting indefinitely."""
    assert (MIN_POOL_SIZE, MAX_POOL_SIZE) == (1, 4)
    assert POOL_WAIT_TIMEOUT == 2.0


def test_a_schema_name_longer_than_postgres_allows_is_refused() -> None:
    """Catches the check that is missing when only the slug pattern is applied: a 64
    character name is a perfectly good slug and PostgreSQL takes it without complaint,
    silently TRUNCATING it to 63 bytes. The tables then exist under a name the caller
    never asked for, and `make purge-residency`, which looks under the name it was
    given, finds nothing to drop and reports success while the table grows forever.
    The at-cap value is asserted as ACCEPTED in the same case, so an off-by-one that
    refuses every legal name is not what makes this pass."""
    assert MAX_SCHEMA_LENGTH == 63

    assert checked_schema("a" * MAX_SCHEMA_LENGTH) == "a" * MAX_SCHEMA_LENGTH
    with pytest.raises(ValueError, match=str(MAX_SCHEMA_LENGTH)):
        checked_schema("a" * (MAX_SCHEMA_LENGTH + 1))


# --------------------------------------------------------------------------
# the schema name is interpolated, so it is checked
# --------------------------------------------------------------------------

REJECTED_SCHEMAS: list[tuple[str, str]] = [
    ("empty", ""),
    ("uppercase", "OpsConsole"),
    ("a space", "ops console"),
    ("a quoted identifier delimiter", 'opsconsole"'),
    ("a single quote", "opsconsole'"),
    ("a semicolon, which would close the statement", "opsconsole; select 1"),
    ("a double dash, which would comment out the rest", "opsconsole--x"),
    ("a dot, which would name a second database object", "app.public"),
    ("a trailing newline", "opsconsole\n"),
    ("a trailing CRLF", "opsconsole\r\n"),
    ("a trailing space", "opsconsole "),
    ("a leading separator", "-opsconsole"),
]


@pytest.mark.parametrize(
    ("case", "schema_name"), REJECTED_SCHEMAS, ids=[case for case, _ in REJECTED_SCHEMAS]
)
def test_a_schema_name_that_is_not_a_slug_is_refused_before_any_statement(
    case: str, schema_name: str
) -> None:
    """Catches a runner that interpolates its `schema` argument straight into DDL. A
    schema name cannot be a bound parameter, so this pre-check is the ONLY thing standing
    between the argument and the statement; the connection double fails the test on any
    statement, which is what makes this "nothing ran" rather than "the statement errored".
    The newline and CRLF rows are the ones `re.match` with a trailing `$` would accept, and a schema name with a line break in it is exactly the kind of
    object nobody can drop afterwards."""
    with pytest.raises(ValueError):
        apply_migrations(as_connection(ForbiddenConnection()), schema=schema_name)


def test_the_schema_pattern_accepts_the_names_this_suite_and_the_default_use() -> None:
    """The other direction, so a pattern that refuses everything - which would satisfy
    every row above - cannot pass. Catches a pattern with no underscore in its separator
    class: the throwaway schemas in tests/storage/conftest.py are `opsconsole_test_<hex>`,
    and the whole Postgres half of this directory would skip on a ValueError that looked
    like a broken fixture."""
    assert SCHEMA_PATTERN.fullmatch(DEFAULT_SCHEMA) is not None
    assert SCHEMA_PATTERN.fullmatch(new_schema_name()) is not None


# --------------------------------------------------------------------------
# applying, and applying again
# --------------------------------------------------------------------------


@requires_postgres
def test_the_runner_answers_the_versions_it_applied(pool: Pool, schema: str) -> None:
    """Catches a runner that answers every KNOWN version rather than every version it
    applied - the difference only shows on the second call, which is the case below, and
    this one is what makes that comparison mean something. Also catches a runner that
    reports success while creating nothing, by asserting the two tables exist."""
    with pool.connection() as conn:
        applied = apply_migrations(conn, schema=schema)

    assert applied == ALL_VERSIONS
    assert {"cost_ledger", SCHEMA_MIGRATIONS} <= table_names(pool, schema)


@requires_postgres
def test_a_second_run_applies_nothing(pool: Pool, schema: str) -> None:
    """The acceptance criterion for `make migrate` being safe to run twice. Catches a
    runner with no `schema_migrations` bookkeeping at all - it would raise `DuplicateTable`
    on the second call, so a deploy that re-ran migrations would fail after the first -
    and one that records versions but never reads them back."""
    with pool.connection() as conn:
        first = apply_migrations(conn, schema=schema)
    with pool.connection() as conn:
        second = apply_migrations(conn, schema=schema)

    assert first == ALL_VERSIONS
    assert second == ()


@requires_postgres
def test_a_second_run_leaves_the_data_alone(pool: Pool, migrated_schema: str) -> None:
    """Catches a runner made idempotent the cheap way - dropping and recreating the table,
    or `TRUNCATE` before create - which passes the case above perfectly and destroys the
    cost rows of everything booked since the last deploy."""
    with pool.connection() as conn:
        conn.execute(
            f'insert into "{migrated_schema}".cost_ledger '
            "(ts, tenant, alias, provider, model, input_tokens, output_tokens, cached, "
            "cost_usd, price_version, estimated, trace_id) "
            "values (now(), 'evals', 'agent-default', 'local', 'm', 1, 1, false, "
            "0, 'v1', false, 't')"
        )

    with pool.connection() as conn:
        apply_migrations(conn, schema=migrated_schema)

    with pool.connection() as conn:
        count = conn.execute(f'select count(*) from "{migrated_schema}".cost_ledger').fetchone()
    assert count is not None
    assert count[0] == 1


@requires_postgres
def test_every_applied_version_is_recorded(pool: Pool, migrated_schema: str) -> None:
    """Catches bookkeeping written for some files and not others, which makes the runner
    re-apply exactly the migration it forgot on the next deploy. Compared against
    `migrations()` rather than against a literal, so the second SQL file an earlier change added
    is covered here without this case being edited, and so is every file after it."""
    assert applied_versions(pool, migrated_schema) == [
        migration.version for migration in migrations()
    ]


@requires_postgres
def test_the_migration_lands_only_in_the_named_schema(pool: Pool, schema: str) -> None:
    """Catches a runner that takes `schema` and then writes its DDL unqualified, landing
    everything in whatever `search_path` happens to be. It is what lets this directory
    isolate by schema at all, and in production it is the difference between the ledger
    and someone else's `public`. The neighbour schema really EXISTS: a name nothing
    created answers an empty table list whatever the runner did, which would make this
    case pass on its own."""
    neighbour = new_schema_name()
    with pool.connection() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(neighbour)))
    try:
        with pool.connection() as conn:
            apply_migrations(conn, schema=schema)

        assert "cost_ledger" in table_names(pool, schema)
        assert table_names(pool, neighbour) == set()
        assert "cost_ledger" not in table_names(pool, "public")
    finally:
        with pool.connection() as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(neighbour)))


# --------------------------------------------------------------------------
# the fifth migration removes residency
# --------------------------------------------------------------------------


def relation_columns(pool: Pool, schema: str, relation: str) -> set[str]:
    """Every column of a table or materialised view, from the catalogue.

    `pg_attribute` rather than `information_schema.columns`, which does not list a
    materialised view's columns at all.
    """
    with pool.connection() as conn:
        rows = conn.execute(
            """
            select a.attname
            from pg_attribute a
            join pg_class c on c.oid = a.attrelid
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s and c.relname = %s and a.attnum > 0 and not a.attisdropped
            """,
            (schema, relation),
        ).fetchall()
    return {str(row[0]) for row in rows}


@requires_postgres
def test_the_fifth_migration_drops_the_residency_log(pool: Pool, schema: str) -> None:
    """Catches a migration that leaves the log or one of its day partitions behind, or that
    never runs: after all five, no `residency_log*` relation is left. The fifth version is
    asserted as applied, so a runner that stopped at four cannot pass."""
    assert [m.version for m in migrations()] == list(ALL_VERSIONS)
    with pool.connection() as conn:
        apply_migrations(conn, schema=schema)

    assert FIFTH_VERSION in applied_versions(pool, schema)
    assert [name for name in table_names(pool, schema) if name.startswith("residency_log")] == []


@requires_postgres
def test_migration_007_drops_incidents_and_keeps_releases(pool: Pool, schema: str) -> None:
    """Catches a migration that never runs or drops the wrong table: after all migrations
    the `incidents` table is gone, while `releases` (Quality's markers and blocked releases
    read it) still exists. Both halves are asserted on one schema, so a `DROP` of both
    tables, or of neither, fails one of the two lines."""
    with pool.connection() as conn:
        apply_migrations(conn, schema=schema)

    assert SEVENTH_VERSION in applied_versions(pool, schema)
    with pool.connection() as conn:
        incidents = conn.execute("select to_regclass(%s)", (f"{schema}.incidents",)).fetchone()
        releases = conn.execute("select to_regclass(%s)", (f"{schema}.releases",)).fetchone()
    assert incidents is not None
    assert incidents[0] is None
    assert releases is not None
    assert releases[0] is not None


@requires_postgres
def test_the_ledger_and_eval_cases_have_no_region(pool: Pool, migrated_schema: str) -> None:
    """Catches a migration that drops `region` from the ledger but not from the rollup or
    `eval_cases`, or that rebuilds the rollup without its unique key: `REFRESH ... CONCURRENTLY`
    is refused by PostgreSQL without one, so the refresh at the end is what proves the key
    came back over the new group. A real column (`tenant`) is asserted on each relation, so
    an empty column read cannot pass."""
    for relation in ("cost_ledger", "eval_cases", "cost_ledger_daily"):
        columns = relation_columns(pool, migrated_schema, relation)
        assert "tenant" in columns or relation == "eval_cases", relation
        assert columns, relation
        assert "region" not in columns, relation

    with pool.connection() as conn:
        # A view built WITH NO DATA takes a plain refresh first; only then is
        # CONCURRENTLY legal, and that needs the rebuilt unique key.
        conn.execute(f'refresh materialized view "{migrated_schema}".cost_ledger_daily')
        conn.execute(
            f'refresh materialized view concurrently "{migrated_schema}".cost_ledger_daily'
        )


# --------------------------------------------------------------------------
# the locked runner the gateway calls at start
# --------------------------------------------------------------------------


@requires_postgres
def test_the_locked_runner_applied_twice_applies_nothing_the_second_time(
    pool: Pool, schema: str
) -> None:
    """Catches a locked runner that skips the version bookkeeping or re-runs files, so a
    gateway restarted on an up-to-date database would fail or change data."""
    with pool.connection() as conn:
        first = apply_migrations_locked(conn, schema=schema)
    with pool.connection() as conn:
        second = apply_migrations_locked(conn, schema=schema)

    assert first == ALL_VERSIONS
    assert second == ()


@requires_postgres
def test_two_locked_runners_at_once_apply_each_file_once(pool: Pool, schema: str) -> None:
    """Catches a runner with no lock (or a lock that gives up instead of waiting): two
    replicas starting together would both apply a file, or one would fail, so each version
    must be recorded once and the two answers together must hold every version with no
    overlap."""
    start = threading.Barrier(2)
    answers: list[tuple[str, ...]] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            with pool.connection() as conn:
                start.wait(timeout=10)
                answers.append(apply_migrations_locked(conn, schema=schema))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    assert len(answers) == 2
    together = [version for answer in answers for version in answer]
    assert sorted(together) == sorted(ALL_VERSIONS)
    assert sorted(applied_versions(pool, schema)) == sorted(ALL_VERSIONS)
