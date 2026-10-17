r"""The connection seam and the migration runner: this project's first Postgres writes.

Until this module, Postgres existed here only as a readiness probe. Three things
arrive together, and every later Postgres write inherits them.

**The pool.** `build_pool` answers a `psycopg_pool.ConnectionPool`. The residency
log is a write on every served request, so a connect per write would put a TCP
handshake and an authentication round trip on the hot path. The readiness probe
in `app/api/routes/health.py` deliberately does NOT borrow from it: a probe that reported on
the pool would go green while the server behind it was gone.

**The runner.** Plain `.sql` files applied in file-name order, with a
`schema_migrations(version, applied_at)` table recording what has run. One
transaction per file, so a file that fails leaves nothing of itself behind and is
retried whole on the next `make migrate`. Chosen over Alembic: one
table's DDL does not pay for an env, autogenerate and a revision graph.

**The schema.** Every statement is qualified with a schema name the caller gives,
defaulting to `opsconsole`. A schema name cannot be a bound parameter - it is
interpolated into DDL - so `SCHEMA_PATTERN` and `MAX_SCHEMA_LENGTH` are checked
before the file is read and before a single statement runs. That check is also
what lets the test suite migrate into a throwaway schema per test rather than
rebuilding a database. The rejection table carries a trailing
newline, a trailing CRLF and a trailing space.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, NamedTuple

from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

DEFAULT_SCHEMA: Final[str] = "opsconsole"
"""Where the ledger's tables live."""

MIGRATIONS: Final[Path] = Path(__file__).resolve().parents[2] / "migrations"
"""The directory of `.sql` files. `parents[2]` is `backend/`, so this is
`backend/migrations/`, outside the package; the image COPYs it so it ships."""

SCHEMA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
"""Lowercase slug, matched with `re.fullmatch`.

The same shape as `routing.resolver.IDENTIFIER_PATTERN` and for a sharper reason:
this string is interpolated into DDL, so `.match` with a trailing `$` - which
accepts a trailing newline - would accept a schema name carrying a line break and
whatever followed it. Spelled here rather than imported because the two are
allowed to diverge: a routing identifier is a metric label, this is a database
object, and only one of them is a SQL injection seam.
"""

MAX_SCHEMA_LENGTH: Final[int] = 63
"""PostgreSQL's own identifier limit, in bytes, and the reason for the check.

A longer name is not rejected by the server - it is silently TRUNCATED to this
many bytes, so the caller creates its tables in a schema with a different name
from the one it asked for and `make purge-residency` then finds nothing to drop.
"""

SCHEMA_MIGRATIONS: Final[str] = "schema_migrations"
"""The bookkeeping table, one row per applied version."""

SCHEMA_PLACEHOLDER: Final[str] = "{schema}"
"""What a `.sql` file writes where the quoted schema name belongs."""

MIN_POOL_SIZE: Final[int] = 1
MAX_POOL_SIZE: Final[int] = 4
"""Small on purpose: one gateway replica issues one short INSERT per served
request, and a pool larger than the worker threads that can be in flight only
holds idle server-side sessions open."""

POOL_WAIT_TIMEOUT: Final[float] = 2.0
"""Seconds to wait for a free connection before raising.

A saturated pool would otherwise hold a request indefinitely, masking the
saturation as a slow response. Two seconds is generous for the one short INSERT
this pool is used for; if a connection is not free in two seconds, the database
is the problem and the caller should know.
"""

DB_CONNECT_TIMEOUT: Final[float] = 5.0
"""Seconds to wait for a new TCP connection to PostgreSQL.

Matches the adapter connect timeout: an unanswered TCP open is a dead
host, and five seconds is already generous. Used in `connect_timeout` of the
psycopg conninfo, not the pool wait.
"""

STATEMENT_TIMEOUT_MS: Final[int] = 2000
"""Maximum milliseconds any one statement may run before PostgreSQL cancels it.

Set as `options='-c statement_timeout=2000ms'` in the conninfo so every
connection from the pool carries it, not just one.
"""


class Migration(NamedTuple):
    """One `.sql` file: its version, which is the file stem, and its text."""

    version: str
    sql: str


def migrations() -> tuple[Migration, ...]:
    """Every migration that ships, ordered by file name.

    The file stem IS the version string recorded in `schema_migrations`, so
    renaming an applied file makes it run again as a new version. Ordered by
    name rather than by mtime, because a checkout has no useful mtimes.
    """
    return tuple(
        Migration(version=path.stem, sql=path.read_text(encoding="utf-8"))
        for path in sorted(MIGRATIONS.glob("*.sql"))
    )


def checked_schema(schema: str) -> str:
    """The schema name, or `ValueError` before it can reach a statement.

    Raised rather than returned as a typed gateway error because nothing a client
    sends decides this: the schema is an operator's configuration, and a bad one
    is a misconfiguration at startup rather than a request to refuse.
    """
    if len(schema.encode("utf-8")) > MAX_SCHEMA_LENGTH:
        raise ValueError(f"schema: longer than {MAX_SCHEMA_LENGTH} bytes")
    if SCHEMA_PATTERN.fullmatch(schema) is None:
        raise ValueError(f"schema: {schema!r} is not a lowercase slug")
    return schema


def _rendered(text: str, schema: str) -> sql.SQL:
    """One migration file with its schema placeholder filled in.

    `sql.SQL` over an interpolated string is safe here and nowhere else in this
    package: `schema` has already been through `checked_schema`, so the only
    characters that can reach the statement are lowercase letters, digits, `-`
    and `_`. Every other value this package sends is a bound parameter.
    """
    return sql.SQL(text.replace(SCHEMA_PLACEHOLDER, f'"{schema}"'))


def build_pool(database_url: str) -> ConnectionPool[Connection[TupleRow]]:
    """An opened connection pool for the residency log and the cost ledger.

    Three timeouts:
    - `timeout` (pool wait): 2 s — a connection not free in 2 s means the pool
      is saturated; the caller gets an error rather than waiting indefinitely.
    - `connect_timeout` in the conninfo: 5 s — matches the adapter connect
      timeout; an unanswered TCP open is a dead host.
    - `statement_timeout` in the conninfo: 2000 ms — PostgreSQL cancels any
      statement that runs longer than 2 seconds, so a stuck query gives up fast.
    """
    conninfo = (
        f"{database_url} "
        f"connect_timeout={int(DB_CONNECT_TIMEOUT)} "
        f"options='-c statement_timeout={STATEMENT_TIMEOUT_MS}ms'"
    )
    return ConnectionPool(
        conninfo,
        min_size=MIN_POOL_SIZE,
        max_size=MAX_POOL_SIZE,
        open=True,
        timeout=POOL_WAIT_TIMEOUT,
    )


def apply_migrations(
    conn: Connection[TupleRow], *, schema: str = DEFAULT_SCHEMA
) -> tuple[str, ...]:
    """Apply every migration this schema has not seen, and answer the ones applied.

    Answers what THIS call applied, so a second call answers `()` - which is the
    whole of "running `make migrate` twice applies nothing the second time and
    leaves the data alone". Idempotence comes from the
    bookkeeping table, never from dropping and recreating: the second shape
    passes the first of those two cases and destroys every row logged since the
    last deploy.

    One transaction per file, so a migration that fails part way leaves neither
    its DDL nor its `schema_migrations` row behind.

    Raises `ValueError` for a schema name that is not a lowercase slug, before
    any statement runs at all.
    """
    checked = checked_schema(schema)
    conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(checked)))
    conn.execute(
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "version text PRIMARY KEY, "
            "applied_at timestamptz NOT NULL DEFAULT now())"
        ).format(sql.Identifier(checked, SCHEMA_MIGRATIONS))
    )
    rows = conn.execute(
        sql.SQL("SELECT version FROM {}").format(sql.Identifier(checked, SCHEMA_MIGRATIONS))
    ).fetchall()
    already = {str(row[0]) for row in rows}

    applied: list[str] = []
    for migration in migrations():
        if migration.version in already:
            continue
        with conn.transaction():
            conn.execute(_rendered(migration.sql, checked))
            conn.execute(
                sql.SQL("INSERT INTO {} (version) VALUES (%s)").format(
                    sql.Identifier(checked, SCHEMA_MIGRATIONS)
                ),
                (migration.version,),
            )
        applied.append(migration.version)
    return tuple(applied)
