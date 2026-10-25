"""Fixtures for the ledger suite: one pool for the session, one schema per test.

Most files here need a real PostgreSQL. It is required by an
explicit conditional on `OPSCONSOLE_TEST_DATABASE_URL`: the
`requires_postgres` marker below skips with that variable's NAME in the reason,
so a run that checked nothing says so in its own output rather than leaving it
to be inferred. Deliberately not `OPSCONSOLE_DATABASE_URL`, which `Settings`
itself reads and which the root `tests/conftest.py` clears for the whole session: exporting that one to run this directory would hand every
app in the suite a `postgres` readiness probe and a Postgres ledger writer
neither asked for.

a project rule: ONE session-scoped pool, opened once. Per-test isolation is a
unique SCHEMA that `apply_migrations` runs into - never a fresh database, never a
fresh pool, never a wipe of a shared schema, which would delete a neighbouring
suite's tables. The schema name is a uuid4 hex under a fixed prefix, so it is a
lowercase slug and satisfies `SCHEMA_PATTERN` without the fixture having to
special-case anything.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

from app.storage.db import apply_migrations, build_pool

Pool = ConnectionPool[Connection[TupleRow]]
"""The pool type `build_pool` answers, spelled once so the fixtures agree."""

DATABASE_URL_VAR = "OPSCONSOLE_TEST_DATABASE_URL"
"""The switch, named so `Settings` can never read it."""

DATABASE_URL = os.environ.get(DATABASE_URL_VAR)

requires_postgres = pytest.mark.skipif(
    DATABASE_URL is None,
    reason=f"{DATABASE_URL_VAR} is not set, so no PostgreSQL is available",
)

SCHEMA_PREFIX = "opsconsole_test"


def new_schema_name() -> str:
    """A schema name no other test is using, and a lowercase slug by construction."""
    return f"{SCHEMA_PREFIX}_{uuid.uuid4().hex}"


@pytest.fixture(scope="session")
def pool() -> Iterator[Pool]:
    """The one pool this directory uses, opened once."""
    if DATABASE_URL is None:
        pytest.skip(f"{DATABASE_URL_VAR} is not set, so no PostgreSQL is available")
    built = build_pool(DATABASE_URL)
    try:
        yield built
    finally:
        built.close()


@pytest.fixture
def schema(pool: Pool) -> Iterator[str]:
    """An empty schema of this test's own, removed with everything in it afterwards."""
    name = new_schema_name()
    with pool.connection() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    try:
        yield name
    finally:
        with pool.connection() as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))


@pytest.fixture
def migrated_schema(pool: Pool, schema: str) -> str:
    """`schema` with the shipped migrations applied, for the cases about the table."""
    with pool.connection() as conn:
        apply_migrations(conn, schema=schema)
    return schema


def table_names(pool: Pool, schema: str) -> set[str]:
    """Every relation in one schema, partitions included.

    `pg_class` rather than `information_schema.tables`, because a partition is a
    relation the latter reports too but with a kind this helper's callers want to
    ask about separately.
    """
    with pool.connection() as conn:
        rows = conn.execute(
            """
            select c.relname
            from pg_class c
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s and c.relkind in ('r', 'p')
            """,
            (schema,),
        ).fetchall()
    return {str(row[0]) for row in rows}
