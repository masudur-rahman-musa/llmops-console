"""`004_console_events.sql` and `PostgresConsoleEventWriter` against a real server.

Every case here runs the SHIPPED migrations into a throwaway schema and then
either writes through the real writer or issues one raw statement, never against
a hand-built table. That is the point of the file: the migration and the
writer's INSERT are written in two places and nothing type-checks between them,
so a column added to one and not the other, or a positional binding that swapped
two `text` columns, is invisible to PostgreSQL and to mypy both.

The raw-SQL cases are deliberate, not a shortcut. `status` and
`eval_run_id` each have a Python pre-check in front of them, so a case going
through the writer would be refused before any statement ran and would prove
only that the pre-check exists - which tests/storage/test_console_events.py
already proves. Going round the pre-check is the ONLY way to ask whether the
table itself would have stopped the value, which is the question a migration
case exists to answer: the pre-check is the message, the constraint is the
guarantee.

Skipped without `OPSCONSOLE_TEST_DATABASE_URL`. One
session pool, one throwaway schema per test - the fixtures in
tests/storage/conftest.py, unchanged.

The contract these cases are written against, before it exists:

* `migrations/004_console_events.sql` creating `{schema}.releases`, with a CHECK on `status`, a
  nullable FK from `releases.eval_run_id` to `eval_runs (id)` and the composite
  primary key `(git_sha, deployed_at)` decision B fixes;
* `PostgresConsoleEventWriter(pool, *, schema=...)` with `write_release` and `eval_run_exists`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from psycopg import sql

from app.storage.console_events import (
    RELEASES,
    PostgresConsoleEventWriter,
    ReleaseRow,
)

from .conftest import Pool, requires_postgres, table_names

DEPLOYED = datetime(2026, 9, 7, 9, 15, 0, tzinfo=UTC)
REDEPLOYED = DEPLOYED + timedelta(hours=6)

GIT_SHA = "abc1234def5678"

UNSTORED_RUN_ID = "ffffffffffffffffffffffffffffffff"
"""A 32-hex id no `eval_runs` row in a fresh schema carries."""


def a_release(**overrides: Any) -> ReleaseRow:  # noqa: ANN401
    """One well-formed release row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "git_sha": GIT_SHA,
        "deployed_at": DEPLOYED,
        "eval_run_id": None,
        "pr_url": None,
        "status": "deployed",
    }
    fields.update(overrides)
    return ReleaseRow(**fields)


def writer_on(pool: Pool, schema: str) -> PostgresConsoleEventWriter:
    """The real writer, pointed at this test's own schema."""
    return PostgresConsoleEventWriter(pool, schema=schema)


def one_value(pool: Pool, statement: sql.Composed, params: tuple[Any, ...]) -> Any:  # noqa: ANN401
    """The first column of the first row, for the cases asserting one stored value."""
    with pool.connection() as conn:
        row = conn.execute(statement, params).fetchone()
    assert row is not None
    return row[0]


@requires_postgres
def test_the_migration_creates_the_releases_table_in_the_schema(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches `004_console_events.sql` never being picked up at all - added to the
    repository but not to the directory `apply_migrations` globs, or named so it sorts
    before the `eval_runs` migration its foreign key references, either of which leaves
    every route below answering 500 against a real database while the whole in-memory
    suite stays green. The names are read from the module's own constants, so a table
    renamed in the SQL and not in the writer fails here rather than at the first INSERT.
    """
    assert RELEASES in table_names(pool, migrated_schema)


@requires_postgres
def test_the_status_check_refuses_an_unknown_word(pool: Pool, migrated_schema: str) -> None:
    """The same gap on `releases.status`, and it is asserted separately because one
    CHECK written and the other forgotten is exactly the shape a two-table migration
    fails in. `rolledback` is one character from a legal value, so a constraint spelled
    with the wrong member list - or no constraint - stores a status the deploys page
    cannot group by.
    """
    with pytest.raises(psycopg.errors.CheckViolation), pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (git_sha, deployed_at, eval_run_id, pr_url, status) "
                "VALUES (%s, %s, %s, %s, %s)"
            ).format(sql.Identifier(migrated_schema, RELEASES)),
            (GIT_SHA, DEPLOYED, None, None, "rolledback"),
        )


@requires_postgres
def test_two_deploys_of_one_sha_are_two_rows(pool: Pool, migrated_schema: str) -> None:
    """Catches decision B being implemented as a `git_sha` primary key with an upsert.
    Under that shape the second write here would not raise - it would OVERWRITE the
    first deploy's timestamp, so a rollback that re-deploys an earlier sha erases the
    original deploy and the deploys page shows one row where two happened. Only a count
    of 2 tells the two shapes apart; a count of 1 with no exception is precisely the
    wrong implementation.
    """
    writer = writer_on(pool, migrated_schema)

    writer.write_release(a_release(deployed_at=DEPLOYED))
    writer.write_release(a_release(deployed_at=REDEPLOYED))

    stored = one_value(
        pool,
        sql.SQL("SELECT count(*) FROM {} WHERE git_sha = %s").format(
            sql.Identifier(migrated_schema, RELEASES)
        ),
        (GIT_SHA,),
    )
    assert stored == 2


@requires_postgres
def test_a_release_naming_an_unstored_run_is_refused_by_the_foreign_key(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches `eval_run_id` declared as a plain `text` column with no REFERENCES
    clause. The route's pre-check would still answer 422 for an unknown run, so the
    gateway suite passes either way; what the FK protects is every other writer - CI
    posting a run id from a job whose eval rows never stored, a manual insert - and
    without it the console joins a release to nothing and shows a deploy with no quality
    number, silently. The statement is raw so the pre-check cannot be what refuses it.
    """
    with pytest.raises(psycopg.errors.ForeignKeyViolation), pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (git_sha, deployed_at, eval_run_id, pr_url, status) "
                "VALUES (%s, %s, %s, %s, %s)"
            ).format(sql.Identifier(migrated_schema, RELEASES)),
            (GIT_SHA, DEPLOYED, UNSTORED_RUN_ID, None, "deployed"),
        )


@requires_postgres
def test_a_release_with_a_null_eval_run_id_is_stored(pool: Pool, migrated_schema: str) -> None:
    """Catches the foreign key declared NOT NULL, and catches a writer that binds the
    empty string for an absent run. Both make a sha deployed before any eval unstorable
    or stored as `''`, which passes no FK and reads as a run id that does not exist
    rather than as "no run"; the assertion is SQL NULL specifically, so `''` fails here.
    """
    writer = writer_on(pool, migrated_schema)

    writer.write_release(a_release(eval_run_id=None))

    stored = one_value(
        pool,
        sql.SQL("SELECT eval_run_id FROM {} WHERE git_sha = %s").format(
            sql.Identifier(migrated_schema, RELEASES)
        ),
        (GIT_SHA,),
    )
    assert stored is None
