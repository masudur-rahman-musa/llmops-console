"""`004_console_events.sql` and `PostgresConsoleEventWriter` against a real server.

Every case here runs the SHIPPED migrations into a throwaway schema and then
either writes through the real writer or issues one raw statement, never against
a hand-built table. That is the point of the file: the migration and the
writer's INSERT are written in two places and nothing type-checks between them,
so a column added to one and not the other, or a positional binding that swapped
two `text` columns, is invisible to PostgreSQL and to mypy both.

The three raw-SQL cases are deliberate, not a shortcut. `severity`, `status` and
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

* `migrations/004_console_events.sql` creating `{schema}.incidents` and
  `{schema}.releases`, with a CHECK on `severity`, a CHECK on `status`, a
  nullable FK from `releases.eval_run_id` to `eval_runs (id)` and the composite
  primary key `(git_sha, deployed_at)` decision B fixes;
* `PostgresConsoleEventWriter(pool, *, schema=...)` with `write_incident`,
  `resolve_incident`, `write_release` and `eval_run_exists`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from psycopg import sql

from app.storage.console_events import (
    INCIDENTS,
    RELEASES,
    IncidentRow,
    PostgresConsoleEventWriter,
    ReleaseRow,
)

from .conftest import Pool, requires_postgres, table_names

STARTED = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)
RESOLVED = STARTED + timedelta(hours=3)
DEPLOYED = datetime(2026, 9, 7, 9, 15, 0, tzinfo=UTC)
REDEPLOYED = DEPLOYED + timedelta(hours=6)

INCIDENT_ID = "0af7651916cd43dd8448eb211c80319c"
GIT_SHA = "abc1234def5678"

UNSTORED_RUN_ID = "ffffffffffffffffffffffffffffffff"
"""A 32-hex id no `eval_runs` row in a fresh schema carries."""


def an_incident(**overrides: Any) -> IncidentRow:  # noqa: ANN401
    """One well-formed incident row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "id": INCIDENT_ID,
        "started_at": STARTED,
        "resolved_at": None,
        "severity": "sev2",
        "title": "gateway 5xx on the eu pool",
        "summary": "the eu pool answered 503 for eleven minutes",
        "postmortem_url": None,
    }
    fields.update(overrides)
    return IncidentRow(**fields)


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
def test_the_migration_creates_both_tables_in_the_schema(pool: Pool, migrated_schema: str) -> None:
    """Catches `004_console_events.sql` never being picked up at all - added to the
    repository but not to the directory `apply_migrations` globs, or named so it sorts
    before the `eval_runs` migration its foreign key references, either of which leaves
    every route below answering 500 against a real database while the whole in-memory
    suite stays green. The names are read from the module's own constants, so a table
    renamed in the SQL and not in the writer fails here rather than at the first INSERT.
    """
    assert {INCIDENTS, RELEASES} <= table_names(pool, migrated_schema)


@requires_postgres
def test_the_severity_check_refuses_a_fourth_word(pool: Pool, migrated_schema: str) -> None:
    """Catches the severity CHECK being left off the table, which makes the Python
    pre-check the only guard and therefore no guard at all for the two writers that do
    not go through it - a psql session, a later migration's backfill, the console's own
    seed data. This statement goes ROUND the pre-check on purpose; without the
    constraint PostgreSQL stores `sev4` happily and the timeline renders a severity no
    filter matches.
    """
    with pytest.raises(psycopg.errors.CheckViolation), pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (id, started_at, resolved_at, severity, title, summary, "
                "postmortem_url) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            ).format(sql.Identifier(migrated_schema, INCIDENTS)),
            (INCIDENT_ID, STARTED, None, "sev4", "a title", "a summary", None),
        )


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


@requires_postgres
def test_an_open_incident_stores_a_null_resolved_at_and_the_patch_sets_it(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches `resolved_at` declared NOT NULL or defaulted to `now()`, which makes
    every incident arrive already closed and leaves the timeline with nothing open to
    show; and catches `resolve_incident` issuing an UPDATE that touches the wrong column
    or no row, which the in-memory case cannot see because it has no SQL. NULL before
    and the exact timestamp after are asserted as a sequence on ONE row, so a writer
    that inserted a second row on resolve fails on the first read too.
    """
    writer = writer_on(pool, migrated_schema)
    column = sql.SQL("SELECT resolved_at FROM {} WHERE id = %s").format(
        sql.Identifier(migrated_schema, INCIDENTS)
    )

    writer.write_incident(an_incident(resolved_at=None))
    assert one_value(pool, column, (INCIDENT_ID,)) is None

    writer.resolve_incident(INCIDENT_ID, RESOLVED)
    assert one_value(pool, column, (INCIDENT_ID,)) == RESOLVED
