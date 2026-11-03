"""`013_eval_bug.sql` against a real server.

Only PostgreSQL can show the widened `result` check and the two new columns, so these cases
run the SHIPPED migrations into a throwaway schema and then write through the real store.
Skipped without `OPSCONSOLE_TEST_DATABASE_URL`.

Contract these cases are written against:

* the `eval_plays.result` check also takes `eval_bug`, and still refuses any other word;
* `eval_plays` gains `fault` and `fault_reason`, both nullable text, so a play stored before
  this change (or one the fault question was never asked about) keeps NULL in both.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from app.evals.report import build_report
from app.evals.results import PlayRecord
from app.storage.eval_store import (
    EVAL_PLAYS,
    PostgresEvalStore,
    play_rows_from_report,
    rows_from_report,
)

from .conftest import Pool, requires_postgres

STARTED = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
FINISHED = datetime(2026, 11, 2, 9, 20, tzinfo=UTC)
REPORT_PATH = Path("/srv/evals/nightly/report.json")


def a_play(case_id: str, result: str) -> PlayRecord:
    return PlayRecord.model_validate(
        {
            "case_id": case_id,
            "owner": "access",
            "play": 1,
            "result": result,
            "broken": (),
            "flagged": (),
            "turns": 1,
            "tool_calls": 0,
            "seconds": 2.0,
            "cost_usd": None,
            "checks": (),
            "conversation": (),
        }
    )


def store_plays(pool: Pool, schema: str, *plays: PlayRecord) -> None:
    report = build_report(
        plays,
        started_at=STARTED,
        finished_at=FINISHED,
        set_name="cases.jsonl",
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=None,
        tenant="all",
        target="https://helpdesk.example",
    )
    run_row, cases = rows_from_report(report, report_path=REPORT_PATH)
    PostgresEvalStore(pool, schema=schema).write(run_row, cases, play_rows_from_report(report))


def fault_columns(pool: Pool, schema: str) -> list[tuple[str, object]]:
    """Case id and `(fault, fault_reason)` of every stored play, by column name."""
    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL("select case_id, result, fault, fault_reason from {} order by case_id").format(
                sql.Identifier(schema, EVAL_PLAYS)
            )
        ).fetchall()
    return [(str(r[0]), tuple(r[1:])) for r in rows]


@requires_postgres
def test_an_eval_bug_play_with_a_fault_is_stored(pool: Pool, migrated_schema: str) -> None:
    """Catches a result check left at four words (an eval bug play then fails to store and
    stops the whole run's write), a missing `fault` or `fault_reason` column, and a check
    widened too far. An `eval_bug` play is stored; its fault and reason are written and
    read back; a play with no fault answer keeps NULL in both; and the word `maybe` is
    still refused by the table itself."""
    schema = migrated_schema
    store_plays(pool, schema, a_play("alpha", "eval_bug"), a_play("beta", "needs_a_look"))

    with pool.connection() as conn:
        conn.execute(
            sql.SQL("update {} set fault = %s, fault_reason = %s where case_id = %s").format(
                sql.Identifier(schema, EVAL_PLAYS)
            ),
            ("eval", "the person made up a login problem", "alpha"),
        )

    stored = fault_columns(pool, schema)
    assert stored == [
        ("alpha", ("eval_bug", "eval", "the person made up a login problem")),
        ("beta", ("needs_a_look", None, None)),
    ]

    with pytest.raises(psycopg.errors.CheckViolation), pool.connection() as conn:
        conn.execute(
            sql.SQL("update {} set result = %s").format(sql.Identifier(schema, EVAL_PLAYS)),
            ("maybe",),
        )
