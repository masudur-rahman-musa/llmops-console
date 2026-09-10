"""`PostgresEvalStore` and `003_eval_runs.sql` against a real server.

Every case here runs the SHIPPED migrations into a throwaway schema and then
writes through the real store, never against a hand-built table. That is the
whole point of the file: the SQL file and the store's two statements are written
in two different places and nothing type-checks between them, so a column added
to one and not the other, or a positional binding that swaps two `text` columns,
is invisible to PostgreSQL and to mypy both. The `numeric` scale is the same gap
in the other direction - a column narrower than six places rounds a cost the
ledger computed exactly, and a run's spend then stops matching the ledger's own.

These cases live in tests/ledger/ rather than beside the pure ones (an earlier change
decision D): a project rule wants ONE session-scoped pool for the suite, and the
fixtures in tests/ledger/conftest.py already have it. Per-test isolation is a
throwaway SCHEMA, never a fresh pool and never a wipe. Skipped without
`OPSCONSOLE_TEST_DATABASE_URL`.

The statement-level view of the same store - that the pre-checks refuse before a
connection is borrowed at all - is in tests/evals/test_store.py, where a pool
double can see it; a real server cannot tell a statement that was never issued
from one that was rejected. The `run_kind` CHECK below is the mirror image: the
store's pre-check means no write can ever reach it, so it is exercised with an
UPDATE rather than through the store, and it is the guard that still stands when
somebody loads a row with `psql`.

The contract these cases are written against, before it exists:

* `003_eval_runs.sql` creates `{schema}.eval_runs` and `{schema}.eval_cases` with
  the columns named below, `eval_cases.run_id` a foreign key to `eval_runs.id`
  ON DELETE CASCADE, an index on `eval_cases (run_id)` and one on
  `eval_runs (started_at DESC)`, and a CHECK over `run_kind`'s three words
  (decision C);
* `opsconsole.evals.store` publishes `EVAL_RUNS` and `EVAL_CASES`, the two table
  names its statements build their identifiers from;
* `PostgresEvalStore(pool, *, schema=...).write(run_row, case_rows)` writes both
  in ONE transaction (decision G).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from opsconsole.evals.assertions import AssertionResult, AssertionStatus, CaseAssertions
from opsconsole.evals.cases import EvalCategory
from opsconsole.evals.outcome import CaseOutcome
from opsconsole.evals.report import REPORT_JSON_NAME, RunReport, build_report
from opsconsole.evals.residency import ResidencyVerdict
from opsconsole.evals.runner import CaseRun, RunResult
from opsconsole.evals.store import EVAL_CASES, EVAL_RUNS, PostgresEvalStore, rows_from_report

from .conftest import Pool, requires_postgres, table_names

STARTED = datetime(2026, 3, 4, 9, 12, 31, 204512, tzinfo=UTC)
FINISHED = datetime(2026, 3, 4, 9, 14, 2, 881044, tzinfo=UTC)

SET_NAME = "helpdesk-golden-v1.jsonl"
SET_SHA = "e4b2c0d9a71f3856cbb4d20e9f61a7c845d3e0b2"
GIT_SHA = "b71f0c3a9d5e42188a6c0f37d21be9054c8a7f6d"
TENANT = "nordlicht"
TARGET = "https://helpdesk.example/gateway"
PROVIDER = "anthropic_eu"
REGION = "eu"
REPORT_PATH = Path("/srv/evals/nightly") / REPORT_JSON_NAME

COST = Decimal("1234.567891")
"""Six decimal places with a digit in every one of them: a column declared
`numeric(18, 2)` rounds this to 1234.57 and PostgreSQL says nothing about it."""

LATENCY_MS = 137

EVAL_RUNS_COLUMNS = (
    "id",
    "started_at",
    "finished_at",
    "set_name",
    "set_sha",
    "set_sha_source",
    "run_kind",
    "target",
    "tenant",
    "git_sha",
    "metrics",
    "passed",
    "report_path",
    "alias_resolution",
    "judge_kappa",
)

EVAL_CASES_COLUMNS = (
    "run_id",
    "case_id",
    "category",
    "passed",
    "tags",
    "latency_ms",
    "cost_usd",
    "provider",
    "region",
    "details",
)


def make_run(case_id: str, *, verdict: str, category: EvalCategory) -> CaseRun:
    """One per-case record, as `run_set` would have built it.

    `broken` is the turn that never answered: no assertions, no verdict and no
    measurements at all, which is the row whose four columns must reach the table
    as NULL.
    """
    if verdict == "broken":
        return CaseRun(
            case_id=case_id,
            category=category,
            assertions=None,
            residency=None,
            outcome=None,
            error=f"turn failed: HTTP 502 for {case_id}",
            passed=False,
        )
    return CaseRun(
        case_id=case_id,
        category=category,
        assertions=CaseAssertions(
            case_id=case_id,
            results=(AssertionResult(name="intent", status=AssertionStatus.PASSED),),
        ),
        residency=ResidencyVerdict(
            case_id=case_id,
            tenant=TENANT,
            data_class="internal",
            observed=REGION,
            allowed=("eu", "self"),
            violation=False,
        ),
        outcome=CaseOutcome(
            reply="Done.",
            latency_ms=LATENCY_MS,
            cost_usd=COST,
            provider=PROVIDER,
            region=REGION,
        ),
        error=None,
        passed=True,
    )


RESULT = RunResult(
    runs=(
        make_run("alpha", verdict="pass", category=EvalCategory.HELPDESK),
        make_run("delta", verdict="broken", category=EvalCategory.INJECTION),
    )
)
"""One case of each shape a stored run has to survive: measured, and never answered."""


def a_report(*, run_kind: str = "ci") -> RunReport:
    """One built report, with a fresh run id of its own (`16c` builds one per call)."""
    return build_report(
        RESULT,
        started_at=STARTED,
        finished_at=FINISHED,
        set_name=SET_NAME,
        set_sha=SET_SHA,
        set_sha_source="commit",
        run_kind=run_kind,
        git_sha=GIT_SHA,
        tenant=TENANT,
        target=TARGET,
    )


def run_rows(pool: Pool, schema: str) -> list[tuple[object, ...]]:
    """Every `eval_runs` row, in the column order the migration declares.

    Selecting by NAME rather than `select *` is deliberate: a column renamed in
    the SQL file is a failure here rather than a console page that renders
    nothing.
    """
    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL("select {} from {} order by started_at").format(
                sql.SQL(", ").join(sql.Identifier(name) for name in EVAL_RUNS_COLUMNS),
                sql.Identifier(schema, EVAL_RUNS),
            )
        ).fetchall()
    return [tuple(row) for row in rows]


def case_rows(pool: Pool, schema: str) -> list[tuple[object, ...]]:
    """Every `eval_cases` row, ordered by the case id so two reads compare directly."""
    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL("select {} from {} order by case_id").format(
                sql.SQL(", ").join(sql.Identifier(name) for name in EVAL_CASES_COLUMNS),
                sql.Identifier(schema, EVAL_CASES),
            )
        ).fetchall()
    return [tuple(row) for row in rows]


def case_column(row: tuple[object, ...], name: str) -> object:
    """One case row's value by column name, so a case reads as the column it means."""
    return row[EVAL_CASES_COLUMNS.index(name)]


def store_report(pool: Pool, schema: str, report: RunReport) -> None:
    """Derive both rows from a report and write them through the real store."""
    run_row, cases = rows_from_report(report, report_path=REPORT_PATH)
    PostgresEvalStore(pool, schema=schema).write(run_row, cases)


# --------------------------------------------------------------------------
# what the migration builds
# --------------------------------------------------------------------------


@requires_postgres
def test_both_tables_exist_with_their_foreign_key_and_their_two_indexes(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches the three halves of the migration that nothing else can see at rest. Two
    tables and no foreign key means a case row can point at a run that never existed, and
    deleting a run leaves its cases behind forever - ON DELETE CASCADE is asserted by its
    catalogue code rather than by the constraint's name, so a key that references the
    right table with the wrong action fails. The two indexes are the two questions this
    table is ever asked (a run's cases, and the most recent runs); without them the
    console's eval page sequentially scans every run ever stored, which is a page that
    gets slower every night and never fails."""
    assert {EVAL_RUNS, EVAL_CASES} <= table_names(pool, migrated_schema)

    with pool.connection() as conn:
        keys = conn.execute(
            """
            select c.confdeltype, c.confrelid::regclass::text
            from pg_constraint c
            join pg_class t on t.oid = c.conrelid
            join pg_namespace n on n.oid = t.relnamespace
            where n.nspname = %s and t.relname = %s and c.contype = 'f'
            """,
            (migrated_schema, EVAL_CASES),
        ).fetchall()
        indexes = conn.execute(
            "select tablename, indexdef from pg_indexes where schemaname = %s",
            (migrated_schema,),
        ).fetchall()

    assert len(keys) == 1
    assert str(keys[0][0]) == "c"
    assert EVAL_RUNS in str(keys[0][1])

    case_indexes = [str(row[1]) for row in indexes if str(row[0]) == EVAL_CASES]
    run_indexes = [str(row[1]) for row in indexes if str(row[0]) == EVAL_RUNS]
    assert any("(run_id)" in definition for definition in case_indexes)
    assert any("started_at DESC" in definition for definition in run_indexes)


@requires_postgres
def test_the_run_kind_check_refuses_a_fourth_value(pool: Pool, migrated_schema: str) -> None:
    """Decision C's CHECK, and the only guard left once somebody edits a row by hand or a
    later writer forgets the pre-check. Catches `run_kind` declared as plain `text`: row
    18 filters on these three words, so a fourth one is a run that no query ever counts
    and no constraint ever complained about. Written as an UPDATE of a row the store
    itself wrote, because the store's own pre-check means no INSERT can ever carry a
    fourth value - the row is legal until this statement, so the CHECK is the only thing
    that can refuse it."""
    store_report(pool, migrated_schema, a_report())

    with pytest.raises(psycopg.errors.CheckViolation), pool.connection() as conn:
        conn.execute(
            sql.SQL("update {} set run_kind = %s").format(
                sql.Identifier(migrated_schema, EVAL_RUNS)
            ),
            ("smoke",),
        )


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------


@requires_postgres
def test_a_report_round_trips_into_the_two_tables(pool: Pool, migrated_schema: str) -> None:
    """The plan's schema-drift risk in one case. Catches a statement whose column list
    disagrees with the SQL file: `set_name`, `set_sha`, `git_sha`, `tenant` and `target`
    are all `text`, so two of them swapped is a run attributed to the wrong tenant and
    PostgreSQL accepts it without a murmur. `metrics` is asserted as the same mapping
    because it is what the eval gate reads back, and a `jsonb` column handed a Python dict with
    no adapter is the first thing that breaks here. Decision F's two columns are asserted
    NULL in the same row, so a store that filled them with zeros fails."""
    report = a_report()

    store_report(pool, migrated_schema, report)

    assert run_rows(pool, migrated_schema) == [
        (
            report.id,
            STARTED,
            FINISHED,
            SET_NAME,
            SET_SHA,
            "commit",
            "ci",
            TARGET,
            TENANT,
            GIT_SHA,
            dict(report.metrics),
            False,
            str(REPORT_PATH),
            None,
            None,
        )
    ]


@requires_postgres
def test_the_cost_reads_back_as_a_decimal_at_six_places(pool: Pool, migrated_schema: str) -> None:
    """The criterion, and the plan's `numeric` risk. Catches a column narrower than six
    decimal places, which rounds a cost the ledger computed exactly and makes a run's
    spend stop reconciling with `cost_ledger`, and catches `double precision`, which
    round-trips this value to something that merely prints the same. The exponent is
    asserted because `Decimal("1234.5678910") == COST`, so equality alone cannot see a
    scale that changed; the type is asserted because a float would compare equal to a
    `Decimal` for kinder numbers than this one."""
    store_report(pool, migrated_schema, a_report())

    stored = case_column(case_rows(pool, migrated_schema)[0], "cost_usd")

    assert isinstance(stored, Decimal)
    assert stored == COST
    assert stored.as_tuple().exponent == -6


FAILING_RESULT = RunResult(
    runs=(
        CaseRun(
            case_id="beta",
            category=EvalCategory.ACCESS_AUTO,
            assertions=CaseAssertions(
                case_id="beta",
                results=(
                    AssertionResult(name="intent", status=AssertionStatus.PASSED),
                    AssertionResult(
                        name="decision",
                        status=AssertionStatus.FAILED,
                        expected="auto_approved",
                        actual="escalated",
                    ),
                ),
            ),
            residency=ResidencyVerdict(
                case_id="beta",
                tenant=TENANT,
                data_class="internal",
                observed=REGION,
                allowed=("eu", "self"),
                violation=False,
            ),
            outcome=CaseOutcome(
                reply="Done.",
                latency_ms=LATENCY_MS,
                cost_usd=COST,
                provider=PROVIDER,
                region=REGION,
            ),
            error=None,
            passed=False,
        ),
    )
)
"""One failing case, built by hand rather than by `make_run`: `run_set` never
builds a `CaseRun` with fewer than eleven assertion results, but `tags_for` only
needs the one that failed, and this is the shortest record that produces exactly
one tag to read back."""


@requires_postgres
def test_the_tags_column_reads_back_the_case_reports_tags_in_order(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches `tags` written as an unordered set, a comma-joined string, or the
    category repeated into every row instead of the report's own computed tags.
    PostgreSQL's `text[]` preserves the order it was given, so a store that built
    the array from a Python `set` reads this back shuffled at least some of the
    time, and a store that never called `taxonomy.tags_for` at all reads back an
    empty array on a case that failed."""
    report = build_report(
        FAILING_RESULT,
        started_at=STARTED,
        finished_at=FINISHED,
        set_name=SET_NAME,
        set_sha=SET_SHA,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=GIT_SHA,
        tenant=TENANT,
        target=TARGET,
    )

    store_report(pool, migrated_schema, report)

    stored = case_column(case_rows(pool, migrated_schema)[0], "tags")
    assert stored == ["wrong_decision"]


@requires_postgres
def test_every_case_row_lands_with_its_own_measurements_and_its_details(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches the four measured columns declared NOT NULL, which would refuse exactly the
    rows a failing target produces, and a store that wrote `0` and `''` into them instead
    - a broken turn would then be averaged in as a free, instant, EU-served case and lift
    every number `evals/metrics.py` reports. `tags` is asserted as the empty array
    rather than as NULL
    (decision F), because a NULL array is a value every later `unnest` has to know about."""
    store_report(pool, migrated_schema, a_report())

    rows = case_rows(pool, migrated_schema)
    measured = next(row for row in rows if case_column(row, "case_id") == "alpha")
    broken = next(row for row in rows if case_column(row, "case_id") == "delta")

    assert case_column(measured, "category") == EvalCategory.HELPDESK.value
    assert case_column(measured, "passed") is True
    assert case_column(measured, "latency_ms") == LATENCY_MS
    assert case_column(measured, "provider") == PROVIDER
    assert case_column(measured, "region") == REGION
    assert case_column(measured, "tags") == []

    assert case_column(broken, "passed") is False
    assert case_column(broken, "latency_ms") is None
    assert case_column(broken, "cost_usd") is None
    assert case_column(broken, "provider") is None
    assert case_column(broken, "region") is None
    details = case_column(broken, "details")
    assert isinstance(details, dict)
    assert details["error"] == "turn failed: HTTP 502 for delta"


# --------------------------------------------------------------------------
# two runs, and a write that fails part way (decision G)
# --------------------------------------------------------------------------


@requires_postgres
def test_two_runs_of_the_same_set_are_two_rows_whose_cases_point_at_their_own(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches a run id derived from the set, the day or the commit rather than generated
    per run: two nightly runs of the same golden set would collide on the primary key, and
    the second would either fail or overwrite the first - which is the comparison the whole
    table exists to support. Catches the other end too, a case row keyed to the LAST run
    written, by asserting each run's case ids against its own id."""
    first = a_report()
    second = a_report()

    store_report(pool, migrated_schema, first)
    store_report(pool, migrated_schema, second)

    assert first.id != second.id
    assert {str(row[0]) for row in run_rows(pool, migrated_schema)} == {first.id, second.id}

    by_run: dict[str, set[str]] = {first.id: set(), second.id: set()}
    for row in case_rows(pool, migrated_schema):
        by_run[str(case_column(row, "run_id"))].add(str(case_column(row, "case_id")))
    assert by_run == {first.id: {"alpha", "delta"}, second.id: {"alpha", "delta"}}


@requires_postgres
def test_a_write_that_fails_part_way_leaves_neither_table_changed(
    pool: Pool, migrated_schema: str
) -> None:
    """Decision G's whole reason. Catches a store issuing the run INSERT and the case
    INSERTs in separate transactions - or in none - which leaves a run row whose cases are
    missing, and every later query counts that as a run with fewer cases and cannot tell
    it from a shorter golden set. The failure is a latency no integer column can hold,
    which no pre-check in the plan covers, so the statement really does fail AFTER the run
    row has been issued; both tables are asserted empty, because "the run row survived" is
    exactly the corruption."""
    report = a_report()
    run_row, cases = rows_from_report(report, report_path=REPORT_PATH)
    doomed = (cases[0], cases[1].model_copy(update={"latency_ms": 2**70}))

    with pytest.raises(psycopg.Error):
        PostgresEvalStore(pool, schema=migrated_schema).write(run_row, doomed)

    assert run_rows(pool, migrated_schema) == []
    assert case_rows(pool, migrated_schema) == []
