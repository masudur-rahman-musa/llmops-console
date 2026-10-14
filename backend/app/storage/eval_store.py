r"""The two rows a run stores: `eval_runs` and `eval_cases` (#39, part).

`16c` writes the two report files; this module writes the two rows built from the
same `RunReport`. It lives in `storage/` with the other Postgres writers - the
folder restructure gathered every Postgres write here, reversing plan
16d decision A's placement in `evals/`. It still reads the record it writes from
`evals.report`, so the dependency points from `storage` up into `evals`; there is
no import contract forbidding it. The migration sits in `migrations/` for
`storage.db.apply_migrations` to find it, and this module imports `storage.db`.

Three things it holds that no single field states:

* **`rows_from_report` is pure and total.** Every column comes from the report and
  nothing is recomputed or re-timed, so the row and the file a run leaves behind
  cannot disagree about the same run
  (`test_the_run_row_carries_every_identity_field_from_the_report`).
* **A case that measured nothing stores NULL, never `0` and never `""`.** A turn
  that never answered has no latency, cost, provider or region; a zero there
  would be averaged in as a free, instant, in-region case and would lift every
  number `evals/scores.py` reports exactly when the target is failing
  (`test_a_case_whose_turn_failed_stores_no_measurements_and_its_error`).
* **Every pre-check runs before a connection is borrowed**,
  which `test_write_refuses_a_malformed_tenant_before_any_sql` asserts as "no
  connection was borrowed at all" rather than as "the INSERT errored". As with
  the cost ledger, each value reaches the statement as a bound parameter, so the
  checks are not what stands between the table and an injection - they are what
  keeps the audit rows honest: `nordlicht\n` and `Nordlicht` are two tenants a
  later query counts as one and as none.

`run_kind` is checked against `report.RunKind`'s own three words rather than
against `app.cli.evals.RUN_KINDS`, because `app.cli.evals` imports this module
and the reverse import would be a cycle;
`test_each_run_kind_the_command_accepts_is_written` is
parametrised over the command's tuple, so the two cannot drift apart in silence.
The CHECK constraint in `003_eval_runs.sql` is the second guard, for the row
somebody loads with `psql`.

`tags` carries row 17b's failure taxonomy (#40), read straight off the report's
own `CaseReport.tags` rather than recomputed here, so the column and the file
cannot disagree about why a case failed
(`test_rows_from_report_fills_tags_from_the_report_case_by_case`).
`metrics` carries row 17d's fixed metric set the same way, in the report's own
JSON form: the two costs are `Decimal` in the mapping and decimal STRINGS in the
column, exactly as they are in the file
(`test_the_stored_metrics_are_the_json_the_report_file_carries`).
`alias_resolution` still ships NULL (decision F), and the migration says so
column by column. `judge_kappa` is filled since 17f, off the report's own
`RunCalibration` record and never out of the `metrics` mapping - a typed field is
what keeps `mypy --strict` honest about a value going into a `double precision`
column, and a mapping lookup that came back `None` would ship NULL on a
calibrated run and nothing would notice
(`test_the_judge_kappa_column_is_filled_from_the_reports_calibration_record`). A
run made with no `--calibration` still stores `None` there, which is the column's
"nothing measured this" and not a zero agreement
(`test_a_calibration_whose_own_kappa_is_null_stores_null_rather_than_zero`,
`test_row_seventeens_remaining_columns_still_arrive_empty_rather_than_absent`).
17c's judge verdict goes into `details` and never into a column of its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Protocol, get_args

from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict

from app.core.validation import checked_identifier
from app.evals.cases import EvalCategory
from app.evals.report import SET_SHA_PATTERN, CaseReport, RunKind, RunReport
from app.model_routing.errors import IdentifierInvalid
from app.storage.db import DEFAULT_SCHEMA, checked_schema

EVAL_RUNS: Final[str] = "eval_runs"
"""The run table `migrations/003_eval_runs.sql` creates."""

EVAL_CASES: Final[str] = "eval_cases"
"""The per-case table beside it, whose `run_id` cascades from `eval_runs.id`.

Both names are constants because the statements below build their identifiers
from them and tests/storage/test_eval_store_postgres.py reads the catalogue by the
same names, so a rename cannot reach only one of the two."""

MAX_SET_NAME_LENGTH: Final[int] = 128
"""The longest golden-set file name a run may record.

Wider than an identifier because a set name is a FILE name - `helpdesk-golden-v1.jsonl`
carries a version and an extension - and narrower than a path because it is one
segment. Nothing else caps this value: it arrives as `--set`'s basename. Pinned
exactly by `test_the_length_caps_are_the_published_ones`, with
the inclusive boundary in `test_a_run_row_value_at_exactly_its_cap_is_accepted`.
"""

MAX_REPORT_PATH_LENGTH: Final[int] = 1024
"""The longest `report.json` path a row may point at.

A whole filesystem path rather than a name, so it is capped near what a
conventional `PATH_MAX` allows rather than at an identifier's width; the eval
gate and the console both open this value. Pinned by the same two cases.
"""

MAX_TARGET_LENGTH: Final[int] = 512
"""The longest `--target` URL a row may record.

`app.cli.evals._checked_target` already refuses anything that is not an absolute http(s)
URL, and says nothing about its length; this is the only bound on it. Pinned by
the same two cases.
"""

RUN_KINDS: Final[tuple[str, ...]] = get_args(RunKind)
"""The three words `run_kind` may be, taken from `report.RunKind` rather than
restated - `app.cli.evals` imports this module, so importing `app.cli.evals.RUN_KINDS` back would be
a cycle. `test_each_run_kind_the_command_accepts_is_written` is parametrised over
the command's own tuple, which is what pins the two together."""

CATEGORIES: Final[frozenset[str]] = frozenset(category.value for category in EvalCategory)
"""The twelve categories a case row may carry, taken from `16a`'s enum.

A report file reaches this store without passing `16a`'s loader, so this is the
only check on that path (`test_write_refuses_a_category_outside_the_twelve_before_any_sql`)."""


class EvalRunRow(BaseModel):
    """One run, as `eval_runs` stores it: the design's columns with decision C's split.

    Frozen, so the row handed to the store is the row the report described. The
    vocabulary fields are plain `str` rather than `Literal`s on purpose: a
    `Literal` would make pydantic the guard, and the store's own pre-check - the
    one that runs before a connection is borrowed - would then never be exercised
    by a test that builds a row.

    `git_sha` is nullable because `git rev-parse` cannot answer outside a
    checkout, and `null` is the absence a query already handles where `unknown`
    would be a value every query has to know about.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    started_at: datetime
    finished_at: datetime
    set_name: str
    set_sha: str
    set_sha_source: str
    run_kind: str
    target: str
    tenant: str
    git_sha: str | None
    metrics: Mapping[str, Any]
    passed: bool
    report_path: str
    alias_resolution: Mapping[str, Any] | None = None
    judge_kappa: float | None = None


class EvalCaseRow(BaseModel):
    """One case in one run, as `eval_cases` stores it.

    The four measured fields are `None` exactly where the case's own outcome is,
    which is what keeps a broken turn out of every average.

    `details` carries the three things the report holds that no column of its own
    does - the assertion results, the residency verdict and the error - so a row
    still says WHY a case failed once `report.json` has been rotated away.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    case_id: str
    category: str
    passed: bool
    tags: tuple[str, ...] = ()
    """The failure taxonomy for this case, in `taxonomy.TAG_ORDER` and empty when
    it passed. Stored in an order-preserving `text[]`, so the array a query reads
    back is the tuple the report published
    (`test_the_tags_column_reads_back_the_case_reports_tags_in_order`)."""
    latency_ms: int | None
    cost_usd: Decimal | None
    provider: str | None
    region: str | None
    details: Mapping[str, Any]


class EvalStore(Protocol):
    """Where a run's rows go. `write` is the whole contract.

    A Protocol rather than a base class, the shape `CostLedgerWriter` established:
    a test double satisfies it without importing anything to inherit from. `write`
    is allowed to RAISE - `app.cli.evals` is what guards it, in one place, and decision E is
    what that guard does with the failure.
    """

    def write(self, run_row: EvalRunRow, case_rows: Sequence[EvalCaseRow]) -> None:
        """Store one run and its cases. May raise; the caller's guard is the only one."""
        ...


def _details(case: CaseReport) -> dict[str, Any]:
    """The four things a case carries that no column of its own holds.

    Dumped in JSON mode rather than as Python objects, because the column is
    `jsonb`: a `Decimal` or a `datetime` inside an assertion result would
    otherwise reach the adapter as an object it cannot serialise, at the end of a
    run already paid for. `17c`'s judge verdict is the fourth key and takes the
    same treatment - a pydantic model handed to `Jsonb` raises inside
    `json.dumps`, which is 17d's trap on this plan's field
    (`test_the_details_carry_the_judge_verdict`). It is `null` and never absent on
    a case nothing judged, so two rows of the same set answer one query.
    """
    return {
        "assertions": None if case.assertions is None else case.assertions.model_dump(mode="json"),
        "residency": None if case.residency is None else case.residency.model_dump(mode="json"),
        "judge": None if case.judge is None else case.judge.model_dump(mode="json"),
        "error": case.error,
    }


def rows_from_report(
    report: RunReport, *, report_path: Path
) -> tuple[EvalRunRow, tuple[EvalCaseRow, ...]]:
    """Turn one built report into the rows that are stored, and nothing else.

    Pure: the run id is the report's own, never a fresh one, or every case row
    would join back to a run nobody can find beside the file that describes it.
    `report_path` is the `report.json` that was WRITTEN rather than the `--out`
    directory - the gate and the console open it expecting a document
    (`test_the_report_path_is_the_report_json_that_was_written`).

    `metrics` is taken in JSON mode and never as the live mapping: since 17d it
    carries two `Decimal` costs, and psycopg's `Jsonb` dumper is `json.dumps`,
    which raises `TypeError` on a `Decimal` - so the live mapping would fail the
    INSERT rather than store a wrong number
    (`test_the_stored_metrics_are_the_json_the_report_file_carries`). Taken this
    way the column is byte for byte what `report.json` holds, which is the same
    property `_case_details` keeps for `details`.
    """
    run_row = EvalRunRow(
        id=report.id,
        started_at=report.started_at,
        finished_at=report.finished_at,
        set_name=report.set_name,
        set_sha=report.set_sha,
        set_sha_source=report.set_sha_source,
        run_kind=report.run_kind,
        target=report.target,
        tenant=report.tenant,
        git_sha=report.git_sha,
        metrics=report.model_dump(mode="json")["metrics"],
        passed=report.passed,
        report_path=str(report_path),
        judge_kappa=None if report.calibration is None else report.calibration.kappa,
    )
    case_rows = tuple(
        EvalCaseRow(
            run_id=report.id,
            case_id=case.case_id,
            category=case.category,
            passed=case.passed,
            tags=case.tags,
            latency_ms=case.latency_ms,
            cost_usd=case.cost_usd,
            provider=case.provider,
            region=case.region,
            details=_details(case),
        )
        for case in report.cases
    )
    return run_row, case_rows


class PostgresEvalStore:
    """One run and its cases, in ONE transaction over a pooled connection. No DDL.

    Decision G: a run whose cases half-landed is counted by every later query as a
    run with fewer cases, and nothing downstream can tell that from a shorter
    golden set. So the run row and one `executemany` over the cases share a
    transaction, and a failure part way leaves neither table changed
    (`test_a_write_that_fails_part_way_leaves_neither_table_changed`).

    Every pre-check runs BEFORE the borrow, and every one of
    them raises `IdentifierInvalid`;
    `test_a_run_row_value_one_character_over_its_cap_is_refused_before_the_borrow`
    asserts that no connection was borrowed at all.
    """

    def __init__(
        self,
        pool: ConnectionPool[Connection[TupleRow]],
        *,
        schema: str = DEFAULT_SCHEMA,
    ) -> None:
        self._pool = pool
        self._schema = checked_schema(schema)

    def write(self, run_row: EvalRunRow, case_rows: Sequence[EvalCaseRow]) -> None:
        """Store one run and its cases. Raises on a value that fails a pre-check."""
        _check_run_row(run_row)
        for case_row in case_rows:
            _check_case_row(case_row)

        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} ("
                    "id, started_at, finished_at, set_name, set_sha, set_sha_source, "
                    "run_kind, target, tenant, git_sha, metrics, passed, report_path, "
                    "alias_resolution, judge_kappa) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                ).format(sql.Identifier(self._schema, EVAL_RUNS)),
                (
                    run_row.id,
                    run_row.started_at,
                    run_row.finished_at,
                    run_row.set_name,
                    run_row.set_sha,
                    run_row.set_sha_source,
                    run_row.run_kind,
                    run_row.target,
                    run_row.tenant,
                    run_row.git_sha,
                    Jsonb(dict(run_row.metrics)),
                    run_row.passed,
                    run_row.report_path,
                    None
                    if run_row.alias_resolution is None
                    else Jsonb(dict(run_row.alias_resolution)),
                    run_row.judge_kappa,
                ),
            )
            with conn.cursor() as cursor:
                cursor.executemany(
                    sql.SQL(
                        "INSERT INTO {} ("
                        "run_id, case_id, category, passed, tags, latency_ms, "
                        "cost_usd, provider, region, details) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    ).format(sql.Identifier(self._schema, EVAL_CASES)),
                    [
                        (
                            case_row.run_id,
                            case_row.case_id,
                            case_row.category,
                            case_row.passed,
                            list(case_row.tags),
                            case_row.latency_ms,
                            case_row.cost_usd,
                            case_row.provider,
                            case_row.region,
                            Jsonb(dict(case_row.details)),
                        )
                        for case_row in case_rows
                    ],
                )


def _check_run_row(row: EvalRunRow) -> None:
    """Every value on the run row, before a connection is borrowed."""
    _checked_length(row.set_name, "set_name", MAX_SET_NAME_LENGTH)
    _checked_length(row.report_path, "report_path", MAX_REPORT_PATH_LENGTH)
    _checked_length(row.target, "target", MAX_TARGET_LENGTH)
    checked_identifier(row.tenant, "tenant")
    _checked_git_sha(row.git_sha)
    _checked_run_kind(row.run_kind)


def _check_case_row(row: EvalCaseRow) -> None:
    """Every value on one case row, before a connection is borrowed.

    `provider` and `region` are checked only when they are present: `None` is not
    a malformed identifier, it is the absence the NULL column exists for, and an
    unconditional check would refuse exactly the rows a failing target produces
    (`test_a_case_that_measured_nothing_is_written_rather_than_refused`).
    """
    checked_identifier(row.case_id, "case_id")
    _checked_category(row.category)
    if row.provider is not None:
        checked_identifier(row.provider, "provider")
    if row.region is not None:
        checked_identifier(row.region, "region")


def _checked_length(value: str, field: str, cap: int) -> str:
    """One free-text value that lands in a `text` column, against its own cap.

    Length only: a set name, a path and a URL have no shared vocabulary, and a
    pattern narrow enough for one would refuse the other two. The cap is what
    keeps three values nothing else bounds from reaching an unbounded column.
    """
    if len(value) > cap:
        raise IdentifierInvalid(f"{field}: longer than {cap} characters")
    return value


def _checked_git_sha(value: str | None) -> str | None:
    """This repo's commit, against `report.SET_SHA_PATTERN`, or the honest absence.

    `None` is accepted because `16c` records it when `git rev-parse` cannot
    answer. It is the column a regression is bisected on, so `A1B2C3D` and a
    trailing newline join to no commit while looking perfectly plausible in a
    table (`test_write_refuses_a_malformed_git_sha_before_any_sql`).
    """
    if value is None:
        return None
    if SET_SHA_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("git_sha: not 7 to 64 lowercase hex characters")
    return value


def _checked_run_kind(value: str) -> str:
    """Why the run happened, against the three words and nothing else.

    A gate run and a nightly run are told apart by this column, so `CI` or `ci\\n` is a gate run
    that no query for `ci` ever finds - the gate would report on a subset of its
    own history and say nothing about the rest.
    """
    if value not in RUN_KINDS:
        raise IdentifierInvalid(f"run_kind: not one of {', '.join(RUN_KINDS)}")
    return value


def _checked_category(value: str) -> str:
    """One case's category, against `16a`'s twelve.

    Row 17 counts per category and the per-category minimums are read off these
    strings, so a thirteenth bucket is a number nobody reports on and
    `helpdesk\\n` is a category with one case in it.
    """
    if value not in CATEGORIES:
        raise IdentifierInvalid(f"category: not one of the {len(CATEGORIES)} the format allows")
    return value
