r"""The two files one run writes: `report.json` and `report.md` (#39, part).

Pure and clockless. `build_report` is handed the two timestamps rather than
reading a clock (decision F), so `runner.run_set` stays the loop `16b` shipped
and every case in `tests/evals/test_report.py` is deterministic.

Three things the module holds that no single field states:

* every count is DERIVED from `RunResult`'s own properties and never accepted
  from a caller (decision E), so the report and the one line `ops-eval` prints
  cannot disagree about one run
  (`test_each_count_equals_the_matching_run_result_property`);
* `assertions` is `None` exactly where `CaseRun.assertions` is - a turn that
  never answered, which is not the same claim as an empty list of assertions
  (`test_a_case_whose_turn_failed_carries_null_assertions_and_its_error`);
* `cost_usd` is serialised as a JSON STRING and never through a float, so a cost
  round-trips to the number the ledger computed
  (`test_cost_usd_is_serialised_as_a_json_string`).

The two file names are constants and nothing a caller supplies shapes a path
(decision B): `make eval` names one directory, row 18's gate reads a known path
inside it, and a rerun overwrites
(`test_a_rerun_overwrites_the_two_files_rather_than_appending`).

`16d` reads this record: it brings migration `003` and writes the `eval_runs` and
`eval_cases` rows from the fields published here and in `docs/eval-report.md`.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from opsconsole.evals.assertions import AssertionResult, AssertionStatus, CaseAssertions
from opsconsole.evals.residency import ResidencyVerdict
from opsconsole.evals.runner import CaseRun, RunResult

REPORT_JSON_NAME: Final[str] = "report.json"
"""The machine-readable report, the file row 18's gate reads."""

REPORT_MD_NAME: Final[str] = "report.md"
"""The human-readable report, the file somebody opens after a red gate.

Both names are fixed rather than timestamped (decision B) and pinned exactly by
`test_the_report_file_names_and_the_run_id_length_are_the_published_ones`."""

RUN_ID_LENGTH: Final[int] = 32
"""Characters in a run id: a `uuid4` hex, which `16d` stores as the `eval_runs` key.

Enforced on the field itself, so a shortened or dashed id is a validation error
rather than a key two runs could share."""

SET_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{7,64}")
r"""What `--set-sha` may be: 7 to 64 lowercase hex characters, inclusive.

Used with `re.fullmatch` and never `^...$` with `.match`, because in Python `$`
also matches before a trailing newline. The value is written
into a report and into `16d`'s `eval_runs` row, so it is checked before the seed
is reset (`test_a_malformed_set_sha_is_refused_before_the_target_is_touched`)."""

REPORT_INDENT: Final[int] = 2
"""How `report.json` is indented. Fixed so the worked report in
`docs/eval-report.md` stays byte-identical to `tests/evals/fixtures/report.json`."""

SetShaSource = Literal["commit", "content"]
"""Where `set_sha` came from: the caller's commit, or the set file's own bytes.

Recorded beside the SHA (decision A) so no reader is ever told a content hash is
a commit (`test_the_set_sha_source_and_run_kind_vocabularies_are_the_published_ones`)."""

RunKind = Literal["ci", "nightly", "manual"]
"""Why the run happened. Row 18 filters `eval_runs` by it."""

FAILURES_HEADING: Final[str] = "## Failures"
"""Where the per-case rows live in `report.md`, and the last section of it.

A clean run says so in ONE line under this heading rather than printing a table
with no rows, which reads as "something failed and the rows are missing"
(`test_a_clean_run_renders_one_line_and_no_empty_failures_table`)."""

CLEAN_RUN_LINE: Final[str] = "Every case passed and every turn stayed in region."
"""The one line a clean run renders in place of the failures table."""


def _new_run_id() -> str:
    """A fresh `uuid4` hex, one per built report.

    Nothing about the run derives it: two runs of the same set on the same
    commit are two rows in `16d`'s table and two directories a reader keeps
    (`test_each_build_gets_its_own_run_id`).
    """
    return uuid.uuid4().hex


class CaseReport(BaseModel):
    """One case's row: what it was, what it cost, and what was wrong with it.

    `latency_ms`, `cost_usd`, `provider` and `region` are measured once by the
    turn and never again, so they are carried rather than recomputed; they are
    `None`, never `0`, on a case whose turn failed
    (`test_a_case_with_no_measurements_serialises_them_as_null_and_not_as_zero`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    passed: bool
    latency_ms: int | None
    cost_usd: Decimal | None
    provider: str | None
    region: str | None
    assertions: CaseAssertions | None
    residency: ResidencyVerdict | None
    error: str | None

    @field_serializer("cost_usd")
    def _serialise_cost(self, value: Decimal | None) -> str | None:
        """Write the cost as its plain decimal text, never through a float.

        `format(value, "f")` and not `str(value)`: Decimal's own text switches to
        scientific notation below `1e-6`, so a real per-token cost would reach a
        report as `1E-7` and stop matching the ledger's own column
        (`test_cost_usd_is_serialised_as_a_json_string`).
        """
        return None if value is None else format(value, "f")


class RunReport(BaseModel):
    """One run: its identity, its counts, and one `CaseReport` per case.

    The identity fields are the columns `16d`'s `eval_runs` row needs, and their
    names are the published contract rather than an implementation detail -
    `docs/eval-report.md` publishes them and
    `test_the_json_document_carries_every_identity_field_by_its_published_name`
    pins the document's own key set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=_new_run_id, min_length=RUN_ID_LENGTH, max_length=RUN_ID_LENGTH)
    started_at: datetime
    finished_at: datetime
    set_name: str
    set_sha: str
    set_sha_source: SetShaSource
    run_kind: RunKind
    git_sha: str | None
    tenant: str
    target: str
    passed: bool
    metrics: Mapping[str, Any]
    cases: tuple[CaseReport, ...]


def _case_report(run: CaseRun) -> CaseReport:
    """One per-case record turned into its report row.

    The four measured fields come from the turn's own `CaseOutcome`, which is
    `None` exactly where the assertions are, so a broken turn reports nothing
    measured rather than reporting zeros.
    """
    outcome = run.outcome
    return CaseReport(
        case_id=run.case_id,
        category=str(run.category),
        passed=run.passed,
        latency_ms=None if outcome is None else outcome.latency_ms,
        cost_usd=None if outcome is None else outcome.cost_usd,
        provider=None if outcome is None else outcome.provider,
        region=None if outcome is None else outcome.region,
        assertions=run.assertions,
        residency=run.residency,
        error=run.error,
    )


def _categories(result: RunResult) -> dict[str, dict[str, int]]:
    """Passes and failures per category, over the categories this run actually used.

    Built from the run rather than from `EvalCategory`'s twelve, so a report does
    not carry ten empty buckets, and a category with no passes is still present
    (`test_the_per_category_breakdown_counts_each_category_once_and_omits_none`).
    """
    breakdown: dict[str, dict[str, int]] = {}
    for run in result.runs:
        counts = breakdown.setdefault(str(run.category), {"passed": 0, "failed": 0})
        counts["passed" if run.passed else "failed"] += 1
    return breakdown


def build_report(
    result: RunResult,
    *,
    started_at: datetime,
    finished_at: datetime,
    set_name: str,
    set_sha: str,
    set_sha_source: str,
    run_kind: str,
    git_sha: str | None,
    tenant: str,
    target: str,
) -> RunReport:
    """Turn one run and its identity into the record both files render.

    Every count is read off `RunResult`'s properties (decision E). `metrics` is a
    mapping rather than a model so row 17 can add #40's fixed metric set and its
    failure taxonomy without changing this schema or `16d`'s column.

    `set_sha_source` and `run_kind` arrive as plain strings and are validated by
    the model's own vocabularies here, which is why the report is built through
    `model_validate`: a caller that has not been through the CLI's `choices` and
    its `SET_SHA_PATTERN` still cannot put a fourth word into a stored row
    (`test_the_set_sha_source_and_run_kind_vocabularies_are_the_published_ones`).
    """
    payload: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": finished_at,
        "set_name": set_name,
        "set_sha": set_sha,
        "set_sha_source": set_sha_source,
        "run_kind": run_kind,
        "git_sha": git_sha,
        "tenant": tenant,
        "target": target,
        "passed": result.failed_count == 0,
        "metrics": {
            "cases": result.cases,
            "passed": result.passed_count,
            "failed": result.failed_count,
            "residency_violations": result.residency_violations,
            "leaked": result.leaked,
            "categories": _categories(result),
        },
        "cases": [_case_report(run) for run in result.runs],
    }
    return RunReport.model_validate(payload)


def report_json(report: RunReport) -> str:
    """`report.json`'s text: the whole record, indented, newline-terminated."""
    return report.model_dump_json(indent=REPORT_INDENT) + "\n"


def _failed_results(assertions: CaseAssertions) -> list[AssertionResult]:
    """The assertions that failed, in the fixed order `assert_case` reports them."""
    return [result for result in assertions.results if result.status is AssertionStatus.FAILED]


def _cell(text: str) -> str:
    """One table cell, with any pipe in it escaped so the row still renders."""
    return text.replace("|", r"\|")


def _reason(case: CaseReport) -> str:
    """Why this case failed, in the words a reader needs before opening the JSON.

    A broken turn reports its error; a case that answered reports every assertion
    that failed and, when the turn left its region, the residency verdict too - a
    violation with every assertion passing has no failed assertion to name
    (`test_the_markdown_names_every_failed_case_and_the_assertion_that_failed`).
    """
    if case.assertions is None:
        return case.error or "the turn never answered"

    reasons = []
    for result in _failed_results(case.assertions):
        if result.expected is None and result.actual is None:
            reasons.append(result.name)
        else:
            reasons.append(f"{result.name} (expected {result.expected}, actual {result.actual})")
    if case.residency is not None and case.residency.violation:
        allowed = ", ".join(case.residency.allowed)
        reasons.append(f"residency: served in {case.residency.observed}, allowed {allowed}")
    return "; ".join(reasons) if reasons else "failed with no stated reason"


def _identity_lines(report: RunReport) -> list[str]:
    """The header a reader needs before any count means anything: which run was this."""
    return [
        f"# Eval run {report.id}",
        "",
        f"- set: {report.set_name}",
        f"- set sha: {report.set_sha} ({report.set_sha_source})",
        f"- run kind: {report.run_kind}",
        f"- git sha: {report.git_sha if report.git_sha is not None else 'unknown'}",
        f"- tenant: {report.tenant}",
        f"- target: {report.target}",
        f"- started: {report.started_at.isoformat()}",
        f"- finished: {report.finished_at.isoformat()}",
        f"- result: {'passed' if report.passed else 'FAILED'}",
    ]


def _count_lines(report: RunReport) -> list[str]:
    """The counts, each beside its own word, so a reader is not counting rows."""
    metrics = report.metrics
    return [
        "## Counts",
        "",
        f"- cases: {metrics['cases']}",
        f"- passed: {metrics['passed']}",
        f"- failed: {metrics['failed']}",
        f"- residency: {metrics['residency_violations']} violations",
        f"- leaked: {'yes' if metrics['leaked'] else 'no'}",
    ]


def _category_lines(report: RunReport) -> list[str]:
    """The per-category breakdown, one row per category the run used."""
    lines = ["## Categories", "", "| category | passed | failed |", "| --- | --- | --- |"]
    categories: Mapping[str, Mapping[str, int]] = report.metrics["categories"]
    lines += [
        f"| {name} | {counts['passed']} | {counts['failed']} |"
        for name, counts in categories.items()
    ]
    return lines


def _failure_lines(report: RunReport) -> list[str]:
    """The failures section, and the last section of the document.

    A clean run renders ONE line here and no table at all, which is the whole
    reason this section is last: everything after the heading is the failures,
    and a reader who sees a header row expects rows under it.
    """
    failed = [case for case in report.cases if not case.passed]
    if not failed:
        return [FAILURES_HEADING, "", CLEAN_RUN_LINE]
    return [
        FAILURES_HEADING,
        "",
        "| case | category | why |",
        "| --- | --- | --- |",
        *(
            f"| {_cell(case.case_id)} | {_cell(case.category)} | {_cell(_reason(case))} |"
            for case in failed
        ),
    ]


def report_markdown(report: RunReport) -> str:
    """`report.md`'s text: who ran what, the counts, the categories, the failures.

    Rendered with f-strings and a joined list rather than a template engine
    (decision C): one document is not a dependency ask.
    """
    sections = [
        _identity_lines(report),
        _count_lines(report),
        _category_lines(report),
        _failure_lines(report),
    ]
    return "\n\n".join("\n".join(lines) for lines in sections) + "\n"


def write_reports(out: Path, report: RunReport) -> None:
    """Write both files into `out`, overwriting whatever a previous run left.

    Exactly two files, both named by the constants above and neither derived from
    a case id, a tenant or a clock (decision B), so row 18's gate reads a known
    path and a rerun into the same directory leaves two documents that parse
    (`test_write_reports_writes_exactly_the_two_published_files`).

    Nothing is caught here: a write that fails is the caller's to report, and the
    CLI turns it into `EXIT_USAGE` naming the path rather than a traceback at the
    end of a paid-for run.
    """
    (out / REPORT_JSON_NAME).write_text(report_json(report), encoding="utf-8")
    (out / REPORT_MD_NAME).write_text(report_markdown(report), encoding="utf-8")
