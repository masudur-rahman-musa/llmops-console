"""`evals/scores.py` and `evals/report.py`: the two scores and the two report files.

Pure: saved case lines go in, a `RunReport` and two strings come out.

Contract these tests are written against:

* `scores.METRIC_NAMES == ("check_score", "judge_score")`.
* `scores.compute_metrics(lines) -> dict` with `check_score` (percent of cases passed, an
  errored case counts as not passed, rounded to 2 places, `None` for no cases),
  `judge_score` (percent of judged replies whose verdict is `pass`, rounded to 2 places,
  `None` when nothing was judged) and the counts `cases`, `passed`, `failed`.
* `report.build_report(lines, *, started_at, finished_at, set_name, set_sha, set_sha_source,
  run_kind, git_sha, tenant, target, judge_prompt_version=None, calibration=None)
  -> RunReport`. `RunReport` keeps its header (`id` of 32 characters, times, set, run kind,
  git sha, tenant, target, `passed`, `metrics`, `calibration`) and holds one `CaseReport` per
  line (`case_id`, `category`, `passed`, `tags` always empty, the four measured fields
  `latency_ms`, `cost_usd`, `provider` all `None`, `error`, `check`, `judged`).
  `RunReport.passed` is true when every case passed. `tenant` is `ALL_TENANTS = "all"`
  for a conversation run: the cases name their own tenants.
* `report_json(report)`, `report_markdown(report)`, `write_reports(out, report)`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.evals.report import (
    FAILURES_HEADING,
    REPORT_JSON_NAME,
    REPORT_MD_NAME,
    RunReport,
    build_report,
    report_json,
    report_markdown,
    write_reports,
)
from app.evals.runner import CaseLine
from app.evals.scores import METRIC_NAMES, compute_metrics

from .builders import case_line

STARTED = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
FINISHED = datetime(2026, 10, 22, 9, 20, tzinfo=UTC)


def report_of(*lines: CaseLine) -> RunReport:
    return build_report(
        lines,
        started_at=STARTED,
        finished_at=FINISHED,
        set_name="cases.jsonl",
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="manual",
        git_sha="b" * 40,
        tenant="all",
        target="https://helpdesk.example",
    )


def test_the_two_metric_names_and_the_two_file_names_are_pinned() -> None:
    """Catches a leftover metric of the old format in the published list (the push, the
    store and the console all read these names) and a renamed report file (the gate and the
    workflow open `report.json` by name)."""
    assert METRIC_NAMES == ("check_score", "judge_score")
    assert REPORT_JSON_NAME == "report.json"
    assert REPORT_MD_NAME == "report.md"


def test_check_score_is_the_percent_of_cases_passed_and_errors_count_as_failed() -> None:
    """Catches a denominator that drops errored cases (3 of 3 answered would read 100) and a
    score taken from turns instead of cases. 4 cases, 3 passed, one of the 4 never
    answered: 75.0, with the judge's own number (40.0) kept apart so a swap is visible."""
    lines = [
        case_line("conv-001", passed=True, verdicts=["pass", "fail"]),
        case_line("conv-002", passed=True, verdicts=["fail", "fail", "pass"]),
        case_line("conv-003", passed=True),
        case_line("conv-004", passed=False, error="HTTP 500"),
    ]

    metrics = compute_metrics(lines)

    assert metrics["check_score"] == 75.0
    assert metrics["judge_score"] == 40.0
    assert (metrics["cases"], metrics["passed"], metrics["failed"]) == (4, 3, 1)


def test_scores_are_rounded_to_two_places() -> None:
    """Catches a long float in the report: 2 of 3 is 66.67, and the gate compares the written
    number, not a different one it computes itself."""
    lines = [case_line(f"conv-00{n}", passed=n < 3) for n in (1, 2, 3)]

    assert compute_metrics(lines)["check_score"] == 66.67


def test_no_cases_and_nothing_judged_give_no_score_rather_than_zero() -> None:
    """Catches a zero for 'nothing measured': a judge_score of 0 would read as a judge that
    failed every reply, and an empty run's check_score of 0 or 100 as a result."""
    metrics = compute_metrics([case_line("conv-001", passed=True)])
    empty = compute_metrics([])

    assert metrics["judge_score"] is None
    assert metrics["check_score"] == 100.0
    assert empty["check_score"] is None


def test_the_report_carries_the_run_the_scores_and_one_row_per_case() -> None:
    """Catches a report whose header and metrics disagree with its cases, or that loses the
    error text: the row of a case that never answered keeps it and the measured fields stay
    empty (a zero latency would read as a free, instant success)."""
    report = report_of(
        case_line("conv-001", passed=True),
        case_line("conv-002", passed=False, error="HTTP 500 for conv-002"),
    )

    assert len(report.id) == 32
    assert report.passed is False
    assert report.tenant == "all"
    assert report.metrics["check_score"] == 50.0
    assert [c.case_id for c in report.cases] == ["conv-001", "conv-002"]
    assert report.cases[1].error == "HTTP 500 for conv-002"
    broken = report.cases[1]
    assert (broken.latency_ms, broken.cost_usd, broken.provider) == (None, None, None)
    assert broken.tags == ()


def test_two_reports_never_share_an_id() -> None:
    """Catches an id derived from the set or the day: two runs stored on one night would
    collide on the primary key."""
    assert (
        report_of(case_line("conv-001", passed=True)).id
        != report_of(case_line("conv-001", passed=True)).id
    )


def test_a_report_with_every_case_passed_is_passed() -> None:
    """Catches a `passed` that follows the score or the judge instead of the cases."""
    assert report_of(case_line("conv-001", passed=True)).passed is True


def test_the_json_round_trips_through_the_model() -> None:
    """Catches a report the gate cannot read back: the gate loads `report.json` through the
    same model, so what is written must validate."""
    report = report_of(case_line("conv-001", passed=True, verdicts=["pass"]))

    assert RunReport.model_validate_json(report_json(report)) == report


def test_the_markdown_lists_failures_and_only_failures() -> None:
    """Catches a report that hides the failed case (the reason a person opens it) or lists
    passing cases under the failures heading. The errored case shows its error text."""
    report = report_of(
        case_line("conv-001", passed=True),
        case_line("conv-002", passed=False, error="HTTP 500 for conv-002"),
    )

    text = report_markdown(report)

    failures = text.split(FAILURES_HEADING, 1)[1]
    assert "conv-002" in failures and "HTTP 500 for conv-002" in failures
    assert "conv-001" not in failures
    assert "check_score" in text and "50.0" in text


def test_a_clean_run_has_no_failures_section() -> None:
    """Catches a failures heading over an empty list."""
    assert FAILURES_HEADING not in report_markdown(report_of(case_line("conv-001", passed=True)))


def test_write_reports_writes_exactly_the_two_files(tmp_path: Path) -> None:
    """Catches a writer that leaves a third file behind or writes under another name."""
    write_reports(tmp_path, report_of(case_line("conv-001", passed=True)))

    assert sorted(p.name for p in tmp_path.iterdir()) == [REPORT_JSON_NAME, REPORT_MD_NAME]


@pytest.mark.parametrize("name", ["decision_accuracy", "refusal_recall", "judge_mean"])
def test_no_metric_of_the_old_format_is_reported(name: str) -> None:
    """Catches the old single-reply metrics surviving in `metrics`: the console reads
    `check_score` where it read `decision_accuracy`, and a stale key would be read by the
    old regression list."""
    report = report_of(case_line("conv-001", passed=True))

    assert name not in report.metrics
