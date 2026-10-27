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
  line (`case_id`, `category`, `passed`, `tags` (the checks a failed case broke), the four
  measured fields `latency_ms`, `cost_usd`, `provider` all `None`, `error`, `check`, `judged`).
  `RunReport.passed` is true when every case passed. `tenant` is `ALL_TENANTS = "all"`
  for a conversation run: the cases name their own tenants.
* `report_json(report)`, `report_markdown(report)`, `write_reports(out, report)`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from app.evals.checks import CaseCheck, TurnCheck
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
from app.evals.runner import CaseLine, JudgedReply
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


# --------------------------------------------------------------------------
# failure tags
# --------------------------------------------------------------------------


def turn_check(number: int, *, early: bool = False, made_up: bool = False) -> TurnCheck:
    """One turn's two checks; the flags say which one broke."""
    return TurnCheck(turn=number, no_early_change=not early, no_made_up_name=not made_up)


def judged(*verdicts: Literal["pass", "fail"] | None) -> tuple[JudgedReply, ...]:
    """One judged reply per verdict, numbered from turn 1."""
    return tuple(
        JudgedReply(turn=n + 1, rubric="r", reply="x", verdict=v, reason=None)
        for n, v in enumerate(verdicts)
    )


def tagged_line(
    case_id: str = "conv-001",
    *,
    passed: bool = False,
    tools_ok: bool = True,
    turns: tuple[TurnCheck, ...] = (),
    verdicts: tuple[Literal["pass", "fail"] | None, ...] = (),
    error: str | None = None,
    answered: bool = True,
) -> CaseLine:
    """A saved case line with a real `CaseCheck` (unless `answered` is false)."""
    check = CaseCheck(tools_in_order=tools_ok, turns=turns) if answered else None
    return CaseLine(
        case_id=case_id,
        category="paraphrases",
        passed=passed,
        error=error,
        check=check,
        judged=judged(*verdicts),
        answer=None,
    )


def test_a_case_that_missed_the_tool_order_is_tagged_wrong_tool() -> None:
    """Catches an implementation that leaves tags empty, or that tags the tool check as
    some other name: only `tools_in_order` is false here, so only `wrong_tool` is right."""
    line = tagged_line(tools_ok=False, turns=(turn_check(1),))

    assert report_of(line).cases[0].tags == ("wrong_tool",)


def test_a_change_before_the_yes_is_tagged_early_change() -> None:
    """Catches an implementation that does not look at the turn checks for an early change,
    or that reads the made-up-name flag by mistake: only `no_early_change` is false."""
    line = tagged_line(turns=(turn_check(1), turn_check(2, early=True)))

    assert report_of(line).cases[0].tags == ("early_change",)


def test_a_made_up_name_is_tagged_made_up_name() -> None:
    """Catches an implementation that reads the early-change flag for this tag, or never
    tags a made-up name: only `no_made_up_name` is false."""
    line = tagged_line(turns=(turn_check(1, made_up=True),))

    assert report_of(line).cases[0].tags == ("made_up_name",)


def test_a_judge_fail_on_a_failed_case_is_tagged_judge_low() -> None:
    """Catches an implementation that never turns a judge "fail" into a tag. Every check
    passes here, so the judge is the only thing that can explain the failure."""
    line = tagged_line(turns=(turn_check(1),), verdicts=("pass", "fail"))

    assert report_of(line).cases[0].tags == ("judge_low",)


def test_a_case_that_broke_every_check_carries_four_tags_in_order_once_each() -> None:
    """Catches a wrong tag order, and a tag repeated once per broken turn or per judge fail
    (two turns and two judge fails break here, yet each tag must appear once)."""
    line = tagged_line(
        tools_ok=False,
        turns=(turn_check(1, early=True, made_up=True), turn_check(2, early=True, made_up=True)),
        verdicts=("fail", "fail"),
    )

    assert report_of(line).cases[0].tags == (
        "wrong_tool",
        "early_change",
        "made_up_name",
        "judge_low",
    )


def test_a_passed_case_carries_no_tags_even_when_the_judge_said_fail() -> None:
    """Catches an implementation that tags every case with a judge fail and forgets that a
    passed case (the judge does not fail a case) must carry no tags."""
    line = tagged_line(passed=True, turns=(turn_check(1),), verdicts=("fail",))

    assert report_of(line).cases[0].tags == ()


def test_a_case_that_never_answered_carries_no_tags() -> None:
    """Catches an implementation that invents a tag for a case the target never answered,
    for example `wrong_tool` because there is no check to say the tools were right."""
    line = tagged_line(error="HTTP 500 for conv-001", answered=False)

    assert report_of(line).cases[0].tags == ()


def test_a_judge_verdict_of_none_is_not_judge_low() -> None:
    """Catches an implementation that tags `judge_low` for any judged reply, or for a
    verdict that is not "pass", instead of only a "fail". An unusable verdict (`None`) is
    not a fail."""
    line = tagged_line(tools_ok=False, turns=(turn_check(1),), verdicts=(None,))

    assert report_of(line).cases[0].tags == ("wrong_tool",)


def test_an_unstable_case_takes_its_tags_from_its_first_failed_run() -> None:
    """Catches an implementation that takes tags from the first run (a pass, no tags), from
    the last run (made_up_name), or from every run together. The first run passed with a
    judge fail, the first failed run broke the tools, the last run made up a name and
    the judge failed it: only `wrong_tool` is right."""
    passed_run = tagged_line(passed=True, turns=(turn_check(1),), verdicts=("fail",))
    tools_run = tagged_line(tools_ok=False, turns=(turn_check(1),))
    name_run = tagged_line(turns=(turn_check(1, made_up=True),), verdicts=("fail",))

    case = report_of(passed_run, tools_run, name_run).cases[0]

    assert case.unstable is True
    assert case.tags == ("wrong_tool",)


def test_the_tags_survive_the_json_round_trip() -> None:
    """Catches tags lost or reshaped between `report.json` and `RunReport`: the gate and the
    store read the file back, and a tuple that comes back as nothing would empty Taxonomy."""
    report = report_of(
        tagged_line("conv-001", tools_ok=False, turns=(turn_check(1, early=True),)),
        tagged_line("conv-002", turns=(turn_check(1),), verdicts=("fail",)),
        tagged_line("conv-003", passed=True, turns=(turn_check(1),)),
    )

    back = RunReport.model_validate_json(report_json(report))

    assert [c.tags for c in back.cases] == [("wrong_tool", "early_change"), ("judge_low",), ()]
