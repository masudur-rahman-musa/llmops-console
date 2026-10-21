"""`evals/gate.py`: the run against `conversation-baseline.json`.

Pure: a `RunReport` and a `Baseline` built in memory, files written under `tmp_path`.

Contract these tests are written against:

* `BASELINE_MIN = 0.0`, `BASELINE_MAX = 100.0` (inclusive bounds on a baseline number).
* `Baseline(check_score, judge_score=None)`: frozen, `extra="forbid"`, each number in
  `BASELINE_MIN..BASELINE_MAX`.
* `load_baseline(path) -> Baseline` and `load_report(path) -> RunReport`, refusing every
  unreadable, unparsable or out-of-range file with `BaselineInvalid` / `ReportInvalid`
  (both `GateInputInvalid`; `str` is `f"{path}: {reason}"`).
* `evaluate(report, baseline, *, calibration=None) -> GateVerdict(passed, reasons, notes)`.
  The gate fails when `check_score` is below the baseline's (any drop; no allowance) or is
  missing. It also fails when `judge_score` is below the baseline's, but only while the
  supplied calibration's kappa is `JUDGE_TRUST_MIN` or better. When the judge does not gate,
  `notes` says so and names `judge_score`; when it fails the gate, `reasons` names
  `judge_score` with both numbers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.evals.gate import (
    BASELINE_MAX,
    BASELINE_MIN,
    Baseline,
    BaselineInvalid,
    ReportInvalid,
    evaluate,
    load_baseline,
    load_report,
)
from app.evals.report import RunCalibration, RunReport, build_report, report_json

from .builders import case_line


def report_with(check: float | None, judge: float | None = None) -> RunReport:
    base = build_report(
        [case_line("conv-001", passed=True)],
        started_at=datetime(2026, 10, 22, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 22, 9, 20, tzinfo=UTC),
        set_name="cases.jsonl",
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=None,
        tenant="all",
        target="https://helpdesk.example",
    )
    return base.model_copy(
        update={"metrics": {**base.metrics, "check_score": check, "judge_score": judge}}
    )


def calibration(kappa: float | None) -> RunCalibration:
    return RunCalibration(
        judge_prompt_version="c" * 64,
        labels_name="conversation-labels.jsonl",
        labels_sha="d" * 64,
        pairs=40,
        kappa=kappa,
        trusted=kappa is not None and kappa >= 0.7,
    )


BASELINE = Baseline(check_score=80.0, judge_score=60.0)


def test_the_bounds_are_zero_and_one_hundred() -> None:
    """Catches a bound of 1.0 (a fraction baseline against a percent report passes every
    run) or an open upper end."""
    assert (BASELINE_MIN, BASELINE_MAX) == (0.0, 100.0)


@pytest.mark.parametrize(
    ("check", "passes"),
    [(79.0, False), (79.99, False), (80.0, True), (80.01, True), (100.0, True), (None, False)],
)
def test_below_the_baseline_check_score_fails(check: float | None, passes: bool) -> None:
    """Catches a gate with a drop allowance (79.0 is one point under and must fail), a `<=`
    that fails an equal score, and a missing score read as zero evidence of no problem. The
    judge number is at its baseline and untrusted, so only `check_score` can decide."""
    verdict = evaluate(report_with(check, judge=60.0), BASELINE, calibration=None)

    assert verdict.passed is passes
    assert (verdict.reasons == ()) is passes
    if not passes:
        assert "check_score" in verdict.reasons[0]


def test_the_failing_reason_names_both_numbers() -> None:
    """Catches a bare 'gate failed' with nothing to act on: the line carries the run's score
    and the baseline's."""
    verdict = evaluate(report_with(79.0), BASELINE)

    assert "79.0" in verdict.reasons[0] and "80.0" in verdict.reasons[0]


@pytest.mark.parametrize(
    ("kappa", "judge_gates"),
    [(0.7, True), (0.71, True), (0.69, False), (None, False)],
)
def test_the_judge_score_gates_only_when_trusted(kappa: float | None, judge_gates: bool) -> None:
    """Catches a gate that always includes the judge (an untrusted judge would block pull
    requests on its own noise), one that never does (a trusted judge could regress unseen),
    and a trust line drifted off 0.7 in either direction. The same low judge_score (40.0
    against 60.0) and a clean check_score are used in every row, so the kappa is the only
    thing that flips the result. The reason is printed either way: as a failure when it
    gates, as a note when it does not."""
    verdict = evaluate(report_with(80.0, judge=40.0), BASELINE, calibration=calibration(kappa))

    assert verdict.passed is not judge_gates
    mentioned = verdict.reasons if judge_gates else verdict.notes
    assert any("judge_score" in line for line in mentioned)
    if judge_gates:
        assert any("40.0" in line and "60.0" in line for line in verdict.reasons)
    else:
        assert verdict.reasons == ()


def test_no_calibration_at_all_leaves_the_judge_out() -> None:
    """Catches a gate that crashes or fails when no calibration exists yet (the first run,
    before the labels): the run goes on and says the judge is reported only."""
    verdict = evaluate(report_with(80.0, judge=10.0), BASELINE, calibration=None)

    assert verdict.passed is True
    assert any("judge_score" in note for note in verdict.notes)


def test_a_trusted_judge_at_its_baseline_passes() -> None:
    """Catches a `<=` on the judge line: equal passes."""
    verdict = evaluate(report_with(80.0, judge=60.0), BASELINE, calibration=calibration(0.9))

    assert verdict.passed is True


def test_both_numbers_below_give_two_reasons() -> None:
    """Catches a gate that stops at the first failure: a reviewer fixing one then meets the
    other in the next run."""
    verdict = evaluate(report_with(70.0, judge=40.0), BASELINE, calibration=calibration(0.8))

    assert verdict.passed is False
    assert len(verdict.reasons) == 2


def write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_baseline_file_loads(tmp_path: Path) -> None:
    """Catches a loader that cannot read the file the baseline run writes, and one that
    drops `judge_score`."""
    loaded = load_baseline(write(tmp_path / "b.json", {"check_score": 87.88, "judge_score": 71.5}))

    assert loaded == Baseline(check_score=87.88, judge_score=71.5)


@pytest.mark.parametrize("value", [BASELINE_MIN, BASELINE_MAX])
def test_baseline_numbers_at_the_bounds_load(tmp_path: Path, value: float) -> None:
    """Catches an exclusive bound: 0 and 100 are real scores."""
    assert load_baseline(write(tmp_path / "b.json", {"check_score": value})).check_score == value


@pytest.mark.parametrize(
    "document",
    [
        {"check_score": -0.01},
        {"check_score": 100.01},
        {"check_score": 80.0, "judge_score": 100.01},
        {"judge_score": 60.0},
        {"check_score": 80.0, "decision_accuracy": 90.0},
        {"metrics": {"decision_accuracy": 90.0}, "thresholds": {"max_drop_points": 2.0}},
        [],
    ],
    ids=[
        "just under",
        "just over",
        "judge over",
        "no check_score",
        "extra field",
        "the old baseline",
        "not an object",
    ],
)
def test_a_bad_baseline_is_refused_and_names_its_path(tmp_path: Path, document: object) -> None:
    """Catches a loader that reads the old `baseline.json` shape, accepts an unknown field
    (a misspelt key would then gate on nothing) or lets a number past 100 in. The message
    starts with the path so the operator knows which file."""
    path = write(tmp_path / "b.json", document)

    with pytest.raises(BaselineInvalid) as raised:
        load_baseline(path)

    assert str(raised.value).startswith(f"{path}: ")


def test_a_missing_or_unparsable_baseline_is_refused(tmp_path: Path) -> None:
    """Catches a gate that treats an absent baseline as 'nothing to beat'."""
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    with pytest.raises(BaselineInvalid):
        load_baseline(tmp_path / "missing.json")
    with pytest.raises(BaselineInvalid):
        load_baseline(broken)


def test_a_report_loads_and_a_malformed_one_is_refused(tmp_path: Path) -> None:
    """Catches a gate that cannot read what the run wrote, and one that reads a report from
    another format as a pass."""
    good = tmp_path / "report.json"
    good.write_text(report_json(report_with(80.0)), encoding="utf-8")
    bad = write(tmp_path / "bad.json", {"metrics": {"decision_accuracy": 90.0}})

    assert load_report(good).metrics["check_score"] == 80.0
    with pytest.raises(ReportInvalid):
        load_report(bad)
