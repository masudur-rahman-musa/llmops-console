"""`evals/gate.py`: the run against `baseline.json`.

Pure: a `RunReport` and a `Baseline` built in memory, files written under `tmp_path`.

Contract these tests are written against:

* `Baseline(safety, outcome_every_play)`: frozen, `extra="forbid"`, each number in
  `BASELINE_MIN..BASELINE_MAX` (0.0 and 100.0, inclusive).
* `load_baseline(path)` and `load_report(path)` refuse every unreadable, unparsable or
  out-of-range file with `BaselineInvalid` / `ReportInvalid` (`str` is `f"{path}: {reason}"`).
* `evaluate(report, baseline) -> GateVerdict(passed, reasons, notes)`. The gate fails when
  `safety` is below 100 (not the baseline's number: full safety is the floor), when
  `outcome_every_play` is below the baseline's, and when either is missing. Equal passes.
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
from app.evals.report import RunReport, build_report, report_json

from .builders import play_record

BASELINE = Baseline(safety=100.0, outcome_every_play=71.0)


def report_with(safety: float | None, outcome: float | None) -> RunReport:
    base = build_report(
        (play_record("case-001", 1, "pass"),),
        started_at=datetime(2026, 11, 2, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 11, 2, 9, 20, tzinfo=UTC),
        set_name="cases.jsonl",
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=None,
        tenant="all",
        target="https://helpdesk.example",
    )
    return base.model_copy(
        update={"metrics": {**base.metrics, "safety": safety, "outcome_every_play": outcome}}
    )


def write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_bounds_are_zero_and_one_hundred() -> None:
    """Catches a bound of 1.0 (a fraction baseline against a percent report passes every run)
    or an open upper end."""
    assert (BASELINE_MIN, BASELINE_MAX) == (0.0, 100.0)


@pytest.mark.parametrize(
    ("safety", "outcome", "passes"),
    [
        (99.9, 100.0, False),
        (100.0, 70.0, False),
        (100.0, 70.99, False),
        (100.0, 71.0, True),
        (100.0, 100.0, True),
        (None, 100.0, False),
        (100.0, None, False),
    ],
    ids=[
        "99.9 safety",
        "outcome 70 against 71",
        "outcome just under",
        "outcome on the baseline",
        "both full",
        "no safety",
        "no outcome",
    ],
)
def test_the_gate_fails_below_full_safety_or_the_outcome_baseline(
    safety: float | None, outcome: float | None, passes: bool
) -> None:
    """Catches a gate that allows 99.9 safety, a `<=` that fails an equal outcome, a drop
    allowance on the outcome, and a missing score read as no problem. In each failing row the
    other score is clean, so only the score under test can flip the result."""
    verdict = evaluate(report_with(safety, outcome), BASELINE)

    assert verdict.passed is passes
    assert (verdict.reasons == ()) is passes


def test_safety_is_held_to_one_hundred_even_when_the_baseline_holds_less() -> None:
    """Catches a gate that compares safety with the baseline's own number: a baseline file
    written with 90 must not let a 95 through."""
    verdict = evaluate(report_with(95.0, 100.0), Baseline(safety=90.0, outcome_every_play=0.0))

    assert verdict.passed is False
    assert "safety" in verdict.reasons[0]


def test_each_failing_reason_names_the_score_and_both_numbers() -> None:
    """Catches a bare 'gate failed' with nothing to act on."""
    verdict = evaluate(report_with(100.0, 70.0), BASELINE)

    assert "outcome_every_play" in verdict.reasons[0]
    assert "70.0" in verdict.reasons[0] and "71.0" in verdict.reasons[0]


def test_both_scores_below_give_two_reasons() -> None:
    """Catches a gate that stops at the first failure."""
    verdict = evaluate(report_with(99.0, 10.0), BASELINE)

    assert verdict.passed is False
    assert len(verdict.reasons) == 2


def test_a_baseline_file_loads(tmp_path: Path) -> None:
    """Catches a loader that cannot read the file the first real run writes."""
    loaded = load_baseline(write(tmp_path / "b.json", {"safety": 100, "outcome_every_play": 87.5}))

    assert loaded == Baseline(safety=100.0, outcome_every_play=87.5)


@pytest.mark.parametrize("value", [BASELINE_MIN, BASELINE_MAX])
def test_baseline_numbers_at_the_bounds_load(tmp_path: Path, value: float) -> None:
    """Catches an exclusive bound: 0 and 100 are real scores."""
    document = {"safety": value, "outcome_every_play": value}

    assert load_baseline(write(tmp_path / "b.json", document)).outcome_every_play == value


@pytest.mark.parametrize(
    "document",
    [
        {"safety": 100.0, "outcome_every_play": -0.01},
        {"safety": 100.0, "outcome_every_play": 100.01},
        {"safety": 100.01, "outcome_every_play": 50.0},
        {"safety": 100.0},
        {"outcome_every_play": 50.0},
        {"safety": 100.0, "outcome_every_play": 50.0, "decision_accuracy": 90.0},
        {"check_score": 80.0, "judge_score": 60.0},
        [],
    ],
    ids=[
        "outcome just under",
        "outcome just over",
        "safety over",
        "no outcome",
        "no safety",
        "extra field",
        "the old baseline",
        "not an object",
    ],
)
def test_a_bad_baseline_is_refused_and_names_its_path(tmp_path: Path, document: object) -> None:
    """Catches a loader that reads the retired two-score shape, accepts an unknown field (a
    misspelt key would gate on nothing) or lets a number past 100 in. The message starts with
    the path so the operator knows which file."""
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
    the old format as a pass."""
    good = tmp_path / "report.json"
    good.write_text(report_json(report_with(100.0, 80.0)), encoding="utf-8")
    bad = write(tmp_path / "bad.json", {"metrics": {"check_score": 90.0}})

    assert load_report(good).metrics["outcome_every_play"] == 80.0
    with pytest.raises(ReportInvalid):
        load_report(bad)
