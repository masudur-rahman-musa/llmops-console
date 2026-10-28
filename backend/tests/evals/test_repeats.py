"""Repeat runs, the Wilson range and unstable cases.

Contract these tests are written against:

* `run_conversations(..., repeats=1)`: rounds 1..repeats in order, each round in file order.
  A reset is sent at the start of any round that has no saved line yet. A saved line is
  keyed by (case id, round); a line saved without a `round` key is round 1.
* `scores.wilson_range(passes, runs)` and `scores.pass_rate(passes, runs)`, in percent at two
  places. `compute_metrics` groups lines by case id: a case passes when all its runs passed,
  it is unstable with at least one pass and one fail, and `check_score` counts stable cases
  only.
* `CaseReport` carries `runs`, `pass_rate`, `range_low`, `range_high`, `unstable`; the
  markdown has `## Unstable cases` before `## Failures`; the gate adds one note.
* `ops-eval conversations --repeats N` accepts 1 to 10 and refuses the rest with exit 2
  before any request.

Everything runs on the `StubTarget` fake and `tmp_path`: no network, no wait.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest

from app.cli.evals import EXIT_OK, EXIT_USAGE
from app.cli.evals_args import UsageError, _checked_repeats
from app.evals.conversations import ConversationCase
from app.evals.gate import Baseline, evaluate, gate_comment
from app.evals.report import (
    FAILURES_HEADING,
    UNSTABLE_HEADING,
    RunReport,
    build_report,
    report_markdown,
)
from app.evals.runner import MAX_REPEATS, CaseLine, run_conversations
from app.evals.scores import compute_metrics, pass_rate, wilson_range
from app.evals.target import ConversationAnswer, TargetClient

from .builders import BASE_URL, StubTarget, answer, call, case_line, make_case, turn
from .test_cli import World, conversations, good_target, read_report, two_cases

RUN_ID = "run-repeat"
MAKEFILE = Path(__file__).resolve().parents[3] / "Makefile"


def ok_answer(case: ConversationCase) -> ConversationAnswer:
    return answer(case.id, *[turn(n + 1, f"Fine. {n + 1}") for n in range(len(case.turns))])


def three_cases() -> list[ConversationCase]:
    return [make_case(f"conv-00{n}") for n in (1, 2, 3)]


def ok_stub(cases: list[ConversationCase]) -> StubTarget:
    return StubTarget({case.id: ok_answer(case) for case in cases})


def play(
    stub: StubTarget, cases: list[ConversationCase], reports: Path, *, repeats: int
) -> tuple[CaseLine, ...]:
    return run_conversations(
        client=TargetClient(BASE_URL, client=stub.client()),
        cases=cases,
        run_id=RUN_ID,
        reports_dir=reports,
        repeats=repeats,
    )


def request_kinds(stub: StubTarget) -> list[str]:
    """`reset` or the case id, one entry per request, in the order they were sent."""
    return [
        "reset" if r.url.path == "/admin/test/reset" else str(json.loads(r.content)["case_id"])
        for r in stub.requests
    ]


def saved_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return rows


def line(
    case_id: str,
    round_number: int,
    *,
    passed: bool,
    error: str | None = None,
    verdicts: Sequence[Literal["pass", "fail"] | None] = (),
) -> CaseLine:
    """One saved result of one round."""
    return case_line(case_id, passed=passed, error=error, verdicts=verdicts).model_copy(
        update={"round": round_number}
    )


def mixed_lines() -> list[CaseLine]:
    """conv-001 passes twice, conv-002 fails twice, conv-003 passes, fails, passes."""
    return [
        line("conv-001", 1, passed=True),
        line("conv-002", 1, passed=False, error="first boom"),
        line("conv-003", 1, passed=True, verdicts=("pass",)),
        line("conv-001", 2, passed=True),
        line("conv-002", 2, passed=False, error="second boom"),
        line("conv-003", 2, passed=False, verdicts=("fail",)),
        line("conv-003", 3, passed=True, verdicts=("pass",)),
    ]


def report_of(lines: list[CaseLine]) -> RunReport:
    return build_report(
        lines,
        started_at=datetime(2026, 10, 23, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 23, 9, 20, tzinfo=UTC),
        set_name="cases.jsonl",
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=None,
        tenant="all",
        target=BASE_URL,
    )


# --- the runner -------------------------------------------------------------------


def test_three_repeats_play_every_case_three_times_with_one_reset_per_round(
    tmp_path: Path,
) -> None:
    """Catches a runner that resets once for the whole run (round 2 would start from what
    round 1 left behind), resets before every case, or plays the rounds interleaved instead
    of round by round in file order."""
    cases = three_cases()
    stub = ok_stub(cases)

    lines = play(stub, cases, tmp_path, repeats=3)

    ids = ["conv-001", "conv-002", "conv-003"]
    assert request_kinds(stub) == ["reset", *ids, "reset", *ids, "reset", *ids]
    assert [(entry.case_id, entry.round) for entry in lines] == [
        (case_id, number) for number in (1, 2, 3) for case_id in ids
    ]
    assert [row["round"] for row in saved_rows(tmp_path / f"{RUN_ID}.jsonl")] == [
        1, 1, 1, 2, 2, 2, 3, 3, 3
    ]  # fmt: skip


def test_one_repeat_is_the_default_and_plays_round_one_only(tmp_path: Path) -> None:
    """Catches a default that is not 1: the nightly and kind runs stay at one play, and a
    default of 3 would triple their cost without anyone asking."""
    cases = three_cases()
    stub = ok_stub(cases)

    lines = run_conversations(
        client=TargetClient(BASE_URL, client=stub.client()),
        cases=cases,
        run_id=RUN_ID,
        reports_dir=tmp_path,
    )

    assert stub.resets == 1
    assert len(stub.played) == 3
    assert {entry.round for entry in lines} == {1}


class DiesOnPlay(StubTarget):
    """Raises a plain error on the Nth play of one case, like a killed process."""

    def __init__(self, cases: list[ConversationCase], *, case_id: str, nth: int) -> None:
        super().__init__({case.id: ok_answer(case) for case in cases})
        self.case_id = case_id
        self.nth = nth

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/admin/test/reset":
            body = json.loads(request.content)
            if body["case_id"] == self.case_id and self.played.count(self.case_id) + 1 == self.nth:
                self.requests.append(request)
                raise RuntimeError("the process died")
        return super().__call__(request)


def test_a_stopped_repeat_run_resumes_inside_its_round_without_a_reset(
    tmp_path: Path,
) -> None:
    """Catches a resume that resets again (round 2 already started, so the reset would wipe
    what it left), replays round 1, or starts round 2 over from its first case."""
    cases = three_cases()
    first = DiesOnPlay(cases, case_id="conv-002", nth=2)
    with pytest.raises(RuntimeError):
        play(first, cases, tmp_path, repeats=2)
    assert len(saved_rows(tmp_path / f"{RUN_ID}.jsonl")) == 4
    second = ok_stub(cases)

    lines = play(second, cases, tmp_path, repeats=2)

    assert second.resets == 0
    assert second.played == ["conv-002", "conv-003"]
    assert [(entry.case_id, entry.round) for entry in lines] == [
        ("conv-001", 1),
        ("conv-002", 1),
        ("conv-003", 1),
        ("conv-001", 2),
        ("conv-002", 2),
        ("conv-003", 2),
    ]


def test_a_saved_line_with_no_round_is_round_one(tmp_path: Path) -> None:
    """Catches a loader that refuses an old run file or reads a missing round as anything
    but 1: conv-001 is saved without a `round` key, so with 2 repeats it must NOT be played
    again in round 1, while round 2 still starts with its own reset."""
    cases = three_cases()
    path = tmp_path / f"{RUN_ID}.jsonl"
    seed = DiesOnPlay(cases, case_id="conv-002", nth=1)
    with pytest.raises(RuntimeError):
        play(seed, cases, tmp_path, repeats=1)
    rows = saved_rows(path)
    assert [row["case_id"] for row in rows] == ["conv-001"]
    for row in rows:
        row.pop("round", None)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    stub = ok_stub(cases)

    lines = play(stub, cases, tmp_path, repeats=2)

    assert request_kinds(stub) == [
        "conv-002",
        "conv-003",
        "reset",
        "conv-001",
        "conv-002",
        "conv-003",
    ]
    assert [entry.round for entry in lines] == [1, 1, 1, 2, 2, 2]


def test_a_case_line_made_without_a_round_is_round_one() -> None:
    """Catches a `CaseLine` whose round is required or defaults to anything but 1."""
    assert case_line("conv-001", passed=True).round == 1


# --- the flag ---------------------------------------------------------------------


def test_the_most_repeats_allowed_is_ten() -> None:
    """Catches a changed cap: the documented range is 1 to 10."""
    assert MAX_REPEATS == 10


@pytest.mark.parametrize("value", [1, 3, 10])
def test_repeats_inside_one_to_ten_are_accepted(value: int) -> None:
    """Catches a range that is exclusive at either end: 1 and 10 are real choices."""
    assert _checked_repeats(value) == value


@pytest.mark.parametrize("value", [0, 11, -1])
def test_repeats_outside_one_to_ten_are_refused_and_name_the_flag(value: int) -> None:
    """Catches an off-by-one cap: 0 plays nothing and would report a run of no cases, and 11
    is one more than the cap."""
    with pytest.raises(UsageError, match="--repeats"):
        _checked_repeats(value)


def test_repeats_of_ten_is_accepted_and_zero_and_eleven_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches the flag checked after the target is touched: 0 and 11 exit 2 with one
    stderr line and not one request (no reset, no play), and 10 runs ten rounds."""
    cases = two_cases(tmp_path)
    for refused in ("0", "11"):
        world = World(good_target())
        world.install(monkeypatch)
        out = tmp_path / f"out-{refused}"

        code = conversations(out, cases, "--repeats", refused)

        printed = capsys.readouterr()
        assert code == EXIT_USAGE
        assert world.target.requests == []
        assert "--repeats" in printed.err
        assert not out.exists() or not list(out.glob("*.jsonl"))

    world = World(good_target())
    world.install(monkeypatch)

    code = conversations(tmp_path / "out-ten", cases, "--repeats", "10")

    assert code == EXIT_OK
    assert world.target.resets == 10
    assert len(world.target.played) == 20


# --- the numbers ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("passes", "runs", "rate", "low", "high"),
    [
        (3, 3, 100.0, 43.85, 100.0),
        (2, 3, 66.67, 20.77, 93.85),
        (0, 3, 0.0, 0.0, 56.15),
    ],
)
def test_wilson_range_at_known_points(
    passes: int, runs: int, rate: float, low: float, high: float
) -> None:
    """Catches a plain normal range (3 of 3 would have no width at all) or a wrong
    z value: these three points are the Wilson 95% interval at 3 runs."""
    assert pass_rate(passes, runs) == rate
    assert wilson_range(passes, runs) == (low, high)


class FlipsConvOne(StubTarget):
    """`conv-001` passes its first play and fails its second, so it is unstable at 2 runs."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/admin/test/reset":
            body = json.loads(request.content)
            if body["case_id"] == "conv-001":
                good = "conv-001" not in self.played
                self.requests.append(request)
                text = answer("conv-001", turn(1, "The rule says."))
                if good:
                    text = answer("conv-001", turn(1, "The rule says.", call("search_policy")))
                return httpx.Response(200, json=text.model_dump(mode="json"))
        return super().__call__(request)


def test_a_case_that_passes_and_fails_is_unstable_and_left_out_of_check_score() -> None:
    """Catches an unstable case counted as a fail (the gate would still flip on chance) or
    as a pass: of the two stable cases one passed, so check_score is 50.0, and the
    unstable one is counted on its own."""
    metrics = compute_metrics(mixed_lines())

    assert metrics["check_score"] == 50.0
    assert metrics["unstable"] == 1
    assert metrics["cases"] == 3
    assert metrics["passed"] == 1
    assert metrics["failed"] == 1


def test_one_run_marks_nothing_unstable_and_keeps_todays_score() -> None:
    """Catches a change to the one-run numbers: the baseline file must not move, so with
    one run per case the metrics are exactly today's plus `unstable: 0`."""
    lines = [
        line("conv-001", 1, passed=True, verdicts=("pass", "fail")),
        line("conv-002", 1, passed=False, verdicts=("pass",)),
    ]

    assert compute_metrics(lines) == {
        "check_score": 50.0,
        "judge_score": 66.67,
        "cases": 2,
        "passed": 1,
        "failed": 1,
        "unstable": 0,
    }


def test_an_all_unstable_run_breaches_with_no_evidence() -> None:
    """Catches an all-unstable run read as a perfect score or as zero: with no stable case
    there is no check_score, and the gate must fail on missing evidence."""
    lines = [
        line("conv-001", 1, passed=True),
        line("conv-001", 2, passed=False),
    ]
    metrics = compute_metrics(lines)
    report = report_of(lines)

    verdict = evaluate(report, Baseline(check_score=80.0))

    assert metrics["check_score"] is None
    assert metrics["unstable"] == 1
    assert report.passed is True
    assert verdict.passed is False
    assert any("no evidence" in reason for reason in verdict.reasons)


# --- the report -------------------------------------------------------------------


def test_the_report_has_one_row_per_case_with_runs_rate_and_range() -> None:
    """Catches a report that lists one row per run (7 rows here) or loses the numbers: the
    unstable case has 3 runs, a 66.67 rate and the Wilson range, and a stable case with one
    failing run first reports that run's error."""
    report = report_of(mixed_lines())

    assert [case.case_id for case in report.cases] == ["conv-001", "conv-002", "conv-003"]
    by_id = {case.case_id: case for case in report.cases}
    unstable = by_id["conv-003"]
    assert (unstable.runs, unstable.pass_rate, unstable.unstable) == (3, 66.67, True)
    assert (unstable.range_low, unstable.range_high) == (20.77, 93.85)
    assert unstable.passed is False
    assert len(unstable.judged) == 3
    assert by_id["conv-001"].unstable is False
    assert by_id["conv-001"].passed is True
    assert by_id["conv-002"].error == "first boom"
    assert report.passed is False


def test_a_report_with_only_unstable_failures_still_passes() -> None:
    """Catches `RunReport.passed` counting an unstable case as a failure: no stable case
    failed here, so the run passed."""
    report = report_of([line("conv-001", 1, passed=True), line("conv-001", 2, passed=False)])

    assert report.passed is True


def test_the_report_lists_unstable_cases_in_their_own_section() -> None:
    """Catches an unstable case listed under Failures (it must not block) or not listed at
    all (it would be invisible): it sits in its own section, before Failures, with its
    pass rate and range."""
    text = report_markdown(report_of(mixed_lines()))

    assert UNSTABLE_HEADING == "## Unstable cases"
    assert text.index(UNSTABLE_HEADING) < text.index(FAILURES_HEADING)
    unstable_part = text[text.index(UNSTABLE_HEADING) : text.index(FAILURES_HEADING)]
    failures_part = text[text.index(FAILURES_HEADING) :]
    assert "conv-003" in unstable_part
    assert "66.67" in unstable_part
    assert "20.77" in unstable_part and "93.85" in unstable_part
    assert "conv-002" not in unstable_part
    assert "conv-002" in failures_part
    assert "conv-003" not in failures_part


def test_a_report_with_no_unstable_case_has_no_unstable_section() -> None:
    """Catches an empty 'Unstable cases' heading printed on every run."""
    text = report_markdown(report_of([line("conv-001", 1, passed=True)]))

    assert UNSTABLE_HEADING not in text


# --- the gate ---------------------------------------------------------------------


def gate_lines() -> list[CaseLine]:
    return [
        line("conv-001", 1, passed=True),
        line("conv-001", 2, passed=True),
        line("conv-002", 1, passed=True),
        line("conv-002", 2, passed=False),
        line("conv-004", 1, passed=False),
        line("conv-004", 2, passed=True),
    ]


def test_the_gate_names_unstable_cases_in_a_note_not_a_breach() -> None:
    """Catches an unstable case turned into a breach (the run is above its baseline, so it
    must pass) or left out of the verdict: exactly one note names both ids and says they
    are out of check_score, and the PR comment carries it."""
    report = report_of(gate_lines())
    baseline = Baseline(check_score=80.0)

    verdict = evaluate(report, baseline)

    assert report.metrics["check_score"] == 100.0
    assert verdict.passed is True
    assert verdict.reasons == ()
    named = [note for note in verdict.notes if "conv-002" in note or "conv-004" in note]
    assert len(named) == 1
    assert "conv-002" in named[0] and "conv-004" in named[0]
    assert "check_score" in named[0]
    assert named[0] in gate_comment(verdict, report, baseline)


def test_the_gate_has_no_unstable_note_when_nothing_is_unstable() -> None:
    """Catches a note added on every run: with one clean run no note may mention unstable
    cases."""
    report = report_of([line("conv-001", 1, passed=True)])

    verdict = evaluate(report, Baseline(check_score=80.0))

    assert not any("unstable" in note.lower() for note in verdict.notes)


# --- the command and the recipe ---------------------------------------------------


def test_the_cli_passes_repeats_and_the_eval_recipe_uses_three(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a flag parsed and dropped (the stub would see 2 plays, not 6) and a recipe
    that does not pass the variable or does not default it to 3."""
    world = World(good_target())
    world.install(monkeypatch)
    out = tmp_path / "out"

    code = conversations(out, two_cases(tmp_path), "--repeats", "3")

    assert code == EXIT_OK
    assert world.target.resets == 3
    assert len(world.target.played) == 6
    assert read_report(out).metrics["cases"] == 2

    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert re.search(r"^EVAL_REPEATS \?= 3$", makefile, flags=re.MULTILINE)
    recipe = _recipe(makefile, "eval")
    assert any("ops-eval conversations" in text and "--repeats $(EVAL_REPEATS)" in text
               for text in recipe)  # fmt: skip


def _recipe(makefile: str, target: str) -> list[str]:
    lines = makefile.splitlines()
    start = next(i for i, text in enumerate(lines) if re.match(rf"{target}:(\s|$)", text))
    recipe: list[str] = []
    for text in lines[start + 1 :]:
        if text.startswith("\t"):
            recipe.append(text.strip())
        elif text.strip() and not text.startswith("#"):
            break
    return recipe


def test_the_summary_line_shows_unstable_only_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a summary that never mentions unstable cases (CI logs would hide the flip) or
    always does (a one-run line would change): it ends with ` · unstable 1` here and has no
    such word on the steady run."""
    steady = World(good_target())
    steady.install(monkeypatch)
    conversations(tmp_path / "steady", two_cases(tmp_path), "--repeats", "2")
    steady_out = capsys.readouterr().out.strip()
    flipping = World(FlipsConvOne({
        "conv-001": answer("conv-001", turn(1, "x")),
        "conv-002": answer("conv-002", turn(1, "Hello!")),
    }))  # fmt: skip
    flipping.install(monkeypatch)

    conversations(tmp_path / "flip", two_cases(tmp_path), "--repeats", "2", run_id="run-flip")

    flip_out = capsys.readouterr().out.strip()
    assert "unstable" not in steady_out
    assert flip_out.endswith(" · unstable 1")
