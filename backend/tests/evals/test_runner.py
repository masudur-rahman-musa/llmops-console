"""`evals/target.py` and `evals/runner.py`: play every case, save each one at once, resume.

Driven through `httpx.MockTransport` (`StubTarget`), so the whole loop runs with no help
desk, no network and no sleep.

Contract these tests are written against:

* `target.RESET_PATH = "/admin/test/reset"`, `target.CONVERSATION_PATH =
  "/v1/eval/conversation"`, `TargetClient(base_url, *, client, timeout=...)` with
  `reset()` (raises `TargetUnreachable` on a non-2xx or a transport failure) and
  `play(case) -> ConversationAnswer` (posts exactly `case_id`, `tenant`, `principal_user`,
  `turns`; raises `TurnFailed` on a non-2xx, a body that is not JSON, a body that is not an
  answer, an answer for another case id, or an answer without exactly one record per turn,
  numbered 1..n).
* `runner.run_conversations(*, client, cases, run_id, reports_dir, judge=None)
  -> tuple[CaseLine, ...]`. The run id is checked with `checked_identifier` (a slug of at
  most 64 characters; `IdentifierInvalid` otherwise) before anything is sent or written.
  The file is `reports_dir / f"{run_id}.jsonl"`, one `CaseLine` JSON object per line, written
  and flushed as each case finishes. If the file has no line yet the target is reset once,
  first; otherwise it is not reset at all. Cases already in the file are not played again
  (an error line counts: it is a saved result). A `TurnFailed` becomes an error line and the
  run goes on; any other exception propagates. The return value holds every case in file
  order, saved ones included.
* `judge.Grader`: a Protocol with `grade(rubric, reply) -> JudgeVerdict | None`; `JudgeClient`
  satisfies it.
* `CaseLine`: `case_id`, `category`, `passed`, `error`, `check` (a `CaseCheck`, or none when
  the target failed), `judged` (a tuple of `JudgedReply(turn, rubric, reply, verdict, reason)`,
  verdict `None` when the judge was unusable) and `answer` (the target's answer, kept so a
  resumed run still knows which role names the run has seen).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, NoReturn

import httpx
import pytest

from app.core.validation import MAX_IDENTIFIER_LENGTH, IdentifierInvalid
from app.evals.conversations import ConversationCase
from app.evals.judge import Grader, JudgeVerdict
from app.evals.runner import CaseLine, run_conversations
from app.evals.target import (
    CONVERSATION_PATH,
    RESET_PATH,
    ConversationAnswer,
    TargetClient,
    TargetUnreachable,
    TurnFailed,
)

from .builders import BASE_URL, StubTarget, answer, call, make_case, turn

RUN_ID = "run-one"


def ok_answer(case: ConversationCase, reply: str = "Fine.") -> ConversationAnswer:
    return answer(case.id, *[turn(n + 1, f"{reply} {n + 1}") for n in range(len(case.turns))])


def three_cases() -> list[ConversationCase]:
    return [make_case(f"conv-00{n}") for n in (1, 2, 3)]


def saved(path: Path) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return lines


def run(
    stub: StubTarget,
    cases: list[ConversationCase],
    reports: Path,
    *,
    run_id: str = RUN_ID,
    judge: Grader | None = None,
) -> tuple[CaseLine, ...]:
    return run_conversations(
        client=TargetClient(BASE_URL, client=stub.client()),
        cases=cases,
        run_id=run_id,
        reports_dir=reports,
        judge=judge,
    )


# --- the target client ------------------------------------------------------------


def test_the_two_paths_are_the_help_desks_own() -> None:
    """Catches the retired `/v1/eval/turn` route coming back, or a mistyped reset path that
    404s every run."""
    assert RESET_PATH == "/admin/test/reset"
    assert CONVERSATION_PATH == "/v1/eval/conversation"


def test_a_case_is_posted_with_exactly_the_four_fields_the_route_takes() -> None:
    """Catches a body with extra keys (the route's model refuses them) or one that sends the
    expected tool calls, the graded rubrics or the category to the system under test, which
    would let it see the answers."""
    case = make_case("conv-007", turns=("one", "two"))
    stub = StubTarget({"conv-007": ok_answer(case)})

    TargetClient(BASE_URL, client=stub.client()).play(case)

    [request] = stub.requests
    assert str(request.url) == BASE_URL + CONVERSATION_PATH
    assert json.loads(request.content) == {
        "case_id": "conv-007",
        "tenant": "nordlicht",
        "principal_user": "seed:production.employee.5",
        "turns": ["one", "two"],
    }


def test_a_reset_that_does_not_answer_two_hundred_aborts_the_run() -> None:
    """Catches a reset treated as optional: a run over a seed the last run changed is not
    comparable to the baseline. A 404 means the route is off in this environment, and a
    redirect means something else answered; neither is a reset."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = TargetClient(BASE_URL, client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(TargetUnreachable):
        client.reset()


def make_bad_answers(case: ConversationCase) -> list[tuple[str, int | dict[str, Any] | str]]:
    good = ok_answer(case).model_dump(mode="json")
    return [
        ("server error", 500),
        ("not json", "<html>"),
        ("an extra field", {**good, "surprise": 1}),
        ("another case's id", {**good, "case_id": "conv-099"}),
        ("no turn records", {**good, "turns": []}),
        ("one record short", {**good, "turns": good["turns"][:1]}),
        ("one record too many", {**good, "turns": [*good["turns"], good["turns"][0]]}),
        (
            "records out of order",
            {**good, "turns": [good["turns"][1], good["turns"][0]]},
        ),
    ]


BAD_ANSWER_LABELS = [
    "server error",
    "not json",
    "an extra field",
    "another case's id",
    "no turn records",
    "one record short",
    "one record too many",
    "records out of order",
]


@pytest.mark.parametrize("index", range(8), ids=BAD_ANSWER_LABELS)
def test_an_answer_that_cannot_be_trusted_is_a_failed_turn(index: int) -> None:
    """Catches a client that accepts an answer with no or too few turn records: every check
    passes over nothing, so a target that answered `{"turns": []}` would score higher than
    one that answered correctly. Each row breaks exactly one thing about a good answer."""
    case = make_case("conv-001", turns=("a", "b"))
    _, bad = make_bad_answers(case)[index]

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(bad, int):
            return httpx.Response(bad)
        if isinstance(bad, str):
            return httpx.Response(200, content=bad)
        return httpx.Response(200, json=bad)

    client = TargetClient(BASE_URL, client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(TurnFailed):
        client.play(case)


# --- saved as each case finishes ---------------------------------------------------


def test_each_case_is_saved_when_it_finishes(tmp_path: Path) -> None:
    """Catches a runner that writes the file at the end: case 2 fails at the target (an
    error line) and case 3 crashes the whole process. The file must already hold case 1
    and case 2's error line, because the paid results of a run that dies are the point of
    saving as you go."""

    def crash(request: httpx.Request) -> NoReturn:
        raise RuntimeError("the machine went away")

    cases = three_cases()
    stub = StubTarget({"conv-001": ok_answer(cases[0]), "conv-002": 502, "conv-003": crash})

    with pytest.raises(RuntimeError):
        run(stub, cases, tmp_path)

    lines = saved(tmp_path / f"{RUN_ID}.jsonl")
    assert [line["case_id"] for line in lines] == ["conv-001", "conv-002"]
    assert lines[0]["error"] is None
    assert lines[1]["error"] is not None
    assert lines[1]["passed"] is False
    assert lines[1]["check"] is None


def test_a_failed_case_does_not_stop_the_run(tmp_path: Path) -> None:
    """Catches a runner that aborts on the first failed case: the other two are still
    played and counted, and the failed one stays in the result as a failure."""
    cases = three_cases()
    stub = StubTarget(
        {"conv-001": ok_answer(cases[0]), "conv-002": 500, "conv-003": ok_answer(cases[2])}
    )

    lines = run(stub, cases, tmp_path)

    assert [line.case_id for line in lines] == ["conv-001", "conv-002", "conv-003"]
    assert [line.error is None for line in lines] == [True, False, True]
    assert stub.played == ["conv-001", "conv-002", "conv-003"]


def test_the_target_is_reset_once_before_the_first_case_of_a_new_run(tmp_path: Path) -> None:
    """Catches a runner that never resets (the seed carries the last run's changes), one that
    resets before every case (cases act on what earlier ones filed), and one that resets
    after the first case."""
    cases = three_cases()
    stub = StubTarget({c.id: ok_answer(c) for c in cases})

    run(stub, cases, tmp_path)

    paths = [r.url.path for r in stub.requests]
    assert paths == [RESET_PATH, CONVERSATION_PATH, CONVERSATION_PATH, CONVERSATION_PATH]


def test_a_restarted_run_skips_saved_cases_and_never_resets(tmp_path: Path) -> None:
    """Catches a resume that plays everything again (paid twice), and the worse one that
    resets the help desk first: the cases act on requests earlier cases filed, so a reset
    would wipe what the saved cases left behind. Run one dies on case 3; run two, same
    run id, must send only case 3 and no reset."""

    def crash(request: httpx.Request) -> NoReturn:
        raise RuntimeError("the machine went away")

    cases = three_cases()
    first = StubTarget(
        {"conv-001": ok_answer(cases[0]), "conv-002": ok_answer(cases[1]), "conv-003": crash}
    )
    with pytest.raises(RuntimeError):
        run(first, cases, tmp_path)
    second = StubTarget({"conv-003": ok_answer(cases[2])})

    lines = run(second, cases, tmp_path)

    assert second.played == ["conv-003"]
    assert second.resets == 0
    assert [line.case_id for line in lines] == ["conv-001", "conv-002", "conv-003"]
    assert [line["case_id"] for line in saved(tmp_path / f"{RUN_ID}.jsonl")] == [
        "conv-001",
        "conv-002",
        "conv-003",
    ]


def test_a_saved_error_line_is_a_saved_result_and_is_not_played_again(tmp_path: Path) -> None:
    """Catches a resume that replays failed cases: replaying would quietly give a flaky case
    a second sample and lift the score. The first run saves conv-002 as an error."""
    cases = three_cases()
    first = StubTarget({c.id: (500 if c.id == "conv-002" else ok_answer(c)) for c in cases})
    run(first, cases, tmp_path)
    second = StubTarget({})

    lines = run(second, cases, tmp_path)

    assert second.requests == []
    assert [line.error is None for line in lines] == [True, False, True]


def test_a_different_run_id_is_a_fresh_run(tmp_path: Path) -> None:
    """Catches a resume keyed on the directory and not the run id: a new id plays all cases
    again and resets once."""
    cases = three_cases()
    run(StubTarget({c.id: ok_answer(c) for c in cases}), cases, tmp_path)
    again = StubTarget({c.id: ok_answer(c) for c in cases})

    run(again, cases, tmp_path, run_id="run-two")

    assert again.resets == 1
    assert again.played == ["conv-001", "conv-002", "conv-003"]


# --- the run id is a file name -----------------------------------------------------


def test_a_run_id_of_the_longest_allowed_length_is_accepted(tmp_path: Path) -> None:
    """Catches an off-by-one on the cap: the run id names the file, and 64 characters is the
    gateway's own identifier limit."""
    case = make_case("conv-001")
    run_id = "r" * MAX_IDENTIFIER_LENGTH

    run(StubTarget({"conv-001": ok_answer(case)}), [case], tmp_path, run_id=run_id)

    assert (tmp_path / f"{run_id}.jsonl").is_file()


@pytest.mark.parametrize(
    "run_id",
    [
        "r" * (MAX_IDENTIFIER_LENGTH + 1),
        "run-one\n",
        "run-one\r\n",
        "run-one ",
        "../run-one",
        "run/one",
        "Run-One",
        "",
    ],
)
def test_a_run_id_that_is_not_a_short_slug_is_refused_before_anything_happens(
    tmp_path: Path, run_id: str
) -> None:
    """Catches a run id used as a file name unchecked (a path out of the reports folder, or
    two spellings of one file) and a pattern that lets a trailing newline, CRLF or space
    through. Nothing is sent to the target and nothing is written."""
    case = make_case("conv-001")
    stub = StubTarget({"conv-001": ok_answer(case)})

    with pytest.raises(IdentifierInvalid):
        run(stub, [case], tmp_path, run_id=run_id)

    assert stub.requests == []
    assert list(tmp_path.iterdir()) == []


# --- checks and the judge inside the loop -----------------------------------------


def test_a_case_passes_only_when_its_checks_pass(tmp_path: Path) -> None:
    """Catches a runner that marks every answered case as passed, or does not run the checks:
    conv-001 calls the expected tool, conv-002 does not."""
    expects = ({"name": "search_policy", "args_subset": {}},)
    good = make_case("conv-001", turns=("q",), tool_calls=expects)
    bad = make_case("conv-002", turns=("q",), tool_calls=expects)
    stub = StubTarget(
        {
            "conv-001": answer("conv-001", turn(1, "Policy.", call("search_policy"))),
            "conv-002": answer("conv-002", turn(1, "No idea.")),
        }
    )

    lines = run(stub, [good, bad], tmp_path)

    assert [line.passed for line in lines] == [True, False]
    assert lines[1].check is not None
    assert lines[1].check.tools_in_order is False


def test_a_role_name_from_an_earlier_case_is_known_to_a_resumed_run(tmp_path: Path) -> None:
    """Catches a resume that forgets what saved cases returned: `Ledger Viewer` came from
    conv-001's tool in run one. In run two (same run id) conv-002 names it with no tool
    result of its own, which is made up, and only the saved answer can tell the checker
    the role exists."""
    first = make_case("conv-001", turns=("roles?",))
    second = make_case("conv-002", turns=("roles?",))
    roles = call("list_requestable_roles", result={"roles": [{"role_name": "Ledger Viewer"}]})
    run(
        StubTarget({"conv-001": answer("conv-001", turn(1, "Ledger Viewer exists.", roles))}),
        [first],
        tmp_path,
    )

    lines = run(
        StubTarget({"conv-002": answer("conv-002", turn(1, "Try Ledger Viewer."))}),
        [first, second],
        tmp_path,
    )

    assert lines[0].passed is True
    assert lines[1].passed is False
    assert lines[1].check is not None
    assert lines[1].check.turns[0].no_made_up_name is False


class FakeJudge:
    def __init__(self, verdicts: dict[str, Literal["pass", "fail"] | None]) -> None:
        self.verdicts = verdicts
        self.asked: list[tuple[str, str]] = []

    def grade(self, rubric: str, reply: str) -> JudgeVerdict | None:
        self.asked.append((rubric, reply))
        verdict = self.verdicts.get(reply)
        if verdict is None:
            return None
        return JudgeVerdict(verdict=verdict, votes=(verdict,) * 3, reasons=("r", "r", "r"))


def test_only_graded_replies_are_judged_and_kept_word_for_word(tmp_path: Path) -> None:
    """Catches a runner that judges every reply (paid for turns no rubric covers), one that
    sends the judge anything but the rubric and the reply, and one that loses a verdict:
    turns 1 and 3 are graded, turn 2 is not, and an unusable judge answer is kept as none,
    not as a pass or a fail."""
    case = make_case(
        "conv-001", turns=("a", "b", "c"), graded=((1, "rubric one"), (3, "rubric three"))
    )
    stub = StubTarget(
        {
            "conv-001": answer(
                "conv-001", turn(1, "reply one"), turn(2, "reply two"), turn(3, "reply three")
            )
        }
    )
    judge = FakeJudge({"reply one": "pass", "reply three": None})

    [line] = run(stub, [case], tmp_path, judge=judge)

    assert judge.asked == [("rubric one", "reply one"), ("rubric three", "reply three")]
    assert [(j.turn, j.rubric, j.reply, j.verdict) for j in line.judged] == [
        (1, "rubric one", "reply one", "pass"),
        (3, "rubric three", "reply three", None),
    ]


def test_without_a_judge_nothing_is_judged(tmp_path: Path) -> None:
    """Catches a runner that invents verdicts when no judge is given: the judge score must
    stay empty so the gate does not read a number nobody measured."""
    case = make_case("conv-001")
    stub = StubTarget({"conv-001": ok_answer(case)})

    [line] = run(stub, [case], tmp_path)

    assert line.judged == ()
