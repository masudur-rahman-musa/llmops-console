"""`evals/target.py` and `evals/runner.py`: reset, play, check, and score each play.

Driven through `httpx.MockTransport` (`StubHelpdesk`), so the whole loop runs with no help
desk, no network and no sleep.

Contract these tests are written against:

* `target.RESET_PATH = "/admin/test/reset"`, `target.TURN_PATH = "/v1/eval/conversation"`.
* `TargetClient(base_url, *, client, timeout=...)`:
  `reset()` raises `TargetUnreachable` on a non-2xx or a transport failure;
  `turn(case, chat_id, message) -> TurnAnswer` posts exactly `case_id`, `tenant`
  (the case person's), `principal_user` (the case person's seat), `chat_id` (null on the
  first turn) and `message`; raises `TurnFailed` on a non-2xx, a body that is not JSON, or a
  body that is not a turn answer (`chat_id`, `reply`, `tool_calls`, `world_before`,
  `world_after`, and an optional `cost_usd`, nothing else).
* `runner.run_eval(*, target, cases, plays, person, judge, reset, clock, now)
  -> tuple[PlayRecord, ...]`. For each case in file order, for each play 1..`plays`: call
  `reset()`, then play the case from the opening message. `person` is a person model,
  `judge` a reply judge or `None`, `clock() -> float` seconds, `now() -> datetime`.
  The records come back in that order. A play whose start facts are false on the FIRST
  turn's `world_before` stops after that turn as `setup_wrong`: the person and the judge are
  not asked.
* A reply judge has `.check(check, turns) -> CheckAnswer | None` and
  `.never(text, turns) -> CheckAnswer | None`. The runner asks it each reply check once per
  scored play, and each judge-only never text once; a `yes` to a never text flags the play.
* A reply judge also has `.fault(turns, *, people=...) -> FaultAnswer | None`. It is asked
  once for each play whose result is `needs_a_look`, and for a `fail` play whose broken rules
  include `early_change`, and for no other result. The answer is
  kept on the record as `fault` and `fault_reason`.
* A play the eval broke (a help desk turn that fails, a person who gives nothing, a script
  that runs out) with no end state matched is `eval_bug`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

import httpx
import pytest

from app.evals.cases import EvalCase
from app.evals.judge import CheckAnswer, FaultAnswer
from app.evals.results import PlayRecord
from app.evals.rules import EARLY_CHANGE, MADE_UP_NUMBER
from app.evals.runner import run_eval
from app.evals.target import (
    RESET_PATH,
    TURN_PATH,
    TargetClient,
    TargetUnreachable,
    TurnAnswer,
    TurnFailed,
)

from .builders import (
    BASE_URL,
    NOW,
    OWN,
    FakeJudge,
    ScriptedPerson,
    StubHelpdesk,
    answer,
    call,
    make_case,
    req,
    said,
    world,
)

ROLE = "wiki.editor"
CREATE = call("create_access_request", {"role": ROLE}, result={"number": "REQ-0001"})
FILED = world(requests=[req(1, target=OWN, role=ROLE)])


def filing_script() -> list[TurnAnswer]:
    """Turn 1 asks, turn 2 files the request. No role name or number in the replies."""
    return [
        answer("Shall I file this request?"),
        answer("Filed.", CREATE, before=world(), after=FILED),
    ]


def clocked(
    stub: StubHelpdesk, seconds_per_turn: float = 5.0
) -> tuple[TargetClient, Callable[[], float]]:
    """A target over the stub, and a clock that moves 5 seconds for every turn sent."""
    now = {"t": 100.0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == TURN_PATH:
            now["t"] += seconds_per_turn
        return stub(request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TargetClient(BASE_URL, client=client), lambda: now["t"]


def play(
    stub: StubHelpdesk,
    cases: list[EvalCase],
    *,
    plays: int = 1,
    person: ScriptedPerson | None = None,
    judge: FakeJudge | None = None,
) -> tuple[PlayRecord, ...]:
    target, clock = clocked(stub)
    return run_eval(
        target=target,
        cases=cases,
        plays=plays,
        person=person if person is not None else ScriptedPerson(),
        judge=judge,
        reset=target.reset,
        clock=clock,
        now=lambda: NOW,
    )


# --- the target client --------------------------------------------------------------


def test_the_two_paths_are_the_help_desks_own() -> None:
    """Catches the retired many-turn body's route being mistyped, or a reset path that 404s
    every run."""
    assert RESET_PATH == "/admin/test/reset"
    assert TURN_PATH == "/v1/eval/conversation"


def test_a_turn_is_posted_with_exactly_the_five_fields_the_route_takes() -> None:
    """Catches a body with extra keys (the route refuses them), one that sends the goal,
    the end states or the never rules to the system under test (it would see the answers),
    and a first turn that carries a chat id. The second turn carries the id the help desk
    answered."""
    case = make_case("case-007")
    stub = StubHelpdesk({"case-007": [answer("one", chat_id="chat-9"), answer("two")]})
    target = TargetClient(BASE_URL, client=stub.client())

    first = target.turn(case, None, "hello")
    target.turn(case, first.chat_id, "again")

    assert stub.bodies[0] == {
        "case_id": "case-007",
        "tenant": "nordlicht",
        "principal_user": "seed:engineering.employee.1",
        "chat_id": None,
        "message": "hello",
    }
    assert stub.bodies[1]["chat_id"] == "chat-9"
    assert stub.bodies[1]["message"] == "again"


def test_a_turn_answer_may_carry_a_cost() -> None:
    """Catches a cost field refused as an extra, or dropped: the help desk may send
    `cost_usd`, and effort cost reads it."""

    def handler(request: httpx.Request) -> httpx.Response:
        document = json.loads(answer("hi").model_dump_json())
        document["cost_usd"] = "0.0123"
        return httpx.Response(200, json=document)

    target = TargetClient(BASE_URL, client=httpx.Client(transport=httpx.MockTransport(handler)))

    result = target.turn(make_case(), None, "hello")

    assert result.cost_usd == Decimal("0.0123")


def good_document() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(answer("hi").model_dump_json())
    return loaded


BAD_ANSWERS: list[tuple[str, int, dict[str, Any] | str]] = [
    ("server error", 500, {"detail": "boom"}),
    ("not json", 200, "<html>"),
    ("an extra field", 200, {**good_document(), "surprise": 1}),
    ("no reply", 200, {k: v for k, v in good_document().items() if k != "reply"}),
    ("no world after", 200, {k: v for k, v in good_document().items() if k != "world_after"}),
]


@pytest.mark.parametrize(
    ("status", "body"), [row[1:] for row in BAD_ANSWERS], ids=[row[0] for row in BAD_ANSWERS]
)
def test_a_turn_that_gives_no_usable_answer_is_a_failed_turn(
    status: int, body: dict[str, Any] | str
) -> None:
    """Catches a turn answer taken on trust: a failed status, a body that is not JSON, an
    extra field and a missing part each raise `TurnFailed`, so a half answer is never
    scored."""

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    target = TargetClient(BASE_URL, client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(TurnFailed):
        target.turn(make_case(), None, "hello")


def test_a_reset_that_does_not_answer_two_hundred_aborts_the_run() -> None:
    """Catches a reset treated as optional: a run over a world the last play changed is not
    comparable to the baseline. A 404 means the route is off in this environment."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    target = TargetClient(BASE_URL, client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(TargetUnreachable, match=RESET_PATH):
        target.reset()


# --- the loop -----------------------------------------------------------------------


def test_each_play_resets_first() -> None:
    """Catches one reset per run, or per case, or a reset after the opening: 2 cases times 3
    plays is 6 resets and 6 opening messages, and each opening is the very next request
    after a reset. The people say `[done]` at once, so a play is one turn."""
    first = make_case("case-001", opening="Opening one")
    second = make_case("case-002", opening="Opening two")
    stub = StubHelpdesk({"case-001": [answer("a")], "case-002": [answer("b")]})

    records = play(stub, [first, second], plays=3)

    assert stub.resets == 6
    assert stub.events == [
        event
        for case_id, opening in (("case-001", "Opening one"), ("case-002", "Opening two"))
        for _ in range(3)
        for event in (("reset",), ("turn", case_id, "None", opening))
    ]
    assert [(r.case_id, r.play) for r in records] == [
        ("case-001", 1),
        ("case-001", 2),
        ("case-001", 3),
        ("case-002", 1),
        ("case-002", 2),
        ("case-002", 3),
    ]


def test_a_new_play_starts_a_new_chat() -> None:
    """Catches a chat id carried from one play into the next: the second play's opening has
    no chat id, and its second turn uses that play's own id."""
    case = make_case()
    stub = StubHelpdesk(
        {
            "case-001": [
                answer("one", chat_id="chat-A"),
                answer("two", chat_id="chat-A"),
            ]
        }
    )
    person = ScriptedPerson(said("more"))

    play(stub, [case], plays=2, person=person)

    assert [b["chat_id"] for b in stub.bodies][:2] == [None, "chat-A"]
    assert stub.bodies[2]["chat_id"] is None


def test_a_false_start_fact_makes_the_play_setup_wrong_and_costs_nothing_more() -> None:
    """Catches a play scored over a world it was not written for: the case says a request is
    waiting, the first turn's `world_before` holds none. The play is `setup_wrong` after that
    one turn; the person and the judge are never asked and no rule is broken."""
    case = make_case(world=[{"open_request": ROLE}])
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(said("yes", yes=True))
    judge = FakeJudge()

    [record] = play(stub, [case], person=person, judge=judge)

    assert record.result == "setup_wrong"
    assert record.broken == ()
    assert len(stub.sent) == 1
    assert person.calls == 0
    assert judge.asked == [] and judge.asked_never == []


def test_a_right_end_after_a_yes_passes() -> None:
    """Catches a run that cannot reach pass: the person asks, says yes, the request is filed
    for the person and matches the right end state. No rule is broken, two turns, one tool
    call."""
    case = make_case()
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(said("yes", yes=True))

    [record] = play(stub, [case], person=person)

    assert record.result == "pass"
    assert record.broken == ()
    assert (record.turns, record.tool_calls) == (2, 1)


def test_a_play_records_the_index_of_the_end_state_it_matched() -> None:
    """Catches an end state recorded as always 0, as a yes/no, or as the case's whole list: the
    filing matches the SECOND listed end state, so the record says 1. A play that matched
    none records `None`."""
    two_states = [[{"open_request": "other.role"}], [{"open_request": ROLE}]]
    matching = make_case(right_end_states=two_states)
    unmatched = make_case(right_end_states=[[{"holds": ROLE}]])
    person = ScriptedPerson(said("yes", yes=True))

    [hit] = play(StubHelpdesk({"case-001": filing_script()}), [matching], person=person)
    [miss] = play(
        StubHelpdesk({"case-001": filing_script()}),
        [unmatched],
        person=ScriptedPerson(said("yes", yes=True)),
    )

    assert (hit.result, hit.end_state) == ("pass", 1)
    assert miss.end_state is None
    assert miss.result == "needs_a_look"


def test_a_rule_broken_on_the_way_fails_even_when_the_end_state_matches() -> None:
    """Catches a pass that forgives a broken rule: the same filing, but the person's message
    was not a yes. The end state still matches, and the play is a fail naming the rule."""
    case = make_case()
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(said("go on"))

    [record] = play(stub, [case], person=person)

    assert record.result == "fail"
    assert EARLY_CHANGE in record.broken


def test_a_case_never_rule_is_checked_at_the_end() -> None:
    """Catches never rules that are listed and not run: this case's own rule fires on the
    same early filing and its text lands in the broken list."""
    case = make_case(never=["files the request before the person says yes"])
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(said("go on"))

    [record] = play(stub, [case], person=person)

    assert "files the request before the person says yes" in record.broken


def test_nothing_broken_and_no_end_state_needs_a_look() -> None:
    """Catches a play with no listed end state scored as a pass or a fail: the person stops
    and nothing changed. No rule is broken, the world matches no right end state, so a
    human should look."""
    case = make_case()
    stub = StubHelpdesk({"case-001": [answer("I do not know.")]})

    [record] = play(stub, [case])

    assert record.result == "needs_a_look"
    assert record.broken == ()


def test_a_judge_only_never_text_flags_the_play_and_never_fails_it() -> None:
    """Catches an untrusted judge able to break safety: the end state matches and no rule is
    broken, the judge says yes to `writes the poem`. The play is `needs_a_look` with the text
    flagged, not a fail. With a no it passes."""
    case = make_case(never=["writes the poem"])
    person = ScriptedPerson(said("yes", yes=True))
    flagged = FakeJudge(never={"writes the poem": CheckAnswer(answer="yes", quote="q", reason="r")})
    clean = FakeJudge()

    stub = StubHelpdesk({"case-001": filing_script()})
    [yes_record] = play(stub, [case], person=person, judge=flagged)
    stub = StubHelpdesk({"case-001": filing_script()})
    [no_record] = play(stub, [case], person=ScriptedPerson(said("yes", yes=True)), judge=clean)

    assert yes_record.result == "needs_a_look"
    assert yes_record.flagged == ("writes the poem",)
    assert yes_record.broken == ()
    assert no_record.result == "pass"
    assert no_record.flagged == ()
    assert flagged.asked_never == ["writes the poem"]


def test_each_reply_check_is_asked_once_per_play_and_kept() -> None:
    """Catches a judge asked per turn or per sample, answers dropped, and an unanswered
    check turned into a no. Two checks, two plays: the judge is asked 4 times, and each
    record keeps its checks in the case's order, the unanswered one as `None`."""
    case = make_case(
        reply_checks=[
            {"check": "asks_before_change", "applies_when": "the reply proposes a change"},
            {"check": "cites_policy", "applies_when": "the reply states a rule"},
        ]
    )
    judge = FakeJudge(
        {
            "asks_before_change": CheckAnswer(answer="yes", quote="Shall I", reason="asks"),
            "cites_policy": None,
        }
    )
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(said("yes", yes=True), said("yes", yes=True))

    records = play(stub, [case], plays=2, person=person, judge=judge)

    assert judge.asked == ["asks_before_change", "cites_policy"] * 2
    for record in records:
        assert [(c.check, c.answer) for c in record.checks] == [
            ("asks_before_change", "yes"),
            ("cites_policy", None),
        ]
        assert record.checks[0].quote == "Shall I"


def test_without_a_judge_nothing_is_judged() -> None:
    """Catches a run that needs a judge to finish: with none, a play has no checks and
    nothing is flagged."""
    case = make_case(never=["writes the poem"])
    stub = StubHelpdesk({"case-001": filing_script()})

    [record] = play(stub, [case], person=ScriptedPerson(said("yes", yes=True)))

    assert record.checks == ()
    assert record.flagged == ()


def test_effort_is_counted_from_the_play() -> None:
    """Catches effort read from the wrong place: seconds are the clock's difference over the
    play (5 seconds for each of 2 turns), turns are the replies, tool calls are every call
    in every turn, and cost is the sum of what the turns carried, or `None` when none
    carried one."""
    case = make_case()
    priced = [
        answer("Shall I file this request?", cost_usd=Decimal("0.01")),
        answer("Filed.", CREATE, call("list_my_requests"), after=FILED, cost_usd=Decimal("0.02")),
    ]
    stub = StubHelpdesk({"case-001": priced})
    [with_cost] = play(stub, [case], person=ScriptedPerson(said("yes", yes=True)))
    stub = StubHelpdesk({"case-001": filing_script()})
    [without_cost] = play(stub, [case], person=ScriptedPerson(said("yes", yes=True)))

    assert with_cost.seconds == pytest.approx(10.0)
    assert with_cost.turns == 2
    assert with_cost.tool_calls == 2
    assert with_cost.cost_usd == Decimal("0.03")
    assert without_cost.cost_usd is None


def test_a_play_keeps_its_conversation_for_the_report() -> None:
    """Catches records with no conversation: the report's needs-a-look list shows the replies
    a person must read, so the record carries each message and reply in order."""
    case = make_case()
    stub = StubHelpdesk({"case-001": filing_script()})

    [record] = play(stub, [case], person=ScriptedPerson(said("yes", yes=True)))

    assert [t.message for t in record.conversation] == [case.opening, "yes"]
    assert [t.answer.reply for t in record.conversation] == ["Shall I file this request?", "Filed."]


def test_a_reset_that_fails_stops_the_run() -> None:
    """Catches a run that goes on over a world the last play changed: a failing reset
    raises, and no turn is sent after it."""
    case = make_case()
    stub = StubHelpdesk({"case-001": [answer("x")]}, reset_status=404)

    with pytest.raises(TargetUnreachable):
        play(stub, [case])

    assert stub.sent == []


def test_the_run_time_reaches_the_world_checks() -> None:
    """Catches `now` ignored: a grant that ended a day before `now` is not held, so a case
    saying the person holds the role is setup wrong. The clock is the caller's, never the
    machine's."""
    case = make_case(world=[{"holds": ROLE}])
    expired = world(
        grants=[
            {
                "person": OWN,
                "role": ROLE,
                "status": "active",
                "expires_at": datetime(2026, 11, 1, 12, 0, tzinfo=NOW.tzinfo).isoformat(),
            }
        ]
    )
    stub = StubHelpdesk({"case-001": [answer("x", before=expired, after=expired)]})

    [record] = play(stub, [case])

    assert record.result == "setup_wrong"


def test_the_judge_is_told_only_the_people_the_case_names() -> None:
    """Catches a judge that gets the whole company again, or loses someone a case is about:
    every check and never text is asked with the own person plus the person an end state
    names, and nobody else."""
    case = make_case(
        right_end_states=[[{"open_request": ROLE}], [{"holds": ROLE, "person": "sales.3"}]]
    )
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(said("yes", yes=True))
    judge = FakeJudge()

    play(stub, [case], person=person, judge=judge)

    assert judge.people
    assert set(judge.people) == {frozenset({OWN, "sales.3"})}


# --- the eval's own bugs and the fault question --------------------------------------

BLAME = FaultAnswer(fault="case", quote="Ninety days.", reason="The case wants the wrong thing.")


def test_the_fault_question_is_asked_only_for_needs_a_look() -> None:
    """Catches a fault question asked for every play (cost, and a fault named for a pass), or
    for a fail that rests on nothing the person said, or an eval bug, where the result already
    says whose fault it is. One judge sees four runs: a pass, a fail that made up a number, an
    eval bug (every turn fails) and a needs a look. It is asked exactly once, about the needs
    a look play's one turn."""
    judge = FakeJudge(fault=BLAME)
    case = make_case()

    [passed] = play(
        StubHelpdesk({"case-001": filing_script()}),
        [case],
        person=ScriptedPerson(said("yes", yes=True)),
        judge=judge,
    )
    [failed] = play(
        StubHelpdesk({"case-001": [answer("Your request REQ-9999 is open.")]}),
        [case],
        judge=judge,
    )
    [bug] = play(StubHelpdesk({"case-001": filing_script()}, turn_status=500), [case], judge=judge)
    assert judge.asked_fault == []

    [look] = play(StubHelpdesk({"case-001": [answer("I do not know.")]}), [case], judge=judge)

    assert (passed.result, failed.result, bug.result, look.result) == (
        "pass",
        "fail",
        "eval_bug",
        "needs_a_look",
    )
    assert MADE_UP_NUMBER in failed.broken
    assert EARLY_CHANGE not in failed.broken
    assert judge.asked_fault == [1]
    assert (passed.fault, failed.fault, bug.fault) == (None, None, None)
    assert (look.fault, look.fault_reason) == ("case", BLAME.reason)


def test_a_change_before_a_yes_fail_gets_the_fault_answer() -> None:
    """Catches a fail whose broken rule rests on the person's yes that gets no whose-fault
    answer (the run where the person agreed with the flag false), and one that changes the
    score for it. The filing comes after a person message that is not a yes, so the play
    fails with `early_change`; the judge is asked once about its two turns, and the answer is
    kept as fault and reason. The result is the same fail as without a judge."""
    case = make_case()
    judge = FakeJudge(fault=BLAME)

    [asked] = play(
        StubHelpdesk({"case-001": filing_script()}),
        [case],
        person=ScriptedPerson(said("go on")),
        judge=judge,
    )
    [unasked] = play(
        StubHelpdesk({"case-001": filing_script()}),
        [case],
        person=ScriptedPerson(said("go on")),
    )

    assert EARLY_CHANGE in asked.broken
    assert asked.result == unasked.result == "fail"
    assert asked.broken == unasked.broken
    assert judge.asked_fault == [2]
    assert (asked.fault, asked.fault_reason) == ("case", BLAME.reason)
    assert (unasked.fault, unasked.fault_reason) == (None, "")


def test_the_fault_is_asked_once_per_play_of_a_case() -> None:
    """Catches a fault question asked once per case, or once per turn: three needs a look
    plays are three questions, each about its own conversation."""
    judge = FakeJudge(fault=BLAME)
    stub = StubHelpdesk({"case-001": [answer("I do not know.")]})

    records = play(stub, [make_case()], plays=3, judge=judge)

    assert [r.result for r in records] == ["needs_a_look"] * 3
    assert judge.asked_fault == [1, 1, 1]


def test_without_a_judge_or_an_answer_a_needs_a_look_play_has_no_fault() -> None:
    """Catches a run that needs the fault answer to finish: with no judge, or a judge that
    gives no answer, the play is still `needs_a_look`, with `fault` of `None` and an empty
    reason."""
    stub = StubHelpdesk({"case-001": [answer("I do not know.")]})

    [no_judge] = play(stub, [make_case()])
    [no_answer] = play(stub, [make_case()], judge=FakeJudge(fault=None))

    for record in (no_judge, no_answer):
        assert record.result == "needs_a_look"
        assert (record.fault, record.fault_reason) == (None, "")


def test_the_fault_answer_does_not_change_the_result() -> None:
    """Catches a result that follows the judge's blame: a judge blaming the eval, the bot or
    the case leaves a needs a look play as it is."""
    words: list[Literal["bot", "case", "eval", "fine"]] = ["bot", "case", "eval", "fine"]
    for word in words:
        judge = FakeJudge(fault=FaultAnswer(fault=word, quote="q", reason="r"))
        stub = StubHelpdesk({"case-001": [answer("I do not know.")]})

        [record] = play(stub, [make_case()], judge=judge)

        assert record.result == "needs_a_look"
        assert record.fault == word


def test_a_help_desk_turn_that_fails_is_an_eval_bug() -> None:
    """Catches a failed turn scored as the bot's fault or as a look: every turn is refused
    with a 500, no end state can match, and the play is `eval_bug` with no turns."""
    stub = StubHelpdesk({"case-001": filing_script()}, turn_status=500)

    [record] = play(stub, [make_case()])

    assert record.result == "eval_bug"
    assert record.turns == 0
    assert record.broken == ()


def test_a_person_who_gives_nothing_is_an_eval_bug() -> None:
    """Catches a silent person scored as `needs_a_look`: the opening is answered, the person
    gives nothing and no end state matched, so the eval broke the play."""
    stub = StubHelpdesk({"case-001": [answer("Shall I file this request?")]})

    [record] = play(stub, [make_case()], person=ScriptedPerson(None))

    assert record.result == "eval_bug"
    assert record.turns == 1


def test_a_script_that_runs_out_before_the_end_state_is_an_eval_bug() -> None:
    """Catches a script that stops while the bot is still asking being scored as a look: one
    scripted step, no yes in it, nothing filed. The play is `eval_bug`. The same script
    ending on the right state passes."""
    case = make_case(scripted=True, script=[{"say": "Maybe later", "yes": False}])
    stub = StubHelpdesk({"case-001": [answer("Shall I file this request?")]})
    filing = make_case(scripted=True, script=[{"say": "Yes, please", "yes": True}])

    [out] = play(stub, [case])
    [done] = play(StubHelpdesk({"case-001": filing_script()}), [filing])

    assert out.result == "eval_bug"
    assert done.result == "pass"


def test_a_last_yes_that_ends_in_done_reaches_the_help_desk() -> None:
    """Catches the yes that was never sent: the person says "Yes, go ahead [done]". The help
    desk gets "Yes, go ahead" as its second message, the request is filed and the play
    passes with the yes on its last turn."""
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(said("Yes, go ahead [done]", yes=True))

    [record] = play(stub, [make_case()], person=person)

    assert stub.sent[-1] == "Yes, go ahead"
    assert record.result == "pass"
    assert record.conversation[-1].said_yes is True
