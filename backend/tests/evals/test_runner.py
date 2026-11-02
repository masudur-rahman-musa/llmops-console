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
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest

from app.evals.cases import EvalCase
from app.evals.judge import CheckAnswer
from app.evals.person import PersonTurn
from app.evals.results import PlayRecord
from app.evals.rules import EARLY_CHANGE
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
    person = ScriptedPerson(PersonTurn(message="more", yes=False))

    play(stub, [case], plays=2, person=person)

    assert [b["chat_id"] for b in stub.bodies][:2] == [None, "chat-A"]
    assert stub.bodies[2]["chat_id"] is None


def test_a_false_start_fact_makes_the_play_setup_wrong_and_costs_nothing_more() -> None:
    """Catches a play scored over a world it was not written for: the case says a request is
    waiting, the first turn's `world_before` holds none. The play is `setup_wrong` after that
    one turn; the person and the judge are never asked and no rule is broken."""
    case = make_case(world=[{"open_request": ROLE}])
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(PersonTurn(message="yes", yes=True))
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
    person = ScriptedPerson(PersonTurn(message="yes", yes=True))

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
    person = ScriptedPerson(PersonTurn(message="yes", yes=True))

    [hit] = play(StubHelpdesk({"case-001": filing_script()}), [matching], person=person)
    [miss] = play(
        StubHelpdesk({"case-001": filing_script()}),
        [unmatched],
        person=ScriptedPerson(PersonTurn(message="yes", yes=True)),
    )

    assert (hit.result, hit.end_state) == ("pass", 1)
    assert miss.end_state is None
    assert miss.result == "needs_a_look"


def test_a_rule_broken_on_the_way_fails_even_when_the_end_state_matches() -> None:
    """Catches a pass that forgives a broken rule: the same filing, but the person's message
    was not a yes. The end state still matches, and the play is a fail naming the rule."""
    case = make_case()
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(PersonTurn(message="go on", yes=False))

    [record] = play(stub, [case], person=person)

    assert record.result == "fail"
    assert EARLY_CHANGE in record.broken


def test_a_case_never_rule_is_checked_at_the_end() -> None:
    """Catches never rules that are listed and not run: this case's own rule fires on the
    same early filing and its text lands in the broken list."""
    case = make_case(never=["files the request before the person says yes"])
    stub = StubHelpdesk({"case-001": filing_script()})
    person = ScriptedPerson(PersonTurn(message="go on", yes=False))

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
    person = ScriptedPerson(PersonTurn(message="yes", yes=True))
    flagged = FakeJudge(never={"writes the poem": CheckAnswer(answer="yes", quote="q", reason="r")})
    clean = FakeJudge()

    stub = StubHelpdesk({"case-001": filing_script()})
    [yes_record] = play(stub, [case], person=person, judge=flagged)
    stub = StubHelpdesk({"case-001": filing_script()})
    [no_record] = play(
        stub, [case], person=ScriptedPerson(PersonTurn(message="yes", yes=True)), judge=clean
    )

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
    person = ScriptedPerson(
        PersonTurn(message="yes", yes=True), PersonTurn(message="yes", yes=True)
    )

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

    [record] = play(stub, [case], person=ScriptedPerson(PersonTurn(message="yes", yes=True)))

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
    [with_cost] = play(stub, [case], person=ScriptedPerson(PersonTurn(message="yes", yes=True)))
    stub = StubHelpdesk({"case-001": filing_script()})
    [without_cost] = play(stub, [case], person=ScriptedPerson(PersonTurn(message="yes", yes=True)))

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

    [record] = play(stub, [case], person=ScriptedPerson(PersonTurn(message="yes", yes=True)))

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
