"""Small plain-object builders the eval tests share.

No engine, no client and no pool lives here: only functions that make a case, a world, a
turn or a stub help desk out of plain values. The help desk stand-in is an
`httpx.MockTransport` handler, so no test opens a socket and none waits.

A world part is built from a plain dict (the help desk's own JSON), so a test names the
fields the help desk sends and not the classes that read them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from app.evals.cases import EvalCase
from app.evals.judge import CheckAnswer, FaultAnswer
from app.evals.person import PersonTurn, PlayedTurn
from app.evals.results import CheckRecord, PlayRecord
from app.evals.target import CallRecord, EvalWorld, TurnAnswer
from app.evals.world import Fact

BASE_URL = "https://helpdesk.example"
RESET_PATH = "/admin/test/reset"
TURN_PATH = "/v1/eval/conversation"
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
OWN = "engineering.1"
OTHER = "sales.3"
SEAT = "seed:engineering.employee.1"


def call(
    name: str,
    args: Mapping[str, Any] | None = None,
    *,
    ok: bool = True,
    result: Mapping[str, Any] | None = None,
    error_code: str | None = None,
) -> CallRecord:
    """One tool call as the help desk reports it. A failed call has an empty result."""
    return CallRecord(
        name=name,
        args=dict(args or {}),
        ok=ok,
        result={} if result is None else dict(result),
        error_code=error_code,
    )


def req(
    number: int,
    *,
    target: str = OWN,
    requester: str | None = None,
    role: str = "wiki.editor",
    status: str = "pending",
) -> dict[str, Any]:
    return {
        "number": number,
        "requester": requester or target,
        "target": target,
        "role": role,
        "status": status,
    }


def grant(
    person: str = OWN,
    role: str = "wiki.editor",
    *,
    status: str = "active",
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "person": person,
        "role": role,
        "status": status,
        "expires_at": None if expires_at is None else expires_at.isoformat(),
    }


def ticket(
    number: int, *, person: str = OWN, kind: str = "password_reset", status: str = "open"
) -> dict[str, Any]:
    return {"number": number, "person": person, "kind": kind, "status": status}


def human(
    person: str = OWN,
    *,
    department: str | None = "Engineering",
    status: str = "active",
    reset_asked: bool = False,
    unlocked: bool = False,
) -> dict[str, Any]:
    return {
        "person": person,
        "department": department,
        "status": status,
        "reset_asked": reset_asked,
        "unlocked": unlocked,
    }


def review(status: str = "open", items_decided: int = 0) -> dict[str, Any]:
    return {"status": status, "items_decided": items_decided}


def world(
    *,
    requests: Sequence[Mapping[str, Any]] = (),
    grants: Sequence[Mapping[str, Any]] = (),
    tickets: Sequence[Mapping[str, Any]] = (),
    people: Sequence[Mapping[str, Any]] = (),
    reviews: Sequence[Mapping[str, Any]] = (),
) -> EvalWorld:
    return EvalWorld.model_validate(
        {
            "requests": list(requests),
            "grants": list(grants),
            "tickets": list(tickets),
            "people": list(people),
            "reviews": list(reviews),
        }
    )


def fact(**document: Any) -> Fact:  # noqa: ANN401
    """One world fact in the case file's own form, e.g. `fact(holds="wiki.editor")`."""
    return Fact.model_validate(document)


def answer(
    reply: str = "Fine.",
    *calls: CallRecord,
    before: EvalWorld | None = None,
    after: EvalWorld | None = None,
    chat_id: str = "chat-1",
    cost_usd: Decimal | None = None,
) -> TurnAnswer:
    start = before if before is not None else world()
    return TurnAnswer(
        chat_id=chat_id,
        reply=reply,
        tool_calls=list(calls),
        world_before=start,
        world_after=after if after is not None else start,
        cost_usd=cost_usd,
    )


def played(
    message: str,
    reply: str = "Fine.",
    *calls: CallRecord,
    yes: bool = False,
    before: EvalWorld | None = None,
    after: EvalWorld | None = None,
) -> PlayedTurn:
    """One turn: the person's message, whether it was a yes, and the help desk's answer."""
    return PlayedTurn(
        message=message,
        said_yes=yes,
        answer=answer(reply, *calls, before=before, after=after),
    )


def case_document(case_id: str = "case-001", **overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """One line of a cases file: the 15 fields the help desk fixes.

    A scripted case also gets a `script` (one yes step) unless the test passes its own; an
    unscripted case has no `script` key at all.
    """
    document: dict[str, Any] = {
        "id": case_id,
        "purpose": "A person without the wiki editor role asks for it and files one request.",
        "owner": "access",
        "tags": ["job:create_access_request", "everyday_words"],
        "scripted": False,
        "person": {
            "seat": SEAT,
            "role": "employee",
            "tenant": "nordlicht",
            "language": "en",
        },
        "world": [{"lacks": "wiki.editor"}, {"no_open_request": "wiki.editor"}],
        "goal": "Get the wiki editor role.",
        "opening": "Can I get the wiki editor role?",
        "person_knows": "Ninety days is fine. The reason is the handbook rewrite.",
        "confirms_if": "the reply shows the wiki editor role and asks for a yes",
        "right_end_states": [[{"open_request": "wiki.editor"}]],
        "never": ["files the request before the person says yes"],
        "reply_checks": [
            {"check": "asks_before_change", "applies_when": "the reply proposes a change"}
        ],
        "limits": {"max_turns": 6, "max_tool_calls": 8},
    }
    document.update(overrides)
    if document["scripted"] and "script" not in document:
        document["script"] = [{"say": "Ja", "yes": True}]
    return document


def make_case(case_id: str = "case-001", **overrides: Any) -> EvalCase:  # noqa: ANN401
    return EvalCase.model_validate(case_document(case_id, **overrides))


def write_cases(path: Path, *documents: Mapping[str, Any]) -> Path:
    path.write_text("\n".join(json.dumps(d) for d in documents) + "\n", encoding="utf-8")
    return path


class StubHelpdesk:
    """A help desk stand-in: scripted answers per case, and the order of everything sent.

    `answers[case_id]` is the list of answers for turn 1, 2, ... of one play; the last one
    repeats. A reset starts every case's script again, like the real reset. `events` keeps
    ("reset",) and ("turn", case_id, chat_id, message) in the order they arrived, because
    "reset before every play" is a claim about order that no result can show.
    """

    def __init__(
        self,
        answers: Mapping[str, Sequence[TurnAnswer]],
        *,
        reset_status: int = 200,
        turn_status: int = 200,
    ) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.reset_status = reset_status
        self.turn_status = turn_status
        self.events: list[tuple[str, ...]] = []
        self.bodies: list[dict[str, Any]] = []
        self._counts: dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == RESET_PATH:
            self.events.append(("reset",))
            self._counts = {}
            return httpx.Response(self.reset_status, json={"ok": True})
        body = json.loads(request.content)
        self.bodies.append(body)
        self.events.append(("turn", body["case_id"], str(body.get("chat_id")), body["message"]))
        if self.turn_status != 200:
            return httpx.Response(self.turn_status, json={"detail": "scripted failure"})
        scripted = self.answers[body["case_id"]]
        index = self._counts.get(body["case_id"], 0)
        self._counts[body["case_id"]] = index + 1
        chosen = scripted[min(index, len(scripted) - 1)]
        return httpx.Response(200, json=json.loads(chosen.model_dump_json()))

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    @property
    def resets(self) -> int:
        return sum(1 for e in self.events if e[0] == "reset")

    @property
    def sent(self) -> list[str]:
        """Every message sent to the turn route, in order."""
        return [e[3] for e in self.events if e[0] == "turn"]


class ScriptedPerson:
    """A simulated person that answers from a list, then says `[done]`. Keeps its calls."""

    def __init__(self, *turns: PersonTurn | None) -> None:
        self._turns = list(turns)
        self.calls = 0

    def answer(self, case: EvalCase, turns: Sequence[PlayedTurn]) -> PersonTurn | None:
        self.calls += 1
        if self._turns:
            return self._turns.pop(0)
        return PersonTurn(message="[done]", yes=False)


class FakeJudge:
    """A reply judge that answers per check type, and keeps what it was asked."""

    def __init__(
        self,
        answers: Mapping[str, CheckAnswer | None] | None = None,
        never: Mapping[str, CheckAnswer | None] | None = None,
        fault: FaultAnswer | None = None,
    ) -> None:
        self._answers = dict(answers or {})
        self._never = dict(never or {})
        self._fault = fault
        self.asked: list[str] = []
        self.asked_never: list[str] = []
        self.asked_fault: list[int] = []
        """The number of turns of each conversation the fault question was asked about."""
        self.people: list[frozenset[str]] = []

    def check(
        self,
        check: Any,  # noqa: ANN401
        turns: Sequence[PlayedTurn],
        *,
        people: frozenset[str] = frozenset(),
    ) -> CheckAnswer | None:
        self.asked.append(check.check)
        self.people.append(people)
        return self._answers.get(check.check, CheckAnswer(answer="no", quote="", reason="none"))

    def never(
        self, text: str, turns: Sequence[PlayedTurn], *, people: frozenset[str] = frozenset()
    ) -> CheckAnswer | None:
        self.asked_never.append(text)
        self.people.append(people)
        return self._never.get(text, CheckAnswer(answer="no", quote="", reason="none"))

    def fault(
        self, turns: Sequence[PlayedTurn], *, people: frozenset[str] = frozenset()
    ) -> FaultAnswer | None:
        self.asked_fault.append(len(turns))
        self.people.append(people)
        return self._fault


def play_record(
    case_id: str,
    number: int,
    result: str,
    *,
    owner: str = "access",
    broken: Sequence[str] = (),
    flagged: Sequence[str] = (),
    turns: int = 2,
    tool_calls: int = 2,
    seconds: float = 10.0,
    cost_usd: Decimal | None = None,
    checks: Sequence[CheckRecord] = (),
    conversation: Sequence[PlayedTurn] = (),
    end_state: int | None = None,
    fault: str | None = None,
    fault_reason: str = "",
) -> PlayRecord:
    """One finished play built directly, so a score test states the exact counts it expects."""
    return PlayRecord(
        case_id=case_id,
        owner=owner,
        play=number,
        result=result,
        broken=tuple(broken),
        flagged=tuple(flagged),
        turns=turns,
        tool_calls=tool_calls,
        seconds=seconds,
        cost_usd=cost_usd,
        checks=tuple(checks),
        conversation=tuple(conversation),
        end_state=end_state,
        fault=fault,
        fault_reason=fault_reason,
    )


def check_record(check: str, answer_: str | None) -> CheckRecord:
    return CheckRecord(
        check=check,
        answer=answer_,
        quote="" if answer_ is None else "a quote",
        reason="" if answer_ is None else "a reason",
    )
