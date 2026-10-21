"""Small plain-object builders the conversation eval tests share.

No engine, no client and no pool lives here: only functions that make a case, an
answer or a stub target out of plain values. A fake target is an `httpx.MockTransport`
handler, so no test opens a socket and none waits.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn

import httpx

from app.evals.conversations import ConversationCase
from app.evals.runner import CaseLine, JudgedReply
from app.evals.target import CallRecord, ConversationAnswer, TurnRecord

BASE_URL = "https://helpdesk.example"


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


def turn(number: int, reply: str, *calls: CallRecord) -> TurnRecord:
    return TurnRecord(turn=number, reply=reply, tool_calls=list(calls))


def answer(case_id: str, *turns: TurnRecord) -> ConversationAnswer:
    return ConversationAnswer(case_id=case_id, turns=list(turns))


def case_document(
    case_id: str = "conv-001",
    *,
    turns: Sequence[str] = ("I need wiki access", "yes"),
    tool_calls: Sequence[Mapping[str, Any]] = (),
    confirm_turn: int | None = None,
    graded: Sequence[tuple[int, str]] = ((1, "Does the reply ask first?"),),
    tenant: str = "nordlicht",
    principal_user: str = "seed:production.employee.5",
    category: str = "paraphrases",
) -> dict[str, Any]:
    """One line of a cases file, in the help desk's own shape."""
    return {
        "id": case_id,
        "category": category,
        "tenant": tenant,
        "principal_user": principal_user,
        "turns": list(turns),
        "expected": {"tool_calls": [dict(c) for c in tool_calls], "confirm_turn": confirm_turn},
        "graded": [{"turn": t, "rubric": r} for t, r in graded],
    }


def make_case(
    case_id: str = "conv-001",
    *,
    turns: Sequence[str] = ("I need wiki access", "yes"),
    tool_calls: Sequence[Mapping[str, Any]] = (),
    confirm_turn: int | None = None,
    graded: Sequence[tuple[int, str]] = ((1, "Does the reply ask first?"),),
) -> ConversationCase:
    document = case_document(
        case_id, turns=turns, tool_calls=tool_calls, confirm_turn=confirm_turn, graded=graded
    )
    return ConversationCase.model_validate(document)


def write_cases(path: Path, *documents: Mapping[str, Any]) -> Path:
    path.write_text("\n".join(json.dumps(d) for d in documents) + "\n", encoding="utf-8")
    return path


class StubTarget:
    """A help desk stand-in: one scripted answer per case id, and a record of every request.

    `requests` is the point of the double. "Reset once", "play only the unsaved cases" and
    "never reset on a resume" are claims about which requests were sent, and no result can
    tell a client that reset twice from one that reset once.
    """

    def __init__(
        self,
        answers: Mapping[str, ConversationAnswer | int | Callable[[httpx.Request], NoReturn]],
    ) -> None:
        self.answers = dict(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/admin/test/reset":
            return httpx.Response(200, json={"ok": True})
        body = json.loads(request.content)
        scripted = self.answers[body["case_id"]]
        if isinstance(scripted, int):
            return httpx.Response(scripted, json={"detail": "scripted failure"})
        if callable(scripted):
            scripted(request)
            raise AssertionError("a scripted callable must raise")
        return httpx.Response(200, json=scripted.model_dump(mode="json"))

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    @property
    def resets(self) -> int:
        return sum(1 for r in self.requests if r.url.path == "/admin/test/reset")

    @property
    def played(self) -> list[str]:
        """The case ids sent to the conversation route, in order."""
        return [
            json.loads(r.content)["case_id"]
            for r in self.requests
            if r.url.path == "/v1/eval/conversation"
        ]


def case_line(
    case_id: str,
    *,
    passed: bool,
    error: str | None = None,
    verdicts: Sequence[Literal["pass", "fail"] | None] = (),
    category: str = "paraphrases",
) -> CaseLine:
    """One saved case result built directly (no cross-field rules are assumed on `CaseLine`).

    `verdicts` become the judged replies, one per entry, so a score test can state the exact
    pass and fail counts it expects.
    """
    return CaseLine(
        case_id=case_id,
        category=category,
        passed=passed,
        error=error,
        check=None,
        judged=tuple(
            JudgedReply(
                turn=n + 1,
                rubric=f"rubric {n + 1}",
                reply=f"reply {n + 1}",
                verdict=v,
                reason=None if v is None else "because",
            )
            for n, v in enumerate(verdicts)
        ),
        answer=None,
    )
