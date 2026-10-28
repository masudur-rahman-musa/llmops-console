"""The target contract: the reset, and one conversation per case.

This is the module that opens the socket to the system under test. Two POST
endpoints, both on the help desk:

    {target}/admin/test/reset          once, when a run starts fresh
    {target}/v1/eval/conversation      once per case, one attempt

The answer holds one record for every message the person typed: the reply, and every
tool call of that turn with its arguments, whether it worked and what it returned.
That is what the three checks read.

Every model forbids extras at every level, and an answer must hold exactly one
record per turn, numbered 1 to n, for the same case id.
An answer with no turn
records would pass every check over nothing, and score higher than a correct one.

Two error types, because the two failures are not the same event. A reset that did
not happen aborts the whole run: a run over an unreset seed cannot be compared to the
baseline. A case that failed is ONE case, saved as failed, one attempt, no retry: a
retried case is a second sample of a non-deterministic system and would quietly lift
the score.
"""

from __future__ import annotations

from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.evals.conversations import ConversationCase

RESET_PATH: Final[str] = "/admin/test/reset"
"""Where the seed reset lives on the help desk. A rename here 404s every run."""

CONVERSATION_PATH: Final[str] = "/v1/eval/conversation"
"""Where one case is played. The old single-reply route is not a target any more."""

DEFAULT_CASE_TIMEOUT_SECONDS: Final[float] = 300.0
"""How long one conversation may take. Up to ten real model turns, so far longer than
one reply; still a real number, because httpx's `None` means no timeout at all."""


class TargetUnreachable(Exception):
    """The seed could not be reset, so the run may not start. Names the reset URL."""


class TurnFailed(Exception):
    """One case did not produce a usable answer.

    A non-2xx, a body that is not JSON, an answer the contract does not define, a
    timeout and a refused connection all arrive as this one type, so the runner
    catches one thing to save one failed case. The message names the case id.
    """


class CallRecord(BaseModel):
    """One tool call as the help desk reports it. A refused call has `ok` false."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any]
    error_code: str | None


class TurnRecord(BaseModel):
    """One user turn: the reply and the tool calls made while answering it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: int
    reply: str
    tool_calls: list[CallRecord]


class ConversationAnswer(BaseModel):
    """What the help desk answers a whole case with: one record per user turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    turns: list[TurnRecord]


class TargetClient:
    """The help desk under evaluation, addressed by base URL.

    The `httpx.Client` is injected so the whole contract is pinned against
    `httpx.MockTransport`; both requests go to the base URL joined with this module's
    own path constant, as an absolute URL.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client,
        timeout: float = DEFAULT_CASE_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def reset(self) -> None:
        """Reset the target to its seed, or raise `TargetUnreachable`.

        Anything but a 2xx is a failure, the 404 included: a 404 means the route is
        off in this environment.
        """
        url = self._url(RESET_PATH)
        try:
            response = self._client.post(url, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise TargetUnreachable(f"{url}: could not be reached: {exc}") from exc
        if not response.is_success:
            raise TargetUnreachable(f"{url}: answered {response.status_code}, not a reset")

    def play(self, case: ConversationCase) -> ConversationAnswer:
        """Play one case and answer the target's record of it, or raise `TurnFailed`.

        Only `case_id`, `tenant`, `principal_user` and `turns` are sent: the expected
        calls and the rubrics stay here, so the system under test never sees them.
        The status is checked BEFORE the body is parsed.
        """
        url = self._url(CONVERSATION_PATH)
        payload = {
            "case_id": case.id,
            "tenant": case.tenant,
            "principal_user": case.principal_user,
            "turns": list(case.turns),
        }
        try:
            response = self._client.post(url, json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise TurnFailed(f"{case.id}: {url} could not be reached: {exc}") from exc
        if not response.is_success:
            raise TurnFailed(f"{case.id}: {url} answered {response.status_code}")
        try:
            document = response.json()
        except ValueError as exc:
            raise TurnFailed(f"{case.id}: the answer is not valid JSON: {exc}") from exc
        try:
            answer = ConversationAnswer.model_validate(document)
        except ValidationError as exc:
            raise TurnFailed(f"{case.id}: the answer is not a conversation answer: {exc}") from exc
        if answer.case_id != case.id:
            raise TurnFailed(f"{case.id}: the answer is for {answer.case_id}")
        if [record.turn for record in answer.turns] != list(range(1, len(case.turns) + 1)):
            raise TurnFailed(
                f"{case.id}: the answer holds {len(answer.turns)} turn records, "
                f"not one each for turns 1 to {len(case.turns)}"
            )
        return answer
