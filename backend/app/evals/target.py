"""The target contract: the reset, and one turn at a time.

This is the module that opens the socket to the system under test. Two POST
endpoints, both on the help desk:

    {target}/admin/test/reset          before every play
    {target}/v1/eval/conversation      once per turn

A turn answer holds the reply, every tool call of that turn with its arguments, whether
it worked and what it returned, and the world as the help desk saw it before and after
the turn. That is what the rules and the end states read.

Every model forbids extras at every level. An answer with a part missing would pass
every check over nothing and score higher than a correct one.

Two error types, because the two failures are not the same event. A reset that did
not happen aborts the whole run: a run over an unreset seed cannot be compared to the
baseline. A turn that failed is ONE play, saved as failed, no retry: a retried turn is a
second sample of a non-deterministic system and would quietly lift the score.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from app.evals.cases import EvalCase

RESET_PATH: Final[str] = "/admin/test/reset"
"""Where the seed reset lives on the help desk. A rename here 404s every run."""

TURN_PATH: Final[str] = "/v1/eval/conversation"
"""Where one turn is played."""

DEFAULT_CASE_TIMEOUT_SECONDS: Final[float] = 300.0
"""How long one turn may take. Still a real number, because httpx's `None` means no
timeout at all."""


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


class WorldRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int
    requester: str
    target: str
    role: str
    status: str


class WorldGrant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    person: str
    role: str
    status: str
    expires_at: datetime | None


class WorldTicket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int
    person: str
    kind: str
    status: str


class WorldPerson(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    person: str
    department: str | None
    status: str
    reset_asked: bool
    unlocked: bool


class WorldReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    items_decided: int


class EvalWorld(BaseModel):
    """The tenant's rows the checks read, as the help desk reports them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: list[WorldRequest]
    grants: list[WorldGrant]
    tickets: list[WorldTicket]
    people: list[WorldPerson]
    reviews: list[WorldReview]


class TurnAnswer(BaseModel):
    """What the help desk answers one turn with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chat_id: str
    reply: str
    tool_calls: list[CallRecord]
    world_before: EvalWorld
    world_after: EvalWorld
    cost_usd: Decimal | None = None


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

    def turn(self, case: EvalCase, chat_id: str | None, message: str) -> TurnAnswer:
        """Send one message and answer the target's record of the turn, or raise `TurnFailed`.

        Only the five fields the route takes are sent: the goal, the end states and the
        never rules stay here, so the system under test never sees them. The status is
        checked BEFORE the body is parsed.
        """
        url = self._url(TURN_PATH)
        payload = {
            "case_id": case.id,
            "tenant": case.person.tenant,
            "principal_user": case.person.seat,
            "chat_id": chat_id,
            "message": message,
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
            return TurnAnswer.model_validate(document)
        except ValidationError as exc:
            raise TurnFailed(f"{case.id}: the answer is not a turn answer: {exc}") from exc
