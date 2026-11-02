"""The simulated person: who types the next message, and how a play ends.

The opening message is sent first and is never a yes. After each reply the person answers
from the case's `person_knows` and says yes only when the proposal matches `confirms_if`;
a message that ends in `[done]` ends the play and is not sent. The person runs on the
gateway alias `simulated-person` with a fixed prompt file. A scripted case never asks a
model: its person sends the case's `script` steps in order, each with the yes mark the
case gives it, and then stops.

The conversation travels in the user message between fixed tags, as data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, StrictBool

from app.evals.cases import EvalCase
from app.evals.judge import (
    CHAT_PATH,
    CONVERSATION_CLOSE,
    CONVERSATION_OPEN,
    DEFAULT_JUDGE_TIMEOUT_SECONDS,
    ask_retried,
    chat_body,
    prompt_text,
)
from app.evals.target import TurnAnswer, TurnFailed

PERSON_ALIAS: Final[str] = "simulated-person"
"""The alias the person names. `routing.yaml` decides which provider serves it."""

PERSON_PROMPT: Final[str] = "person-v1.md"

DONE_MARK: Final[str] = "[done]"
"""A message that ends in this ends the play."""


class PersonTurn(BaseModel):
    """What the person types next, and whether it is a yes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message: str
    yes: StrictBool


class PlayedTurn(BaseModel):
    """One turn: the person's message, whether it was a yes, and the help desk's answer."""

    model_config = ConfigDict(frozen=True)

    message: str
    said_yes: bool
    answer: TurnAnswer


class PersonModel(Protocol):
    def answer(self, case: EvalCase, turns: Sequence[PlayedTurn]) -> PersonTurn | None: ...


def converse(
    case: EvalCase,
    *,
    model: PersonModel,
    send: Callable[[str], TurnAnswer],
    should_stop: Callable[[Sequence[PlayedTurn]], bool] = lambda turns: False,
) -> tuple[PlayedTurn, ...]:
    """Play one conversation: send the opening, then answer each reply until the end.

    The play ends on `[done]`, when the person gives no answer, when a scripted person has
    no steps left, when a turn cannot be answered by the help desk, when `should_stop` says
    so after a turn, or after the case's `max_turns` replies.
    """
    turns: list[PlayedTurn] = []
    steps = iter(case.script)
    message, said_yes = case.opening, False
    while True:
        try:
            answer = send(message)
        except TurnFailed:
            break
        turns.append(PlayedTurn(message=message, said_yes=said_yes, answer=answer))
        if len(turns) >= case.limits.max_turns or should_stop(turns):
            break
        if case.scripted:
            step = next(steps, None)
            if step is None:
                break
            message, said_yes = step.say, step.yes
            continue
        reply = model.answer(case, turns)
        if reply is None or reply.message.strip().endswith(DONE_MARK):
            break
        message, said_yes = reply.message, reply.yes
    return tuple(turns)


class SimulatedPerson:
    """The `simulated-person` alias, addressed by the base URL of a gateway that serves it."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client,
        api_key: str,
        timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}{CHAT_PATH}"
        self._client = client
        self._api_key = api_key
        self._timeout = timeout

    def answer(self, case: EvalCase, turns: Sequence[PlayedTurn]) -> PersonTurn | None:
        """The person's next message, or `None` after one retry."""
        lines = [CONVERSATION_OPEN]
        for turn in turns:
            lines.append(f"Person: {turn.message}")
            lines.append(f"Help desk: {turn.answer.reply}")
        lines.append(CONVERSATION_CLOSE)
        user = (
            f"What you know: {case.person_knows}\n"
            f"Your goal: {case.goal}\n"
            f"Language: {case.person.language}\n"
            f"Say yes only if: {case.confirms_if}\n\n" + "\n".join(lines)
        )
        body = chat_body(PERSON_ALIAS, prompt_text(PERSON_PROMPT), user)
        return ask_retried(
            self._client,
            self._url,
            body,
            api_key=self._api_key,
            timeout=self._timeout,
            model=PersonTurn,
        )
