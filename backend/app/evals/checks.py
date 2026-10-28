r"""The three per-turn checks, over plain objects. No network, no clock.

A case passes when every check on every turn passes (`CaseCheck.passed`):

1. **The right tools, in order.** The case's successful calls, across the whole
   conversation, hold `expected.tool_calls` as an ordered subsequence, each
   `args_subset` contained in its call's `args`. Extra calls in between are fine.
A refused call (`ok` false)
   never counts as the expected call.
2. **No change before the yes.** A successful call to one of the help desk's 12 change
   tools in a turn before `confirm_turn` fails that turn, and with a null
   `confirm_turn` any successful change fails its turn. A refused change changed
   nothing and never counts.
3. **No made-up name.** A `REQ-` or `TCK-` number of four or more digits in a reply
   must be held by a tool result in this case, at or before that turn. A role name
   some tool returned anywhere in the run must be held the same way. A role name no
   tool ever returned is not caught here; the judge's rubric covers it.

The pattern for a number needs four digits and is anchored on both sides, so
`REQ-007` is no number and `REQ-00071` is never cut to `REQ-0007`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from app.evals.conversations import ConversationCase
from app.evals.target import CallRecord, ConversationAnswer

CHANGE_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "create_access_request",
        "reset_own_password",
        "unlock_own_account",
        "approve_request",
        "reject_request",
        "revoke_access",
        "onboard_user",
        "transfer_user",
        "offboard_user",
        "open_access_review",
        "decide_review_item",
        "create_ticket",
    }
)
"""The help desk's 12 change tools, as one constant. Every other tool only reads."""

NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])(?:REQ|TCK)-[0-9]{4,}(?![0-9])"
)
"""A request or ticket number: four or more digits, whole on both sides."""

ROLE_NAME_KEY: Final[str] = "role_name"
"""The key a tool result holds a role's display name under."""


class TurnCheck(BaseModel):
    """The two per-turn checks for one turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: int
    no_early_change: bool
    no_made_up_name: bool

    @property
    def passed(self) -> bool:
        """True when both checks hold on this turn."""
        return self.no_early_change and self.no_made_up_name


class CaseCheck(BaseModel):
    """All three checks for one case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tools_in_order: bool
    turns: tuple[TurnCheck, ...]

    @property
    def passed(self) -> bool:
        """True when the tools were right and every turn passed."""
        return self.tools_in_order and all(turn.passed for turn in self.turns)


def _walk(value: Any) -> Iterator[tuple[str | None, Any]]:  # noqa: ANN401 - JSON of any shape
    """Every (key, value) pair at any depth of a JSON value. List items have key `None`."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key, inner
            yield from _walk(inner)
    elif isinstance(value, list):
        for inner in value:
            yield None, inner
            yield from _walk(inner)


def _roles_in_calls(calls: list[CallRecord]) -> set[str]:
    found: set[str] = set()
    for call in calls:
        for key, value in _walk(call.result):
            if key == ROLE_NAME_KEY and isinstance(value, str):
                found.add(value)
    return found


def role_names_in(answer: ConversationAnswer) -> frozenset[str]:
    """Every role name any tool result of the answer holds, at any depth."""
    return frozenset(_roles_in_calls([c for turn in answer.turns for c in turn.tool_calls]))


def _contains(args: dict[str, Any], subset: dict[str, Any]) -> bool:
    return all(key in args and args[key] == value for key, value in subset.items())


def _tools_in_order(case: ConversationCase, answer: ConversationAnswer) -> bool:
    made = [call for turn in answer.turns for call in turn.tool_calls if call.ok]
    position = 0
    for expected in case.expected.tool_calls:
        while position < len(made) and not (
            made[position].name == expected.name
            and _contains(made[position].args, expected.args_subset)
        ):
            position += 1
        if position == len(made):
            return False
        position += 1
    return True


def _numbers_in(calls: list[CallRecord]) -> set[str]:
    found: set[str] = set()
    for call in calls:
        found.update(NUMBER_PATTERN.findall(json.dumps(call.result, ensure_ascii=False)))
    return found


def _names(role: str, text: str) -> bool:
    """True when `text` holds `role` as a whole name, not inside a longer word."""
    return re.search(rf"(?<!\w){re.escape(role)}(?!\w)", text) is not None


def check_case(
    case: ConversationCase,
    answer: ConversationAnswer,
    *,
    run_roles: frozenset[str] = frozenset(),
) -> CaseCheck:
    """Run the three checks on one played case.

    `run_roles` is every role name any tool returned anywhere in the run; the caller
    builds it with `role_names_in`. Pure: the same inputs give the same result.
    """
    confirm = case.expected.confirm_turn
    seen_calls: list[CallRecord] = []
    turns: list[TurnCheck] = []
    for record in answer.turns:
        seen_calls.extend(record.tool_calls)
        changed = any(call.ok and call.name in CHANGE_TOOLS for call in record.tool_calls)
        early = changed and (confirm is None or record.turn < confirm)

        # What the person typed up to this turn is theirs, not an invention: a reply that
        # repeats "REQ-0001" from their own message is quoting, not making it up.
        said = "\n".join(case.turns[: record.turn])
        held_numbers = _numbers_in(seen_calls) | set(NUMBER_PATTERN.findall(said))
        held_roles = _roles_in_calls(seen_calls)
        made_up = any(number not in held_numbers for number in NUMBER_PATTERN.findall(record.reply))
        made_up = made_up or any(
            role not in held_roles and _names(role, record.reply) and not _names(role, said)
            for role in run_roles
        )
        turns.append(
            TurnCheck(turn=record.turn, no_early_change=not early, no_made_up_name=not made_up)
        )
    return CaseCheck(tools_in_order=_tools_in_order(case, answer), turns=tuple(turns))
