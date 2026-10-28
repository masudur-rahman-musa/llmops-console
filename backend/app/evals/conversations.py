r"""The conversation case: its model, its pre-checks and its loader.

One JSONL line is one case, in the help desk's own shape
(`backend/evals/conversations/cases.jsonl` in the companion repo, never copied here):
`id`, `category`, `tenant`, `principal_user`, `turns` (what the person types),
`expected` (`tool_calls` in order, `confirm_turn`) and `graded` (the replies a judge
reads, each with its rubric).

Two things are worth more than the model itself.

* Extras are forbidden at EVERY level.
  A misspelt `confirm_turns` inside `expected` would otherwise load cleanly and the
  check for a change before the yes would quietly stop applying.
* The id and turn limits are the help desk route's own, so a file the route would
  refuse is refused here, before the reset has been paid for.

`load_cases` loads a file whole or not at all, names the 1-based FILE line of a bad
case (blank lines count), and refuses a missing or empty file: a run over zero cases
has no failure and would pass the gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.errors import LibraryInputInvalid
from app.core.validation import IdentifierInvalid, checked_identifier

CASE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"conv-[0-9]{3}")
"""The case id: `conv-` and exactly three digits, matched with `fullmatch` so a trailing
newline cannot slip past a `$`."""

MIN_TURNS: Final[int] = 1
"""The fewest messages a case may hold. An empty conversation could never fail a check."""

MAX_TURNS: Final[int] = 10
"""The most messages a case may hold. The help desk route refuses more."""

MAX_PRINCIPAL_LENGTH: Final[int] = 128
"""The longest `principal_user`, inclusive."""

PRINCIPAL_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:seed:)?[a-z0-9]+(?:[._@-][a-z0-9]+)*")
"""A `principal_user`: one optional `seed:` in front, then a lowercase dotted name."""


class ConversationSetInvalid(LibraryInputInvalid):
    """A cases file could not be loaded.

    An operator error reported to whoever ran the harness, never an answer to a
    caller, so it carries no wire code. `line` is the 1-based file line of the bad
    case, or `None` when the whole file is at fault.
    """

    def __init__(self, reason: str, *, path: Path, line: int | None = None) -> None:
        self.reason = reason
        self.path = path
        self.line = line
        location = f"{path}:{line}" if line is not None else str(path)
        super().__init__(f"{location}: {reason}")


class ExpectedCall(BaseModel):
    """One tool call the case expects, with the arguments that must be in the real call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    args_subset: dict[str, Any]


class Expected(BaseModel):
    """What the case expects: the tool calls in order, and the turn of the person's yes.

    `confirm_turn` is `None` when nothing may change at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_calls: tuple[ExpectedCall, ...]
    confirm_turn: int | None


class GradedTurn(BaseModel):
    """One reply the judge reads: the turn number and the rubric it is read against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: int = Field(ge=1)
    rubric: str


class ConversationCase(BaseModel):
    """One conversation case, as the help desk wrote it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    category: str
    tenant: str
    principal_user: str
    turns: tuple[str, ...] = Field(min_length=MIN_TURNS, max_length=MAX_TURNS)
    expected: Expected
    graded: tuple[GradedTurn, ...]

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if CASE_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("id: not conv- and three digits")
        return value

    @field_validator("category", "tenant")
    @classmethod
    def _check_slug(cls, value: str) -> str:
        try:
            return checked_identifier(value, "name")
        except IdentifierInvalid as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("principal_user")
    @classmethod
    def _check_principal(cls, value: str) -> str:
        if len(value) > MAX_PRINCIPAL_LENGTH or PRINCIPAL_PATTERN.fullmatch(value) is None:
            raise ValueError("principal_user: not a seat name")
        return value

    @model_validator(mode="after")
    def _check_turn_numbers(self) -> ConversationCase:
        count = len(self.turns)
        confirm = self.expected.confirm_turn
        if confirm is not None and not 1 <= confirm <= count:
            raise ValueError(f"expected.confirm_turn: {confirm} is not a turn of 1..{count}")
        for graded in self.graded:
            if graded.turn > count:
                raise ValueError(f"graded: turn {graded.turn} is past the {count} turns")
        return self


def load_cases(path: Path) -> tuple[ConversationCase, ...]:
    """Read a cases file, in file order, or raise `ConversationSetInvalid`.

    Blank lines and lines starting `#` are skipped; the reported line is still the
    file's. A duplicate id names the second line, because saved results are keyed by
    id and the second case would be skipped on a resume as if it had been played.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConversationSetInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    cases: list[ConversationCase] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            case = ConversationCase.model_validate(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ConversationSetInvalid(
                f"is not valid JSON: {exc.msg}", path=path, line=number
            ) from exc
        except ValidationError as exc:
            raise ConversationSetInvalid(
                f"is not a valid case: {_first_problem(exc)}", path=path, line=number
            ) from exc
        if case.id in seen:
            raise ConversationSetInvalid(
                f"repeats the id {case.id!r}, first seen on line {seen[case.id]}",
                path=path,
                line=number,
            )
        seen[case.id] = number
        cases.append(case)

    if not cases:
        raise ConversationSetInvalid("holds no cases", path=path)
    return tuple(cases)


def _first_problem(exc: ValidationError) -> str:
    """The first problem of a validation error as one line."""
    problems = exc.errors()
    first = problems[0]
    field = ".".join(str(part) for part in first.get("loc") or ()) or "the case"
    message = str(first.get("msg", "is not valid")).replace("\n", " ")
    more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{field}: {message}{more}"
