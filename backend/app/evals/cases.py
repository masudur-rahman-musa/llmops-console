"""The eval case: its model and its loader.

One JSONL line is one case, in the help desk's own shape: exactly 15 fields. Extras are
forbidden at every level and nothing has a default, so a misspelt `confirms_if` cannot
quietly let the simulated person say yes to anything. A never text that no rule checks
stops the load, before a run is paid for.

`load_cases` loads a file whole or not at all, names the 1-based FILE line of a bad
case (blank lines count), and refuses a missing or empty file: a run over zero cases has
no failure and would pass the gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.errors import LibraryInputInvalid
from app.evals.never_rules import NEVER_RULES
from app.evals.world import Fact

CASE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
"""A case id: lowercase letters, digits and dashes, matched with `fullmatch`."""


class CaseSetInvalid(LibraryInputInvalid):
    """A cases file could not be loaded.

    An operator error reported to whoever ran the harness, never an answer to a caller,
    so it carries no wire code. `line` is the 1-based file line of the bad case, or
    `None` when the whole file is at fault.
    """

    def __init__(self, reason: str, *, path: Path, line: int | None = None) -> None:
        self.reason = reason
        self.path = path
        self.line = line
        location = f"{path}:{line}" if line is not None else str(path)
        super().__init__(f"{location}: {reason}")


class CasePerson(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seat: str
    role: str
    tenant: str
    language: str


class ReplyCheck(BaseModel):
    """One reply check: its type, and when it applies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    applies_when: str


class Limits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_turns: int = Field(ge=1, le=20)
    max_tool_calls: int = Field(ge=0)


class EvalCase(BaseModel):
    """One eval case, as the help desk wrote it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    purpose: str
    owner: str
    tags: tuple[str, ...]
    scripted: bool
    person: CasePerson
    world: tuple[Fact, ...]
    goal: str
    opening: str
    person_knows: str
    confirms_if: str
    right_end_states: tuple[tuple[Fact, ...], ...]
    never: tuple[str, ...]
    reply_checks: tuple[ReplyCheck, ...]
    limits: Limits

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if CASE_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("id: not lowercase letters, digits and dashes")
        return value

    @field_validator("never")
    @classmethod
    def _check_never(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for text in value:
            if text not in NEVER_RULES:
                raise ValueError(f"never: no rule checks {text!r}")
        return value


def load_cases(path: Path) -> tuple[EvalCase, ...]:
    """Read a cases file, in file order, or raise `CaseSetInvalid`.

    Blank lines and lines starting `#` are skipped; the reported line is still the
    file's. A duplicate id names the second line.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CaseSetInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            case = EvalCase.model_validate(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise CaseSetInvalid(f"is not valid JSON: {exc.msg}", path=path, line=number) from exc
        except ValidationError as exc:
            raise CaseSetInvalid(
                f"is not a valid case: {_first_problem(exc)}", path=path, line=number
            ) from exc
        if case.id in seen:
            raise CaseSetInvalid(
                f"repeats the id {case.id!r}, first seen on line {seen[case.id]}",
                path=path,
                line=number,
            )
        seen[case.id] = number
        cases.append(case)

    if not cases:
        raise CaseSetInvalid("holds no cases", path=path)
    return tuple(cases)


def _first_problem(exc: ValidationError) -> str:
    """The first problem of a validation error as one line."""
    problems = exc.errors()
    first = problems[0]
    field = ".".join(str(part) for part in first.get("loc") or ()) or "the case"
    message = str(first.get("msg", "is not valid")).replace("\n", " ")
    more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{field}: {message}{more}"
