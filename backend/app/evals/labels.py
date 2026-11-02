"""The labels file: one row per reply check, as the owner reviewed it.

One JSONL line is one label: the case and play it came from, the check type, the question
the judge is asked, the conversation it judged (the person's and the bot's turns with tool
results), the answer (`yes`, `no` or `does_not_apply`), a reason, who labelled it (`model`
or `musa`) and when. The project only reads this file: rows are prepared outside it and
reviewed by the owner, and only the owner's rows count in a calibration.

`load_labels` loads a file whole or not at all and names the 1-based FILE line of a bad
row. Extras are forbidden at every level. An empty file is no rows, since the project ships
one; a missing file is an error, so a typo in a path never calibrates nothing in silence.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.errors import LabelsInvalid
from app.evals.runner import MAX_PLAYS

CHECK_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9_]{1,64}")
"""A check type: lowercase letters, digits and `_`, 1 to 64 characters, with `fullmatch`."""

LabelAnswer = Literal["yes", "no", "does_not_apply"]
LabelledBy = Literal["model", "musa"]

OWNER_LABEL: Final[LabelledBy] = "musa"
"""The only labeller whose rows a calibration counts."""


class LabelToolCall(BaseModel):
    """One tool call in a labelled turn, with its result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    args: dict[str, Any]
    ok: bool
    error_code: str | None
    result: Any


class LabelTurn(BaseModel):
    """One turn of a labelled conversation: what the person said and what the bot did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    person: str
    reply: str
    tool_calls: tuple[LabelToolCall, ...]


class LabelRow(BaseModel):
    """One reply check of one play, with the answer its labeller gave."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    play: int = Field(ge=1, le=MAX_PLAYS)
    check_id: str
    question: str
    conversation: tuple[LabelTurn, ...]
    answer: LabelAnswer
    reason: str
    labelled_by: LabelledBy
    labelled_on: date

    @field_validator("check_id")
    @classmethod
    def _check_check_id(cls, value: str) -> str:
        if CHECK_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("not 1 to 64 lowercase letters, digits and `_`")
        return value


def load_labels(path: Path) -> tuple[LabelRow, ...]:
    """Read a labels file, in file order, or raise `LabelsInvalid`.

    Blank lines are skipped; the reported line is still the file's. A second row for the
    same case, play and check names the second line.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelsInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    rows: list[LabelRow] = []
    seen: dict[tuple[str, int, str], int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = LabelRow.model_validate(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise LabelsInvalid(f"is not valid JSON: {exc.msg}", path=path, line=number) from exc
        except ValidationError as exc:
            raise LabelsInvalid(
                f"is not a valid label: {_first_problem(exc)}", path=path, line=number
            ) from exc
        key = (row.case_id, row.play, row.check_id)
        if key in seen:
            raise LabelsInvalid(
                f"duplicate of the label on line {seen[key]} for the same case, play and check",
                path=path,
                line=number,
            )
        seen[key] = number
        rows.append(row)
    return tuple(rows)


def _first_problem(exc: ValidationError) -> str:
    """The first problem of a validation error as one line."""
    problems = exc.errors()
    first = problems[0]
    field = ".".join(str(part) for part in first.get("loc") or ()) or "the label"
    message = str(first.get("msg", "is not valid")).replace("\n", " ")
    more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{field}: {message}{more}"
