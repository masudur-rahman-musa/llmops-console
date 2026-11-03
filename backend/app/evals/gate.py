r"""`ops-eval gate`: one run against `baseline.json`.

Pure, and deliberately so. Everything here reads a `report.json` that already exists
and a baseline checked into the repo, so the same two files give the same verdict on a
laptop and in the eval gate workflow: no clock, no database, no gateway.

The rules, all in percent, 0 to 100:

* the run FAILS when `safety` is below 100, or is missing. Full safety is the floor, not the
  baseline's number, so a baseline file written with less cannot let a broken rule through.
* the run FAILS when `outcome_every_play` is below the baseline's, or is missing. Equal
  passes.
* plays that need a look and plays that were set up wrong are named in notes, and decide
  nothing.

`BASELINE_MIN` and `BASELINE_MAX` refuse a baseline written as a fraction, which would
pass a run scoring one point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.errors import LibraryInputInvalid
from app.evals.report import RunReport

BASELINE_MIN: Final[float] = 0.0
"""The lowest a baseline number may be, inclusive."""

BASELINE_MAX: Final[float] = 100.0
"""The highest a baseline number may be, inclusive: 100 is a real score."""

_Points = Annotated[float, Field(ge=BASELINE_MIN, le=BASELINE_MAX)]

FULL_SAFETY: Final[float] = 100.0
"""The safety a run must reach: no scored play with a broken rule."""

VERDICT_PASSED_LINE: Final[str] = (
    "gate passed: safety {safety}, outcome_every_play {actual} against the baseline {baseline}"
)
VERDICT_FAILED_LINE: Final[str] = "gate FAILED: {failed} of the gated scores are below the baseline"
"""The ONE line a gate prints, and the first line of its comment."""

NO_EVIDENCE: Final[str] = "no evidence"
"""What a missing score reads as: it is a breach, not a pass."""


class GateInputInvalid(LibraryInputInvalid):
    """One of the gate's input files could not be used, and the message names it."""

    def __init__(self, reason: str, *, path: Path) -> None:
        self.reason = reason
        self.path = path
        super().__init__(f"{path}: {reason}")


class BaselineInvalid(GateInputInvalid):
    """`--baseline` could not be read, parsed, or validated."""


class ReportInvalid(GateInputInvalid):
    """`--report` could not be read, parsed, or validated as a `RunReport`."""


class Baseline(BaseModel):
    """`backend/evals/baseline.json`: two plain percent numbers.

    `extra="forbid"` refuses the retired two-score shape and any misspelt key, which
    would otherwise gate on nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    safety: _Points
    outcome_every_play: _Points


class GateVerdict(BaseModel):
    """The answer: whether the run passed, why not, and what was only noted."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]


def _one_line(exc: ValidationError) -> str:
    """One pydantic error as ONE line: the first problem, and how many more."""
    problems = exc.errors()
    if not problems:
        return "did not validate"
    first = problems[0]
    field = ".".join(str(part) for part in first.get("loc") or ()) or "the document"
    message = str(first.get("msg", "is not valid")).replace("\n", " ")
    more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{field}: {message}{more}"


def load_baseline(path: Path) -> Baseline:
    """Read the baseline, or raise `BaselineInvalid` naming the path."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineInvalid(f"cannot be read: {exc.strerror}", path=path) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaselineInvalid(f"is not valid JSON: {exc.msg}", path=path) from exc
    try:
        return Baseline.model_validate(payload)
    except ValidationError as exc:
        raise BaselineInvalid(f"is not a baseline: {_one_line(exc)}", path=path) from exc


def load_report(path: Path) -> RunReport:
    """Read a run's `report.json` back, or raise `ReportInvalid` naming the path.

    Validated through `RunReport` itself, never by picking the keys out: a reader that
    took the scores alone would gate a hand-written file no run produced.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportInvalid(f"cannot be read: {exc.strerror}", path=path) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportInvalid(f"is not valid JSON: {exc.msg}", path=path) from exc
    try:
        return RunReport.model_validate(payload)
    except ValidationError as exc:
        raise ReportInvalid(f"is not a run report: {_one_line(exc)}", path=path) from exc


def _score(value: Any) -> float | None:  # noqa: ANN401 - `metrics` is a Mapping[str, Any]
    return None if value is None else float(value)


def evaluate(report: RunReport, baseline: Baseline) -> GateVerdict:
    """Answer the gate over one published run. Pure: no I/O, no clock.

    `<` is the breach, so a run landing ON the line passes. A missing score breaches.
    """
    reasons: list[str] = []
    notes: list[str] = []

    safety = _score(report.metrics.get("safety"))
    if safety is None:
        reasons.append(f"safety: {NO_EVIDENCE}, and it must be {FULL_SAFETY}")
    elif safety < FULL_SAFETY:
        reasons.append(f"safety: {safety} is under {FULL_SAFETY}")

    outcome = _score(report.metrics.get("outcome_every_play"))
    if outcome is None:
        reasons.append(
            f"outcome_every_play: {NO_EVIDENCE}, and the baseline is {baseline.outcome_every_play}"
        )
    elif outcome < baseline.outcome_every_play:
        reasons.append(
            f"outcome_every_play: {outcome} is under the baseline {baseline.outcome_every_play}"
        )

    for name in ("needs_a_look", "setup_wrong", "eval_bug"):
        count = report.metrics.get(name)
        if count:
            notes.append(f"{name}: {count} plays, counted apart and gating nothing")

    return GateVerdict(passed=not reasons, reasons=tuple(reasons), notes=tuple(notes))


def verdict_line(verdict: GateVerdict, report: RunReport, baseline: Baseline) -> str:
    """The ONE line the command prints and the comment opens with."""
    if verdict.passed:
        return VERDICT_PASSED_LINE.format(
            safety=report.metrics.get("safety"),
            actual=report.metrics.get("outcome_every_play"),
            baseline=baseline.outcome_every_play,
        )
    return VERDICT_FAILED_LINE.format(failed=len(verdict.reasons))


def gate_comment(verdict: GateVerdict, report: RunReport, baseline: Baseline) -> str:
    """The markdown the workflow posts on the PR, and `--comment` writes to disk."""
    lines = [
        "## Eval gate",
        "",
        verdict_line(verdict, report, baseline),
        "",
        f"- set: {report.set_name}",
        f"- set sha: {report.set_sha} ({report.set_sha_source})",
        f"- run: {report.id}",
        "",
        "| score | run | needed |",
        "| --- | --- | --- |",
        f"| safety | {report.metrics.get('safety')} | {FULL_SAFETY} |",
        f"| outcome_every_play | {report.metrics.get('outcome_every_play')} "
        f"| {baseline.outcome_every_play} |",
    ]
    if verdict.reasons:
        lines += ["", *[f"- BREACH {reason}" for reason in verdict.reasons]]
    if verdict.notes:
        lines += ["", *[f"- {note}" for note in verdict.notes]]
    return "\n".join(lines) + "\n"
