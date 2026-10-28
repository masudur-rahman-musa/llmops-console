"""Calibrating the judge against reference labels: the file, the kappa, the trust flag.

A judge nobody checked is an opinion with a version hash. This module grades a file of
labelled replies through the SAME judge a run uses (the fixed prompt, three samples,
the majority) and reports how well the two agree, as one number the gate can refuse to
trust.

Four properties are worth more than the arithmetic.

* **Unweighted kappa over two classes**.
  Pass and fail have no distance between them, so weights would add nothing, and plain
  percent agreement would reward a judge that always says the common word. The sum is
  done in whole numbers and divided once.
* **Chance agreement of 1 is `None`, never 1.0**.
When every label and every
  answer is the same word the agreement is real and says nothing, and a gate reading
  1.0 would trust a judge nothing measured.
* **`MIN_CALIBRATION_LABELS` is applied twice**: to the file's count before a single
  request is paid for, and to the pairs that SURVIVE the judge's drops.
* **A label the judge could not grade is dropped**, never graded as a fail and never as
  the label's own word.

Nothing here opens a socket: the grader is handed in.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.errors import LibraryError, LibraryInputInvalid
from app.evals.conversations import CASE_ID_PATTERN
from app.evals.judge import Grader, Verdict, judge_prompt_version

MIN_CALIBRATION_LABELS: Final[int] = 40
"""How many labels a calibration needs, at both ends. A floor, not a hint: the run
reads a written `calibration.json` as trust."""

JUDGE_TRUST_MIN: Final[float] = 0.7
"""The lowest kappa that is still trusted, INCLUSIVE. Read with `>=`; `None` is never
trusted."""

KAPPA_SCALE: Final[int] = 3
"""Decimal places the kappa is published at; two cannot separate 0.699 from 0.700."""

KAPPA_MIN: Final[float] = -1.0
KAPPA_MAX: Final[float] = 1.0
"""The ends of the kappa scale, inclusive, checked on the way IN: the number lands in
`eval_runs.judge_kappa`, a column with no bounds of its own."""

MAX_LABELS_NAME_LENGTH: Final[int] = 128
"""The longest `labels_name` a read artifact may carry; it is copied into `report.json`."""

CALIBRATION_JSON_NAME: Final[str] = "calibration.json"
"""The one file a calibration writes, named by a constant."""

CALIBRATION_INDENT: Final[int] = 2

Source = Literal["model-reference", "human"]


class CalibrationSetInvalid(LibraryInputInvalid):
    """A labels file or a `calibration.json` could not be loaded.

    `line` is the 1-based file line, or `None` when the file itself is at fault.
    """

    def __init__(self, reason: str, *, path: Path, line: int | None = None) -> None:
        self.reason = reason
        self.path = path
        self.line = line
        location = f"{path}:{line}" if line is not None else str(path)
        super().__init__(f"{location}: {reason}")


class CalibrationRefused(LibraryError):
    """Too few labels, or too few that the judge could grade, for a calibration to mean
    anything. The file was fine; the operator's next move is to look at the gateway."""


class ConversationLabel(BaseModel):
    """One labelled reply: the eight fields of a line in `conversation-labels.jsonl`.

    `turn` and `rubric` are copied from the case's `graded` entry, `reply` word for word
    from the run, `votes` are the label maker's three grades, and `source` says whether
    a model or a person made the label.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    turn: int = Field(ge=1)
    rubric: str
    reply: str
    verdict: Verdict
    votes: tuple[Verdict, ...] = Field(min_length=3, max_length=3)
    reason: str
    source: Source

    @field_validator("case_id")
    @classmethod
    def _check_case_id(cls, value: str) -> str:
        if CASE_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("case_id: not conv- and three digits")
        return value


class CalibrationReport(BaseModel):
    """One calibration: how well the judge agreed, and whether to trust it.

    `judge_prompt_version` keeps it honest across time: a calibration made under one
    prompt and read against a run made under another is a stale number worn as trust.
    `kappa` is `None` when chance agreement was total, and `trusted` is then false.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_prompt_version: str
    labels_name: str
    labels_sha: str
    pairs: int
    kappa: float | None
    trusted: bool


def load_labels(path: Path) -> tuple[ConversationLabel, ...]:
    """Read a labels file, in file order, or raise `CalibrationSetInvalid`.

    Blank lines and `#` lines are skipped; the reported line is the FILE's. Loaded whole
    or not at all. The 40-label floor is NOT here: `calibrate` applies it, so a short
    file loads for inspection and is refused where it would cost money.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationSetInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    labels: list[ConversationLabel] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            labels.append(ConversationLabel.model_validate(json.loads(stripped)))
        except json.JSONDecodeError as exc:
            raise CalibrationSetInvalid(
                f"is not valid JSON: {exc.msg}", path=path, line=number
            ) from exc
        except ValidationError as exc:
            raise CalibrationSetInvalid(
                f"is not a valid label: {exc}", path=path, line=number
            ) from exc
    return tuple(labels)


def cohen_kappa(labels: Sequence[Verdict], judged: Sequence[Verdict]) -> float | None:
    """Cohen's kappa over the two classes, unweighted, or `None` when it is undefined.

    `(agree * n - E) / (n * n - E)` where `E` is the sum, over the two words, of the
    count in `labels` times the count in `judged`: whole numbers until the one division.
    Rounded half up at `KAPPA_SCALE` places through `Decimal`, never through `round()`.
    """
    pairs = len(labels)
    agree = sum(1 for left, right in zip(labels, judged, strict=True) if left == right)
    expected = sum(
        sum(1 for item in labels if item == word) * sum(1 for item in judged if item == word)
        for word in ("pass", "fail")
    )
    if pairs * pairs == expected:
        return None
    kappa = Decimal(agree * pairs - expected) / Decimal(pairs * pairs - expected)
    return float(kappa.quantize(Decimal(1).scaleb(-KAPPA_SCALE), rounding=ROUND_HALF_UP))


def is_trusted(kappa: float | None) -> bool:
    """True when the kappa is at `JUDGE_TRUST_MIN` or better. `None` is not trusted."""
    return kappa is not None and kappa >= JUDGE_TRUST_MIN


def calibrate(
    labels: Sequence[ConversationLabel],
    *,
    judge: Grader,
    labels_name: str,
    labels_sha: str,
) -> CalibrationReport:
    """Grade every label through the judge and report the agreement.

    Raises `CalibrationRefused` BEFORE any request when the file holds fewer than
    `MIN_CALIBRATION_LABELS`, and again when fewer than that many labels could be
    graded: a kappa over 39 pairs read as calibration is worse than none.
    """
    if len(labels) < MIN_CALIBRATION_LABELS:
        raise CalibrationRefused(
            f"{len(labels)} labels, fewer than the {MIN_CALIBRATION_LABELS} a calibration needs"
        )
    truth: list[Verdict] = []
    judged: list[Verdict] = []
    for label in labels:
        verdict = judge.grade(label.rubric, label.reply)
        if verdict is None:
            continue
        truth.append(label.verdict)
        judged.append(verdict.verdict)
    if len(truth) < MIN_CALIBRATION_LABELS:
        raise CalibrationRefused(
            f"only {len(truth)} of {len(labels)} labels could be graded, fewer than the "
            f"{MIN_CALIBRATION_LABELS} a calibration needs"
        )
    kappa = cohen_kappa(truth, judged)
    return CalibrationReport(
        judge_prompt_version=judge_prompt_version(),
        labels_name=labels_name,
        labels_sha=labels_sha,
        pairs=len(truth),
        kappa=kappa,
        trusted=is_trusted(kappa),
    )


def read_calibration(path: Path) -> CalibrationReport:
    """Read one `calibration.json` back, or raise `CalibrationSetInvalid` naming the path.

    Validated through `CalibrationReport` itself (extras forbidden), then the kappa is
    held to its scale and the label file name to its cap.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationSetInvalid(f"cannot be read: {exc.strerror}", path=path) from exc
    try:
        report = CalibrationReport.model_validate(json.loads(text))
    except json.JSONDecodeError as exc:
        raise CalibrationSetInvalid(f"is not valid JSON: {exc.msg}", path=path) from exc
    except ValidationError as exc:
        raise CalibrationSetInvalid(f"is not a calibration: {exc}", path=path) from exc
    if report.kappa is not None and not KAPPA_MIN <= report.kappa <= KAPPA_MAX:
        raise CalibrationSetInvalid(
            f"carries a kappa of {report.kappa}, outside {KAPPA_MIN}..{KAPPA_MAX}", path=path
        )
    if len(report.labels_name) > MAX_LABELS_NAME_LENGTH:
        raise CalibrationSetInvalid(
            f"names a labels file longer than {MAX_LABELS_NAME_LENGTH} characters", path=path
        )
    return report


def write_calibration(out: Path, report: CalibrationReport) -> None:
    """Write exactly `CALIBRATION_JSON_NAME` into `out`, overwriting what is there."""
    (out / CALIBRATION_JSON_NAME).write_text(
        report.model_dump_json(indent=CALIBRATION_INDENT) + "\n", encoding="utf-8"
    )
