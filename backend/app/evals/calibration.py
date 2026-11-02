"""Judge calibration: how far the judge agrees with the owner, per check type.

The judge is asked about every row of the labels file labelled by the owner, one request
per row, with the same prompt and tags a run uses. Its answers are set against the owner's,
apart for each check type, and each type gets Cohen's kappa and a trusted flag. A run then
counts reply quality only over the trusted types, and only while the judge prompt is the
one the calibration was made with.

* **Kappa is unweighted.** The three answers have no order, so `yes` against
  `does_not_apply` is as wrong as `yes` against `no`; a weight would call it a near miss.
* **Chance agreement of 1 has no kappa.** When every label and every answer is the same
  word there is nothing to tell agreement from luck, so the type is not trusted.
* **An unanswered check is dropped.** A judge outage is a fact about our gateway, not about
  the judge's reading, so it is counted apart and left out of the pairs.

Nothing here opens a socket: the judge is handed in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict

from app.core.errors import CalibrationInvalid
from app.evals.judge import JUDGE_ALIAS, CheckAnswer
from app.evals.labels import OWNER_LABEL, LabelRow, LabelTurn
from app.model_routing.policy import load_policy

TRUSTED_KAPPA: Final[float] = 0.7
"""The lowest kappa a check type is trusted at, inclusive."""

KAPPA_DECIMALS: Final[int] = 4
"""Decimals a stored kappa keeps. Trust is read off the stored number."""

CALIBRATION_INDENT: Final[int] = 2


class LabelJudge(Protocol):
    """Anything that answers a label's question about its conversation. `JudgeClient` is one."""

    def label(self, question: str, conversation: Sequence[LabelTurn]) -> CheckAnswer | None: ...


class CheckTypeResult(BaseModel):
    """One check type's numbers: rows labelled, answered by the judge, agreed, kappa, trust."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: int
    answered: int
    agreed: int
    kappa: float | None
    trusted: bool


class Calibration(BaseModel):
    """What `calibration.json` holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_model: str
    judge_prompt_sha256: str
    labels_sha256: str
    made_on: date
    check_types: dict[str, CheckTypeResult]


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Unweighted Cohen's kappa over `(label, judge)` pairs, or `None` when there is none.

    `None` for no pairs and for a chance agreement of 1, where kappa divides by zero.
    """
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(1 for label, judged in pairs if label == judged) / total
    labels = Counter(label for label, _ in pairs)
    judged = Counter(answer for _, answer in pairs)
    chance = sum(labels[word] * judged[word] for word in labels) / (total * total)
    if chance >= 1:
        return None
    return (observed - chance) / (1 - chance)


def is_trusted(kappa: float | None) -> bool:
    """Whether a kappa reaches `TRUSTED_KAPPA`; no kappa is never trusted."""
    return kappa is not None and kappa >= TRUSTED_KAPPA


def calibrate(
    rows: Sequence[LabelRow],
    judge: LabelJudge,
    *,
    judge_model: str,
    judge_prompt_sha256: str,
    labels_sha256: str,
    made_on: date,
) -> Calibration:
    """Ask the judge about every owner row and count the answers per check type.

    Raises `CalibrationInvalid` before any request when no row is the owner's: a
    calibration over the model's own labels would only measure the model against itself.
    """
    owned = [row for row in rows if row.labelled_by == OWNER_LABEL]
    if not owned:
        raise CalibrationInvalid(
            f"no label is labelled_by {OWNER_LABEL}, so there is nothing to calibrate"
        )

    labels: Counter[str] = Counter()
    pairs: dict[str, list[tuple[str, str]]] = {}
    for row in owned:
        labels[row.check_id] += 1
        pairs.setdefault(row.check_id, [])
        answer = judge.label(row.question, row.conversation)
        if answer is not None:
            pairs[row.check_id].append((row.answer, answer.answer))

    check_types: dict[str, CheckTypeResult] = {}
    for check_id, found in pairs.items():
        kappa = cohens_kappa(found)
        stored = None if kappa is None else round(kappa, KAPPA_DECIMALS)
        check_types[check_id] = CheckTypeResult(
            labels=labels[check_id],
            answered=len(found),
            agreed=sum(1 for label, judged in found if label == judged),
            kappa=stored,
            trusted=is_trusted(stored),
        )
    return Calibration(
        judge_model=judge_model,
        judge_prompt_sha256=judge_prompt_sha256,
        labels_sha256=labels_sha256,
        made_on=made_on,
        check_types=check_types,
    )


def calibration_json(calibration: Calibration) -> str:
    """`calibration.json`'s text: indented, newline-terminated."""
    return calibration.model_dump_json(indent=CALIBRATION_INDENT) + "\n"


def judge_model_from_routing(path: Path) -> str:
    """The judge alias's first candidate in `routing.yaml`, as `provider/model`.

    The provider files sit in `providers/` beside the routing file, as the gateway reads them.
    """
    policy = load_policy(path, path.parent / "providers")
    first = policy.aliases[JUDGE_ALIAS].candidates[0]
    return f"{first.provider}/{first.model}"
