"""Play every case in file order, save each one the moment it finishes, and resume.

The loop is the whole module. The result of a case is a `CaseLine`, written as one
flushed JSON line to `<reports_dir>/<run-id>.jsonl` as soon as the case is done, so a run that
dies keeps every paid
result. Starting again with the same run id plays only the cases not yet in the file,
and never resets the help desk inside a round, because the cases act on what earlier
ones filed. A new run id, or an empty file, resets once before the first case.

With `repeats` above one the whole case list is played that many times. Each pass is a
round: it starts with its own reset and plays the cases in file order. A round that already
has a saved line is resumed with no reset. A line saved before rounds existed carries no
round number and is read as round 1, so an old run file still resumes.

Rules the record keeps that no single field states:

* a failed case is a saved failure, and a resume does not replay it: replaying would
  quietly give a flaky case a second sample;
* a case passes exactly when its three checks pass; the judge never decides it;
* the target's answer is kept on the line, so a resumed run still knows which role names
  the run has seen.

The run id names a file, so it is checked as a short slug before anything is sent or
written.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.errors import LibraryInputInvalid
from app.core.validation import checked_identifier
from app.evals.checks import CaseCheck, check_case, role_names_in
from app.evals.conversations import ConversationCase, ConversationSetInvalid
from app.evals.judge import Grader, Verdict
from app.evals.target import ConversationAnswer, TargetClient, TurnFailed

MAX_REPEATS: Final[int] = 10
"""The most rounds one run may play. Each round is a paid run of every case."""


class RepeatsInvalid(LibraryInputInvalid):
    """`repeats` was outside 1 to `MAX_REPEATS`."""


def checked_repeats(value: int) -> int:
    """Refuse a round count outside 1 to `MAX_REPEATS`, before anything is sent."""
    if not 1 <= value <= MAX_REPEATS:
        raise RepeatsInvalid(f"repeats: {value} is not between 1 and {MAX_REPEATS}")
    return value


class JudgedReply(BaseModel):
    """One graded reply and what the judge said. `verdict` is `None` when it was unusable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: int
    rubric: str
    reply: str
    verdict: Verdict | None
    reason: str | None


class CaseLine(BaseModel):
    """What happened to one case: its checks, its judged replies, or why it has neither.

    `check` is `None` exactly when the target failed, and `error` carries that failure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    passed: bool
    error: str | None
    check: CaseCheck | None
    judged: tuple[JudgedReply, ...]
    answer: ConversationAnswer | None
    round: int = 1
    """Which pass over the case list this line belongs to, from 1."""


def _judge_replies(
    case: ConversationCase, answer: ConversationAnswer, judge: Grader | None
) -> tuple[JudgedReply, ...]:
    """Judge only the `graded` replies, each against its own rubric, and nothing else."""
    if judge is None:
        return ()
    judged: list[JudgedReply] = []
    for graded in case.graded:
        reply = answer.turns[graded.turn - 1].reply
        verdict = judge.grade(graded.rubric, reply)
        judged.append(
            JudgedReply(
                turn=graded.turn,
                rubric=graded.rubric,
                reply=reply,
                verdict=None if verdict is None else verdict.verdict,
                reason=(
                    None
                    if verdict is None
                    else next(
                        (
                            r
                            for v, r in zip(verdict.votes, verdict.reasons, strict=True)
                            if v == verdict.verdict
                        ),
                        None,
                    )
                ),
            )
        )
    return tuple(judged)


def _saved_lines(path: Path) -> dict[tuple[str, int], CaseLine]:
    """The lines already in the file, by case id and round.

    A last line cut short by a killed process is dropped and the file is rewritten
    without it; a bad line anywhere else is refused, since it is not a crash's doing.
    """
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").splitlines()
    kept: dict[tuple[str, int], CaseLine] = {}
    for number, text in enumerate(raw, start=1):
        if not text.strip():
            continue
        try:
            line = CaseLine.model_validate_json(text)
        except ValidationError as exc:
            if number == len(raw):
                path.write_text("".join(f"{t}\n" for t in raw[:-1]), encoding="utf-8")
                break
            raise ConversationSetInvalid(
                "holds a saved result that cannot be read", path=path, line=number
            ) from exc
        kept[(line.case_id, line.round)] = line
    return kept


def load_run(path: Path) -> tuple[CaseLine, ...]:
    """The saved lines of one finished or stopped run, in file order.

    The label maker reads a run's replies and tool results from here, so the labels
    come from the same run the baseline was set on and cost no second one. Only round 1
    is answered: one line per case, as the label maker expects.
    """
    return tuple(line for line in _saved_lines(path).values() if line.round == 1)


def _append(path: Path, line: CaseLine) -> None:
    """Add one line to the file and push it to disk before the next case starts."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_conversations(
    *,
    client: TargetClient,
    cases: Sequence[ConversationCase],
    run_id: str,
    reports_dir: Path,
    judge: Grader | None = None,
    repeats: int = 1,
) -> tuple[CaseLine, ...]:
    """Play the cases not yet saved, `repeats` rounds, and answer every line, round by round.

    A `TurnFailed` becomes an error line and the run goes on; a `TargetUnreachable` from
    the reset, and any other exception, propagate. Nothing is sent or written before the
    run id and the round count have been checked. Role names are gathered per round,
    because each round starts from a reset help desk.
    """
    checked_identifier(run_id, "run_id")
    checked_repeats(repeats)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{run_id}.jsonl"

    saved = _saved_lines(path)
    lines: list[CaseLine] = []
    for round_number in range(1, repeats + 1):
        in_round = [line for (_, r), line in saved.items() if r == round_number]
        if not in_round:
            client.reset()
        roles: set[str] = set()
        for line in in_round:
            if line.answer is not None:
                roles |= role_names_in(line.answer)
        for case in cases:
            existing = saved.get((case.id, round_number))
            if existing is not None:
                lines.append(existing)
                continue
            line = _play(client, case, judge, roles, round_number)
            _append(path, line)
            lines.append(line)
    return tuple(lines)


def _play(
    client: TargetClient,
    case: ConversationCase,
    judge: Grader | None,
    roles: set[str],
    round_number: int,
) -> CaseLine:
    """Play one case and build its line. `roles` gains the names its answer showed."""
    try:
        answer = client.play(case)
    except TurnFailed as exc:
        return CaseLine(
            case_id=case.id,
            category=case.category,
            passed=False,
            error=str(exc),
            check=None,
            judged=(),
            answer=None,
            round=round_number,
        )
    roles |= role_names_in(answer)
    check = check_case(case, answer, run_roles=frozenset(roles))
    return CaseLine(
        case_id=case.id,
        category=case.category,
        passed=check.passed,
        error=None,
        check=check,
        judged=_judge_replies(case, answer, judge),
        answer=answer,
        round=round_number,
    )
