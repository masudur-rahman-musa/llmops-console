"""The loop: for each case and each play, reset, converse, check, and score the play.

Every play starts from freshly reset data, so no play can see what an earlier one changed. A
play's start facts are checked on the world the help desk returned before the first turn: a
false fact makes the play `setup_wrong`, which stops it after that one turn and costs the
person and the judge nothing.

After the conversation, code checks the shared rules and the case's own never rules and
matches the world after to the right end states; then the judge, when there is one, answers
each reply check on its own and each never text only it can see. A judge that answers `yes`
to a never text flags the play and never fails it. A play that still needs a look, or fails
because a change came before a yes, gets one more question, whose fault it is; the answer
is kept beside the play and changes nothing.
A play the eval itself cut short is an eval bug, and its record says why in plain words.

Nothing here opens a socket or reads a clock: the target, the person, the judge, the reset
and both clocks are handed in, so the whole loop runs in tests with fakes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Final, Protocol

from app.evals.cases import EvalCase, ReplyCheck
from app.evals.judge import CheckAnswer, FaultAnswer
from app.evals.never_rules import code_never_rules, judge_never_texts
from app.evals.person import EndReason, PersonModel, PlayedTurn, converse
from app.evals.results import EVAL_BUG, FAIL, NEEDS_A_LOOK, CheckRecord, PlayRecord, play_result
from app.evals.rules import EARLY_CHANGE, end_rules, turn_rules
from app.evals.target import TargetClient, TurnAnswer, TurnFailed
from app.evals.world import false_start_facts, matching_end_state, person_name

logger = logging.getLogger(__name__)

MAX_PLAYS: Final[int] = 10
"""The most plays one case may get in a run."""

DEFAULT_PLAYS: Final[int] = 3
"""How many times each case is played unless the caller says otherwise."""

EVAL_BUG_REASONS: Final[Mapping[EndReason, str]] = {
    "turn_failed": "a help desk turn failed, so the play was cut short",
    "person_silent": "the simulated person gave no answer, so the play was cut short",
    "script_ran_out": "the script ran out of messages before any right end state",
}
"""What an eval bug play's record says about why the eval broke it."""


class ReplyJudge(Protocol):
    """Anything that answers a reply check and a never text. `JudgeClient` is one."""

    def check(
        self, check: ReplyCheck, turns: Sequence[PlayedTurn], *, people: frozenset[str] = ...
    ) -> CheckAnswer | None: ...

    def never(
        self, text: str, turns: Sequence[PlayedTurn], *, people: frozenset[str] = ...
    ) -> CheckAnswer | None: ...

    def fault(
        self, turns: Sequence[PlayedTurn], *, people: frozenset[str] = ...
    ) -> FaultAnswer | None: ...


def named_people(case: EvalCase) -> frozenset[str]:
    """The case's own person and everyone its world facts and end states name: the only
    people whose rows the judge is shown."""
    facts = [*case.world, *(fact for state in case.right_end_states for fact in state)]
    return frozenset(
        {person_name(case.person.seat), *(f.person for f in facts if f.person is not None)}
    )


def run_eval(
    *,
    target: TargetClient,
    cases: Sequence[EvalCase],
    plays: int,
    person: PersonModel,
    judge: ReplyJudge | None,
    reset: Callable[[], None],
    clock: Callable[[], float],
    now: Callable[[], datetime],
) -> tuple[PlayRecord, ...]:
    """Play every case `plays` times, in file order, and answer the finished plays.

    `reset` runs before each play and may raise; the run stops there, because a run over a
    world the last play changed cannot be compared to the baseline.
    """
    records: list[PlayRecord] = []
    for case in cases:
        for number in range(1, plays + 1):
            reset()
            records.append(
                _play_once(
                    target=target,
                    case=case,
                    number=number,
                    person=person,
                    judge=judge,
                    clock=clock,
                    now=now,
                )
            )
    return tuple(records)


def _setup_is_wrong(case: EvalCase, turns: Sequence[PlayedTurn], now: datetime) -> bool:
    if not turns:
        return False
    return bool(
        false_start_facts(
            case.world,
            own=person_name(case.person.seat),
            world=turns[0].answer.world_before,
            now=now,
        )
    )


def _play_once(
    *,
    target: TargetClient,
    case: EvalCase,
    number: int,
    person: PersonModel,
    judge: ReplyJudge | None,
    clock: Callable[[], float],
    now: Callable[[], datetime],
) -> PlayRecord:
    chat: dict[str, str | None] = {"id": None}

    def send(message: str) -> TurnAnswer:
        try:
            answer = target.turn(case, chat["id"], message)
        except TurnFailed as exc:
            logger.warning("eval turn failed: %s", exc)
            raise
        chat["id"] = answer.chat_id
        return answer

    started = clock()
    played = converse(
        case,
        model=person,
        send=send,
        should_stop=lambda so_far: _setup_is_wrong(case, so_far, now()),
    )
    seconds = clock() - started
    turns = played.turns

    effort = {
        "turns": len(turns),
        "tool_calls": sum(len(t.answer.tool_calls) for t in turns),
        "seconds": seconds,
        "cost_usd": _cost(turns),
    }
    if _setup_is_wrong(case, turns, now()):
        return _record(case, number, "setup_wrong", (), (), (), turns, effort)

    broken = [*turn_rules(turns), *end_rules(case, turns), *code_never_rules(case, turns)]
    end_state = (
        matching_end_state(
            case.right_end_states,
            own=person_name(case.person.seat),
            before=turns[0].answer.world_before,
            after=turns[-1].answer.world_after,
            now=now(),
        )
        if turns
        else None
    )
    checks: list[CheckRecord] = []
    flagged: list[str] = []
    if judge is not None and turns:
        people = named_people(case)
        for check in case.reply_checks:
            answer = judge.check(check, turns, people=people)
            checks.append(_check_record(check.check, answer))
        for text in judge_never_texts(case):
            verdict = judge.never(text, turns, people=people)
            if verdict is not None and verdict.answer == "yes":
                flagged.append(text)
    result = play_result(
        setup_wrong=False,
        broken=broken,
        flagged=flagged,
        end_state_matched=end_state is not None,
        ended=played.ended,
    )
    record = _record(case, number, result, broken, flagged, checks, turns, effort, end_state)
    if result == EVAL_BUG:
        return record.model_copy(update={"fault_reason": EVAL_BUG_REASONS.get(played.ended, "")})
    if _wants_fault(result, broken) and judge is not None and turns:
        fault = judge.fault(turns, people=named_people(case))
        if fault is not None:
            return record.model_copy(update={"fault": fault.fault, "fault_reason": fault.reason})
    return record


def _wants_fault(result: str, broken: Sequence[str]) -> bool:
    """Whether the whose-fault question is asked: a play to look at, or a failure that rests
    on the person's yes, where the person may be what went wrong."""
    return result == NEEDS_A_LOOK or (result == FAIL and EARLY_CHANGE in broken)


def _cost(turns: Sequence[PlayedTurn]) -> Decimal | None:
    costs = [t.answer.cost_usd for t in turns if t.answer.cost_usd is not None]
    return sum(costs, Decimal(0)) if costs else None


def _check_record(name: str, answer: CheckAnswer | None) -> CheckRecord:
    if answer is None:
        return CheckRecord(check=name, answer=None, quote="", reason="")
    return CheckRecord(check=name, answer=answer.answer, quote=answer.quote, reason=answer.reason)


def _record(
    case: EvalCase,
    number: int,
    result: str,
    broken: Sequence[str],
    flagged: Sequence[str],
    checks: Sequence[CheckRecord],
    turns: Sequence[PlayedTurn],
    effort: Mapping[str, object],
    end_state: int | None = None,
) -> PlayRecord:
    return PlayRecord.model_validate(
        {
            "case_id": case.id,
            "owner": case.owner,
            "play": number,
            "result": result,
            "end_state": end_state,
            "broken": tuple(broken),
            "flagged": tuple(flagged),
            "checks": tuple(checks),
            "conversation": tuple(turns),
            **effort,
        }
    )
