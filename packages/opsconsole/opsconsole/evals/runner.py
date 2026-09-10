"""One reset, every case in file order, one `RunResult` (#33, #38).

The loop is the whole module: reset once before the first case (decision F), send
each case in the order the file listed it, map the envelope, score it with 16a's
`assert_case` - imported, never re-implemented - and check the region every turn
reports. Nothing here writes a file: `16c` owns `report.json` and `report.md`,
`16d` owns the `eval_runs` and `eval_cases` rows, and this module owns only the
record both of them read.

Three rules the record keeps that no single field states:

* a broken turn is a FAILED case and the run continues, one attempt, never a
  retry and never omitted from the counts (decision E) - omitting it would shrink
  the denominator every threshold in the eval gate is measured against
  (`test_a_case_that_times_out_is_a_failed_case_and_the_run_continues`);
* a residency violation makes that case not passed even when every assertion
  passed: a turn that left its allowed region is not a result the run may count
  as correct (#38,
  `test_a_residency_violation_fails_the_run_even_when_every_assertion_passed`);
* every count on `RunResult` is DERIVED from the per-case records, so no caller
  can report a number the records do not support.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from opsconsole.evals.assertions import AssertionStatus, CaseAssertions, assert_case
from opsconsole.evals.cases import EvalCase, EvalCategory
from opsconsole.evals.judge import JUDGE_LOW_SCORE, JudgeClient, JudgeVerdict
from opsconsole.evals.outcome import CaseOutcome
from opsconsole.evals.residency import EVAL_DATA_CLASS, ResidencyVerdict, check_residency
from opsconsole.evals.target import TargetClient, TurnFailed, TurnFailureKind
from opsconsole.routing.policy import Policy

LEAK_ASSERTION: str = "cross_tenant_leak"
"""The assertion whose failure is a tenant boundary breach rather than a score.

One of `assertions.ASSERTION_NAMES`, named here because `RunResult.leaked` reads
it and the CLI puts a leak on the same exit code as a residency violation
(`test_a_cross_tenant_case_is_supplied_the_markers_so_the_leak_assertion_is_live`)."""


class CaseRun(BaseModel):
    """What happened to one case: its verdict, its region, or why neither exists.

    `assertions` is `None` exactly when the turn failed, and `error` carries that
    failure's message - so a report can tell "the agent answered wrongly" from
    "the target never answered", which are not the same regression. `outcome` is
    `None` in exactly the same place and for the same reason: a turn that never
    answered measured nothing, and a report that filled its latency and its cost
    with zeros would read as a free, instant success.

    `category` and `outcome` are carried rather than re-derived because the
    report is built from this record and nothing else (`16c` decision E), and
    neither the golden set nor the turn is available to it
    (`test_a_case_report_carries_the_measured_fields_the_assertions_and_the_verdict`).

    `failure_kind` is the raised `TurnFailed`'s own kind, carried for the same
    reason and set in exactly the same place as `error`: `17b`'s taxonomy tags a
    broken turn `timeout` or `provider_error` and has nothing else to read, since
    a turn that never answered left no assertions behind. It is `None` on every
    case that answered (`test_a_case_that_answered_carries_no_failure_kind`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: EvalCategory
    assertions: CaseAssertions | None
    residency: ResidencyVerdict | None
    outcome: CaseOutcome | None = None
    error: str | None = None
    failure_kind: TurnFailureKind | None = None
    judge: JudgeVerdict | None = None
    passed: bool


class RunResult(BaseModel):
    """Every case's record, and the counts the summary line and `16c` both read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runs: tuple[CaseRun, ...]

    @property
    def cases(self) -> int:
        """How many cases ran. A broken turn is one of them (decision E)."""
        return len(self.runs)

    @property
    def passed_count(self) -> int:
        """How many cases passed every assertion they stated and stayed in region."""
        return sum(1 for run in self.runs if run.passed)

    @property
    def failed_count(self) -> int:
        """The rest of them, whatever the reason."""
        return self.cases - self.passed_count

    @property
    def residency_violations(self) -> int:
        """How many turns were served somewhere the tenant's policy does not allow."""
        return sum(1 for run in self.runs if run.residency is not None and run.residency.violation)

    @property
    def leaked(self) -> bool:
        """True when a cross-tenant case's reply carried another tenant's marker.

        Read off the assertion's VERDICT and never off the case's category, which
        would fail every run containing a cross-tenant case at all - and a gate
        that fails always is a gate nobody keeps
        (`test_a_clean_cross_tenant_case_does_not_report_a_leak`).
        """
        return any(
            result.name == LEAK_ASSERTION and result.status is AssertionStatus.FAILED
            for run in self.runs
            if run.assertions is not None
            for result in run.assertions.results
        )


def run_set(
    *,
    client: TargetClient,
    cases: tuple[EvalCase, ...],
    policy: Policy,
    tenant: str,
    other_tenant_markers: tuple[str, ...] = (),
    judge: JudgeClient | None = None,
) -> RunResult:
    """Reset the target, run every case in file order, and answer what happened.

    The reset is first and happens once (pinned in `tests/evals/test_runner.py` by
    the reset-once case); a `TargetUnreachable` from it is not caught here,
    because a run over an unreset seed is not comparable to the baseline and the
    CLI turns that into exit `2` (decision F).

    Residency is checked on every case that answered, not on the first one and
    then trusted (`test_residency_is_checked_on_every_case_and_not_only_the_first`).

    The judge is optional and OFF unless the CLI built one (`17c` decision B), and
    it scores only the cases that carry a `judge_rubric`, after `assert_case` and
    on the same loop - so a set with no rubric anywhere costs exactly what it cost
    before this row (`test_a_case_with_no_rubric_is_not_judged_at_all`). A median
    below `JUDGE_LOW_SCORE` fails the case (decision D), which is the one thing
    that makes `passed` more than a statement about assertions and residency
    (`test_a_case_that_passed_every_assertion_fails_on_a_low_judge_score`).
    """
    client.reset()

    runs: list[CaseRun] = []
    for case in cases:
        try:
            _, outcome = client.run_case(case, other_tenant_markers=other_tenant_markers)
        except TurnFailed as exc:
            runs.append(
                CaseRun(
                    case_id=case.id,
                    category=case.category,
                    assertions=None,
                    residency=None,
                    outcome=None,
                    error=str(exc),
                    failure_kind=exc.kind,
                    passed=False,
                )
            )
            continue

        assertions = assert_case(case, outcome)
        verdict = (
            judge.score(case, outcome)
            if judge is not None and case.judge_rubric is not None
            else None
        )
        residency = check_residency(
            policy,
            case_id=case.id,
            tenant=tenant,
            data_class=EVAL_DATA_CLASS,
            observed=outcome.region,
        )
        runs.append(
            CaseRun(
                case_id=case.id,
                category=case.category,
                assertions=assertions,
                residency=residency,
                outcome=outcome,
                error=None,
                failure_kind=None,
                judge=verdict,
                passed=(
                    assertions.passed
                    and not residency.violation
                    and (verdict is None or verdict.median >= JUDGE_LOW_SCORE)
                ),
            )
        )
    return RunResult(runs=tuple(runs))
