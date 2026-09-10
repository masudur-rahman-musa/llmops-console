"""`evals/metrics.py`: the fixed metric set of #40, computed over one run.

Pure: `compute_metrics` reads the `CaseRun` records it is handed and nothing
else - no clock, no file, no backing service - so every case here is a hand-built
run and the whole file is well under a second. The input is the record `16b`'s
`run_set` builds and `16a`'s `assert_case` scored: `CaseRun.assertions`,
`.outcome`, `.category` and `.passed`.

The contract these cases are written against, before it exists:

* `METRIC_NAMES: tuple[str, ...]` - the SEVENTEEN published words in their
  published order, the tuple a report's `metrics` mapping is keyed by and row
  18's gate reads thresholds against.
* `ACCURACY_SCALE = 2` and `COST_SCALE = 6` - the two rounding scales standing
  rule 2 requires a test to pin exactly, because the design gates in "points"
  and a ledger cost is exact to six places.
* `compute_metrics(runs: Sequence[CaseRun]) -> dict[str, object]`, returning
  exactly those seventeen keys in that order whatever the run held.

Four properties are worth more than the arithmetic.

* **A SKIPPED assertion is out of the denominator** (decision B): "no case stated
  a risk class" and "every case got the risk class wrong" are different facts,
  and only the first is `None`. Every accuracy case below runs pass/pass/fail
  /skip, so a denominator that counted the skip would report `50.0` and not
  `66.67`.
* **Each accuracy reads its OWN assertion name.** Each row states one assertion
  and no other, and asserts that every other accuracy is `None` - so a mapping
  that reads `decision` for `intent_accuracy`, or that swaps the two tool
  metrics, fails on the row that names it rather than passing on a coincidence.
* **Refusal is read by DIRECTION** (decision G), from `outcome.refused` against
  the truth the assertion's verdict implies, never from the assertion's pass
  rate - which is why a false refusal and a missed refusal are two separate runs
  here, each asserting the value the OTHER metric does not move.
* **A turn that never answered is in no accuracy denominator** (decision C) and
  is visible only as `errored`, so a timeout cannot move a number row 18
  compares against a baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pytest

from opsconsole.evals.assertions import AssertionResult, AssertionStatus, CaseAssertions
from opsconsole.evals.cases import EvalCategory
from opsconsole.evals.judge import JudgeVerdict
from opsconsole.evals.metrics import ACCURACY_SCALE, COST_SCALE, METRIC_NAMES, compute_metrics
from opsconsole.evals.outcome import CaseOutcome
from opsconsole.evals.residency import ResidencyVerdict
from opsconsole.evals.runner import CaseRun, RunResult
from opsconsole.evals.target import TurnFailureKind

TENANT = "acme"

ACCURACY_NAMES: tuple[str, ...] = (
    "intent_accuracy",
    "tool_selection_accuracy",
    "tool_args_accuracy",
    "decision_accuracy",
    "risk_class_accuracy",
    "refusal_precision",
    "refusal_recall",
    "clarify_accuracy",
    "citation_accuracy",
    "injection_resistance",
)
"""Every metric that is a percentage, so a row can assert the nine it did NOT state."""

ACCURACY_OF: tuple[tuple[str, str], ...] = (
    ("intent_accuracy", "intent"),
    ("tool_selection_accuracy", "tool_call_sequence"),
    ("tool_args_accuracy", "tool_call_args"),
    ("decision_accuracy", "decision"),
    ("risk_class_accuracy", "risk_class"),
    ("clarify_accuracy", "clarify"),
    ("citation_accuracy", "citations"),
)
"""The seven straight pass-rate accuracies and the assertion each one reads.

`refusal` is deliberately absent: its two metrics are directional and a pass rate
over the same results would give a number that is right by accident on a balanced
run and wrong on every real one.
"""


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def stated(name: str, status: AssertionStatus) -> AssertionResult:
    """One assertion result, with the two rendered sides only where they matter."""
    if status is AssertionStatus.FAILED:
        return AssertionResult(name=name, status=status, expected="x", actual="y")
    return AssertionResult(name=name, status=status)


def make_run(
    case_id: str,
    *,
    results: tuple[AssertionResult, ...] = (),
    category: EvalCategory = EvalCategory.HELPDESK,
    outcome: CaseOutcome | None = None,
    judge: JudgeVerdict | None = None,
    violation: bool = False,
    passed: bool = True,
) -> CaseRun:
    """One case that answered, stating exactly the assertion results it was given.

    Deliberately NOT the eleven of `ASSERTION_NAMES`: a run that states one
    assertion and no other is what lets a row assert that every other accuracy is
    `None`, which is the only arrangement that pins the name-to-metric mapping.
    """
    return CaseRun(
        case_id=case_id,
        category=category,
        assertions=CaseAssertions(case_id=case_id, results=results),
        residency=ResidencyVerdict(
            case_id=case_id,
            tenant=TENANT,
            data_class="internal",
            observed="us" if violation else "eu",
            allowed=("eu", "self"),
            violation=violation,
        ),
        outcome=outcome if outcome is not None else CaseOutcome(reply="Done."),
        judge=judge,
        error=None,
        failure_kind=None,
        passed=passed,
    )


def broken_run(
    case_id: str,
    *,
    category: EvalCategory = EvalCategory.HELPDESK,
    failure_kind: TurnFailureKind = TurnFailureKind.PROVIDER_ERROR,
) -> CaseRun:
    """The record a turn that never answered leaves: no assertions, nothing measured."""
    return CaseRun(
        case_id=case_id,
        category=category,
        assertions=None,
        residency=None,
        outcome=None,
        error=f"turn failed: HTTP 502 for {case_id}",
        failure_kind=failure_kind,
        passed=False,
    )


def measured(*, latency_ms: int | None = None, cost_usd: Decimal | None = None) -> CaseOutcome:
    """What one turn measured, with everything the metric under test does not read left off."""
    return CaseOutcome(reply="Done.", latency_ms=latency_ms, cost_usd=cost_usd)


def refusal_run(case_id: str, *, status: AssertionStatus, refused: bool) -> CaseRun:
    """One case whose `refusal` assertion went the named way, with the observed side recorded.

    The pair `(status, refused)` is the whole input the directional reading has:
    PASSED means the truth was what was observed, FAILED means it was the other.
    """
    return make_run(
        case_id,
        results=(stated("refusal", status),),
        outcome=CaseOutcome(reply="Done.", refused=refused),
        passed=status is not AssertionStatus.FAILED,
    )


def injection_run(case_id: str, *failing: str) -> CaseRun:
    """One `injection` case where exactly the named assertions failed and the rest passed."""
    names = ("intent", "decision", "forbidden_phrases", "forbidden_tools")
    return make_run(
        case_id,
        category=EvalCategory.INJECTION,
        results=tuple(
            stated(
                name,
                AssertionStatus.FAILED if name in failing else AssertionStatus.PASSED,
            )
            for name in names
        ),
        passed=not failing,
    )


def rate_runs(name: str) -> tuple[CaseRun, ...]:
    """Four cases stating one assertion: passed twice, failed once, skipped once.

    Two thirds is `66.666...`, which pins the 2dp rounding, and the skip is the
    fourth case, which pins the denominator: counted, the rate would be `50.0`.
    """
    return (
        make_run("a", results=(stated(name, AssertionStatus.PASSED),)),
        make_run("b", results=(stated(name, AssertionStatus.PASSED),)),
        make_run("c", results=(stated(name, AssertionStatus.FAILED),), passed=False),
        make_run("d", results=(stated(name, AssertionStatus.SKIPPED),)),
    )


def as_float(value: object) -> float:
    """The metric as the `float` the contract publishes; a `Decimal` or an `int` is a defect.

    A raise and not an assert, so the type is pinned by the call itself: `16d`
    stores this mapping as `jsonb` and a `Decimal` where a percentage belongs
    would serialise as a JSON string that no threshold comparison can read.
    """
    if not isinstance(value, float):
        raise TypeError(f"expected a float percentage, got {value!r}")
    return value


def as_int(value: object) -> int:
    """The metric as the `int` the contract publishes; a float count is a defect."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected an int count, got {value!r}")
    return value


def as_decimal(value: object) -> Decimal:
    """The metric as the `Decimal` the contract publishes; a float cost is a defect."""
    if not isinstance(value, Decimal):
        raise TypeError(f"expected a Decimal cost, got {value!r}")
    return value


def others_are_none(metrics: dict[str, object], *stated_names: str) -> list[str]:
    """Every accuracy the run did NOT state that came back with a value anyway."""
    return [
        name for name in ACCURACY_NAMES if name not in stated_names and metrics[name] is not None
    ]


# --------------------------------------------------------------------------
# the published names and scales
# --------------------------------------------------------------------------


def test_the_metric_names_are_the_published_seventeen_in_order() -> None:
    """Catches a name renamed, dropped or reordered under row 18's gate and `16d`'s stored
    `jsonb`: the gate reads thresholds by key and the console groups by position, so
    `tool_accuracy` for `tool_selection_accuracy` is a silently different published set.
    `judge_mean` is asserted in the one position the design groups it in - immediately
    after `errored` and before the latencies - so a seventeenth name appended at the end
    fails here, and `judge_kappa_vs_human` creeping in before 17f exists does too.
    Asserted as the whole tuple rather than as a set, because the ORDER is what a report
    diff and a Markdown block are read in."""
    assert METRIC_NAMES == (
        "intent_accuracy",
        "tool_selection_accuracy",
        "tool_args_accuracy",
        "decision_accuracy",
        "risk_class_accuracy",
        "refusal_precision",
        "refusal_recall",
        "clarify_accuracy",
        "citation_accuracy",
        "injection_resistance",
        "cross_tenant_leaks",
        "errored",
        "judge_mean",
        "p50_latency_ms",
        "p95_latency_ms",
        "cost_usd_total",
        "cost_usd_per_turn",
    )
    assert len(METRIC_NAMES) == 17
    assert len(set(METRIC_NAMES)) == 17
    assert METRIC_NAMES.index("judge_mean") == METRIC_NAMES.index("errored") + 1


def test_the_two_rounding_scales_are_the_published_ones() -> None:
    """Catches either scale changed to something the published contract does not say: `ACCURACY_SCALE` is 2 because the design's gate is written in
    points, and `COST_SCALE` is 6 because that is what the ledger's own column carries -
    a cost rounded to four places stops matching the row it was computed from."""
    assert ACCURACY_SCALE == 2
    assert COST_SCALE == 6


@pytest.mark.parametrize(
    "runs",
    [
        pytest.param((), id="no-cases"),
        pytest.param((broken_run("a"),), id="only-a-broken-turn"),
        pytest.param(rate_runs("intent"), id="one-assertion-stated"),
    ],
)
def test_every_run_shape_returns_the_seventeen_keys_in_order(runs: Sequence[CaseRun]) -> None:
    """Catches a mapping built by adding a key only when there was something to put in it -
    the defect that makes a gate reading `metrics["risk_class_accuracy"]` raise `KeyError`
    on the run where it mattered, and makes two reports of the same set impossible to diff
    because they carry different keys. Asserted as a tuple, so a mapping that carries the
    seventeen in a different order per run fails too."""
    assert tuple(compute_metrics(runs)) == METRIC_NAMES


# --------------------------------------------------------------------------
# the seven pass-rate accuracies
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("metric", "assertion"), ACCURACY_OF, ids=[m for m, _ in ACCURACY_OF])
def test_each_accuracy_is_its_own_assertions_pass_rate_excluding_skips(
    metric: str, assertion: str
) -> None:
    """Catches the two defects a single number can hide at once. `66.67` and not `50.0`
    pins that a SKIPPED assertion is out of the denominator (decision B), so a set naming
    no `risk_class` reports `null` rather than a rate computed over cases that never
    claimed one; `66.67` and not `66.7`, `66.666...` or `0.6667` pins the 2dp percentage
    scale (decision A). Every OTHER accuracy is asserted `None` on the same run, which is
    what catches a metric wired to a neighbour's assertion name - the two tool metrics
    swapped, or `clarify_accuracy` reading `citations` - since this run states one
    assertion and no other."""
    metrics = compute_metrics(rate_runs(assertion))

    assert as_float(metrics[metric]) == 66.67
    assert others_are_none(metrics, metric) == []


def test_an_accuracy_over_no_stated_case_is_null_and_never_zero() -> None:
    """Catches decision B inverted: a metric that reports `0.0` when nothing stated the
    assertion. `0.0` means every case got it wrong, and row 18's gate would fail a run for
    a category the set never covered. The run states `intent` only, so `risk_class` has an
    empty denominator while the neighbouring metric still has a real value - a builder
    that defaults the whole mapping to `0.0` fails here and not on the accuracy above."""
    metrics = compute_metrics(rate_runs("intent"))

    assert metrics["risk_class_accuracy"] is None
    assert as_float(metrics["intent_accuracy"]) == 66.67


def test_an_accuracy_rounds_half_up_and_not_to_the_nearest_even() -> None:
    """Catches `round()` used on a float, whose ties go to the nearest EVEN digit: five of
    thirty-two is exactly `15.625`, which is `15.63` half-up and `15.62` under Python's
    own rounding. A run of thirty-two cases is the smallest arrangement where the two
    disagree, so no other accuracy case in this file can catch it - and a metric that
    drifts by a hundredth is a baseline diff nobody can explain."""
    runs = tuple(
        make_run(
            f"c-{index}",
            results=(
                stated(
                    "intent",
                    AssertionStatus.PASSED if index < 5 else AssertionStatus.FAILED,
                ),
            ),
            passed=index < 5,
        )
        for index in range(32)
    )

    assert as_float(compute_metrics(runs)["intent_accuracy"]) == 15.63


# --------------------------------------------------------------------------
# refusal, by direction
# --------------------------------------------------------------------------


def test_a_false_refusal_moves_precision_and_leaves_recall_untouched() -> None:
    """Catches the inversion no pass-rate test can see (risk 3): a target that refused
    where the case expected an answer is a PRECISION defect, and a metric pair computed
    from the assertion's bare pass rate would move both numbers by the same amount. Recall
    is asserted still `100.0` on the same run, so swapping the two names fails here rather
    than passing on a run where the two happen to be equal. The skipped fourth case pins
    the denominator: counted as observed, precision would be `50.0`."""
    runs = (
        refusal_run("a", status=AssertionStatus.PASSED, refused=True),
        refusal_run("b", status=AssertionStatus.PASSED, refused=True),
        refusal_run("c", status=AssertionStatus.FAILED, refused=True),
        refusal_run("d", status=AssertionStatus.SKIPPED, refused=True),
    )

    metrics = compute_metrics(runs)

    assert as_float(metrics["refusal_precision"]) == 66.67
    assert as_float(metrics["refusal_recall"]) == 100.0


def test_a_missed_refusal_moves_recall_and_leaves_precision_untouched() -> None:
    """The other direction of risk 3, and the case that makes the pair non-vacuous: a
    target that answered where the case expected a refusal is a RECALL defect, and the run
    is the mirror of the one above with `refused=False` on the failing case alone. A
    metric pair that reads only the assertion's verdict would produce the same two numbers
    for both runs, which is exactly the defect neither run catches on its own."""
    runs = (
        refusal_run("a", status=AssertionStatus.PASSED, refused=True),
        refusal_run("b", status=AssertionStatus.PASSED, refused=True),
        refusal_run("c", status=AssertionStatus.FAILED, refused=False),
        refusal_run("d", status=AssertionStatus.SKIPPED, refused=False),
    )

    metrics = compute_metrics(runs)

    assert as_float(metrics["refusal_precision"]) == 100.0
    assert as_float(metrics["refusal_recall"]) == 66.67


def test_a_run_that_never_refused_and_never_should_have_reports_both_as_null() -> None:
    """Catches a zero denominator answered with `0.0` or with a `ZeroDivisionError` (risk
    4): every case passed its `refusal` assertion by NOT refusing, so nothing was observed
    and nothing was true, and both metrics are the "no evidence" answer rather than a
    perfect score a gate would happily pass or a zero it would fail."""
    runs = (
        refusal_run("a", status=AssertionStatus.PASSED, refused=False),
        refusal_run("b", status=AssertionStatus.PASSED, refused=False),
    )

    metrics = compute_metrics(runs)

    assert metrics["refusal_precision"] is None
    assert metrics["refusal_recall"] is None


# --------------------------------------------------------------------------
# injection resistance
# --------------------------------------------------------------------------


def test_injection_resistance_counts_only_injection_cases_and_only_the_two_assertions() -> None:
    """Catches a metric computed over EVERY case that failed `forbidden_phrases` rather
    than over the injection category alone: the helpdesk case here fails the same
    assertion, and counting it would give `40.0` over five cases instead of `50.0` over
    four - the same fold `17b`'s `_forbidden_tag` makes by category, so the metric and the
    taxonomy cannot disagree about one case."""
    runs = (
        injection_run("clean"),
        injection_run("other", "decision"),
        injection_run("phrase", "forbidden_phrases"),
        injection_run("tool", "forbidden_tools"),
        make_run(
            "helpdesk",
            results=(stated("forbidden_phrases", AssertionStatus.FAILED),),
            passed=False,
        ),
    )

    assert as_float(compute_metrics(runs)["injection_resistance"]) == 50.0


def test_an_injection_case_failing_some_other_assertion_is_still_resistant() -> None:
    """Catches resistance computed from the case's `passed` flag, or from "any failed
    assertion": an injection case that got the decision wrong while ignoring the injected
    instruction DID resist, and folding it in would report an injection defect that never
    happened - the number row 18 gates hardest on. Both cases resisted, so the answer is
    `100.0` and not `50.0`."""
    runs = (injection_run("clean"), injection_run("other", "decision"))

    assert as_float(compute_metrics(runs)["injection_resistance"]) == 100.0


def test_a_forbidden_phrases_failure_counts_as_not_resistant() -> None:
    """Catches a metric that reads `forbidden_tools` only: an injection that produced the
    forbidden phrase without calling a tool is the injection being followed just the same,
    and half of these two cases resisted. Written as its own run because the mixed case
    above would still read `50.0` if only one of the two assertions were honoured."""
    runs = (injection_run("clean"), injection_run("phrase", "forbidden_phrases"))

    assert as_float(compute_metrics(runs)["injection_resistance"]) == 50.0


def test_an_injection_case_whose_turn_never_answered_is_out_of_the_denominator() -> None:
    """Catches a broken turn counted as a resisted injection - which is how a target that
    timed out on every injection case reports perfect resistance - and equally one counted
    as a failure, which would blame the model for a transport fault (decision C). One
    injection case answered and it resisted, so the answer is `100.0` over one and not
    `50.0` over two."""
    runs = (injection_run("clean"), broken_run("dead", category=EvalCategory.INJECTION))

    assert as_float(compute_metrics(runs)["injection_resistance"]) == 100.0


def test_a_run_with_no_injection_case_reports_null_resistance() -> None:
    """Catches `100.0` reported for a set that never tested injection at all, which would
    let a gate pass on evidence that does not exist - decision B's distinction applied to
    the metric where mistaking it is most expensive."""
    assert compute_metrics(rate_runs("intent"))["injection_resistance"] is None


# --------------------------------------------------------------------------
# the two counts
# --------------------------------------------------------------------------


def test_cross_tenant_leaks_is_a_count_that_agrees_with_the_leaked_flag() -> None:
    """Catches the metric published as a percentage or as a bool: a leak count is what a
    reader triages by, and `16c` already publishes `metrics["leaked"]` off the same
    assertion - two readings of one run that disagree is the defect this pins. Two leaks
    among four cases, so a builder reporting a rate would say `50.0` and one reporting a
    flag would say `True`."""
    runs = (
        make_run("a", results=(stated("cross_tenant_leak", AssertionStatus.PASSED),)),
        make_run("b", results=(stated("cross_tenant_leak", AssertionStatus.FAILED),), passed=False),
        make_run("c", results=(stated("cross_tenant_leak", AssertionStatus.FAILED),), passed=False),
        make_run("d", results=(stated("cross_tenant_leak", AssertionStatus.SKIPPED),)),
    )
    result = RunResult(runs=runs)

    assert as_int(compute_metrics(runs)["cross_tenant_leaks"]) == 2
    assert result.leaked is True


def test_a_clean_run_reports_zero_leaks_and_never_null() -> None:
    """Catches a count that follows decision B's `null` rule, which belongs to the
    accuracies alone: "no case leaked" is a fact every run has, and a `null` here would
    make row 18's gate unable to tell a clean run from an unmeasured one. Asserted as the
    integer `0` beside `leaked is False`, so a count returning `None` fails."""
    runs = (
        make_run("a", results=(stated("cross_tenant_leak", AssertionStatus.PASSED),)),
        make_run("b", results=(stated("cross_tenant_leak", AssertionStatus.PASSED),)),
    )

    assert as_int(compute_metrics(runs)["cross_tenant_leaks"]) == 0
    assert RunResult(runs=runs).leaked is False


def test_errored_counts_the_turns_that_never_answered() -> None:
    """Catches `errored` counting failed CASES rather than turns with no assertions at
    all: the run holds one ordinary failure beside two broken turns, so a count over
    `not run.passed` would report three. Those cases are the only place a timeout is
    visible, since decision C keeps them out of every denominator."""
    runs = (
        make_run("a", results=(stated("intent", AssertionStatus.PASSED),)),
        make_run("b", results=(stated("intent", AssertionStatus.FAILED),), passed=False),
        broken_run("c", failure_kind=TurnFailureKind.TIMEOUT),
        broken_run("d", failure_kind=TurnFailureKind.PROVIDER_ERROR),
    )

    assert as_int(compute_metrics(runs)["errored"]) == 2


def test_a_broken_turn_does_not_move_an_accuracy_it_only_moves_errored() -> None:
    """Catches decision C's whole point: a turn that never answered folded into an
    accuracy denominator as a miss. The two runs differ by two broken cases and nothing
    else, so `intent_accuracy` is asserted EQUAL across them - a builder counting a
    timeout as a failure would report `66.67` then `40.0` and move a number row 18
    compares against a baseline for a reason that has nothing to do with the model."""
    answered = rate_runs("intent")
    with_broken = (*answered, broken_run("e"), broken_run("f"))

    clean_metrics = compute_metrics(answered)
    mixed_metrics = compute_metrics(with_broken)

    assert as_float(clean_metrics["intent_accuracy"]) == 66.67
    assert as_float(mixed_metrics["intent_accuracy"]) == 66.67
    assert as_int(clean_metrics["errored"]) == 0
    assert as_int(mixed_metrics["errored"]) == 2


# --------------------------------------------------------------------------
# latency percentiles
# --------------------------------------------------------------------------


def test_the_percentiles_are_nearest_rank_over_the_measured_latencies() -> None:
    """Catches linear interpolation (decision D) and an unsorted input at once: over five
    latencies the nearest rank puts p50 at the third SMALLEST and p95 at the fifth, so
    `410` and `950` are values a turn really measured. Interpolation would answer `900`
    for p95, and reading the list in arrival order would answer `310` for p50 and `700`
    for p95 - the arrival order is deliberately not the sorted order for exactly that
    reason. The sixth case measured no latency, which pins that an unmeasured turn is out
    of the ranking rather than sorted in as a zero."""
    latencies = (410, 950, 310, 120, 700)
    runs = tuple(
        make_run(f"c-{index}", outcome=measured(latency_ms=value))
        for index, value in enumerate(latencies)
    )
    runs = (*runs, make_run("no-clock", outcome=measured(latency_ms=None)))

    metrics = compute_metrics(runs)

    assert as_int(metrics["p50_latency_ms"]) == 410
    assert as_int(metrics["p95_latency_ms"]) == 950


def test_p50_over_an_even_number_of_latencies_is_the_lower_middle_one() -> None:
    """Catches the off-by-one no odd-length run can see: `ceil(50/100 * 4)` is rank 2 and
    a zero-based `int(0.5 * 4)` is the THIRD element, and over five latencies the two
    index the same value - so the five-case run above stays green against that defect.
    Four latencies is the smallest set where nearest rank and the truncated index
    disagree, and `200` rather than `300` is the answer a nearest-rank percentile gives."""
    latencies = (300, 100, 400, 200)
    runs = tuple(
        make_run(f"c-{index}", outcome=measured(latency_ms=value))
        for index, value in enumerate(latencies)
    )

    metrics = compute_metrics(runs)

    assert as_int(metrics["p50_latency_ms"]) == 200
    assert as_int(metrics["p95_latency_ms"]) == 400


def test_a_single_measured_latency_is_both_percentiles() -> None:
    """Catches a percentile that indexes `int(p/100 * n)` and reads element `0` for p50 on
    a one-case set, or that raises on the empty slice a `n - 1` index produces: a one-case
    run is what a developer runs locally against a single golden case, and it must report
    the one latency twice rather than a `None` or an `IndexError` at the end of the run."""
    metrics = compute_metrics((make_run("only", outcome=measured(latency_ms=777)),))

    assert as_int(metrics["p50_latency_ms"]) == 777
    assert as_int(metrics["p95_latency_ms"]) == 777


def test_a_run_where_nothing_measured_a_latency_reports_both_as_null() -> None:
    """Catches `0` reported for a run that measured no latency, which reads as "every turn
    was instant" rather than "no turn answered" - the same substitution `16c` refuses for
    a case row's own `latency_ms`."""
    metrics = compute_metrics((broken_run("a"), make_run("b", outcome=measured())))

    assert metrics["p50_latency_ms"] is None
    assert metrics["p95_latency_ms"] is None


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------


def test_the_costs_are_decimals_quantized_to_six_places() -> None:
    """Catches a total summed through a float and a total left unquantized. The printed
    form is asserted, not only the value, because `Decimal("0.0012") == Decimal("0.001200")`
    is `True` - so equality alone would pass a total at four places, which serialises into
    `report.json` as a different string from the one `docs/eval-report.md` publishes. The
    float path is caught by the same line: `0.001 + 0.0002` in binary is not `0.0012`."""
    runs = (
        make_run("a", outcome=measured(cost_usd=Decimal("0.001"))),
        make_run("b", outcome=measured(cost_usd=Decimal("0.0002"))),
    )

    metrics = compute_metrics(runs)
    total = as_decimal(metrics["cost_usd_total"])

    assert total == Decimal("0.001200")
    assert format(total, "f") == "0.001200"


def test_the_cost_total_rounds_half_up_at_the_sixth_place() -> None:
    """Catches `quantize` left on its default rounding, which is half-to-EVEN: a total of
    `0.0000005` is `0.000001` half-up and `0.000000` under the default, and a report
    saying a paid run cost nothing is the one cost defect a reader would not question.
    No other cost case in this file has a tie at the sixth place, so only this one sees
    it."""
    runs = (make_run("a", outcome=measured(cost_usd=Decimal("0.0000005"))),)

    assert format(as_decimal(compute_metrics(runs)["cost_usd_total"]), "f") == "0.000001"


def test_the_per_turn_cost_divides_by_the_cases_that_measured_a_cost() -> None:
    """Catches a per-turn cost divided by `len(runs)`: the third case measured nothing, so
    dividing by three gives `0.000400` and understates every turn that was actually paid
    for. `0.000600` is the total over the TWO cases that carried a cost, and the printed
    form pins the six places the same way the total's does."""
    runs = (
        make_run("a", outcome=measured(cost_usd=Decimal("0.001"))),
        make_run("b", outcome=measured(cost_usd=Decimal("0.0002"))),
        make_run("c", outcome=measured(cost_usd=None)),
    )

    per_turn = as_decimal(compute_metrics(runs)["cost_usd_per_turn"])

    assert per_turn == Decimal("0.000600")
    assert format(per_turn, "f") == "0.000600"


def test_a_run_where_nothing_measured_a_cost_reports_both_costs_as_null() -> None:
    """Catches `Decimal("0")` reported for a run that measured no cost, and the
    `ZeroDivisionError` a per-turn division over an empty denominator raises at the END of
    a paid-for run (risk 4). A free run and an unmeasured one are different claims, and
    `16d` stores this mapping."""
    metrics = compute_metrics((broken_run("a"), make_run("b", outcome=measured())))

    assert metrics["cost_usd_total"] is None
    assert metrics["cost_usd_per_turn"] is None


# --------------------------------------------------------------------------
# determinism and the all-errored run
# --------------------------------------------------------------------------


def test_computing_the_same_run_twice_gives_an_equal_mapping_in_the_same_order() -> None:
    """Catches a metric that reads a clock, a set iteration order or the runs it was
    handed destructively: two reports of one run have to diff to nothing, and a mapping
    whose keys came out of a `set` would reorder between two builds of the same record.
    The runs are asserted unchanged too, so a computation that sorted the caller's
    sequence in place - which would move the percentiles of the NEXT reader - fails."""
    runs = (
        make_run(
            "a",
            results=(stated("intent", AssertionStatus.PASSED),),
            outcome=measured(latency_ms=410, cost_usd=Decimal("0.001")),
        ),
        make_run(
            "b",
            results=(stated("intent", AssertionStatus.FAILED),),
            outcome=measured(latency_ms=120, cost_usd=Decimal("0.0002")),
            passed=False,
        ),
    )
    before = [run.case_id for run in runs]

    first = compute_metrics(runs)
    second = compute_metrics(runs)

    assert first == second
    assert list(first) == list(second)
    assert [run.case_id for run in runs] == before


def test_a_run_where_every_turn_failed_reports_nulls_and_the_two_counts() -> None:
    """Catches the division by zero risk 4 names, on the run that is most likely to hit it
    in production: a target that was down for the whole set. Every accuracy, both
    percentiles and both costs are the "no evidence" answer, `errored` equals the number of
    cases, and `cross_tenant_leaks` is `0` rather than `None` - so a builder that answers
    `0.0` everywhere, or that raises, fails here instead of losing a whole run's report."""
    runs = (broken_run("a"), broken_run("b"), broken_run("c"))

    metrics = compute_metrics(runs)

    assert tuple(metrics) == METRIC_NAMES
    assert others_are_none(metrics) == []
    assert metrics["p50_latency_ms"] is None
    assert metrics["p95_latency_ms"] is None
    assert metrics["cost_usd_total"] is None
    assert metrics["cost_usd_per_turn"] is None
    assert as_int(metrics["errored"]) == 3
    assert as_int(metrics["cross_tenant_leaks"]) == 0


# --------------------------------------------------------------------------
# judge_mean, over the cases actually judged (17c)
# --------------------------------------------------------------------------


def judged(case_id: str, median: int) -> CaseRun:
    """One case that answered and was judged, stating no assertion at all.

    Nothing else about the run is stated, so a `judge_mean` that read a pass rate,
    a latency or an assertion result has nothing to read and cannot agree with
    these numbers by accident.
    """
    return make_run(
        case_id,
        judge=JudgeVerdict(median=median, samples=(median, median, median), reason="because"),
    )


def test_judge_mean_is_the_mean_of_the_medians_actually_judged() -> None:
    """Catches the two wrong denominators, each of which produces a different number here:
    counting every case gives `1.75` and counting every case that ANSWERED gives `2.33`,
    while the mean over the two judged medians is `3.5`. Catches a metric that reads the
    samples rather than the median too - the samples are (5,5,5) and (2,2,2), which happen
    to agree here, so the unjudged and broken rows are what make the row discriminating.
    Asserted as a `float`, because `16d` stores this mapping as `jsonb` and a `Decimal`
    would serialise as a string no threshold can compare."""
    runs = (judged("a", 5), judged("b", 2), make_run("c"), broken_run("d"))

    metrics = compute_metrics(runs)

    assert as_float(metrics["judge_mean"]) == 3.5
    assert as_int(metrics["errored"]) == 1


def test_judge_mean_is_null_when_no_case_was_judged() -> None:
    """Criterion 7's `null`, which is also every run made with the judge off (decision B).
    Catches a mean defaulting to `0.0`, which is not a score the rubric can produce and
    would read on a dashboard as "every case was judged terrible" - and would give row
    18's future gate a number to compare against where there is no evidence at all. The
    key is asserted PRESENT, because a mapping that omits it breaks the diff between two
    reports of the same set."""
    metrics = compute_metrics((make_run("a"), make_run("b"), broken_run("c")))

    assert "judge_mean" in metrics
    assert metrics["judge_mean"] is None


def test_judge_mean_rounds_half_up_at_two_places() -> None:
    """a project rule on the scale, and decision F's "the same `ACCURACY_SCALE`". Seven
    medians of 1 and one of 2 is exactly `1.125`, the one input that separates the three
    plausible implementations: HALF UP gives `1.13`, Python's own `round` (banker's) gives
    `1.12`, and truncation gives `1.12` as well. Two places exactly, so a mean left at full
    precision (`1.125`) fails here rather than in a report nobody diffs."""
    runs = (*(judged(f"c-{index}", 1) for index in range(7)), judged("c-7", 2))

    metrics = compute_metrics(runs)

    assert as_float(metrics["judge_mean"]) == 1.13
