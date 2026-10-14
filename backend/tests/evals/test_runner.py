"""`evals/runner.py`: one reset, every case in file order, one `RunResult`.

Driven end to end through `httpx.MockTransport`, so the whole loop - reset, turn,
envelope, assertions, residency - is exercised with no target, no network and no
backing service. The handler dispatches on the path and on the `case_id` in the
body, which is also what lets a case be given its own answer.

The contract these cases are written against, before it exists:

* `CaseRun(case_id=..., category=..., assertions=..., residency=...,
  outcome=..., error=None, passed=...)`, frozen and `extra="forbid"`;
  `assertions` and `outcome` are `None` exactly when the turn failed, and
  `error` carries that failure's message. `category` and `outcome` were added by
  `16c`, whose report is built from this record and nothing else.
* `RunResult(runs=(...))`, frozen, with `cases`, `passed_count`, `failed_count`,
  `residency_violations` and `leaked` DERIVED from `runs`, so no caller can
  report a count the per-case records do not support.
* `run_set(*, client, cases, policy, tenant, other_tenant_markers=()) -> RunResult`.

Two rules the cases below pin that no single assertion makes obvious:

* a residency violation makes that `CaseRun` not passed even when every
  assertion passed, because a turn that left the region is not a result the run
  may count as correct (#38);
* `assert_case` is imported and never re-implemented, which is why every
  assertion expectation here is thin: what is under test is the LOOP.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import httpx
import pytest

from app.evals.assertions import AssertionStatus
from app.evals.cases import EvalCase, EvalCategory, ExpectedOutcome
from app.evals.judge import JUDGE_LOW_SCORE, JUDGE_PATH, JUDGE_SAMPLES, JudgeClient
from app.evals.residency import EVAL_DATA_CLASS
from app.evals.runner import RunResult, run_set
from app.evals.target import RESET_PATH, TURN_PATH, TargetClient, TurnFailureKind
from app.routing.policy import (
    DataClass,
    DataClassRule,
    Policy,
    RateLimit,
    Residency,
    Tenant,
)

BASE_URL = "https://target.example"

TENANT = "acme"

POLICY = Policy(
    providers={},
    eu_regions=("eu", "self"),
    aliases={},
    tenants={
        TENANT: Tenant(
            residency=Residency.EU,
            data_class_rules={
                EVAL_DATA_CLASS: DataClassRule(allow_regions=("eu", "self")),
                DataClass.PUBLIC: DataClassRule(allow_regions=("us", "eu", "self")),
            },
            budget_usd_month=Decimal("10.00"),
            rate_limit=RateLimit(rpm=60, tpm=60_000),
        )
    },
)
"""One EU tenant. `us` is a region this policy knows (it is in the `public` rule)
but not one an eval turn may use, so the residency case below can only be decided
by the `internal` allow list and never by "that region does not exist"."""


def make_case(
    case_id: str,
    *,
    category: EvalCategory = EvalCategory.HELPDESK,
    expected: ExpectedOutcome | None = None,
    rubric: str | None = None,
) -> EvalCase:
    """One case; `expected` defaults to a thin case, which `assert_case` passes.

    `rubric` defaults to `None`, which is what keeps every case above out of the
    judge's way: a rubric is the only thing that makes a case judgeable (17c
    criterion 1), so the existing cases assert the unjudged path unchanged.
    """
    return EvalCase(
        id=case_id,
        category=category,
        tenant=TENANT,
        principal_user="dana.okafor",
        messages=[{"role": "user", "content": f"question for {case_id}"}],
        expected=expected if expected is not None else ExpectedOutcome(),
        judge_rubric=rubric,
    )


def envelope(**fields: object) -> dict[str, object]:
    """An envelope the policy is happy with, plus whatever a case overrides."""
    payload: dict[str, object] = {
        "decision": "auto_approved",
        "reply": "Done.",
        "latency_ms": 120,
        "cost_usd": "0.0010",
        "provider": "anthropic_eu",
        "region": "eu",
    }
    payload.update(fields)
    return payload


class Target:
    """A recording in-process target: one answer per case id, one call log.

    The log holds `"reset"` and each case id in the order they were requested,
    which is what makes "reset once, before the first case" and "file order"
    assertable rather than assumed.
    """

    def __init__(
        self,
        answers: Mapping[str, dict[str, object]],
        *,
        raises: Mapping[str, Exception] | None = None,
    ) -> None:
        self.answers = answers
        self.raises = raises if raises is not None else {}
        self.calls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == RESET_PATH:
            self.calls.append("reset")
            return httpx.Response(200, json={"status": "reset"})
        assert request.url.path == TURN_PATH
        body: dict[str, Any] = json.loads(request.content)
        case_id = str(body["case_id"])
        self.calls.append(case_id)
        error = self.raises.get(case_id)
        if error is not None:
            raise error
        return httpx.Response(200, json=self.answers[case_id])

    def client(self) -> TargetClient:
        return TargetClient(BASE_URL, client=httpx.Client(transport=httpx.MockTransport(self)))


def run(
    target: Target,
    cases: tuple[EvalCase, ...],
    *,
    markers: tuple[str, ...] = (),
    judge: JudgeClient | None = None,
) -> RunResult:
    """Run a set against a recording target, with the fixed tenant and policy.

    `judge` defaults to `None`, which is the run every case above this line makes:
    the judge is off unless a case hands one in (17c decision B).
    """
    return run_set(
        client=target.client(),
        cases=cases,
        policy=POLICY,
        tenant=TENANT,
        other_tenant_markers=markers,
        judge=judge,
    )


THREE_CASES: tuple[EvalCase, ...] = (make_case("c-1"), make_case("c-2"), make_case("c-3"))

ALL_FINE: Mapping[str, dict[str, object]] = {
    "c-1": envelope(),
    "c-2": envelope(),
    "c-3": envelope(),
}


def test_every_case_runs_once_in_file_order() -> None:
    """Catches a runner that reorders the set (a dict keyed by id, a set, a sort), which
    would make a report's rows disagree with the file an author reads, and one that drops
    or repeats a case - either of which silently changes the denominator every threshold
    in the eval gate is measured against. Order and counts are both asserted, so a runner that
    runs all three in the wrong order still fails."""
    target = Target(ALL_FINE)

    result = run(target, THREE_CASES)

    assert [record.case_id for record in result.runs] == ["c-1", "c-2", "c-3"]
    assert target.calls == ["reset", "c-1", "c-2", "c-3"]
    assert result.cases == 3
    assert result.passed_count == 3
    assert result.failed_count == 0


def test_the_target_is_reset_exactly_once_before_the_first_case() -> None:
    """Catches a reset per case, which throws away the state each case's predecessor built
    and makes a multi-turn seed meaningless, and a reset after the first turn, which is
    the same as no reset at all for that case. The call log's FIRST entry is asserted, not
    just its presence, because "reset happened somewhere" is not decision F's claim."""
    target = Target(ALL_FINE)

    run(target, THREE_CASES)

    assert target.calls.count("reset") == 1
    assert target.calls[0] == "reset"


def test_one_failing_case_does_not_stop_the_others() -> None:
    """Catches a runner that stops at the first failure - a set is a measurement, not a
    build, and stopping turns a 3% regression into an unknown score. The middle case is
    the failing one, so a runner that stops leaves `c-3` unrun and the counts short; the
    other two state nothing, so only `c-2`'s own expectation can fail."""
    cases = (
        make_case("c-1"),
        make_case("c-2", expected=ExpectedOutcome(decision="auto_approved")),
        make_case("c-3"),
    )
    target = Target({**ALL_FINE, "c-2": envelope(decision="escalated")})

    result = run(target, cases)

    assert [record.case_id for record in result.runs] == ["c-1", "c-2", "c-3"]
    assert [record.passed for record in result.runs] == [True, False, True]
    assert result.passed_count == 2
    assert result.failed_count == 1
    assert result.residency_violations == 0


TURN_ERRORS: list[tuple[str, Exception]] = [
    ("a per-case timeout", httpx.ReadTimeout("timed out")),
    ("a transport error", httpx.ConnectError("connection refused")),
]


@pytest.mark.parametrize(("label", "error"), TURN_ERRORS, ids=[row[0] for row in TURN_ERRORS])
def test_a_case_that_times_out_is_a_failed_case_and_the_run_continues(
    label: str, error: Exception
) -> None:
    """Catches the three wrong answers to a broken turn: aborting the run, OMITTING the
    case (which shrinks the denominator and lifts the score), and retrying it (decision E,
    a second sample of a non-deterministic system). The attempt count is asserted because
    a retry that eventually failed looks identical in every other field."""
    target = Target(ALL_FINE, raises={"c-2": error})

    result = run(target, THREE_CASES)

    assert [record.case_id for record in result.runs] == ["c-1", "c-2", "c-3"]
    assert target.calls == ["reset", "c-1", "c-2", "c-3"]
    assert result.cases == 3
    assert result.failed_count == 1

    broken = result.runs[1]
    assert broken.passed is False
    assert broken.assertions is None
    assert broken.error is not None
    assert "c-2" in broken.error


def test_a_broken_turns_kind_reaches_the_case_run() -> None:
    """Catches a runner that catches `TurnFailed` and discards its `.kind`
    (dropping to `None` on every failure), which would leave every broken case's
    report untagged between `timeout` and `provider_error` - the split row 17b's
    taxonomy exists to make. `httpx.ReadTimeout` is httpx's own timeout subtype,
    so a runner reading the wrong attribute or hardcoding `provider_error` fails
    only this case and not the plain-error case below."""
    target = Target(ALL_FINE, raises={"c-2": httpx.ReadTimeout("timed out")})

    result = run(target, THREE_CASES)

    assert result.runs[1].failure_kind == TurnFailureKind.TIMEOUT


def test_a_broken_turns_provider_error_kind_reaches_the_case_run() -> None:
    """The other kind, and the reverse of the case above. Catches a runner that
    always reports `timeout` regardless of the underlying error: a plain
    `httpx.ConnectError` is not a timeout, so a runner hardcoding `timeout` for
    every `TurnFailed` fails here while still passing the read-timeout case."""
    target = Target(ALL_FINE, raises={"c-2": httpx.ConnectError("connection refused")})

    result = run(target, THREE_CASES)

    assert result.runs[1].failure_kind == TurnFailureKind.PROVIDER_ERROR


def test_a_case_that_answered_carries_no_failure_kind() -> None:
    """Catches `failure_kind` defaulted to anything but `None` on the success
    path, which would make every clean case look like a broken turn to row 17b's
    taxonomy and tag a passing run's cases with `timeout` or `provider_error`."""
    target = Target(ALL_FINE)

    result = run(target, THREE_CASES)

    assert [record.failure_kind for record in result.runs] == [None, None, None]


def test_a_residency_violation_fails_the_run_even_when_every_assertion_passed() -> None:
    """Catches residency checked only for cases that already failed, or not checked at all.
    Every case here states nothing, so all eleven assertions pass; the ONLY thing that can
    make this run non-clean is the `us` region on `c-2`, and `us` is a region the policy
    names under another data class, so a check reading the wrong rule passes it (#38,
    a project rule)."""
    target = Target({**ALL_FINE, "c-2": envelope(region="us")})

    result = run(target, THREE_CASES)

    assert result.residency_violations == 1
    violating = result.runs[1]
    assert violating.residency is not None
    assert violating.residency.violation is True
    assert violating.residency.observed == "us"
    assert violating.assertions is not None
    assert violating.assertions.passed is True
    assert violating.passed is False


def test_an_envelope_with_no_region_at_all_is_a_violation() -> None:
    """Catches a runner that only checks residency when a region was reported - the shape a
    half-implemented target actually produces. Skipping here would report a clean run for
    a target that never proved where the data went, and the run would be counted against
    the baseline as if it had."""
    target = Target({**ALL_FINE, "c-2": envelope(region=None)})

    result = run(target, THREE_CASES)

    assert result.residency_violations == 1
    assert result.runs[1].passed is False


def test_residency_is_checked_on_every_case_and_not_only_the_first() -> None:
    """Catches a check hoisted out of the loop (one region sampled, then trusted). Every
    case here is compliant, so this row asserts the verdicts EXIST for all three: without
    it, a runner checking only case one would pass every other residency case above."""
    target = Target(ALL_FINE)

    result = run(target, THREE_CASES)

    assert [record.residency is not None for record in result.runs] == [True, True, True]
    assert result.residency_violations == 0


def test_a_cross_tenant_case_is_supplied_the_markers_so_the_leak_assertion_is_live() -> None:
    """Catches a runner that never passes the seed's other-tenant markers through: the leak
    assertion then reports `skipped` for every cross-tenant case, and the one control that
    catches a tenant boundary breach is silently off while the report still shows the row.
    The reply names another tenant's marker, so with the markers supplied the assertion can
    only be `failed`."""
    case = make_case("x-1", category=EvalCategory.CROSS_TENANT)
    target = Target({"x-1": envelope(reply="Globex's invoice 88 is still open.")})

    result = run(target, (case,), markers=("globex",))

    record = result.runs[0]
    assert record.assertions is not None
    leak = next(item for item in record.assertions.results if item.name == "cross_tenant_leak")
    assert leak.status is AssertionStatus.FAILED
    assert result.leaked is True
    assert record.passed is False


def test_a_clean_cross_tenant_case_does_not_report_a_leak() -> None:
    """Catches a `leaked` flag set by the case's CATEGORY rather than by the assertion's
    verdict, which would fail every run that contains a cross-tenant case at all - and a
    gate that fails always is a gate nobody keeps. Same markers, a reply that names none
    of them."""
    case = make_case("x-1", category=EvalCategory.CROSS_TENANT)
    target = Target({"x-1": envelope(reply="I can only see your own tickets.")})

    result = run(target, (case,), markers=("globex",))

    assert result.leaked is False
    assert result.passed_count == 1


# --------------------------------------------------------------------------
# the judge, on the cases that carry a rubric (17c)
# --------------------------------------------------------------------------


class Judge:
    """A recording in-process judge, answering one score per rubric it is shown.

    Keyed by rubric text rather than by case id, because the case id is exactly
    what a judge request does NOT carry: the rubric and the reply reach it as
    data (17c criterion 5), so matching on the rubric is the only thing that can
    tell which case a sample belongs to - and it is also what proves the runner
    handed the judge the right case's rubric rather than the first one's.

    A rubric with a `None` score answers 500, which is the judge outage of
    decision E; a rubric this map does not know is an error, so a runner that
    judged a case with no rubric fails loudly rather than quietly.
    """

    def __init__(self, scores: Mapping[str, int | None]) -> None:
        self.scores = scores
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == JUDGE_PATH
        body: dict[str, Any] = json.loads(request.content)
        sent = json.dumps(body["messages"])
        matched = [rubric for rubric in self.scores if rubric in sent]
        assert len(matched) == 1, f"the judge was shown {len(matched)} known rubrics"
        self.requests.append(matched[0])
        score = self.scores[matched[0]]
        if score is None:
            return httpx.Response(500, json={"error": "the judge is down"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"score": score, "reason": "because"}),
                        },
                    }
                ]
            },
        )

    def client(self) -> JudgeClient:
        return JudgeClient(
            "https://gateway.example",
            client=httpx.Client(transport=httpx.MockTransport(self)),
            api_key="sk-eval-judge-not-a-real-key",
        )


def test_only_a_case_with_a_rubric_is_judged() -> None:
    """Catches a runner that judges every case in the set, which triples the spend of a
    two-hundred-case run and puts a score on cases no author ever wrote a standard for.
    The judged case is the MIDDLE one, so a runner judging only the first, or only the
    last, fails; and the samples are counted, so a runner that judged the right case once
    instead of three times fails here rather than in `test_judge.py`."""
    cases = (make_case("c-1"), make_case("c-2", rubric="rubric for c-2"), make_case("c-3"))
    judge = Judge({"rubric for c-2": 4})

    result = run(Target(ALL_FINE), cases, judge=judge.client())

    assert judge.requests == ["rubric for c-2"] * JUDGE_SAMPLES
    assert [record.judge is None for record in result.runs] == [True, False, True]
    verdict = result.runs[1].judge
    assert verdict is not None
    assert verdict.median == 4
    assert verdict.samples == (4, 4, 4)


def test_a_case_with_no_rubric_is_not_judged_at_all() -> None:
    """Risk 1, asserted where it costs money: the transport saw NO request. Catches a
    runner that calls the judge for every case and lets the judge decide, which would still
    produce `judge is None` on these records - the verdict alone cannot tell "not judged"
    from "judged and discarded", and only one of those two is free."""
    judge = Judge({"a rubric no case carries": 4})

    result = run(Target(ALL_FINE), THREE_CASES, judge=judge.client())

    assert judge.requests == []
    assert [record.judge for record in result.runs] == [None, None, None]
    assert result.passed_count == 3


def test_a_case_that_passed_every_assertion_fails_on_a_low_judge_score() -> None:
    """Risk 4 and criterion 6. Catches a judge whose verdict is recorded and never read:
    every assertion here passes and the residency check is clean, so the ONLY thing that
    can fail this case is the median of 2 - and a runner that stores `judge` without
    folding it into `passed` reports a clean run beside a case the judge failed. The
    assertions are asserted passed in the same breath, so a run that failed for the wrong
    reason cannot be mistaken for this one."""
    case = make_case("c-1", rubric="a rubric")
    judge = Judge({"a rubric": 2})

    result = run(Target(ALL_FINE), (case,), judge=judge.client())

    record = result.runs[0]
    assert record.judge is not None
    assert record.judge.median == 2
    assert record.assertions is not None
    assert record.assertions.passed is True
    assert record.residency is not None
    assert record.residency.violation is False
    assert record.passed is False
    assert result.failed_count == 1


@pytest.mark.parametrize("score", [JUDGE_LOW_SCORE, 4, 5], ids=["three", "four", "five"])
def test_a_score_of_three_does_not_fail_the_case(score: int) -> None:
    """The inclusive boundary of `JUDGE_LOW_SCORE`, which decision D fixes at "1 and 2
    fail, 3 and above do not". Catches `<=` where `<` belongs - an implementation failing
    every merely-adequate case would turn the judge into a gate nobody can pass - and
    catches a judge that overrides a verdict the assertions already settled: the second
    case in this run failed its `decision` assertion and is scored 5, so a runner that let
    a good score PASS a failed case fails here."""
    cases = (
        make_case("c-1", rubric="the adequate case"),
        make_case(
            "c-2", expected=ExpectedOutcome(decision="auto_approved"), rubric="the wrong case"
        ),
    )
    target = Target({**ALL_FINE, "c-2": envelope(decision="escalated")})
    judge = Judge({"the adequate case": score, "the wrong case": 5})

    result = run(target, cases, judge=judge.client())

    assert [record.passed for record in result.runs] == [True, False]
    assert result.passed_count == 1


def test_a_judge_that_never_answered_leaves_the_verdict_the_assertions_gave_it() -> None:
    """Decision E, in the runner. Catches a judge outage counted as a case failure, which
    would turn every run made while the gateway was down into a red gate that says nothing
    about the agent under test, and catches the opposite - an unjudged case recorded with a
    fabricated median. The failing case is here too, so a runner that answered "everything
    passes when the judge is down" fails as well."""
    cases = (
        make_case("c-1", rubric="the unreachable case"),
        make_case(
            "c-2", expected=ExpectedOutcome(decision="auto_approved"), rubric="the failing case"
        ),
    )
    target = Target({**ALL_FINE, "c-2": envelope(decision="escalated")})
    judge = Judge({"the unreachable case": None, "the failing case": None})

    result = run(target, cases, judge=judge.client())

    assert [record.judge for record in result.runs] == [None, None]
    assert [record.passed for record in result.runs] == [True, False]
    assert result.passed_count == 1


def test_a_run_with_no_judge_configured_judges_nothing() -> None:
    """Criterion 6's last clause and decision B. Catches a runner that builds its own
    judge from the environment when none was handed in, which is how a `make eval` on a
    laptop with no gateway up starts making paid calls - and how the harness's behaviour
    starts depending on a shell variable rather than on its arguments. The cases DO carry
    rubrics, so a runner that judges whenever a rubric exists fails here."""
    cases = (make_case("c-1", rubric="a rubric"), make_case("c-2", rubric="another rubric"))

    result = run(Target(ALL_FINE), cases)

    assert [record.judge for record in result.runs] == [None, None]
    assert result.passed_count == 2
