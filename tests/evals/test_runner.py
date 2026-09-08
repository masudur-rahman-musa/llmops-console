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

from opsconsole.evals.assertions import AssertionStatus
from opsconsole.evals.cases import EvalCase, EvalCategory, ExpectedOutcome
from opsconsole.evals.residency import EVAL_DATA_CLASS
from opsconsole.evals.runner import RunResult, run_set
from opsconsole.evals.target import RESET_PATH, TURN_PATH, TargetClient
from opsconsole.routing.policy import (
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
) -> EvalCase:
    """One case; `expected` defaults to a thin case, which `assert_case` passes."""
    return EvalCase(
        id=case_id,
        category=category,
        tenant=TENANT,
        principal_user="dana.okafor",
        messages=[{"role": "user", "content": f"question for {case_id}"}],
        expected=expected if expected is not None else ExpectedOutcome(),
        judge_rubric=None,
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


def run(target: Target, cases: tuple[EvalCase, ...], *, markers: tuple[str, ...] = ()) -> RunResult:
    """Run a set against a recording target, with the fixed tenant and policy."""
    return run_set(
        client=target.client(),
        cases=cases,
        policy=POLICY,
        tenant=TENANT,
        other_tenant_markers=markers,
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
    in row 18 is measured against. Order and counts are both asserted, so a runner that
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


def test_a_residency_violation_fails_the_run_even_when_every_assertion_passed() -> None:
    """Catches residency checked only for cases that already failed, or not checked at all.
    Every case here states nothing, so all ten assertions pass; the ONLY thing that can
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
