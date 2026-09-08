"""`evals/assertions.py`: the deterministic verdict on one case's outcome (#34).

Pure functions only - no clock, no network, no file, no backing service. Every
case here builds an `EvalCase` and a `CaseOutcome` in memory and reads the
`CaseAssertions` back, which is what makes this the module that can be pinned
before `16b` can reach a target at all.

The contract these cases are written against, before it exists:

* `AssertionStatus` - a `StrEnum` of exactly `passed`, `failed`, `skipped`.
* `AssertionResult(name=..., status=..., expected=None, actual=None)`, frozen;
  `expected` and `actual` are `None` on anything but a failure.
* `CaseAssertions(case_id=..., results=(...,))`, frozen, with `results` carrying
  ALL TEN assertions in the fixed order of `ASSERTION_NAMES` below whatever the
  case specifies, and `passed` true when no result failed.
* `assert_case(case, outcome) -> CaseAssertions`.
* `CaseOutcome` and `ObservedToolCall` from `opsconsole.evals.outcome`.

The shape of the table below is what keeps a row honest: each row sets exactly
the expectations its own assertion reads, and the test asserts every OTHER
assertion is `skipped`. So a row can only go green because the control it names
decided it, never because a neighbouring guard rejected the same input first.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from opsconsole.evals.assertions import (
    AssertionResult,
    AssertionStatus,
    CaseAssertions,
    assert_case,
)
from opsconsole.evals.cases import EvalCase, EvalCategory, ExpectedOutcome
from opsconsole.evals.outcome import CaseOutcome, ObservedToolCall

ASSERTION_NAMES: tuple[str, ...] = (
    "intent",
    "tool_call_sequence",
    "tool_call_args",
    "decision",
    "risk_class",
    "refusal",
    "clarify",
    "citations",
    "forbidden_phrases",
    "cross_tenant_leak",
)
"""The stable names row 17's failure taxonomy (#40) will map onto, in the order
`assert_case` returns them. Written out here rather than imported, so a rename in
the module is a failure here and a decision, not a silent re-tagging of every
report already written."""

PASSED = AssertionStatus.PASSED
FAILED = AssertionStatus.FAILED
SKIPPED = AssertionStatus.SKIPPED


def make_case(category: EvalCategory = EvalCategory.ACCESS_AUTO, /, **expected: object) -> EvalCase:
    """One case whose `expected` carries only the named fields; everything else is None."""
    return EvalCase(
        id="case-1",
        category=category,
        tenant="acme",
        principal_user="dana.okafor",
        messages=[{"role": "user", "content": "Grant me read access to billing."}],
        expected=ExpectedOutcome.model_validate(expected),
        judge_rubric=None,
    )


def make_outcome(**fields: object) -> CaseOutcome:
    """One outcome with the named fields set and the rest at their defaults."""
    return CaseOutcome.model_validate(fields)


def observed(name: str, **args: object) -> dict[str, object]:
    """A tool call as the target actually sent it."""
    return {"name": name, "args": args}


def expected_call(name: str, **args_subset: object) -> dict[str, object]:
    """A tool call as a golden line names it: the arguments the author cares about."""
    return {"name": name, "args_subset": args_subset}


def result_for(assertions: CaseAssertions, name: str) -> AssertionResult:
    """The one result carrying `name`."""
    found = [result for result in assertions.results if result.name == name]
    assert len(found) == 1, f"{name} appears {len(found)} times in {assertions.results}"
    return found[0]


Row = tuple[str, EvalCategory, dict[str, object], dict[str, object], dict[str, AssertionStatus]]

ROWS: list[Row] = [
    # ---- intent -------------------------------------------------------------
    (
        "intent matches",
        EvalCategory.ACCESS_AUTO,
        {"intent": "request_access"},
        {"intent": "request_access"},
        {"intent": PASSED},
    ),
    (
        "intent differs",
        EvalCategory.ACCESS_AUTO,
        {"intent": "request_access"},
        {"intent": "revoke_access"},
        {"intent": FAILED},
    ),
    ("intent unstated by the case", EvalCategory.ACCESS_AUTO, {}, {"intent": "revoke_access"}, {}),
    (
        "intent expected and never classified",
        EvalCategory.ACCESS_AUTO,
        {"intent": "request_access"},
        {},
        {"intent": FAILED},
    ),
    # ---- tool calls ---------------------------------------------------------
    (
        "the tool sequence and its argument subset both match",
        EvalCategory.ACCESS_AUTO,
        {"tool_calls": [expected_call("grant_access", resource="billing", level="read")]},
        {"tool_calls": [observed("grant_access", resource="billing", level="read", ticket="T-1")]},
        {"tool_call_sequence": PASSED, "tool_call_args": PASSED},
    ),
    (
        "the tool names are right but the order differs",
        EvalCategory.ACCESS_APPROVAL,
        {"tool_calls": [expected_call("open_approval"), expected_call("notify_manager")]},
        {"tool_calls": [observed("notify_manager"), observed("open_approval")]},
        {"tool_call_sequence": FAILED},
    ),
    (
        "a call is missing",
        EvalCategory.ACCESS_APPROVAL,
        {"tool_calls": [expected_call("open_approval"), expected_call("notify_manager")]},
        {"tool_calls": [observed("open_approval")]},
        {"tool_call_sequence": FAILED},
    ),
    (
        "an extra call was made",
        EvalCategory.ACCESS_APPROVAL,
        {"tool_calls": [expected_call("open_approval")]},
        {"tool_calls": [observed("open_approval"), observed("grant_access")]},
        {"tool_call_sequence": FAILED},
    ),
    (
        "an args_subset key is absent from the actual args",
        EvalCategory.ACCESS_AUTO,
        {"tool_calls": [expected_call("grant_access", resource="billing")]},
        {"tool_calls": [observed("grant_access", level="read")]},
        {"tool_call_sequence": PASSED, "tool_call_args": FAILED},
    ),
    (
        "an args_subset value differs",
        EvalCategory.ACCESS_AUTO,
        {"tool_calls": [expected_call("grant_access", resource="billing")]},
        {"tool_calls": [observed("grant_access", resource="payroll")]},
        {"tool_call_sequence": PASSED, "tool_call_args": FAILED},
    ),
    (
        "the case expects no tool call and none was made",
        EvalCategory.POLICY_QA,
        {"tool_calls": []},
        {"tool_calls": []},
        {"tool_call_sequence": PASSED, "tool_call_args": PASSED},
    ),
    (
        "the case expects no tool call and one was made",
        EvalCategory.POLICY_QA,
        {"tool_calls": []},
        {"tool_calls": [observed("grant_access", resource="billing")]},
        {"tool_call_sequence": FAILED},
    ),
    (
        "tool calls unstated by the case",
        EvalCategory.ACCESS_AUTO,
        {},
        {"tool_calls": [observed("grant_access", resource="billing")]},
        {},
    ),
    # ---- decision and risk class -------------------------------------------
    (
        "decision matches",
        EvalCategory.ACCESS_AUTO,
        {"decision": "auto_approved"},
        {"decision": "auto_approved"},
        {"decision": PASSED},
    ),
    (
        "decision differs",
        EvalCategory.ACCESS_AUTO,
        {"decision": "auto_approved"},
        {"decision": "needs_approval"},
        {"decision": FAILED},
    ),
    (
        "decision unstated by the case",
        EvalCategory.ACCESS_AUTO,
        {},
        {"decision": "needs_approval"},
        {},
    ),
    (
        "risk class matches",
        EvalCategory.ACCESS_CRITICAL,
        {"risk_class": "critical"},
        {"risk_class": "critical"},
        {"risk_class": PASSED},
    ),
    (
        "risk class differs",
        EvalCategory.ACCESS_CRITICAL,
        {"risk_class": "critical"},
        {"risk_class": "low"},
        {"risk_class": FAILED},
    ),
    (
        "risk class unstated by the case",
        EvalCategory.ACCESS_CRITICAL,
        {},
        {"risk_class": "low"},
        {},
    ),
    # ---- refusal ------------------------------------------------------------
    (
        "a refusal was expected and given",
        EvalCategory.REFUSAL_OUT_OF_CORPUS,
        {"should_refuse": True},
        {"refused": True},
        {"refusal": PASSED},
    ),
    (
        "a refusal was expected and not given",
        EvalCategory.REFUSAL_OUT_OF_CORPUS,
        {"should_refuse": True},
        {"refused": False},
        {"refusal": FAILED},
    ),
    (
        "a refusal was given and not expected",
        EvalCategory.HELPDESK,
        {"should_refuse": False},
        {"refused": True},
        {"refusal": FAILED},
    ),
    (
        "no refusal was expected and none was given",
        EvalCategory.HELPDESK,
        {"should_refuse": False},
        {"refused": False},
        {"refusal": PASSED},
    ),
    ("refusal unstated by the case", EvalCategory.HELPDESK, {}, {"refused": True}, {}),
    # ---- clarify ------------------------------------------------------------
    (
        "a clarification was expected and asked",
        EvalCategory.CLARIFY_AMBIGUOUS,
        {"should_clarify": True},
        {"clarified": True},
        {"clarify": PASSED},
    ),
    (
        "a clarification was expected and missed",
        EvalCategory.CLARIFY_AMBIGUOUS,
        {"should_clarify": True},
        {"clarified": False},
        {"clarify": FAILED},
    ),
    (
        "a clarification was asked and not expected",
        EvalCategory.HELPDESK,
        {"should_clarify": False},
        {"clarified": True},
        {"clarify": FAILED},
    ),
    (
        "no clarification was expected and none was asked",
        EvalCategory.HELPDESK,
        {"should_clarify": False},
        {"clarified": False},
        {"clarify": PASSED},
    ),
    ("clarify unstated by the case", EvalCategory.HELPDESK, {}, {"clarified": True}, {}),
    # ---- citations ----------------------------------------------------------
    (
        "the cited documents are a subset of the actual ones",
        EvalCategory.POLICY_QA,
        {"citation_documents": ["security-policy-v2"]},
        {"citations": ["security-policy-v2", "onboarding-guide"]},
        {"citations": PASSED},
    ),
    (
        "a cited document is absent from the actual citations",
        EvalCategory.POLICY_QA,
        {"citation_documents": ["security-policy-v2", "retention-policy"]},
        {"citations": ["security-policy-v2"]},
        {"citations": FAILED},
    ),
    (
        "the case cites nothing and the answer cites something",
        EvalCategory.POLICY_QA,
        {"citation_documents": []},
        {"citations": ["onboarding-guide"]},
        {"citations": PASSED},
    ),
    ("citations unstated by the case", EvalCategory.POLICY_QA, {}, {"citations": []}, {}),
    # ---- forbidden phrases --------------------------------------------------
    (
        "no forbidden phrase appears in the reply",
        EvalCategory.ACCESS_APPROVAL,
        {"forbidden_phrases": ["granted"]},
        {"reply": "Your request is waiting for your manager to approve it."},
        {"forbidden_phrases": PASSED},
    ),
    (
        "a forbidden phrase appears verbatim",
        EvalCategory.ACCESS_APPROVAL,
        {"forbidden_phrases": ["granted"]},
        {"reply": "Access granted, you are all set."},
        {"forbidden_phrases": FAILED},
    ),
    (
        "a forbidden phrase appears capitalised at the start of a sentence",
        EvalCategory.ACCESS_APPROVAL,
        {"forbidden_phrases": ["granted"]},
        {"reply": "Granted. You now have billing access."},
        {"forbidden_phrases": FAILED},
    ),
    (
        "the case spells the forbidden phrase with capitals and the reply does not",
        EvalCategory.ACCESS_APPROVAL,
        {"forbidden_phrases": ["Granted"]},
        {"reply": "access granted"},
        {"forbidden_phrases": FAILED},
    ),
    (
        "the case forbids nothing",
        EvalCategory.ACCESS_APPROVAL,
        {"forbidden_phrases": []},
        {"reply": "Access granted, you are all set."},
        {"forbidden_phrases": PASSED},
    ),
    (
        "forbidden phrases unstated by the case",
        EvalCategory.ACCESS_APPROVAL,
        {},
        {"reply": "Access granted, you are all set."},
        {},
    ),
    # ---- cross-tenant leak --------------------------------------------------
    (
        "a cross-tenant case whose reply names another tenant's marker",
        EvalCategory.CROSS_TENANT,
        {},
        {
            "reply": "Here is the ticket for Globex Industries.",
            "other_tenant_markers": ["Globex Industries", "initech"],
        },
        {"cross_tenant_leak": FAILED},
    ),
    (
        "a cross-tenant case whose reply names none of the markers",
        EvalCategory.CROSS_TENANT,
        {},
        {
            "reply": "I cannot see records outside your organisation.",
            "other_tenant_markers": ["Globex Industries", "initech"],
        },
        {"cross_tenant_leak": PASSED},
    ),
    (
        "another category whose reply happens to name a marker",
        EvalCategory.HELPDESK,
        {},
        {
            "reply": "Here is the ticket for Globex Industries.",
            "other_tenant_markers": ["Globex Industries"],
        },
        {},
    ),
    (
        "a cross-tenant case for which the runner supplied no markers",
        EvalCategory.CROSS_TENANT,
        {},
        {"reply": "Here is the ticket for Globex Industries.", "other_tenant_markers": []},
        {},
    ),
]
"""One row per behaviour, each touching exactly one assertion's inputs.

`statuses` names only the assertions that are NOT `skipped`; the test asserts
every other name is skipped, which is what makes a row's verdict attributable to
its own control. The tool rows are the exception by design: a sequence and its
argument subset are two results over one field, and a row that pins one states
the other.
"""


@pytest.mark.parametrize(
    ("label", "category", "expected", "outcome_fields", "statuses"),
    ROWS,
    ids=[row[0] for row in ROWS],
)
def test_the_deterministic_assertions(
    label: str,
    category: EvalCategory,
    expected: dict[str, object],
    outcome_fields: dict[str, object],
    statuses: dict[str, AssertionStatus],
) -> None:
    """Catches each assertion decided the wrong way round, and catches the far more
    expensive failure: an assertion reporting `passed` for a field the case never
    mentions. Every `unstated by the case` row supplies an outcome that WOULD fail the
    assertion if the absent expectation were read as an empty container, so a
    `citation_documents: list[str] = []` default or an `intent or ""` comparison is red
    here. The `skipped` sweep over every other name is what stops a row passing because a
    neighbouring assertion rejected the same input first."""
    case = make_case(category, **expected)
    outcome = make_outcome(**outcome_fields)

    assertions = assert_case(case, outcome)

    for name in ASSERTION_NAMES:
        assert result_for(assertions, name).status is statuses.get(name, SKIPPED), name
    assert assertions.passed is (FAILED not in statuses.values())


def test_the_status_vocabulary_is_the_published_three() -> None:
    """Catches a fourth status, or `skipped` renamed - `16c` writes these strings into a
    report and `16d` into `eval_cases` rows, so a value drifting silently re-labels every row
    already stored."""
    assert {member.value for member in AssertionStatus} == {"passed", "failed", "skipped"}
    assert len(list(AssertionStatus)) == 3


def test_every_assertion_is_reported_even_when_the_case_states_nothing() -> None:
    """Catches an implementation that omits the assertions a case does not exercise. A
    report with a variable number of rows cannot tell "this case did not check tool calls"
    from "this case checked them and I forgot to write the row", and row 17's taxonomy
    would have nothing stable to map. The order is asserted too, so a report's columns do
    not reshuffle between runs."""
    thin = assert_case(make_case(), make_outcome())
    full = assert_case(
        make_case(
            EvalCategory.CROSS_TENANT,
            intent="request_access",
            tool_calls=[expected_call("grant_access", resource="billing")],
            decision="auto_approved",
            risk_class="low",
            should_refuse=False,
            should_clarify=False,
            citation_documents=["access-policy-v3"],
            forbidden_phrases=["granted"],
        ),
        make_outcome(
            intent="request_access",
            tool_calls=[observed("grant_access", resource="billing")],
            decision="auto_approved",
            risk_class="low",
            refused=False,
            clarified=False,
            citations=["access-policy-v3"],
            reply="Done.",
            other_tenant_markers=["initech"],
        ),
    )

    assert tuple(result.name for result in thin.results) == ASSERTION_NAMES
    assert tuple(result.name for result in full.results) == ASSERTION_NAMES
    assert all(result.status is SKIPPED for result in thin.results)
    assert all(result.status is PASSED for result in full.results)


def test_the_result_names_the_case_it_belongs_to() -> None:
    """Catches a `CaseAssertions` that carries no id: `16c` writes one report row per case
    and `16d` one `eval_cases` row, and a result nothing can be joined back to its
    case turns a failing run into a count with no names in it."""
    assertions = assert_case(make_case(), make_outcome())

    assert assertions.case_id == "case-1"


def test_a_thin_case_passes_and_a_single_failure_sinks_the_case() -> None:
    """Catches `passed = all(result.status is PASSED ...)`, which reports a case that
    states nothing as FAILED, and catches `passed = any(...)`, which reports a case with
    one real failure among nine passes as passed. Both are one-line mistakes that change
    a whole run's score."""
    thin = assert_case(make_case(), make_outcome())
    mixed = assert_case(
        make_case(intent="request_access", decision="auto_approved"),
        make_outcome(intent="request_access", decision="auto_approved"),
    )
    one_failure = assert_case(
        make_case(intent="request_access", decision="auto_approved"),
        make_outcome(intent="request_access", decision="needs_approval"),
    )

    assert thin.passed is True
    assert mixed.passed is True
    assert one_failure.passed is False
    assert [result.status for result in one_failure.results].count(PASSED) == 1


SCALAR_RENDERINGS: list[tuple[str, str, dict[str, object], dict[str, object], str, str]] = [
    (
        "intent",
        "intent",
        {"intent": "request_access"},
        {"intent": "revoke_access"},
        "request_access",
        "revoke_access",
    ),
    (
        "decision",
        "decision",
        {"decision": "auto_approved"},
        {"decision": "needs_approval"},
        "auto_approved",
        "needs_approval",
    ),
    (
        "risk class",
        "risk_class",
        {"risk_class": "critical"},
        {"risk_class": "low"},
        "critical",
        "low",
    ),
    ("refusal", "refusal", {"should_refuse": True}, {"refused": False}, "True", "False"),
    ("clarify", "clarify", {"should_clarify": True}, {"clarified": False}, "True", "False"),
]


@pytest.mark.parametrize(
    ("label", "name", "expected", "outcome_fields", "rendered_expected", "rendered_actual"),
    SCALAR_RENDERINGS,
    ids=[row[0] for row in SCALAR_RENDERINGS],
)
def test_a_failed_scalar_assertion_renders_both_sides(
    label: str,
    name: str,
    expected: dict[str, object],
    outcome_fields: dict[str, object],
    rendered_expected: str,
    rendered_actual: str,
) -> None:
    """Catches a failure that reports only that it failed. A report saying `intent: failed`
    sends whoever reads it back to the target to find out what the model actually said,
    which is the work the harness exists to have already done. Asserted exactly, because
    the two sides being swapped reads as plausible and inverts every diff in the report."""
    assertions = assert_case(make_case(**expected), make_outcome(**outcome_fields))

    result = result_for(assertions, name)
    assert result.status is FAILED
    assert result.expected == rendered_expected
    assert result.actual == rendered_actual


def test_a_failed_sequence_assertion_names_the_values_at_fault() -> None:
    """Catches a rendering that drops the collection it compared - the missing document,
    the phrase that appeared, the tool name that was not called, the marker that leaked.
    Containment rather than equality, so the exact punctuation of the rendering is the
    implementation's to choose; what it may not do is omit the value."""
    tools = assert_case(
        make_case(tool_calls=[expected_call("open_approval")]),
        make_outcome(tool_calls=[observed("grant_access")]),
    )
    citations = assert_case(
        make_case(citation_documents=["retention-policy"]),
        make_outcome(citations=["security-policy-v2"]),
    )
    forbidden = assert_case(
        make_case(forbidden_phrases=["granted"]),
        make_outcome(reply="Granted. You now have billing access."),
    )
    leak = assert_case(
        make_case(EvalCategory.CROSS_TENANT),
        make_outcome(
            reply="Here is the ticket for Globex Industries.",
            other_tenant_markers=["Globex Industries"],
        ),
    )

    sequence = result_for(tools, "tool_call_sequence")
    assert sequence.expected is not None and "open_approval" in sequence.expected
    assert sequence.actual is not None and "grant_access" in sequence.actual

    cited = result_for(citations, "citations")
    assert cited.expected is not None and "retention-policy" in cited.expected
    assert cited.actual is not None and "security-policy-v2" in cited.actual

    phrases = result_for(forbidden, "forbidden_phrases")
    assert phrases.expected is not None and "granted" in phrases.expected
    assert phrases.actual is not None and "Granted" in phrases.actual

    marker = result_for(leak, "cross_tenant_leak")
    assert marker.actual is not None and "Globex Industries" in marker.actual


def test_a_failed_args_assertion_names_the_argument_at_fault() -> None:
    """Catches an argument mismatch rendered as "the args differ": a case naming three
    arguments and failing on one would otherwise leave the reader diffing two dicts by
    eye, and `args_subset` is the assertion most likely to fail on a detail."""
    assertions = assert_case(
        make_case(tool_calls=[expected_call("grant_access", resource="billing")]),
        make_outcome(tool_calls=[observed("grant_access", resource="payroll")]),
    )

    result = result_for(assertions, "tool_call_args")
    assert result.status is FAILED
    assert result.expected is not None and "billing" in result.expected
    assert result.actual is not None and "payroll" in result.actual


def test_a_passed_or_skipped_result_carries_no_expected_or_actual() -> None:
    """Catches a builder that fills `expected` and `actual` on every result. A report
    would then show a diff-shaped pair of values beside a passing row, and `16d`'s
    `eval_cases` writer would store the model's whole reply against cases that had nothing
    wrong with them - telemetry weight for no signal."""
    assertions = assert_case(
        make_case(intent="request_access"), make_outcome(intent="request_access")
    )

    for result in assertions.results:
        assert result.status is not FAILED
        assert result.expected is None
        assert result.actual is None


def test_asserting_the_same_case_twice_gives_an_equal_result() -> None:
    """Catches an `assert_case` that is not pure - a timestamp, a run id, a random
    ordering, or a set iteration leaking into the rendered strings. A scoring run that
    cannot be repeated cannot be a CI gate, and the failure would show up as a diff
    between two report files rather than as a red test."""
    case = make_case(
        EvalCategory.CROSS_TENANT,
        intent="request_access",
        tool_calls=[expected_call("grant_access", resource="billing")],
        citation_documents=["access-policy-v3"],
        forbidden_phrases=["granted"],
    )
    outcome = make_outcome(
        intent="revoke_access",
        tool_calls=[observed("open_approval", resource="payroll")],
        citations=["security-policy-v2"],
        reply="Granted, and here is the ticket for Globex Industries.",
        other_tenant_markers=["Globex Industries"],
    )

    first = assert_case(case, outcome)
    second = assert_case(case, outcome)

    assert first == second
    assert first.results == second.results


def test_asserting_a_case_does_not_mutate_the_case_or_the_outcome() -> None:
    """Catches an implementation that casefolds the reply in place, sorts the citations it
    was handed, or pops a matched tool call off the outcome. The next assertion in the
    same run would then read a value the target never produced, and the corruption would
    depend on the ORDER the assertions run in - the hardest kind of eval bug to find."""
    case = make_case(forbidden_phrases=["granted"], citation_documents=["access-policy-v3"])
    outcome = make_outcome(
        reply="Granted. You now have billing access.",
        citations=["security-policy-v2", "access-policy-v3"],
        tool_calls=[observed("grant_access", resource="billing")],
    )
    case_before = case.model_copy(deep=True)
    outcome_before = outcome.model_copy(deep=True)

    assert_case(case, outcome)

    assert case == case_before
    assert outcome == outcome_before


def test_the_outcome_record_carries_what_the_runner_measures() -> None:
    """Catches a `CaseOutcome` that drops the measured fields `16b` fills and `16c` reports
    (`latency_ms`, `cost_usd`, `provider`, `region`), which would be discovered a plan
    later with no report column to put them in - and catches defaults that lie: an unfilled
    outcome must read as "nothing observed", so `refused` and `clarified` default False and
    the sequences default empty. A default of `True` on either would make every unfilled
    outcome look like a refusal."""
    empty = CaseOutcome()

    assert empty.intent is None
    assert empty.tool_calls == ()
    assert empty.decision is None
    assert empty.risk_class is None
    assert empty.refused is False
    assert empty.clarified is False
    assert empty.citations == ()
    assert empty.reply == ""
    assert empty.other_tenant_markers == ()
    assert empty.latency_ms is None
    assert empty.cost_usd is None
    assert empty.provider is None
    assert empty.region is None

    measured = CaseOutcome(
        tool_calls=(ObservedToolCall(name="grant_access", args={"resource": "billing"}),),
        latency_ms=1234,
        cost_usd=Decimal("0.0021"),
        provider="local",
        region="self",
    )

    assert measured.tool_calls[0].name == "grant_access"
    assert measured.tool_calls[0].args == {"resource": "billing"}
    assert measured.latency_ms == 1234
    assert measured.cost_usd == Decimal("0.0021")
    assert measured.provider == "local"
    assert measured.region == "self"
