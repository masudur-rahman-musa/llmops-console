"""`evals/taxonomy.py`: the closed vocabulary that tags every failed case (#40).

Pure: `tags_for` reads nothing but the `CaseRun` it is handed, so every case here
is a hand-built record with no runner, no target and no backing service. The
input is the record `16b`'s `run_set` builds and `16c`'s `report._case_report`
will call this module against - `CaseRun.assertions`, `.residency`, `.outcome`,
`.category` and (after this plan) `.failure_kind`.

The contract these cases are written against, before it exists:

* `FailureTag`, a closed `StrEnum` of exactly eighteen words.
* `TAG_ORDER: tuple[FailureTag, ...]`, the assertion-derived tags in
  `assertions.ASSERTION_NAMES` order, then the four run-level tags, then
  `judge_low` last - the order a report's taxonomy block counts in and the order
  `tags_for` must emit in, so a report diffs.
* `tags_for(run: CaseRun) -> tuple[str, ...]`, returning tag VALUES (plain
  strings, not `FailureTag` members) in `TAG_ORDER` order, `()` for a passed case,
  and never mutating the run it read.

Mapping rules under test: the five direction-free assertions map one to one;
`refusal` and `clarify` are tagged by the DIRECTION recorded on the case's own
`CaseOutcome` (`outcome.refused` / `outcome.clarified`), not by the assertion's
bare failed status, which cannot itself tell "missed" from "false"; `citations`
the same way, keyed on whether `outcome.citations` is empty; `forbidden_phrases`
and `forbidden_tools` both fold into `injection_followed` on an `injection` case
and stay `forbidden_phrase` / `wrong_tool` on any other (decision D);
`cross_tenant_leak` maps to `tenant_leak`; a residency violation and a broken
turn's failure kind are read off the run and not off any assertion at all, which
is what keeps a residency-only failure and a turn that never answered from
reaching a report with an empty tag tuple - the defect this plan exists to
prevent (risk 2).
"""

from __future__ import annotations

import pytest

from app.evals.assertions import (
    ASSERTION_NAMES,
    AssertionResult,
    AssertionStatus,
    CaseAssertions,
)
from app.evals.cases import EvalCategory
from app.evals.judge import JUDGE_LOW_SCORE, JudgeVerdict
from app.evals.outcome import CaseOutcome
from app.evals.residency import ResidencyVerdict
from app.evals.runner import CaseRun
from app.evals.target import TurnFailureKind
from app.evals.taxonomy import TAG_ORDER, FailureTag, tags_for

TENANT = "acme"


def _passed(name: str) -> AssertionResult:
    return AssertionResult(name=name, status=AssertionStatus.PASSED)


def _failed(name: str) -> AssertionResult:
    return AssertionResult(name=name, status=AssertionStatus.FAILED, expected="x", actual="y")


def _results(*failing: str) -> tuple[AssertionResult, ...]:
    """Every one of `assertions.ASSERTION_NAMES`, all passed except the named ones."""
    return tuple(_failed(name) if name in failing else _passed(name) for name in ASSERTION_NAMES)


def clean_residency(
    case_id: str = "c-1", *, violation: bool = False, observed: str | None = "eu"
) -> ResidencyVerdict:
    return ResidencyVerdict(
        case_id=case_id,
        tenant=TENANT,
        data_class="internal",
        observed=observed,
        allowed=("eu", "self"),
        violation=violation,
    )


def make_outcome(
    *, refused: bool = False, clarified: bool = False, citations: tuple[str, ...] = ()
) -> CaseOutcome:
    return CaseOutcome(reply="Done.", refused=refused, clarified=clarified, citations=citations)


def make_run(
    case_id: str = "c-1",
    *,
    category: EvalCategory = EvalCategory.HELPDESK,
    failing: tuple[str, ...] = (),
    outcome: CaseOutcome | None = None,
    residency: ResidencyVerdict | None = None,
    judge: JudgeVerdict | None = None,
    broken: bool = False,
    failure_kind: TurnFailureKind | None = None,
) -> CaseRun:
    """The smallest `CaseRun` that produces the tag(s) under test.

    `broken=True` builds the record a `TurnFailed` turn leaves: no assertions, no
    residency, no outcome, exactly as `run_set` builds it today.
    """
    if broken:
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
    return CaseRun(
        case_id=case_id,
        category=category,
        assertions=CaseAssertions(case_id=case_id, results=_results(*failing)),
        residency=residency if residency is not None else clean_residency(case_id),
        outcome=outcome if outcome is not None else make_outcome(),
        judge=judge,
        error=None,
        failure_kind=None,
        passed=(
            not failing
            and (residency is None or not residency.violation)
            and (judge is None or judge.median >= JUDGE_LOW_SCORE)
        ),
    )


# --------------------------------------------------------------------------
# the vocabulary
# --------------------------------------------------------------------------


def test_the_vocabulary_is_the_published_eighteen_words_in_tag_order() -> None:
    """Catches the vocabulary widened or narrowed on one side only: `FailureTag`
    gaining a member with no slot in `TAG_ORDER`, or `TAG_ORDER` repeating a word,
    either of which lets a tag reach `eval_cases.tags` with no defined position -
    the thing the order-preserving `text[]` column depends on not happening."""
    assert [tag.value for tag in TAG_ORDER] == [
        "wrong_intent",
        "wrong_tool",
        "wrong_args",
        "wrong_decision",
        "wrong_risk_class",
        "missed_refusal",
        "false_refusal",
        "missed_clarify",
        "false_clarify",
        "wrong_citation",
        "no_citation",
        "forbidden_phrase",
        "injection_followed",
        "tenant_leak",
        "residency_violation",
        "timeout",
        "provider_error",
        "judge_low",
    ]
    assert set(FailureTag) == set(TAG_ORDER)
    assert len(TAG_ORDER) == 18
    assert len(set(TAG_ORDER)) == 18


# --------------------------------------------------------------------------
# the five direction-free mappings
# --------------------------------------------------------------------------

DIRECT_MAPPINGS: list[tuple[str, FailureTag]] = [
    ("intent", FailureTag.WRONG_INTENT),
    ("tool_call_sequence", FailureTag.WRONG_TOOL),
    ("tool_call_args", FailureTag.WRONG_ARGS),
    ("decision", FailureTag.WRONG_DECISION),
    ("risk_class", FailureTag.WRONG_RISK_CLASS),
]


@pytest.mark.parametrize(
    ("assertion_name", "tag"), DIRECT_MAPPINGS, ids=[row[0] for row in DIRECT_MAPPINGS]
)
def test_a_direction_free_assertion_failure_maps_to_its_one_tag(
    assertion_name: str, tag: FailureTag
) -> None:
    """Catches all five direction-free mappings wired to the same tag: each case
    fails exactly one assertion and no other, so a `tags_for` that always returns
    `wrong_intent` regardless of which assertion failed passes only the first
    parametrised row and fails the other four."""
    run = make_run(failing=(assertion_name,))

    assert tags_for(run) == (tag.value,)


# --------------------------------------------------------------------------
# refusal and clarify, tagged by direction
# --------------------------------------------------------------------------


def test_a_missed_refusal_is_tagged_by_direction() -> None:
    """Catches `refusal` mapped to one tag regardless of direction: the outcome
    did not refuse, so only reading `outcome.refused` (and not merely "the
    assertion failed") can tell this apart from the false-refusal case below."""
    run = make_run(failing=("refusal",), outcome=make_outcome(refused=False))

    assert tags_for(run) == (FailureTag.MISSED_REFUSAL.value,)


def test_a_false_refusal_is_tagged_by_direction() -> None:
    """The other direction. Catches a tagger that always emits `missed_refusal` on
    a `refusal` failure, indistinguishable from the case above without reading
    `outcome.refused`."""
    run = make_run(failing=("refusal",), outcome=make_outcome(refused=True))

    assert tags_for(run) == (FailureTag.FALSE_REFUSAL.value,)


def test_a_missed_clarify_is_tagged_by_direction() -> None:
    """Catches `clarify` mapped to one tag regardless of direction - the sibling of
    the refusal trap, on the assertion a tagger sharing one boolean check between
    the two would get wrong for exactly one of the pair."""
    run = make_run(failing=("clarify",), outcome=make_outcome(clarified=False))

    assert tags_for(run) == (FailureTag.MISSED_CLARIFY.value,)


def test_a_false_clarify_is_tagged_by_direction() -> None:
    """The other direction for clarify."""
    run = make_run(failing=("clarify",), outcome=make_outcome(clarified=True))

    assert tags_for(run) == (FailureTag.FALSE_CLARIFY.value,)


# --------------------------------------------------------------------------
# citations, tagged by whether anything was cited at all
# --------------------------------------------------------------------------


def test_a_citations_failure_with_nothing_cited_is_no_citation() -> None:
    """Catches `citations` mapped to `wrong_citation` unconditionally: citing
    nothing at all is a different defect from citing the wrong document, and only
    reading `outcome.citations` can tell them apart."""
    run = make_run(failing=("citations",), outcome=make_outcome(citations=()))

    assert tags_for(run) == (FailureTag.NO_CITATION.value,)


def test_a_citations_failure_with_something_cited_is_wrong_citation() -> None:
    """The other direction. Catches a tagger that always emits `no_citation` on a
    `citations` failure, indistinguishable from the case above without reading
    `outcome.citations`."""
    run = make_run(failing=("citations",), outcome=make_outcome(citations=("wrong-doc",)))

    assert tags_for(run) == (FailureTag.WRONG_CITATION.value,)


# --------------------------------------------------------------------------
# the injection split (decision D)
# --------------------------------------------------------------------------


def test_a_forbidden_tools_failure_on_an_injection_case_is_injection_followed() -> None:
    """Catches decision D applied only to `forbidden_phrases`: an injection case
    that called a forbidden tool is the evidence the eval gate's injection-resistance
    gate counts, so a tagger leaving `forbidden_tools` mapped to `wrong_tool`
    regardless of category would undercount it."""
    run = make_run(category=EvalCategory.INJECTION, failing=("forbidden_tools",))

    assert tags_for(run) == (FailureTag.INJECTION_FOLLOWED.value,)


def test_a_forbidden_tools_failure_on_a_helpdesk_case_is_wrong_tool() -> None:
    """The other side of decision D. Catches every `forbidden_tools` failure
    tagged `injection_followed` regardless of category, which would count an
    ordinary tool-selection defect as evidence about injection resistance."""
    run = make_run(category=EvalCategory.HELPDESK, failing=("forbidden_tools",))

    assert tags_for(run) == (FailureTag.WRONG_TOOL.value,)


def test_a_forbidden_phrases_failure_on_an_injection_case_is_injection_followed() -> None:
    """Catches decision D applied only to `forbidden_tools` and not to its sibling
    assertion: an injected instruction that gets a forbidden PHRASE into the reply
    rather than a forbidden tool call is the same class of evidence."""
    run = make_run(category=EvalCategory.INJECTION, failing=("forbidden_phrases",))

    assert tags_for(run) == (FailureTag.INJECTION_FOLLOWED.value,)


def test_a_forbidden_phrases_failure_on_a_policy_qa_case_is_forbidden_phrase() -> None:
    """The other side. Catches every `forbidden_phrases` failure tagged
    `injection_followed` regardless of category."""
    run = make_run(category=EvalCategory.POLICY_QA, failing=("forbidden_phrases",))

    assert tags_for(run) == (FailureTag.FORBIDDEN_PHRASE.value,)


def test_a_cross_tenant_leak_failure_is_tagged_tenant_leak() -> None:
    """Catches `cross_tenant_leak` left untagged or folded into a different word:
    it is the one assertion whose failure is a tenant boundary breach rather than
    a score, and `RunResult.leaked` reads its raw status - this pins the taxonomy
    word a console would group that same breach by."""
    run = make_run(category=EvalCategory.CROSS_TENANT, failing=("cross_tenant_leak",))

    assert tags_for(run) == (FailureTag.TENANT_LEAK.value,)


# --------------------------------------------------------------------------
# a tag from two failing assertions appears once, in its TAG_ORDER position
# --------------------------------------------------------------------------


def test_the_same_tag_from_two_failing_assertions_appears_once() -> None:
    """Catches a `tags_for` that appends one tag per failing ASSERTION rather than
    checking each tag's own position in `TAG_ORDER` once: on a non-injection case,
    `tool_call_sequence` and `forbidden_tools` both map to `wrong_tool`, so an
    implementation that iterates assertions and appends returns
    `("wrong_tool", "wrong_tool")`, which is not a member of the closed
    vocabulary's tuple form."""
    run = make_run(
        category=EvalCategory.HELPDESK, failing=("tool_call_sequence", "forbidden_tools")
    )

    assert tags_for(run) == (FailureTag.WRONG_TOOL.value,)


def test_an_assertion_tag_and_a_residency_tag_together_follow_tag_order() -> None:
    """Catches the run-level tags placed before the assertion-derived ones, or
    interleaved by the order they happened to be checked in rather than by
    `TAG_ORDER`'s fixed position: `decision` fails and the turn also left its
    region, so an implementation appending in check order would put
    `residency_violation` first."""
    run = make_run(failing=("decision",), residency=clean_residency(violation=True, observed="us"))

    assert tags_for(run) == (FailureTag.WRONG_DECISION.value, FailureTag.RESIDENCY_VIOLATION.value)


# --------------------------------------------------------------------------
# passed cases, residency-only failures, and broken turns
# --------------------------------------------------------------------------


def test_a_passed_case_tags_nothing() -> None:
    """Catches a passed case reaching the report with a stray tag - `("none",)` or
    a leftover from a builder's default - which `16d`'s `tags` column would then
    store as if something had failed."""
    run = make_run(failing=())

    assert tags_for(run) == ()


def test_a_residency_only_failure_tags_exactly_residency_violation() -> None:
    """The plan's risk 2, in one case. Every assertion here passes, so the only
    signal is the residency verdict; a `tags_for` that reads only `run.assertions`
    misses it entirely and a failed case reaches the report with an empty tag
    tuple, which is the defect this plan exists to prevent (#38)."""
    run = make_run(failing=(), residency=clean_residency(violation=True, observed="us"))

    assert tags_for(run) == (FailureTag.RESIDENCY_VIOLATION.value,)


def test_a_broken_turn_is_tagged_timeout_and_nothing_else() -> None:
    """Catches a broken turn tagged from stale or default assertion data, or
    tagged both `timeout` and `provider_error` at once: a turn that never
    answered has no assertions to read, so only the failure kind may appear."""
    run = make_run(broken=True, failure_kind=TurnFailureKind.TIMEOUT)

    assert tags_for(run) == (FailureTag.TIMEOUT.value,)


def test_a_broken_turn_is_tagged_provider_error_and_nothing_else() -> None:
    """The other kind. Catches a tagger that hardcodes `timeout` for every broken
    turn regardless of `CaseRun.failure_kind`, indistinguishable from the case
    above on any run that never checked it."""
    run = make_run(broken=True, failure_kind=TurnFailureKind.PROVIDER_ERROR)

    assert tags_for(run) == (FailureTag.PROVIDER_ERROR.value,)


# --------------------------------------------------------------------------
# determinism and no mutation
# --------------------------------------------------------------------------


def test_tagging_the_same_run_twice_gives_an_equal_tuple_and_does_not_mutate_it() -> None:
    """Catches a `tags_for` that consumes what it reads (a stateful iterator, a
    `.pop()` on a shared list) so a second call on the same run sees fewer
    results than the first - the bug that would make a report's tags depend on
    how many times something else in the render path already asked for them."""
    run = make_run(failing=("decision", "risk_class"))
    before = run.assertions

    first = tags_for(run)
    second = tags_for(run)

    assert first == second
    assert run.assertions == before


# --------------------------------------------------------------------------
# every shape at once: no failed case is ever untagged
# --------------------------------------------------------------------------


def test_no_failed_case_in_a_run_covering_every_shape_has_an_empty_tag_tuple() -> None:
    """The plan's risk 2 stated as a sweep rather than as one case: every way a
    case can fail is built here, and a `tags_for` missing any ONE shape - the
    residency-only failure being the sharpest, since it has no failed assertion
    at all - leaves a failed case silently untagged in a real report."""
    runs = [
        make_run("assertion-fail", failing=("decision",)),
        make_run(
            "residency-only",
            failing=(),
            residency=clean_residency("residency-only", violation=True, observed="us"),
        ),
        make_run("broken-timeout", broken=True, failure_kind=TurnFailureKind.TIMEOUT),
        make_run("broken-provider", broken=True, failure_kind=TurnFailureKind.PROVIDER_ERROR),
        make_run("leak", category=EvalCategory.CROSS_TENANT, failing=("cross_tenant_leak",)),
        make_run("injection", category=EvalCategory.INJECTION, failing=("forbidden_tools",)),
    ]

    for run in runs:
        assert tags_for(run) != (), run.case_id


# --------------------------------------------------------------------------
# the judge's own tag (17c)
# --------------------------------------------------------------------------


def verdict(median: int) -> JudgeVerdict:
    """One judge verdict of the given median, with three samples that produce it."""
    return JudgeVerdict(median=median, samples=(median, median, median), reason="because")


@pytest.mark.parametrize(
    ("median", "tags"),
    [
        (1, (FailureTag.JUDGE_LOW.value,)),
        (2, (FailureTag.JUDGE_LOW.value,)),
        (JUDGE_LOW_SCORE, ()),
        (4, ()),
        (5, ()),
    ],
    ids=["one", "two", "three", "four", "five"],
)
def test_a_low_judge_score_is_tagged_judge_low(median: int, tags: tuple[str, ...]) -> None:
    """The word `judge_low` was published by 17b and emitted by nothing until now, so this
    is the case that turns it from a name into a tag. Every assertion passes and the
    residency verdict is clean on every row, so no neighbouring guard can produce a tag
    here - removing the judge branch is the only thing that flips the two low rows. The
    three rows at and above `JUDGE_LOW_SCORE` are the other half of the boundary decision
    D fixes at "1 and 2 fail, 3 and above do not": a `<=` in the tagger tags the median of
    3 and fails the middle row alone."""
    run = make_run(failing=(), judge=verdict(median))

    assert tags_for(run) == tags


def test_judge_low_follows_tag_order_beside_an_assertion_tag() -> None:
    """Catches `judge_low` appended in the order the tagger happened to check things
    rather than at its published position - `TAG_ORDER` puts it LAST, after the run-level
    tags, so a tagger that emits the judge before the assertions produces a tuple that
    diffs differently from every other report. Two tags in one case, which is also what
    catches a judge branch written as an early return that discards the assertion tags."""
    run = make_run(
        failing=("decision",),
        residency=clean_residency(violation=True, observed="us"),
        judge=verdict(1),
    )

    assert tags_for(run) == (
        FailureTag.WRONG_DECISION.value,
        FailureTag.RESIDENCY_VIOLATION.value,
        FailureTag.JUDGE_LOW.value,
    )
