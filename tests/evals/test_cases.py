"""`evals/cases.py`: the golden-line model, its pre-checks and the JSONL loader.

Library only. Nothing here builds an app, sends a request or touches a backing
service; the only I/O is reading a fixture file from `tests/evals/fixtures/`.
What these cases pin is the format an adopter writes by hand (#74) and the
refusal of every line shape that would otherwise be loaded as a case asserting
less than its author meant.

The contract these cases are written against, before it exists:

* `EvalCategory` - a `StrEnum` closed at twelve values.
* `ExpectedToolCall(name=..., args_subset=...)` and `ExpectedOutcome(...)`, both
  `extra="forbid"`, every `ExpectedOutcome` field optional and defaulting to
  `None` so an absent field means "asserts nothing" rather than "asserts empty".
* `EvalCase(id=..., category=..., tenant=..., principal_user=..., messages=...,
  expected=..., judge_rubric=None)`, `extra="forbid"`, raising
  `IdentifierInvalid` from its field validators before the instance exists.
* `load_golden_set(path) -> tuple[EvalCase, ...]`, raising `EvalSetInvalid` with
  a `line` attribute and a `str()` beginning `f"{path}:{line}: "`.
* `MAX_PRINCIPAL_LENGTH` and `PRINCIPAL_PATTERN` - the constants a project rule
  requires a test to pin exactly.

a project rule on `PRINCIPAL_PATTERN`: matched with `re.fullmatch`, and the
rejection table below carries a trailing newline, a trailing CRLF and a trailing
space, because a check written `^...$` with `.match` accepts the first of those,
and `16c` stores `principal_user` as an `eval_cases` column while `16b` puts a
case id into a report path.

`id` and `tenant` are checked against the gateway's OWN `IDENTIFIER_PATTERN` and
`MAX_IDENTIFIER_LENGTH`, imported here rather than restated, so an id a golden
set may name and a tenant the gateway authenticated can never disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from opsconsole.evals.cases import (
    MAX_PRINCIPAL_LENGTH,
    PRINCIPAL_PATTERN,
    EvalCase,
    EvalCategory,
    EvalSetInvalid,
    ExpectedOutcome,
    ExpectedToolCall,
    load_golden_set,
)
from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.resolver import MAX_IDENTIFIER_LENGTH

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

CANONICAL_PATH = FIXTURE_DIR / "canonical.jsonl"
"""The worked example of an earlier change decision F, one line, on line 3 of that file.

`docs/eval-format.md` carries the SAME line. The drift this catches is the whole
point of the fixture: the first real Helpdesk set that fails to load names the
field it disagrees about, here rather than at the first `ops-eval run`.
"""


def test_the_category_vocabulary_is_the_documented_twelve() -> None:
    """Catches a thirteenth category slipping in behind a documented set, and a renamed
    or dropped one - either of which makes a shipped golden set unloadable while
    `docs/eval-format.md` still promises the old word. Asserted in both directions (the
    value set and the length), so a subset check cannot pass a member that was added."""
    assert {member.value for member in EvalCategory} == {
        "access_auto",
        "access_approval",
        "access_blocked_sod",
        "access_critical",
        "helpdesk",
        "policy_qa",
        "refusal_out_of_corpus",
        "clarify_ambiguous",
        "injection",
        "cross_tenant",
        "lifecycle",
        "review",
    }
    assert len(list(EvalCategory)) == 12


def test_the_canonical_example_line_loads_field_by_field() -> None:
    """Catches a model that drifts from the line an adopter is told to copy: a renamed
    field, a `tool_calls` entry that is not `{name, args_subset}`, a `judge_rubric` that
    is not optional. Every field of the example is read back, so a model that silently
    drops one (an unparsed `args_subset`, a `messages` list coerced to strings) fails
    here rather than by asserting nothing at run time."""
    cases = load_golden_set(CANONICAL_PATH)

    assert len(cases) == 1
    case = cases[0]
    assert case.id == "access-auto-001"
    assert case.category is EvalCategory.ACCESS_AUTO
    assert case.tenant == "acme"
    assert case.principal_user == "dana.okafor"
    assert case.messages == [
        {
            "role": "user",
            "content": "Please give me read access to the finance-reports SharePoint site.",
        }
    ]
    assert case.judge_rubric is not None
    assert "finance-reports" in case.judge_rubric

    expected = case.expected
    assert expected.intent == "request_access"
    assert expected.tool_calls == [
        ExpectedToolCall(
            name="grant_access", args_subset={"resource": "finance-reports", "level": "read"}
        )
    ]
    assert expected.decision == "auto_approved"
    assert expected.risk_class == "low"
    assert expected.should_refuse is False
    assert expected.should_clarify is False
    assert expected.citation_documents == ["access-policy-v3"]
    assert expected.forbidden_phrases == ["ticket closed"]


def test_blank_and_comment_lines_are_skipped_and_the_order_is_the_files() -> None:
    """Catches a loader that returns the cases in any order but the file's - a dict keyed
    by id, a set, a sort - which would make a report's row order differ from the set an
    author reads. The fixture puts a blank line and an INDENTED comment between cases, so
    a reader that only skips lines literally starting with `#`, or that treats a blank
    line as a parse error, fails here too."""
    cases = load_golden_set(FIXTURE_DIR / "valid_set.jsonl")

    assert [case.id for case in cases] == ["alpha", "beta", "gamma"]


def test_an_absent_judge_rubric_is_none_rather_than_a_load_error() -> None:
    """Catches `judge_rubric` made required, which would refuse every deterministic-only
    case in a real set - and catches a default of `""`, which a later judge step would
    read as an empty rubric to score against rather than as "no judge for this case"."""
    cases = load_golden_set(FIXTURE_DIR / "valid_set.jsonl")

    assert cases[1].id == "beta"
    assert cases[1].judge_rubric is None
    assert cases[0].judge_rubric is not None


def test_an_absent_expected_field_is_none_and_not_an_empty_container() -> None:
    """Catches `tool_calls: list[...] = []` and `citation_documents: list[str] = []`
    defaults. An empty list is indistinguishable from "the author expects no tool calls",
    which `assert_case` must report as a PASS; `None` is what makes it report `skipped`,
    and the difference is the whole reason a thin case cannot inflate a score."""
    cases = load_golden_set(FIXTURE_DIR / "valid_set.jsonl")

    beta = cases[1].expected
    assert beta.citation_documents == ["security-policy-v2"]
    assert beta.intent is None
    assert beta.tool_calls is None
    assert beta.decision is None
    assert beta.risk_class is None
    assert beta.should_refuse is None
    assert beta.should_clarify is None
    assert beta.forbidden_phrases is None


REJECTED_FILES: list[tuple[str, str, int]] = [
    ("a malformed JSON line", "malformed_json.jsonl", 4),
    ("an unknown category", "unknown_category.jsonl", 2),
    ("an unknown top-level field", "unknown_field.jsonl", 2),
    ("an unknown field inside expected", "unknown_expected_field.jsonl", 4),
    ("an unknown field inside a tool_calls entry", "unknown_tool_call_field.jsonl", 1),
    ("a duplicate id", "duplicate_id.jsonl", 4),
    ("an id that is not a lowercase slug", "bad_identifier.jsonl", 2),
]
"""One file per refusal, each with the 1-based line number of its offending line.

The three extras files each carry EVERY required field correctly and ONE extra
key, so a refusal can only come from `extra="forbid"` and never from a missing
field - the neighbouring guard that would otherwise pass these rows whether or
not extras are forbidden at that level. The comment and blank lines in the
fixtures are what make the expected numbers file lines rather than case indices.
"""


@pytest.mark.parametrize(
    ("label", "filename", "line"), REJECTED_FILES, ids=[row[0] for row in REJECTED_FILES]
)
def test_a_bad_line_raises_eval_set_invalid_naming_its_line_number(
    label: str, filename: str, line: int
) -> None:
    """Catches a loader that counts CASES rather than file lines, which points an author
    at the wrong row of a set with comments in it, and one that reports no location at
    all. The number is asserted exactly, in the message and on the attribute, so an
    off-by-one from `enumerate` starting at 0 fails. The duplicate-id row also catches a
    loader that reports the FIRST occurrence: the offending line is the second one."""
    path = FIXTURE_DIR / filename

    with pytest.raises(EvalSetInvalid) as raised:
        load_golden_set(path)

    assert raised.value.line == line
    assert str(raised.value).startswith(f"{path}:{line}: ")


def test_nothing_is_returned_when_a_later_line_is_bad() -> None:
    """Catches a loader that appends as it parses and returns the cases it managed - the
    fixture's line 3 is a valid case and line 4 is broken, so a partial return would hand
    back one case and a run would score a set nobody wrote. Whole or not at all."""
    with pytest.raises(EvalSetInvalid):
        load_golden_set(FIXTURE_DIR / "malformed_json.jsonl")


def test_an_identifier_refusal_is_wrapped_rather_than_leaked() -> None:
    """Catches a loader that lets `IdentifierInvalid` escape: a caller loading a set would
    then have to catch two exception types to report one file problem, and the failure
    would carry no line number at all. The cause is kept, so the operator still learns
    which field was wrong."""
    with pytest.raises(EvalSetInvalid) as raised:
        load_golden_set(FIXTURE_DIR / "bad_identifier.jsonl")

    assert isinstance(raised.value.__cause__, IdentifierInvalid)


def test_a_missing_file_raises_rather_than_returning_no_cases() -> None:
    """Catches a loader that treats an unreadable path as an empty set - a mistyped
    `--set` argument in `16b` would then run zero cases and report a perfect score. The
    line is `None` because no line is at fault, which is what distinguishes a file
    problem from a line problem for the caller that prints it."""
    path = FIXTURE_DIR / "there-is-no-such-set.jsonl"

    with pytest.raises(EvalSetInvalid) as raised:
        load_golden_set(path)

    assert raised.value.line is None
    assert str(path) in str(raised.value)


def test_a_file_with_no_cases_raises_rather_than_returning_empty() -> None:
    """Catches `return cases` on a file of comments and blank lines. An empty golden set
    passes every threshold there is, so the gate a set exists to be would go green on a
    file whose cases were all commented out."""
    with pytest.raises(EvalSetInvalid) as raised:
        load_golden_set(FIXTURE_DIR / "no_cases.jsonl")

    assert raised.value.line is None


MINIMAL: dict[str, Any] = {
    "id": "case-1",
    "category": "helpdesk",
    "tenant": "acme",
    "principal_user": "dana.okafor",
    "messages": [{"role": "user", "content": "Printer is offline."}],
    "expected": {},
}
"""The smallest line the format accepts, built here so a boundary row changes ONE
field and every other field is known good."""


def case_with(**overrides: object) -> EvalCase:
    """Build an `EvalCase` from the minimal line with the named fields replaced."""
    return EvalCase.model_validate({**MINIMAL, **overrides})


MAX_IDENTIFIER = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP_IDENTIFIER = "a" * (MAX_IDENTIFIER_LENGTH + 1)
MAX_PRINCIPAL = "a" * MAX_PRINCIPAL_LENGTH
OVER_CAP_PRINCIPAL = "a" * (MAX_PRINCIPAL_LENGTH + 1)


def test_the_length_caps_are_the_published_ones() -> None:
    """Catches a cap invented here instead of imported from the resolver, and a principal
    cap quietly widened - `16c` stores both as `eval_cases` columns, so a value the model
    accepts and the column does not is an insert that fails after the run is paid for."""
    assert MAX_IDENTIFIER_LENGTH == 64
    assert MAX_PRINCIPAL_LENGTH == 128


def test_the_id_length_boundary_is_inclusive() -> None:
    """Catches an off-by-one at the cap in either direction: a check written `>=` refuses
    an id of exactly the permitted length, and a check on a longer constant lets an id
    past that the resolver's own vocabulary would refuse."""
    assert case_with(id=MAX_IDENTIFIER).id == MAX_IDENTIFIER

    with pytest.raises(IdentifierInvalid):
        case_with(id=OVER_CAP_IDENTIFIER)


def test_the_tenant_length_boundary_is_inclusive() -> None:
    """Catches the same off-by-one on the field the gateway already authenticated: a
    tenant of exactly 64 characters is a name a running gateway serves, so refusing it
    here would make a legitimate tenant's golden set unloadable."""
    assert case_with(tenant=MAX_IDENTIFIER).tenant == MAX_IDENTIFIER

    with pytest.raises(IdentifierInvalid):
        case_with(tenant=OVER_CAP_IDENTIFIER)


def test_the_principal_length_boundary_is_inclusive() -> None:
    """Catches the principal being checked against the identifier's cap of 64 - real
    principals are email-shaped and routinely longer - and catches 129 being accepted,
    which is the value `16c`'s column would truncate."""
    assert case_with(principal_user=MAX_PRINCIPAL).principal_user == MAX_PRINCIPAL

    with pytest.raises(IdentifierInvalid):
        case_with(principal_user=OVER_CAP_PRINCIPAL)


REJECTED_IDENTIFIERS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP_IDENTIFIER),
    ("uppercase", "Case-1"),
    ("a space", "case 1"),
    ("a colon, which would forge a second key segment", "case:1"),
    ("a path separator, which would escape a report directory", "../case-1"),
    ("a glob character", "case-*"),
    ("a trailing newline", "case-1\n"),
    ("a trailing CRLF", "case-1\r\n"),
    ("a trailing space", "case-1 "),
    ("a leading separator", "-case-1"),
]


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_IDENTIFIERS, ids=[row[0] for row in REJECTED_IDENTIFIERS]
)
def test_a_malformed_id_is_refused(label: str, value: str) -> None:
    """Catches an id that never passes through a pattern at all: `16b` puts it in a report
    file name, so `../case-1` writes outside the report directory and `case-1\\n` puts a
    line break in a path and in an `eval_cases` row. The trailing-newline row is standing
    rule 4's - a check written `^...$` with `.match` accepts it."""
    with pytest.raises(IdentifierInvalid):
        case_with(id=value)


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_IDENTIFIERS, ids=[row[0] for row in REJECTED_IDENTIFIERS]
)
def test_a_malformed_tenant_is_refused(label: str, value: str) -> None:
    """Catches the tenant being taken on trust because it "comes from a file we wrote":
    the set is an adopter's file (#74), and this tenant reaches a routing resolution and
    a ledger row in `16b` and `16c`. Same rejection table as the id, so the two fields
    cannot drift into two vocabularies."""
    with pytest.raises(IdentifierInvalid):
        case_with(tenant=value)


ACCEPTED_PRINCIPALS: list[tuple[str, str]] = [
    ("a dotted name", "dana.okafor"),
    ("an email-shaped principal", "sam@acme.example"),
    ("a numeric directory id", "u-1042"),
    ("an underscore", "svc_desk"),
    ("exactly the cap", MAX_PRINCIPAL),
]


@pytest.mark.parametrize(
    ("label", "value"), ACCEPTED_PRINCIPALS, ids=[row[0] for row in ACCEPTED_PRINCIPALS]
)
def test_a_realistic_principal_is_accepted(label: str, value: str) -> None:
    """Catches `principal_user` being validated with the resolver's slug pattern, which
    admits neither a dot nor an `@`. That would refuse the form every real directory uses
    and make the format unwritable for the adopters #74 exists for."""
    assert case_with(principal_user=value).principal_user == value


REJECTED_PRINCIPALS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP_PRINCIPAL),
    ("uppercase", "Dana.Okafor"),
    ("a space", "dana okafor"),
    ("a colon", "dana:okafor"),
    ("a path separator", "dana/okafor"),
    ("a trailing newline", "dana.okafor\n"),
    ("a trailing CRLF", "dana.okafor\r\n"),
    ("a trailing space", "dana.okafor "),
    ("a leading separator", ".dana"),
    ("a trailing separator", "dana."),
    ("two separators in a row", "dana..okafor"),
]


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_PRINCIPALS, ids=[row[0] for row in REJECTED_PRINCIPALS]
)
def test_a_malformed_principal_is_refused(label: str, value: str) -> None:
    """Catches a principal check that is only a length cap, which would let a newline or a
    path separator into the `eval_cases` column `16c` writes and the report `16b` names.
    The trailing-newline and trailing-CRLF rows are a project rule's."""
    with pytest.raises(IdentifierInvalid):
        case_with(principal_user=value)


def test_the_principal_pattern_is_anchored_at_both_ends() -> None:
    """Catches `PRINCIPAL_PATTERN` used with `.match` and written `^...$`: in Python `$`
    also matches before a trailing newline, so `dana.okafor\\n` would pass a check that
    looks correct. Asserted on the pattern itself as well as through the model, because
    the model could be doing its own stripping and hiding a pattern that is wrong for the
    next caller to reuse."""
    assert PRINCIPAL_PATTERN.fullmatch("dana.okafor") is not None
    assert PRINCIPAL_PATTERN.fullmatch("dana.okafor\n") is None
    assert PRINCIPAL_PATTERN.fullmatch("dana.okafor\r\n") is None
    assert PRINCIPAL_PATTERN.fullmatch("dana.okafor ") is None
    assert PRINCIPAL_PATTERN.fullmatch("") is None


def test_an_unknown_field_is_refused_on_every_level_of_the_model() -> None:
    """Catches `extra="forbid"` set on `EvalCase` alone. A typo inside `expected` or
    inside a `tool_calls` entry is the dangerous one: the case still loads, the misspelt
    assertion is simply absent, and the case reports `skipped` for the very thing its
    author wrote it to pin. Asserted at the model level as well as through the loader, so
    a caller building an `EvalCase` in code (`16b`'s tests) gets the same refusal."""
    with pytest.raises(ValueError):
        ExpectedOutcome.model_validate({"intent": "troubleshoot", "intents": "troubleshoot"})

    with pytest.raises(ValueError):
        ExpectedToolCall.model_validate({"name": "grant_access", "args_subset": {}, "args": {}})

    with pytest.raises(ValueError):
        case_with(judge_rubrik="a misspelt rubric")


def test_expected_is_required_and_may_be_empty() -> None:
    """Catches `expected` given a default, which would turn a line that forgot it into a
    case asserting nothing at all instead of a load error - and catches an empty
    `expected` being refused, which is the legitimate shape of a case scored only by its
    judge rubric."""
    assert case_with().expected == ExpectedOutcome()

    payload = {key: value for key, value in MINIMAL.items() if key != "expected"}
    with pytest.raises(ValueError):
        EvalCase.model_validate(payload)
