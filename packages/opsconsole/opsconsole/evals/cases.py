r"""The golden-set line: its model, its pre-checks and its loader (#34, #74).

One JSONL line is one case. `docs/eval-format.md` documents the same shape for
adopters who never read this module, and the worked example there is the line in
`tests/evals/fixtures/canonical.jsonl`, loaded field by field by
`test_the_canonical_example_line_loads_field_by_field` so the two cannot drift.

Two properties are worth more than the model itself.

Extras are forbidden at EVERY level - the case, its `expected`, and each entry of
`expected.tool_calls`. A misspelt field inside `expected` would otherwise load
cleanly and simply assert nothing, so the case its author wrote to pin a decision
would report `skipped` and inflate the score of every run afterwards; refused at
each level by `test_an_unknown_field_is_refused_on_every_level_of_the_model`.

Every `ExpectedOutcome` field defaults to `None` and never to an empty container,
because `assertions.assert_case` reads `None` as "this case asserts nothing about
it" and `[]` as "the author expects none". `[]` as a default would silently turn
the first into the second; pinned by
`test_an_absent_expected_field_is_none_and_not_an_empty_container`.

`id` and `tenant` are checked against the gateway's OWN `IDENTIFIER_PATTERN` and
`MAX_IDENTIFIER_LENGTH`, imported rather than restated, so a name a golden set
may use and a name a running gateway authenticated cannot disagree.
`principal_user` is email-shaped and gets its own `PRINCIPAL_PATTERN` at
`MAX_PRINCIPAL_LENGTH`. Both use `re.fullmatch`: `16b` puts a
case id in a report path and `16c` stores the id, the tenant and the principal as
`eval_cases` columns, so a trailing newline or a `..` segment has to be refused
before an `EvalCase` exists at all. Boundaries are inclusive and pinned exactly by
`test_the_length_caps_are_the_published_ones`,
`test_the_id_length_boundary_is_inclusive`,
`test_the_tenant_length_boundary_is_inclusive` and
`test_the_principal_length_boundary_is_inclusive`.

`load_golden_set` loads a set whole or not at all
(`test_nothing_is_returned_when_a_later_line_is_bad`), and refuses a missing file
and a file of nothing but comments rather than returning no cases
(`test_a_missing_file_raises_rather_than_returning_no_cases`,
`test_a_file_with_no_cases_raises_rather_than_returning_empty`) - a run over zero
cases scores every threshold there is.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH

MAX_PRINCIPAL_LENGTH: Final[int] = 128
"""The longest `principal_user` a golden line may name.

Wider than `MAX_IDENTIFIER_LENGTH` because a principal is a directory name and
routinely an email address. `16c` stores it as an `eval_cases` column, so the
value is pinned exactly by `test_the_length_caps_are_the_published_ones`.
"""

PRINCIPAL_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+(?:[._@-][a-z0-9]+)*")
"""The vocabulary of a `principal_user`: the identifier's, plus `.` and `@`.

Used with `re.fullmatch` and never with `.match` - in Python `$` also matches
before a trailing newline, so an anchored `.match` would accept
`dana.okafor\n` into a report path and a database column.
Asserted on the pattern itself by
`test_the_principal_pattern_is_anchored_at_both_ends`, so a caller that reuses it
inherits the refusal rather than the model's own stripping.
"""


class EvalCategory(StrEnum):
    """The twelve case categories a golden set may use.

    Closed, so a misspelt category is a load error naming its line rather than a
    thirteenth bucket nothing reports on. `docs/eval-format.md` publishes the same
    twelve words and their per-category minimums; the set and its size are pinned
    by `test_the_category_vocabulary_is_the_documented_twelve`.
    """

    ACCESS_AUTO = "access_auto"
    ACCESS_APPROVAL = "access_approval"
    ACCESS_BLOCKED_SOD = "access_blocked_sod"
    ACCESS_CRITICAL = "access_critical"
    HELPDESK = "helpdesk"
    POLICY_QA = "policy_qa"
    REFUSAL_OUT_OF_CORPUS = "refusal_out_of_corpus"
    CLARIFY_AMBIGUOUS = "clarify_ambiguous"
    INJECTION = "injection"
    CROSS_TENANT = "cross_tenant"
    LIFECYCLE = "lifecycle"
    REVIEW = "review"


class ExpectedToolCall(BaseModel):
    """One tool call a case expects, and the arguments it cares about.

    `args_subset` is a subset by design: a case names the
    arguments that carry its meaning and ignores the ticket ids and timestamps a
    real call also carries.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    args_subset: dict[str, Any] = {}


class ExpectedOutcome(BaseModel):
    """What a case asserts about the answer. Every field is optional.

    `None` means "asserts nothing about this", which `assert_case` reports as
    `skipped`. It is never a pass, so a thin case cannot lift a run's score.
    """

    model_config = ConfigDict(extra="forbid")

    intent: str | None = None
    tool_calls: list[ExpectedToolCall] | None = None
    decision: str | None = None
    risk_class: str | None = None
    should_refuse: bool | None = None
    should_clarify: bool | None = None
    citation_documents: list[str] | None = None
    forbidden_phrases: list[str] | None = None


def _checked_identifier(value: str, field: str) -> str:
    """Validate an id or a tenant against the gateway's own identifier vocabulary."""
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(f"{field}: longer than {MAX_IDENTIFIER_LENGTH} characters")
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{field}: not a lowercase slug")
    return value


def _checked_principal(value: str) -> str:
    """Validate a `principal_user` against its own, email-shaped vocabulary."""
    if len(value) > MAX_PRINCIPAL_LENGTH:
        raise IdentifierInvalid(f"principal_user: longer than {MAX_PRINCIPAL_LENGTH} characters")
    if PRINCIPAL_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("principal_user: not a directory principal")
    return value


class EvalCase(BaseModel):
    """One golden line.

    `expected` is required and may be empty: a case scored only by its judge
    rubric writes `{}`, but a line that FORGOT it is a load error rather than a
    case asserting nothing (`test_expected_is_required_and_may_be_empty`).

    The three identifier fields are checked before the instance exists, so a
    caller building an `EvalCase` in code gets the same refusal the loader gives
    a file (`test_a_malformed_id_is_refused`, `test_a_malformed_tenant_is_refused`,
    `test_a_malformed_principal_is_refused`).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    category: EvalCategory
    tenant: str
    principal_user: str
    messages: list[dict[str, str]]
    expected: ExpectedOutcome
    judge_rubric: str | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _checked_identifier(value, "id")

    @field_validator("tenant")
    @classmethod
    def _check_tenant(cls, value: str) -> str:
        return _checked_identifier(value, "tenant")

    @field_validator("principal_user")
    @classmethod
    def _check_principal_user(cls, value: str) -> str:
        return _checked_principal(value)


class EvalSetInvalid(Exception):
    """A golden set could not be loaded.

    A plain exception rather than a `GatewayError`: an unloadable set is an
    operator error reported to whoever ran the harness, never an answer to a
    caller, so it carries no wire code. `PolicyInvalid` is the same shape for the
    same reason, and `tests/gateway/test_error_contract.py` walks
    `GatewayError.__subclasses__()` asserting exactly one class per code - a
    codeless subclass would break that set.

    `line` is the 1-based FILE line of the offending line, or `None` when the
    file itself is at fault. That is the number an author counts to in an editor,
    comments and blank lines included; pinned by
    `test_a_bad_line_raises_eval_set_invalid_naming_its_line_number`.
    """

    def __init__(self, reason: str, *, path: Path, line: int | None = None) -> None:
        self.reason = reason
        self.path = path
        self.line = line
        location = f"{path}:{line}" if line is not None else str(path)
        super().__init__(f"{location}: {reason}")


def load_golden_set(path: Path) -> tuple[EvalCase, ...]:
    """Read a golden set, in file order, or raise.

    Blank lines and lines whose first non-space character is `#` are skipped, so
    a set can be commented; the reported line number is still the file's, which
    is what makes it findable. Every per-line failure - malformed JSON, an unknown
    category, an unknown field, a rejected identifier, a duplicate id - becomes an
    `EvalSetInvalid` naming that line, with the original raised as its `__cause__`
    (`test_an_identifier_refusal_is_wrapped_rather_than_leaked`), so a caller
    catches one type to report one file problem.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalSetInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise EvalSetInvalid(f"is not valid JSON: {exc.msg}", path=path, line=number) from exc
        try:
            case = EvalCase.model_validate(payload)
        except (ValidationError, IdentifierInvalid) as exc:
            raise EvalSetInvalid(f"is not a valid case: {exc}", path=path, line=number) from exc
        if case.id in seen:
            raise EvalSetInvalid(
                f"repeats the id {case.id!r}, first seen on line {seen[case.id]}",
                path=path,
                line=number,
            )
        seen[case.id] = number
        cases.append(case)

    if not cases:
        raise EvalSetInvalid("carries no cases", path=path)
    return tuple(cases)
