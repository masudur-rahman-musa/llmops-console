"""What one case actually produced: the seam between the runner and the verdict.

A dumb record with no logic. `16b` fills it from a real target run and `16c`
writes its measured fields into a report and an `eval_cases` row; `assertions`
reads it and nothing else, which is what lets the whole deterministic verdict be
pinned before anything here can reach a network.

Defaults read as "nothing observed" rather than as an observation: `refused` and
`clarified` are False and every sequence is empty, so an outcome the runner never
filled cannot look like a refusal. Pinned by
`test_the_outcome_record_carries_what_the_runner_measures`, which also pins the
four measured fields (`latency_ms`, `cost_usd`, `provider`, `region`) this plan
asserts nothing about and `16c` has no report column for if they are missing.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict


class ObservedToolCall(BaseModel):
    """One tool call as the target actually sent it, arguments and all.

    The counterpart of `cases.ExpectedToolCall`, and deliberately not the same
    class: a case names the arguments it cares about (`args_subset`), an outcome
    carries every argument that was sent (`args`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    args: dict[str, Any] = {}


class CaseOutcome(BaseModel):
    """The observed answer to one case.

    Frozen, because `assert_case` is pure and may not sort a citation list or
    casefold a reply in place - the corruption would depend on the order the
    assertions ran in (`test_asserting_a_case_does_not_mutate_the_case_or_the_outcome`).

    `other_tenant_markers` is supplied here rather than read from the golden line: the Helpdesk owns the line format, and the runner is
    what knows which other tenants the seed carries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: str | None = None
    tool_calls: tuple[ObservedToolCall, ...] = ()
    decision: str | None = None
    risk_class: str | None = None
    refused: bool = False
    clarified: bool = False
    citations: tuple[str, ...] = ()
    reply: str = ""
    other_tenant_markers: tuple[str, ...] = ()
    latency_ms: int | None = None
    cost_usd: Decimal | None = None
    provider: str | None = None
    region: str | None = None
