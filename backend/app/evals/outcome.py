"""What one case actually produced: the seam between the runner and the verdict.

A dumb record with no logic. `evals.target` fills it from a real target run;
`assertions` reads it and nothing else, which is what lets the whole deterministic
verdict be pinned before anything here can reach a network.

Defaults read as "nothing observed" rather than as an observation: `refused` and
`clarified` are False and every sequence is empty, so an outcome the runner never
filled cannot look like a refusal.
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
    assertions ran in.

    `other_tenant_markers` is supplied here rather than read from the golden line:
    the Helpdesk owns the line format, and the runner is what knows which other
    tenants the seed carries.
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
