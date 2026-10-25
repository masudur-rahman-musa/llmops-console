"""The console's page and body models: one shape per `/console/*` page.

**Sources are flags, never errors**. Every page body carries
`sources: {postgres, prometheus}`, each `ok`, `unconfigured` or `error`, so a
source that is down empties its own fields while the route stays 200 and the
other source still renders. The read models each page composes are defined here;
`app.storage.dashboard_reads` imports them from here.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class SourceState(StrEnum):
    OK = "ok"
    UNCONFIGURED = "unconfigured"
    ERROR = "error"


class _Body(BaseModel):
    model_config = ConfigDict(frozen=True)


# --- read models: the shapes Postgres rows become, used by storage and page bodies ---


class LastEval(_Body):
    id: str
    finished_at: datetime
    passed: bool
    check_score: float | None


class CostDay(_Body):
    day: date
    tenant: str
    provider: str
    requests: int
    cached_requests: int
    cost_usd: float


class Saving(_Body):
    day: date
    tenant: str
    saved_usd: float


class AliasCost(_Body):
    alias: str
    cost_usd: float


class EvalRunSummary(_Body):
    id: str
    started_at: datetime
    run_kind: str
    git_sha: str | None
    passed: bool
    metrics: dict[str, Any]


class ReleaseMarker(_Body):
    git_sha: str
    deployed_at: datetime
    status: str


class CategoryCount(_Body):
    category: str
    passed: int
    total: int


class TagCount(_Body):
    tag: str
    count: int


class Regression(_Body):
    metric: str
    value: float
    previous: float


class Incident(_Body):
    id: str
    started_at: datetime
    resolved_at: datetime | None
    severity: str
    title: str
    summary: str
    postmortem_url: str | None
    open: bool


class RequestRow(_Body):
    """One ledger row as the Requests page lists it.

    The reader answers `trace_url` as null; the route fills it, because the link
    needs a setting and the id check lives in one place."""

    ts: datetime
    tenant: str
    alias: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool
    cost_usd: float
    prompt_version: str | None
    trace_id: str
    trace_url: str | None


class Release(_Body):
    git_sha: str
    deployed_at: datetime
    status: str
    pr_url: str | None
    eval_run_id: str | None
    eval_passed: bool | None
    check_score: float | None


# --- page bodies ---


class Sources(_Body):
    postgres: SourceState
    prometheus: SourceState


class HomePage(_Body):
    sources: Sources
    requests_today: int | None
    cost_usd_today: float | None
    p95_seconds: float | None
    error_rate: float | None
    last_eval: LastEval | None
    open_incidents: int | None


class Budget(_Body):
    tenant: str
    cap_usd: float
    spent_usd: float | None
    ratio: float | None


class CostPage(_Body):
    sources: Sources
    rollup_populated: bool | None
    days: tuple[CostDay, ...]
    savings: tuple[Saving, ...]
    savings_usd_total: float | None
    estimated: bool
    budgets: tuple[Budget, ...]
    cost_per_request: tuple[AliasCost, ...]


class BlockedItem(_Body):
    git_sha: str
    deployed_at: datetime
    pr_url: str | None
    eval_run_id: str | None
    regressed: tuple[Regression, ...]


class GateScore(_Body):
    """One score the eval gate checks: the newest run's value beside its threshold.

    `passed` is null when the value or the threshold is missing; a value exactly on
    the threshold passes, as in `app.evals.gate`."""

    name: str
    label: str
    value: float | None
    threshold: float | None
    passed: bool | None
    note: str | None


class QualityPage(_Body):
    sources: Sources
    gate: tuple[GateScore, ...]
    runs: tuple[EvalRunSummary, ...]
    release_markers: tuple[ReleaseMarker, ...]
    categories: tuple[CategoryCount, ...]
    taxonomy: tuple[TagCount, ...]
    blocked: tuple[BlockedItem, ...]


class Circuit(_Body):
    provider: str
    state: str


class FallbackCount(_Body):
    tenant: str
    alias: str
    provider: str
    count: float


class Slo(_Body):
    """One SLO's error budget. `budget_remaining` is negative when overspent, never clamped."""

    name: str
    objective: float
    window: str
    observed: float | None
    budget_remaining: float | None


class ReliabilityPage(_Body):
    sources: Sources
    slos: tuple[Slo, ...]
    latency_p95: tuple[tuple[float, float], ...]
    error_rate: tuple[tuple[float, float], ...]
    circuits: tuple[Circuit, ...]
    fallbacks: tuple[FallbackCount, ...]


class IncidentsPage(_Body):
    sources: Sources
    incidents: tuple[Incident, ...]


class ReleasesPage(_Body):
    sources: Sources
    releases: tuple[Release, ...]


class ModelCandidate(_Body):
    provider: str
    model: str


class ModelAlias(_Body):
    name: str
    purpose: str | None
    candidates: tuple[ModelCandidate, ...]


class ProviderFileRow(_Body):
    name: str
    provider: str
    kind: str
    enabled: bool


class ModelsPage(_Body):
    """The live routing policy. It reads no Postgres or Prometheus, so it has no `sources`."""

    loaded_at: datetime
    aliases: tuple[ModelAlias, ...]
    provider_files: tuple[ProviderFileRow, ...]


class RedactionCount(_Body):
    kind: str
    count: float


class RefusalCount(_Body):
    reason: str
    count: float


class ModelTokens(_Body):
    model: str
    input: float
    output: float


class RequestsPage(_Body):
    """What the gateway hid, tokens per model, the cache hit rate and the newest requests.

    `guardrail_blocks` and `cache_hit_rate` are null when Prometheus has no series:
    no traffic is not a 0% hit rate."""

    sources: Sources
    redactions: tuple[RedactionCount, ...]
    refusals: tuple[RefusalCount, ...]
    guardrail_blocks: float | None
    tokens: tuple[ModelTokens, ...]
    cache_hit_rate: float | None
    requests: tuple[RequestRow, ...]
