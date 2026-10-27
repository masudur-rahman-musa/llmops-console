"""The console's page and body models: one shape per `/console/*` page.

**Sources are flags, never errors**. Every page body carries
`sources: {postgres, prometheus}`, each `ok`, `unconfigured` or `error`
(`prometheus` is null on a page that never reads it), so a
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
    """The newest eval run. `check_score` is in percent points, as stored (57.14).

    `passed` is the eval gate verdict, `None` when there is no baseline to check against."""

    id: str
    finished_at: datetime
    passed: bool | None
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


class UserCost(_Body):
    """One random user id's served spend in the window. Never a name or an email."""

    tenant: str
    user_id: str
    requests: int
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
    eval_run_id: str | None = None
    """The eval run that let this release through, so the Quality chart draws the marker
    at that run. `None` when no run was recorded. Pinned by
    tests/storage/test_dashboard_reads_postgres.py::test_a_release_marker_names_its_eval_run."""


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
    status: int
    latency_ms: int | None


class ModelCandidate(_Body):
    provider: str
    model: str


class ModelAlias(_Body):
    name: str
    purpose: str | None
    candidates: tuple[ModelCandidate, ...]


class Routing(_Body):
    """The live routing policy: each alias in file order, its providers in fallback order.

    Read from the policy store, not from Postgres or Prometheus, so no source flag
    covers it."""

    loaded_at: datetime
    aliases: tuple[ModelAlias, ...]


# --- page bodies ---


class Sources(_Body):
    """`prometheus` is null on a page that reads no Prometheus at all (Cost, Quality)."""

    postgres: SourceState
    prometheus: SourceState | None


class HomePage(_Body):
    sources: Sources
    requests_today: int | None
    cost_usd_today: float | None
    p95_seconds: float | None
    error_rate: float | None
    last_eval: LastEval | None
    routing: Routing
    grafana_url: str | None


class Budget(_Body):
    tenant: str
    cap_usd: float
    spent_usd: float | None
    ratio: float | None


class CostPage(_Body):
    """`month_spend_usd` is every tenant's not-cached ledger spend this UTC month, the
    budgets' own read summed; null when Postgres is down or unconfigured.
    `prompt_cache_saved_usd` is what the provider's prompt cache saved, null likewise."""

    sources: Sources
    rollup_populated: bool | None
    days: tuple[CostDay, ...]
    savings: tuple[Saving, ...]
    savings_usd_total: float | None
    estimated: bool
    budgets: tuple[Budget, ...]
    month_spend_usd: float | None
    prompt_cache_saved_usd: float | None
    cost_per_request: tuple[AliasCost, ...]
    top_users: tuple[UserCost, ...]


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


class RedactionCount(_Body):
    kind: str
    count: float


class RefusalCount(_Body):
    reason: str
    count: float


class ModelTokens(_Body):
    """`input` is plain input; `cache_read` and `cache_write` are the prompt-cache tokens."""

    model: str
    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class DayRequests(_Body):
    """One UTC day: ledger rows that day, and the 5xx answers Prometheus counted.

    `errors` is null when Prometheus could not be read."""

    day: date
    requests: int
    errors: float | None


class RequestsPage(_Body):
    """What the gateway hid, tokens per model, the cache hit rate and the newest requests.

    `guardrail_blocks` is null when Prometheus has no series, `cache_hit_rate` when the
    ledger has no rows in the window: no traffic is not a 0% hit rate. `requests` is one
    page of the ledger: `requests_page` of `requests_pages`, newest first, narrowed by the
    request's filters. `slow_ms` is the window's p95 latency (null with no latencies),
    `tenants` and `models` are the window's distinct values for the filter lists, and
    `langfuse_url` opens the same requests in Langfuse (null without a project URL). `per_day` has
    every window day, oldest first, and is empty with `tokens` when the ledger is down.
    `prompt_cache_share` is cache reads over all input in `tokens`, null when there is none."""

    sources: Sources
    redactions: tuple[RedactionCount, ...]
    refusals: tuple[RefusalCount, ...]
    guardrail_blocks: float | None
    tokens: tuple[ModelTokens, ...]
    cache_hit_rate: float | None
    prompt_cache_share: float | None
    per_day: tuple[DayRequests, ...]
    requests: tuple[RequestRow, ...]
    requests_page: int
    requests_pages: int
    slow_ms: float | None
    tenants: tuple[str, ...]
    models: tuple[str, ...]
    langfuse_url: str | None
