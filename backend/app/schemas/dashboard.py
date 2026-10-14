"""The console's page and body models: one shape per `/console/*` page.

**Sources are flags, never errors**. Every page body carries
`sources: {postgres, prometheus}`, each `ok`, `unconfigured` or `error`, so a
source that is down empties its own fields while the route stays 200 and the
other source still renders. The read models each page composes
(`LastEval`, `CostDay`, ...) live in `app.gateway.console_reads` (an earlier change moves
them to `app.storage.dashboard_reads`).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.gateway.console_reads import (
    AliasCost,
    CategoryCount,
    CostDay,
    EvalRunSummary,
    Incident,
    LastEval,
    RegionCount,
    Regression,
    Release,
    ReleaseMarker,
    Saving,
    TagCount,
)


class SourceState(StrEnum):
    OK = "ok"
    UNCONFIGURED = "unconfigured"
    ERROR = "error"


class _Body(BaseModel):
    model_config = ConfigDict(frozen=True)


class Sources(_Body):
    postgres: SourceState
    prometheus: SourceState


class HomePage(_Body):
    sources: Sources
    requests_today: int | None
    cost_usd_today: float | None
    p95_seconds: float | None
    error_rate: float | None
    residency_violations_today: int | None
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


class QualityPage(_Body):
    sources: Sources
    runs: tuple[EvalRunSummary, ...]
    release_markers: tuple[ReleaseMarker, ...]
    categories: tuple[CategoryCount, ...]
    taxonomy: tuple[TagCount, ...]
    blocked: tuple[BlockedItem, ...]


class RedactionCount(_Body):
    tenant: str
    kind: str
    count: float


class ResidencyPage(_Body):
    sources: Sources
    by_region: tuple[RegionCount, ...]
    violations: int | None
    redactions: tuple[RedactionCount, ...]


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
