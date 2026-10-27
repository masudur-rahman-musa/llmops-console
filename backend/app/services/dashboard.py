"""The console's reads: the two source helpers and the budgets.

`_postgres` and `_prometheus` each turn one source into a `(state, value)` pair,
so a page body can render the other source's fields when one is down.
`budgets` sets each tenant's cap beside its ledger spend for the month.
`gate_scores` is the quality page's eval gate rows, `gate_verdict` the home page's word.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

import psycopg
from fastapi import Request

from app.evals.gate import Baseline, BaselineInvalid, load_baseline
from app.limits.budget import budget_ratio, to_micro_usd
from app.model_routing.policy import PolicyStore
from app.schemas.dashboard import Budget, EvalRunSummary, GateScore, SourceState
from app.services.prometheus_reads import PrometheusClient, PrometheusError
from app.storage.dashboard_reads import ConsoleReader, RequestFilters

logger = logging.getLogger(__name__)


def postgres[T](
    request: Request, page: str, read: Callable[[ConsoleReader], T]
) -> tuple[SourceState, T | None]:
    """One page's Postgres read, as a state and a value.

    Only `psycopg.Error` is a source failure; the pool's own timeout is one of
    them. Anything else is a bug and stays a 500.
    """
    reader: ConsoleReader = request.app.state.console_reader
    if not reader.configured:
        return SourceState.UNCONFIGURED, None
    try:
        return SourceState.OK, read(reader)
    except psycopg.Error as exc:
        logger.error("console %s: the postgres read failed: %s", page, type(exc).__name__)
        return SourceState.ERROR, None


def prometheus[T](
    request: Request, page: str, read: Callable[[PrometheusClient], T]
) -> tuple[SourceState, T | None]:
    """One page's Prometheus reads, as a state and a value. One failure fails them all."""
    client: PrometheusClient | None = request.app.state.prometheus
    if client is None:
        return SourceState.UNCONFIGURED, None
    try:
        return SourceState.OK, read(client)
    except PrometheusError as exc:
        logger.error("console %s: the prometheus read failed: %s", page, exc)
        return SourceState.ERROR, None


def first(samples: list[tuple[dict[str, str], float]]) -> float | None:
    return samples[0][1] if samples else None


def budgets(
    request: Request, tenant: str | None, spend: Mapping[str, float] | None
) -> tuple[Budget, ...]:
    """Each tenant's cap from the policy against its not-cached ledger spend this UTC month.

    `spend` is the cost page's Postgres read, not the Redis counter: the
    counter still enforces the cap but starts at zero after a Redis restart. A tenant
    missing from it spent nothing; `spend` itself null (Postgres down or unconfigured)
    is an unknown spend, `null` rather than zero. Pinned by
    `test_budgets_show_the_ledger_month_spend` and
    `test_budgets_are_null_when_postgres_is_down`.
    """
    store: PolicyStore = request.app.state.policy
    budgets: list[Budget] = []
    for name, tenant_policy in store.policy.tenants.items():
        if tenant is not None and name != tenant:
            continue
        cap = tenant_policy.budget_usd_month
        spent = None if spend is None else spend.get(name, 0.0)
        budgets.append(
            Budget(
                tenant=name,
                cap_usd=float(cap),
                spent_usd=spent,
                ratio=None
                if spent is None
                else budget_ratio(
                    spend_micro_usd=to_micro_usd(Decimal(str(spent))),
                    cap_micro_usd=to_micro_usd(cap),
                ),
            )
        )
    return tuple(budgets)


GATE_SCORES: Final[tuple[tuple[str, str], ...]] = (
    ("check_score", "Check score"),
    ("judge_score", "Judge score"),
)
"""The scores the eval gate checks, in the order the quality page lists them."""

JUDGE_NOTE: Final[str] = (
    "Counts only while the judge agrees with people (kappa 0.7 or better). "
    "Kappa measures that agreement: 1 means the judge and people always agree, "
    "0 means they agree no more than chance would."
)
"""Fixed, because a run row stores no kappa to say whether the judge counted."""


def gate_scores(baseline_path: Path, runs: tuple[EvalRunSummary, ...]) -> tuple[GateScore, ...]:
    """The newest run's gated scores beside the baseline's thresholds.

    A baseline that cannot be read or is not valid never fails the page: its
    thresholds are null. Pinned by tests/api/test_quality_gate.py.
    """
    newest = max(runs, key=lambda run: run.started_at, default=None)
    return _gate_rows(baseline_path, None if newest is None else newest.metrics)


def gate_verdict(baseline_path: Path, metrics: Mapping[str, Any] | None) -> bool | None:
    """The eval gate's verdict on one run's metrics, as the quality page's gate rows give it.

    False when any gated score is below its threshold, true when none is and at least one
    passed, `None` when nothing could be checked (no baseline, no gated score). The run's
    stored `passed` means "every stable case passed", a different question. Pinned by
    tests/api/test_console_routes.py::test_home_verdict_matches_the_eval_gate.
    """
    verdicts = [row.passed for row in _gate_rows(baseline_path, metrics) if row.passed is not None]
    if not verdicts:
        return None
    return all(verdicts)


def _count(metrics: Mapping[str, Any], name: str) -> int | None:
    """A stored case count, or `None` when the run never stored it."""
    raw = metrics.get(name)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def check_score_note(metrics: Mapping[str, Any] | None) -> str | None:
    """What the check score counts: "20 of 35 stable cases; 10 unstable, not counted".

    The check score leaves out unstable cases (passed on some tries, failed on others), so
    it is passed over stable cases, not over all of them. With no unstable case it is a
    plain "35 of 45 cases"; any count missing gives no note. Pinned by
    tests/api/test_quality_gate.py::test_the_check_score_note_counts_stable_cases.
    """
    if metrics is None:
        return None
    cases, passed, unstable = (_count(metrics, name) for name in ("cases", "passed", "unstable"))
    if cases is None or passed is None or unstable is None:
        return None
    if unstable == 0:
        return f"{passed} of {cases} cases"
    return f"{passed} of {cases - unstable} stable cases; {unstable} unstable, not counted"


def _gate_rows(baseline_path: Path, metrics: Mapping[str, Any] | None) -> tuple[GateScore, ...]:
    """One gate row per gated score: the value from `metrics`, the threshold from the baseline."""
    try:
        baseline: Baseline | None = load_baseline(baseline_path)
    except BaselineInvalid as exc:
        logger.warning("console quality: eval baseline %s", exc)
        baseline = None
    rows = []
    for name, label in GATE_SCORES:
        raw = None if metrics is None else metrics.get(name)
        value = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
        threshold = None if baseline is None else getattr(baseline, name)
        rows.append(
            GateScore(
                name=name,
                label=label,
                value=value,
                threshold=threshold,
                passed=None if value is None or threshold is None else value >= threshold,
                note=JUDGE_NOTE if name == "judge_score" else check_score_note(metrics),
            )
        )
    return tuple(rows)


def langfuse_traces_url(
    project_url: str | None, filters: RequestFilters, slow_ms: float | None
) -> str | None:
    """The Langfuse traces page with the request list's filters set, or `None` without a project.

    Each filter is `column;type;key;operator;value`, percent-encoded apart from `,;:.`, and
    they are joined by commas: the day as two UTC-midnight bounds, then tenant, model, cache,
    errors and slow (latency in seconds). Nothing filtered is the bare `/traces` page. This
    is Langfuse's page address, not a stable API; the checks on tenant and model keep `;`
    and `,` out of every value. Pinned by tests/api/test_console_filters.py."""
    if not project_url:
        return None
    parts: list[str] = []
    if filters.day is not None:
        start = f"{filters.day.isoformat()}T00:00:00.000Z"
        end = f"{(filters.day + timedelta(days=1)).isoformat()}T00:00:00.000Z"
        parts += [f"Timestamp;datetime;;>=;{start}", f"Timestamp;datetime;;<;{end}"]
    if filters.tenant is not None:
        parts.append(f"Metadata;stringObject;tenant;=;{filters.tenant}")
    if filters.model is not None:
        parts.append(f"Metadata;stringObject;model;=;{filters.model}")
    if filters.cached:
        parts.append("Metadata;stringObject;cache;=;hit")
    if filters.errors:
        parts.append("Level;string;;=;ERROR")
    if filters.slow and slow_ms is not None:
        parts.append(f"Latency (s);number;;>;{slow_ms / 1000:g}")
    base = f"{project_url.rstrip('/')}/traces"
    if not parts:
        return base
    return f"{base}?filter={','.join(quote(part, safe=',;:.') for part in parts)}"
