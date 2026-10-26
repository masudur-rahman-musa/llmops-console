"""The console's reads: the two source helpers and the budgets.

`_postgres` and `_prometheus` each turn one source into a `(state, value)` pair,
so a page body can render the other source's fields when one is down.
`budgets` sets each tenant's cap beside its ledger spend for the month.
`gate_scores` is the quality page's eval gate rows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Final

import psycopg
from fastapi import Request

from app.evals.gate import Baseline, BaselineInvalid, load_baseline
from app.limits.budget import budget_ratio, to_micro_usd
from app.model_routing.policy import PolicyStore
from app.schemas.dashboard import Budget, EvalRunSummary, GateScore, SourceState
from app.services.prometheus_reads import PrometheusClient, PrometheusError
from app.storage.dashboard_reads import ConsoleReader

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

JUDGE_NOTE: Final[str] = "Counts only while the judge agrees with people (kappa 0.7 or better)."
"""Fixed, because a run row stores no kappa to say whether the judge counted."""


def gate_scores(baseline_path: Path, runs: tuple[EvalRunSummary, ...]) -> tuple[GateScore, ...]:
    """The newest run's gated scores beside the baseline's thresholds.

    A baseline that cannot be read or is not valid never fails the page: its
    thresholds are null. Pinned by tests/api/test_quality_gate.py.
    """
    try:
        baseline: Baseline | None = load_baseline(baseline_path)
    except BaselineInvalid as exc:
        logger.warning("console quality: eval baseline %s", exc)
        baseline = None
    newest = max(runs, key=lambda run: run.started_at, default=None)
    rows = []
    for name, label in GATE_SCORES:
        raw = None if newest is None else newest.metrics.get(name)
        value = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
        threshold = None if baseline is None else getattr(baseline, name)
        rows.append(
            GateScore(
                name=name,
                label=label,
                value=value,
                threshold=threshold,
                passed=None if value is None or threshold is None else value >= threshold,
                note=JUDGE_NOTE if name == "judge_score" else None,
            )
        )
    return tuple(rows)
