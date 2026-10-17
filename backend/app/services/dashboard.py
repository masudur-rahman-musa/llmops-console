"""The console's reads: the two source helpers, the budgets, and the SLO budgets.

`_postgres` and `_prometheus` each turn one source into a `(state, value)` pair,
so a page body can render the other source's fields when one is down.
`_budgets` reads neither source - the counter is the budget store the chat route
debits - and `_slos` is the three gateway-measured error budgets over the fixed
`SLO_WINDOW`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final

import psycopg
from fastapi import Request

from app.limits.budget import budget_ratio, current_month, to_micro_usd, to_usd
from app.model_routing.policy import PolicyStore
from app.schemas.dashboard import Budget, Slo, SourceState
from app.services.prometheus_reads import (
    AVAILABILITY_ALLOWED_BAD,
    AVAILABILITY_BUDGET_QUERY,
    AVAILABILITY_OBJECTIVE,
    LATENCY_ALLOWED_SLOW,
    LATENCY_BUDGET_QUERY,
    LATENCY_THRESHOLD_SECONDS,
    RESIDENCY_VIOLATIONS_30D_QUERY,
    PrometheusClient,
    PrometheusError,
)
from app.storage.dashboard_reads import ConsoleReader

logger = logging.getLogger(__name__)

SLO_WINDOW: Final[str] = "30d"
"""The window every SLO row reports, matching the `[30d]` in each budget query; it
never follows `days` (`test_slos_rows_are_availability_latency_residency_in_order`)."""


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


def budgets(request: Request, tenant: str | None) -> tuple[Budget, ...]:
    """Each tenant's cap from the policy against its spend counter for this UTC month.

    Read from neither source: the counter is the budget store the chat route
    debits, so this answers even with Postgres down. An unknown spend (a Redis
    outage) is `null` rather than zero (`test_budgets_read_the_policy_cap_and_the_store_spend`).
    """
    store: PolicyStore = request.app.state.policy
    policy = store.policy
    month = current_month()
    budgets: list[Budget] = []
    for name, tenant_policy in policy.tenants.items():
        if tenant is not None and name != tenant:
            continue
        cap = tenant_policy.budget_usd_month
        spend = request.app.state.budget_store.spend(name, month)
        budgets.append(
            Budget(
                tenant=name,
                cap_usd=float(cap),
                spent_usd=None if spend is None else float(to_usd(spend)),
                ratio=None
                if spend is None
                else budget_ratio(spend_micro_usd=spend, cap_micro_usd=to_micro_usd(cap)),
            )
        )
    return tuple(budgets)


def slos(client: PrometheusClient) -> tuple[Slo, ...]:
    """The three gateway-measured SLO budgets over the fixed `SLO_WINDOW`.

    No traffic makes both ratios `0 / 0`, which Prometheus answers `NaN` and the
    client drops, so an empty result is `null` rather than a full budget
    (`test_nan_gives_null_observed_and_budget`). An overspent budget stays negative
    (`test_an_overspent_budget_is_negative_and_unclamped`). A failure raises
    `PrometheusError` into the page's one `_prometheus` read, which empties `slos`
    with everything else
    (`test_a_failing_budget_query_is_200_with_empty_slos_prometheus_error_and_an_error_log`).
    """
    bad = first(client.instant_query(AVAILABILITY_BUDGET_QUERY))
    under = first(client.instant_query(LATENCY_BUDGET_QUERY))
    violations = first(client.instant_query(RESIDENCY_VIOLATIONS_30D_QUERY))
    return (
        Slo(
            name="availability",
            objective=AVAILABILITY_OBJECTIVE,
            window=SLO_WINDOW,
            observed=None if bad is None else 1 - bad,
            budget_remaining=None if bad is None else 1 - bad / AVAILABILITY_ALLOWED_BAD,
        ),
        Slo(
            name="latency",
            objective=LATENCY_THRESHOLD_SECONDS,
            window=SLO_WINDOW,
            observed=under,
            budget_remaining=None if under is None else 1 - (1 - under) / LATENCY_ALLOWED_SLOW,
        ),
        Slo(
            name="residency",
            objective=0.0,
            window=SLO_WINDOW,
            observed=violations,
            budget_remaining=None,
        ),
    )
