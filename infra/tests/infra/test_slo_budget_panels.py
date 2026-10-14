"""An earlier change: the "SLO error budgets" row of `gateway-overview.json`.

The dashboards are read off the session-scoped `built` fixture in `infra/tests/infra/conftest.py`
(one `kubectl kustomize infra/k8s` per session, a project rule) through
`test_k8s_grafana.py`'s `dashboard_documents` and `panels_of`, so a panel written into the JSON
file but not generated into the `grafana-dashboards` ConfigMap is the same failure as a panel
never written.

Each panel's `expr` is compared byte for byte with the constant the console route runs, so
Grafana and `/console/reliability` cannot show two different budgets for one SLO.
"""

from __future__ import annotations

from typing import Any

from opsconsole.gateway.prometheus import (
    AVAILABILITY_BUDGET_QUERY,
    LATENCY_BUDGET_QUERY,
    RESIDENCY_VIOLATIONS_30D_QUERY,
)

from infra.test_k8s_grafana import dashboard_documents, panels_of

DASHBOARD_KEY = "gateway-overview.json"
ROW_TITLE = "SLO error budgets"
BUDGET_EXPRS = [
    AVAILABILITY_BUDGET_QUERY,
    LATENCY_BUDGET_QUERY,
    RESIDENCY_VIOLATIONS_30D_QUERY,
]


def overview_panels(built: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every panel of the generated gateway overview, nested panels included."""
    dashboards = dashboard_documents(built)
    assert DASHBOARD_KEY in dashboards, f"{DASHBOARD_KEY} is not in {sorted(dashboards)}"
    return panels_of(dashboards[DASHBOARD_KEY])


def target_exprs(panel: dict[str, Any]) -> list[str]:
    """The `expr` of every target a panel declares."""
    return [str(target.get("expr", "")) for target in panel.get("targets", [])]


def budget_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The panels with any target reading one of the three budget queries."""
    return [panel for panel in panels if any(expr in BUDGET_EXPRS for expr in target_exprs(panel))]


def y_of(panel: dict[str, Any]) -> int:
    grid = panel.get("gridPos", {})
    assert isinstance(grid, dict) and "y" in grid, f"{panel.get('title')!r} has no gridPos.y"
    return int(grid["y"])


def test_gateway_overview_has_the_slo_row_with_three_stat_panels(
    built: list[dict[str, Any]],
) -> None:
    """Catches the row missing or titled otherwise, a budget drawn as a timeseries rather than
    a `stat` (a 30-day ratio has one value, not a line), a panel duplicated or left out, and
    the three panels placed ABOVE their row header, where Grafana files them under whatever row
    precedes it."""
    panels = overview_panels(built)

    rows = [
        panel for panel in panels if panel.get("type") == "row" and panel.get("title") == ROW_TITLE
    ]
    assert len(rows) == 1, f"{DASHBOARD_KEY} has {len(rows)} rows titled {ROW_TITLE!r}"

    found = budget_panels(panels)
    assert len(found) == 3, [panel.get("title") for panel in found]
    for panel in found:
        assert panel.get("type") == "stat", f"{panel.get('title')!r} is {panel.get('type')!r}"
        assert y_of(panel) > y_of(rows[0]), f"{panel.get('title')!r} sits above {ROW_TITLE!r}"


def test_each_slo_panel_expr_equals_its_prometheus_constant(
    built: list[dict[str, Any]],
) -> None:
    """Catches a panel `expr` retyped by hand and drifting from the console's constant (a
    `le="6"` for `le="6.0"`, a `[7d]` window, a dropped provider filter), a panel carrying a
    second target beside the budget query, and one budget query drawn twice while another is
    missing: each constant must be the sole target of exactly one panel."""
    found = budget_panels(overview_panels(built))

    for panel in found:
        assert len(target_exprs(panel)) == 1, f"{panel.get('title')!r}: {target_exprs(panel)}"
    assert sorted(target_exprs(panel)[0] for panel in found) == sorted(BUDGET_EXPRS)
