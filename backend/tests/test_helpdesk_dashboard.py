"""The help desk agent dashboard's stuck chat panels, and the scrape wiring behind them.

Files only: the dashboard JSON, the Prometheus config held in its ConfigMap, and the help
desk manifest. Read from this file's own location. No build tool runs, so these cases need
neither `kubectl` nor a cluster.

A panel that names a series nobody scrapes renders empty with no error anywhere, so the
panels and the scrape jobs are pinned together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "infra" / "k8s" / "monitoring" / "dashboards" / "helpdesk-agent.json"
PROMETHEUS = REPO_ROOT / "infra" / "k8s" / "monitoring" / "prometheus.yaml"
HELPDESK = REPO_ROOT / "infra" / "k8s" / "apps" / "helpdesk.yaml"

DATASOURCE_UID = "prometheus"
FIRST_NEW_ROW_Y = 24
"""The six existing panels fill rows 0 to 23; the new ones start below them."""


def documents(path: Path) -> list[dict[str, Any]]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def dashboard_panels() -> list[dict[str, Any]]:
    panels = json.loads(DASHBOARD.read_text(encoding="utf-8"))["panels"]
    assert isinstance(panels, list)
    return panels


def exprs(panel: dict[str, Any]) -> list[str]:
    return [str(target.get("expr", "")) for target in panel.get("targets", [])]


def new_panels() -> list[dict[str, Any]]:
    return [panel for panel in dashboard_panels() if panel["gridPos"]["y"] >= FIRST_NEW_ROW_Y]


def test_the_dashboard_has_the_three_stuck_chat_panels() -> None:
    """Catches a panel reading a series the help desk does not export (an empty panel), a
    panel missing, one left on another datasource, and one placed over the old panels. Three
    panels at row 24 or below, each on the datasource uid `prometheus`; one reads
    `helpdesk_turn_endings_total` by `outcome`, one reads p50 and p95 from the buckets of
    `helpdesk_turns_to_file`, and one sums `helpdesk_chats_waiting` with no tenant label."""
    panels = new_panels()

    assert len(panels) == 3
    for panel in panels:
        assert panel["targets"], panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == DATASOURCE_UID, panel["title"]

    by_series = {
        series: [panel for panel in panels if any(series in expr for expr in exprs(panel))]
        for series in (
            "helpdesk_turn_endings_total",
            "helpdesk_turns_to_file_bucket",
            "helpdesk_chats_waiting",
        )
    }
    assert {series: len(found) for series, found in by_series.items()} == {
        "helpdesk_turn_endings_total": 1,
        "helpdesk_turns_to_file_bucket": 1,
        "helpdesk_chats_waiting": 1,
    }
    endings = " ".join(exprs(by_series["helpdesk_turn_endings_total"][0]))
    assert "outcome" in endings
    turns = exprs(by_series["helpdesk_turns_to_file_bucket"][0])
    assert any("histogram_quantile(0.5" in expr for expr in turns)
    assert any("histogram_quantile(0.95" in expr for expr in turns)
    waiting = " ".join(exprs(by_series["helpdesk_chats_waiting"][0]))
    assert "sum(helpdesk_chats_waiting)" in waiting.replace(" (", "(")
    assert "tenant" not in waiting


def test_prometheus_scrapes_the_help_desk_agent_api_and_scheduler() -> None:
    """Catches a dashboard with no data behind it: the agent api on port 8000 and the
    scheduler must each be a scrape job with `/metrics` and a fully qualified Service name
    in `apps`, since Prometheus runs in `monitoring`."""
    config_map = next(
        doc for doc in documents(PROMETHEUS) if doc["metadata"]["name"] == "prometheus-config"
    )
    scrape = yaml.safe_load(config_map["data"]["prometheus.yml"])["scrape_configs"]
    targets = {
        target
        for job in scrape
        for group in job["static_configs"]
        for target in group["targets"]
        if job.get("metrics_path", "/metrics") == "/metrics"
    }

    assert "helpdesk-agent-api.apps.svc.cluster.local:8000" in targets
    scheduler = [
        target
        for target in targets
        if target.startswith("helpdesk-scheduler.apps.svc.cluster.local:")
    ]
    assert len(scheduler) == 1


def test_the_scheduler_has_a_metrics_port_and_service_and_still_no_probes() -> None:
    """Catches a scheduler scraped at a port nothing listens on (a Service with no
    container port behind it), a missing Service (the scrape name resolves to nothing), and
    probes added by the way: the scheduler serves nothing, so there must still be none."""
    docs = documents(HELPDESK)
    deployment = next(
        doc
        for doc in docs
        if doc["kind"] == "Deployment" and doc["metadata"]["name"] == "helpdesk-scheduler"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    service = next(
        doc
        for doc in docs
        if doc["kind"] == "Service" and doc["metadata"]["name"] == "helpdesk-scheduler"
    )

    ports = {port["name"]: port["containerPort"] for port in container["ports"]}
    assert "metrics" in ports
    service_ports = {port["name"]: port for port in service["spec"]["ports"]}
    assert service_ports["metrics"]["targetPort"] in ("metrics", ports["metrics"])
    assert service["spec"]["selector"] == {"app.kubernetes.io/name": "helpdesk-scheduler"}
    for probe in ("livenessProbe", "readinessProbe", "startupProbe"):
        assert probe not in container
