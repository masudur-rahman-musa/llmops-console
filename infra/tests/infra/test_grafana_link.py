"""An earlier change: the Overview's "Health details in Grafana" link.

On the server the link is the gateway's own Grafana dashboard, built from the Grafana Ingress host
and the dashboard `uid`; locally it is the compose Grafana's gateway overview on port 3002.
An earlier change: the build runs no Grafana, so its link is empty.
An earlier change: the local stack runs Grafana, so `backend/.env.demo` holds the 3002 link.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from infra.test_k8s_gateway_prometheus import GATEWAY_CONFIG_MAP, gateway_config

REPO_ROOT = Path(__file__).resolve().parents[3]
INGRESS = REPO_ROOT / "infra" / "k8s" / "monitoring" / "ingress.yaml"
DASHBOARD = REPO_ROOT / "infra" / "k8s" / "monitoring" / "dashboards" / "gateway-overview.json"
COMPOSE = REPO_ROOT / "docker-compose.yml"
ENV_DEMO = REPO_ROOT / "backend" / ".env.demo"
KEY = "OPSCONSOLE_GRAFANA_URL"
EXPECTED = "https://grafana.example.com/d/gateway-overview"


def grafana_host() -> str:
    documents = [d for d in yaml.safe_load_all(INGRESS.read_text(encoding="utf-8")) if d]
    ingresses = [d for d in documents if d.get("kind") == "Ingress"]
    assert len(ingresses) == 1, f"{INGRESS} declares {len(ingresses)} Ingresses, not 1"
    rules = ingresses[0]["spec"]["rules"]
    assert len(rules) == 1, f"{INGRESS} declares {len(rules)} rules, not 1"
    return str(rules[0]["host"])


def test_the_server_links_the_gateway_overview_dashboard(built: list[dict[str, Any]]) -> None:
    """Catches the key missing from the built `gateway-config` (the page says "not set" on the
    server), a link to the Grafana root or another dashboard, and a host or uid that drifts from
    the Ingress or the dashboard file while this value keeps the old one."""
    uid = json.loads(DASHBOARD.read_text(encoding="utf-8"))["uid"]
    built_value = gateway_config(built).get(KEY)

    assert f"https://{grafana_host()}/d/{uid}" == EXPECTED, "the Ingress host or the uid moved"
    assert built_value == EXPECTED, f"{GATEWAY_CONFIG_MAP} sets {KEY} to {built_value!r}"


def test_the_server_server_leaves_the_grafana_link_unset(
    server_built: list[dict[str, Any]],
) -> None:
    """Catches the build still carrying the Hetzner link (a dead host on the live console),
    and the key dropped from the patch (the value is then absent, not empty), or set to a
    placeholder or a space instead of exactly the empty string the gateway reads as "no link"."""
    data = gateway_config(server_built)

    assert KEY in data, f"{GATEWAY_CONFIG_MAP} on the server carries {sorted(data)}"
    assert data[KEY] == "", f"{KEY} on the server is {data[KEY]!r}, not the empty string"


def test_the_server_server_runs_no_grafana(server_built: list[dict[str, Any]]) -> None:
    """Catches Grafana added to the build while its link stays empty: the console would
    keep saying "Grafana link not set" next to a Grafana that runs. Fails so the value is filled."""
    workloads = [
        f"{d['kind']}/{d.get('metadata', {}).get('name')}"
        for d in server_built
        if d.get("kind") in ("Deployment", "StatefulSet")
        and d.get("metadata", {}).get("name") == "grafana"
    ]
    assert workloads == [], f"the server now runs {workloads}: fill {KEY} in the server patch"


def test_the_local_grafana_link_points_at_the_gateway_overview() -> None:
    """Catches `backend/.env.demo` with the link empty or missing (the page says "not set"
    next to a running Grafana), repeated, on a port that is not the compose Grafana's (3000 and
    3001 answer other apps), or on a dashboard uid that is not the gateway overview's."""
    if not ENV_DEMO.is_file():
        pytest.skip("backend/.env.demo is not present (CI)")
    lines = ENV_DEMO.read_text(encoding="utf-8").splitlines()
    values = [line.partition("=")[2] for line in lines if line.startswith(f"{KEY}=")]
    uid = json.loads(DASHBOARD.read_text(encoding="utf-8"))["uid"]
    grafana = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["grafana"]
    port = str(grafana["ports"][0]).split(":")[0]

    assert values == [f"http://localhost:{port}/d/{uid}"], f"{KEY} in .env.demo is {values!r}"
    assert values == ["http://localhost:3002/d/gateway-overview"]
