"""An earlier change: the Overview's "Health details in Grafana" link.

On the server the link is the gateway's own Grafana dashboard, built from the Grafana Ingress host
and the dashboard `uid`; locally it is left empty so the page says "Grafana link not set".
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


def test_the_local_grafana_link_is_left_unset() -> None:
    """Catches `backend/.env.demo` still pointing the link at a local port (the help desk answers
    there, so the page links to the wrong app) instead of leaving it empty."""
    if not ENV_DEMO.is_file():
        pytest.skip("backend/.env.demo is not present (CI)")
    lines = ENV_DEMO.read_text(encoding="utf-8").splitlines()
    values = [line.partition("=")[2] for line in lines if line.startswith(f"{KEY}=")]
    assert values == [""], f"{KEY} in .env.demo is {values!r}"
