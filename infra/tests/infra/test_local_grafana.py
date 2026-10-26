"""The local stack shows the dashboards: Grafana in compose, a datasource, `make grafana-check`.

An earlier change. Every case reads files as text, JSON or parsed YAML and starts nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from infra.test_hetzner_module import load_yaml, read

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
MAKEFILE = REPO_ROOT / "Makefile"
K8S_GRAFANA = REPO_ROOT / "infra" / "k8s" / "monitoring" / "grafana.yaml"
DASHBOARDS = REPO_ROOT / "infra" / "k8s" / "monitoring" / "dashboards"
DATASOURCE_FILE = REPO_ROOT / "infra" / "local" / "grafana-datasource.yml"
PROVIDER_FILE = REPO_ROOT / "infra" / "local" / "grafana-dashboards.yml"
CI_CD_DEPLOY = REPO_ROOT / "dev" / "docs" / "ci-cd-deploy.md"
LOCAL_PROMETHEUS = REPO_ROOT / "infra" / "local" / "prometheus.yml"
K8S_PROMETHEUS = REPO_ROOT / "infra" / "k8s" / "monitoring" / "prometheus.yaml"
RULES_SOURCE = "./infra/k8s/monitoring/alerts/rules.yml"
RULES_TARGET = "/etc/prometheus/rules/rules.yml"

GRAFANA_IMAGE = "grafana/grafana:11.4.0"
DATASOURCE_UID = "prometheus"
PATH_PATTERN = r"infra/[\w./-]+"


def service(name: str) -> dict[str, Any]:
    found = load_yaml(COMPOSE_FILE).get("services", {})
    assert name in found, f"docker-compose.yml has no {name} service"
    body = found[name]
    assert isinstance(body, dict)
    return body


def environment(body: dict[str, Any]) -> dict[str, str]:
    env = body.get("environment", {})
    if isinstance(env, list):
        return dict(item.split("=", 1) for item in env)
    return {str(k): str(v) for k, v in env.items()}


def recipe(target: str) -> str:
    """The text of one Makefile target: its line and the tab-indented lines after it."""
    lines = read(MAKEFILE).splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(rf"{target}:", line)), None)
    assert start is not None, f"Makefile has no {target} target"
    body = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("\t") or line.strip() == "":
            body.append(line)
        else:
            break
    return "\n".join(body)


def host_port(mapping: object) -> int:
    parts = str(mapping).split(":")
    return int(parts[-2]) if len(parts) >= 2 else int(parts[0])


def datasource_uids(node: object, title: str = "") -> list[tuple[str, str]]:
    """Every (panel title, datasource uid) found anywhere under a dashboard node."""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        title = str(node.get("title", title))
        ds = node.get("datasource")
        if isinstance(ds, dict) and "uid" in ds:
            found.append((title, str(ds["uid"])))
        for value in node.values():
            found.extend(datasource_uids(value, title))
    elif isinstance(node, list):
        for item in node:
            found.extend(datasource_uids(item, title))
    return found


def bullet(text: str, start_marker: str) -> str:
    """One `- ` bullet of the deploy handbook: from its first line to the next bullet."""
    lines = text.splitlines()
    first = next((i for i, line in enumerate(lines) if line.startswith(start_marker)), None)
    assert first is not None, f"ci-cd-deploy.md has no line starting {start_marker!r}"
    body = [lines[first]]
    for line in lines[first + 1 :]:
        if line.startswith("- ") or line.startswith("#") or line.strip() == "":
            break
        body.append(line)
    return "\n".join(body)


def test_grafana_runs_by_default_with_the_k8s_pin() -> None:
    """Catches a Grafana put behind a profile, on another version than the cluster's, or on
    a host port other than 3002, so the console link and `make grafana-check` miss it."""
    body = service("grafana")
    k8s_images = re.findall(r"image:\s*(grafana/grafana:\S+)", read(K8S_GRAFANA))
    assert k8s_images == [GRAFANA_IMAGE], "the cluster pin moved: change both"
    assert "profiles" not in body
    assert body.get("image") == k8s_images[0]
    assert body.get("ports") == ["127.0.0.1:3002:3000"]


def test_grafana_port_clashes_with_nothing() -> None:
    """Catches Grafana published on 3000 (help desk web) or 3001 (this console), or a second
    service taking 3002, which would fail one of the two at `up`."""
    services = load_yaml(COMPOSE_FILE)["services"]
    grafana_ports = [host_port(p) for p in services["grafana"]["ports"]]
    assert grafana_ports == [3002]
    taken = [
        host_port(p)
        for name, body in services.items()
        if name != "grafana"
        for p in body.get("ports", [])
    ]
    assert 3002 not in taken
    assert {3000, 3001}.isdisjoint(grafana_ports)


def test_the_local_datasource_reads_the_local_prometheus() -> None:
    """Catches a datasource on the wrong uid (every panel goes blank), a URL on localhost
    instead of the compose name, browser access, or no default datasource."""
    sources = load_yaml(DATASOURCE_FILE)["datasources"]
    assert len(sources) == 1
    source = sources[0]
    assert source["type"] == "prometheus"
    assert source["uid"] == DATASOURCE_UID
    assert source["url"] == "http://prometheus:9090"
    assert source["access"] == "proxy"
    assert source["isDefault"] is True


def test_every_dashboard_panel_finds_the_local_datasource() -> None:
    """Catches a dashboard panel naming a datasource uid the local Grafana does not have, so
    that panel shows 'datasource not found' while the others work."""
    uid = load_yaml(DATASOURCE_FILE)["datasources"][0]["uid"]
    files = sorted(DASHBOARDS.glob("*.json"))
    assert files, "no dashboards found"
    wrong: list[str] = []
    seen = 0
    for path in files:
        for title, found in datasource_uids(json.loads(read(path))):
            seen += 1
            if found != uid:
                wrong.append(f"{path.name}: panel {title!r} names {found!r}, not {uid!r}")
    assert seen > 0, "no panel names a datasource"
    assert wrong == []


def test_the_five_dashboards_are_mounted_where_the_provider_looks() -> None:
    """Catches a dashboards mount that is writable or missing, a provider path that differs
    from the mount target (Grafana loads nothing), a sixth or lost dashboard file, and a
    provisioning file mounted writable or outside /etc/grafana/provisioning/."""
    mounts = [str(v) for v in service("grafana").get("volumes", [])]
    dash = [m for m in mounts if m.startswith("./infra/k8s/monitoring/dashboards:")]
    assert len(dash) == 1, mounts
    _, target, mode = dash[0].split(":")
    assert mode == "ro"

    providers = load_yaml(PROVIDER_FILE)["providers"]
    assert [p["options"]["path"] for p in providers] == [target]
    assert len(list(DASHBOARDS.glob("*.json"))) == 5

    for name in ("grafana-datasource.yml", "grafana-dashboards.yml"):
        mine = [m for m in mounts if m.startswith(f"./infra/local/{name}:")]
        assert len(mine) == 1, (name, mounts)
        _, target_path, mode = mine[0].split(":")
        assert target_path.startswith("/etc/grafana/provisioning/"), target_path
        assert mode == "ro"


def test_local_grafana_lets_a_viewer_in_and_no_more() -> None:
    """Catches anonymous access off (the console link asks for a login) and anonymous access
    above Viewer (`Editor` or `Admin`: anyone on the laptop network could change Grafana)."""
    env = environment(service("grafana"))
    assert env.get("GF_AUTH_ANONYMOUS_ENABLED") == "true"
    assert env.get("GF_AUTH_ANONYMOUS_ORG_ROLE") == "Viewer"


def test_the_grafana_check_target_asks_health_and_the_dashboard() -> None:
    """Catches a check that skips /api/health or the dashboard fetch, uses the wrong port or
    uid, or is wired into `make verify` where CI has no stack running."""
    body = recipe("grafana-check")
    assert "curl -fsS http://localhost:3002/api/health" in body
    assert "curl -fsS http://localhost:3002/api/dashboards/uid/gateway-overview" in body
    verify_line = recipe("verify").splitlines()[0]
    assert "grafana-check" not in verify_line.partition("##")[0].split()


def test_every_path_the_changed_files_name_exists() -> None:
    """Catches a path left behind in the compose service, the check recipe or the deploy
    handbook's two edited lines (local link, `infra/local/` files)."""
    texts = {
        "compose grafana service": json.dumps(service("grafana")),
        "grafana-check recipe": recipe("grafana-check"),
    }
    handbook = read(CI_CD_DEPLOY)
    texts["handbook link line"] = bullet(handbook, "- The Overview tab's")
    texts["handbook infra/local line"] = bullet(handbook, "- `infra/local/`")
    for label, text in texts.items():
        for path in re.findall(PATH_PATTERN, text):
            assert (REPO_ROOT / path.rstrip(".,")).exists(), f"{label}: {path}"
    for name in ("grafana-datasource.yml", "grafana-dashboards.yml"):
        assert (REPO_ROOT / "infra" / "local" / name).is_file(), name


def mount_targets(body: dict[str, Any]) -> list[str]:
    return [str(v).split(":")[1] for v in body.get("volumes", [])]


def only_on_this_laptop(ports: list[object]) -> bool:
    return bool(ports) and all(str(p).startswith("127.0.0.1:") for p in ports)


def test_local_prometheus_loads_the_alert_rules() -> None:
    """Catches rules not mounted, mounted writable, or not named in `rule_files`, so the
    Grafana alert list stays empty."""
    mounts = [str(v) for v in service("prometheus").get("volumes", [])]
    assert f"{RULES_SOURCE}:{RULES_TARGET}:ro" in mounts, mounts
    assert load_yaml(LOCAL_PROMETHEUS)["rule_files"] == [RULES_TARGET]


def test_the_mounted_rules_file_holds_every_alert() -> None:
    """Catches the mount pointing at a file that lost alerts or has a rule without `expr`."""
    groups = load_yaml(REPO_ROOT / RULES_SOURCE)["groups"]
    rules = [r for g in groups for r in g["rules"] if "alert" in r]
    assert len(rules) == 12
    assert all(r.get("expr") for r in rules)


def test_local_prometheus_names_no_rule_file_it_cannot_see() -> None:
    """Catches a rule path with no mount, which makes Prometheus refuse to start."""
    targets = mount_targets(service("prometheus"))
    missing = [f for f in load_yaml(LOCAL_PROMETHEUS)["rule_files"] if f not in targets]
    assert missing == []


def test_local_prometheus_scrapes_the_help_desk_scheduler() -> None:
    """Catches a missing scheduler job, a wrong port or path, or a name that differs from
    the cluster job, so the Chats waiting chart stays empty."""
    jobs = {j["job_name"]: j for j in load_yaml(LOCAL_PROMETHEUS)["scrape_configs"]}
    job = jobs["helpdesk-scheduler"]
    assert job["static_configs"] == [{"targets": ["host.docker.internal:8082"]}]
    assert job["metrics_path"] == "/metrics"
    assert "job_name: helpdesk-scheduler" in read(K8S_PROMETHEUS)


def test_local_grafana_listens_only_on_this_laptop() -> None:
    """Catches Grafana published on every network card, where anyone on the Wi-Fi sees the
    dashboards with no login."""
    assert service("grafana")["ports"] == ["127.0.0.1:3002:3000"]
    assert not only_on_this_laptop(["3002:3000"])
    assert not only_on_this_laptop(["0.0.0.0:3002:3000"])
    assert not only_on_this_laptop(["127.0.0.1:3002:3000", "3003:3000"])
