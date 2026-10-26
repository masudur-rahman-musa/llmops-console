"""The local stack shows numbers: Prometheus in compose, the gateway wired to it, `make numbers`.

An earlier change. Every case reads files as text or parsed YAML and starts nothing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from infra.test_hetzner_module import load_yaml, read

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
PROMETHEUS_CONFIG = REPO_ROOT / "infra" / "local" / "prometheus.yml"
NUMBERS_OVERLAY = REPO_ROOT / "infra" / "local" / "numbers.compose.yml"
MAKEFILE = REPO_ROOT / "Makefile"
README = REPO_ROOT / "README.md"

PROMETHEUS_IMAGE = "prom/prometheus:v3.1.0"


def service(name: str) -> dict[str, Any]:
    found = load_yaml(COMPOSE_FILE).get("services", {})
    assert name in found, f"docker-compose.yml has no {name} service"
    body = found[name]
    assert isinstance(body, dict)
    return body


def numbers_recipe() -> str:
    """The text of the `numbers:` target: its line and the tab-indented lines after it."""
    lines = read(MAKEFILE).splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(r"numbers:", line)), None)
    assert start is not None, "Makefile has no numbers target"
    body = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("\t") or line.strip() == "":
            body.append(line)
        else:
            break
    return "\n".join(body)


def test_prometheus_runs_by_default_with_the_k8s_pin() -> None:
    """Catches a Prometheus put behind a profile, on another version than the cluster's,
    on another port, or without the scrape config mounted."""
    body = service("prometheus")
    assert "profiles" not in body
    assert body.get("image") == PROMETHEUS_IMAGE
    assert body.get("ports") == ["9090:9090"]
    mounts = [str(v) for v in body.get("volumes", [])]
    assert any(m.startswith("./infra/local/prometheus.yml:") for m in mounts), mounts


def test_prometheus_scrapes_the_gateway_every_five_seconds() -> None:
    """Catches a scrape config with the wrong job, target, path or interval, so the local
    numbers would be missing or take too long to land."""
    jobs = load_yaml(PROMETHEUS_CONFIG)["scrape_configs"]
    assert len(jobs) == 4
    job = jobs[0]
    assert job["job_name"] == "gateway"
    assert job["metrics_path"] == "/metrics"
    assert job["static_configs"] == [{"targets": ["gateway:8080"]}]
    interval = (
        job.get("scrape_interval") or load_yaml(PROMETHEUS_CONFIG)["global"]["scrape_interval"]
    )
    assert interval == "5s"


def test_local_prometheus_scrapes_the_help_desk() -> None:
    """Catches a local Prometheus that never dials the help desk (guardrail blocks stay n/a),
    a job on the wrong host or port (the agent API is 8000, the MCP server 8081), a target
    that is the compose-network name instead of the host, and a compose Prometheus that cannot
    resolve `host.docker.internal` (Linux has no default for it)."""
    jobs = {job["job_name"]: job for job in load_yaml(PROMETHEUS_CONFIG)["scrape_configs"]}
    assert jobs["helpdesk-agent-api"]["static_configs"] == [
        {"targets": ["host.docker.internal:8000"]}
    ]
    assert jobs["helpdesk-mcp-server"]["static_configs"] == [
        {"targets": ["host.docker.internal:8081"]}
    ]
    for name in ("helpdesk-agent-api", "helpdesk-mcp-server"):
        assert jobs[name].get("metrics_path", "/metrics") == "/metrics", name

    extra_hosts = service("prometheus").get("extra_hosts")
    assert "host.docker.internal:host-gateway" in (extra_hosts or [])


def test_the_gateway_migrates_and_reads_prometheus_in_compose() -> None:
    """Catches a compose gateway that does not migrate at start or cannot find Prometheus,
    which leaves tabs failing on a missing column or showing the source as unconfigured."""
    env = service("gateway")["environment"]
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env)
    assert str(env["OPSCONSOLE_MIGRATE_ON_START"]).lower() == "true"
    assert env["OPSCONSOLE_PROMETHEUS_URL"] == "http://prometheus:9090"


def test_the_numbers_target_runs_eval_traffic_release_rollup_check_in_order() -> None:
    """Catches a `make numbers` that skips a step or runs them in the wrong order (for
    example checking before the traffic or the rollup exists)."""
    recipe = numbers_recipe()
    steps = [
        "$(MAKE) eval",
        "ops-send-traffic",
        "/admin/releases",
        "ops-refresh-rollup",
        "ops-check-console",
    ]
    positions = [recipe.find(step) for step in steps]
    assert -1 not in positions, dict(zip(steps, positions, strict=True))
    assert positions == sorted(positions)


def test_every_path_the_new_files_name_exists() -> None:
    """Catches a path left behind in the Makefile target, a missing `infra/local/` file, or
    a README with no line on `make numbers`."""
    assert PROMETHEUS_CONFIG.is_file()
    assert NUMBERS_OVERLAY.is_file()
    for path in re.findall(r"infra/[\w./-]+", numbers_recipe()):
        assert (REPO_ROOT / path).exists(), path
    assert "make numbers" in read(README)
