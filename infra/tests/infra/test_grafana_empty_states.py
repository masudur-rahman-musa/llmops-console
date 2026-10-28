"""Grafana empty states and the alerting file.

Every case reads files as text, JSON or YAML, or reads the built `kubectl kustomize` stream the
session-scoped `built` fixture in `conftest.py` provides. Nothing starts a service, and no case
carries the `db` marker.

The two things a dashboard can say when it has no data are asserted by VALUE, never by a range:
the grey is `#808080`, the two empty texts are the exact strings below.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MONITORING_DIR = REPO_ROOT / "infra" / "k8s" / "monitoring"
DASHBOARDS = MONITORING_DIR / "dashboards"
ALERTING_FILE = MONITORING_DIR / "grafana-alerting.yml"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

DASHBOARD_FILES = (
    "gateway-overview.json",
    "cost.json",
    "platform.json",
    "evals.json",
    "helpdesk-agent.json",
)

GREY = "#808080"
NO_DATA_YET = "No data yet"
CLUSTER_ONLY = "Cluster only, empty on localhost"
HELPDESK_POD_TITLES = ("Pod restarts, Helpdesk workloads", "Pod phase, Helpdesk workloads")
ALERTING_CONFIGMAP = "grafana-alerting"
ALERTING_KEY_SOURCE = "./infra/k8s/monitoring/grafana-alerting.yml"
COMPOSE_ALERTING_DIR = "/etc/grafana/provisioning/alerting/"
POD_ALERTING_DIR = "/etc/grafana/provisioning/alerting"
CONTACT_POINT = "alerts-sent-by-server-alertmanager"
CONTACT_POINT_TYPE = "prometheus-alertmanager"
CONTACT_POINT_URL = "http://alertmanager.monitoring.svc:9093"
OLD_CONTACT_POINT = "grafana-default-email"
PATH_PATTERN = re.compile(r"infra/[\w./-]+")


def read(path: Path) -> str:
    """Read a file the tests need, failing the one case that needs it when it is absent."""
    assert path.is_file(), f"{path} does not exist"
    return path.read_text(encoding="utf-8")


def dashboard(filename: str) -> dict[str, Any]:
    """One dashboard file parsed as JSON."""
    parsed = json.loads(read(DASHBOARDS / filename))
    assert isinstance(parsed, dict), filename
    return parsed


def walk(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every panel, rows' nested panels included."""
    found: list[dict[str, Any]] = []
    for panel in panels:
        found.append(panel)
        found.extend(walk(panel.get("panels", [])))
    return found


def panels_of(filename: str) -> list[dict[str, Any]]:
    """Every panel of one dashboard file."""
    return walk(dashboard(filename).get("panels", []))


def defaults_of(panel: dict[str, Any]) -> dict[str, Any]:
    """`fieldConfig.defaults`, empty when the panel has none."""
    field_config = panel.get("fieldConfig", {})
    defaults = field_config.get("defaults", {}) if isinstance(field_config, dict) else {}
    return defaults if isinstance(defaults, dict) else {}


def no_value_of(panel: dict[str, Any]) -> object:
    """The panel's `noValue`, `None` when unset."""
    return defaults_of(panel).get("noValue")


def titled(filename: str, title: str) -> dict[str, Any]:
    """The one panel of `filename` with exactly this title."""
    found = [panel for panel in panels_of(filename) if panel.get("title") == title]
    assert len(found) == 1, f"{filename} has {len(found)} panels titled {title!r}"
    return found[0]


def overlaps(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether two panels' grid rectangles share any cell."""
    a, b = first["gridPos"], second["gridPos"]
    return bool(
        a["x"] < b["x"] + b["w"]
        and b["x"] < a["x"] + a["w"]
        and a["y"] < b["y"] + b["h"]
        and b["y"] < a["y"] + a["h"]
    )


def note_at_top(filename: str) -> dict[str, Any]:
    """The text panel at y 0, asserting it is full width, 2 high and overlapped by nothing."""
    panels = panels_of(filename)
    at_top = [
        panel for panel in panels if panel["gridPos"]["y"] == 0 and panel["gridPos"]["x"] == 0
    ]
    notes = [panel for panel in at_top if panel.get("type") == "text"]
    assert len(notes) == 1, f"{filename} has {len(notes)} text panels at y 0, x 0"
    note = notes[0]
    assert note["gridPos"]["w"] == 24, note["gridPos"]
    assert note["gridPos"]["h"] == 2, note["gridPos"]
    crowding = [
        str(panel.get("title")) for panel in panels if panel is not note and overlaps(note, panel)
    ]
    assert crowding == [], f"{filename}: panels overlapping the note: {crowding}"
    return note


def note_text(note: dict[str, Any]) -> str:
    """The markdown a text panel shows."""
    options = note.get("options", {})
    assert isinstance(options, dict)
    return str(options.get("content", ""))


def string_values(node: object) -> list[str]:
    """Every string VALUE in a parsed document, keys excluded."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [text for value in node.values() for text in string_values(value)]
    if isinstance(node, list):
        return [text for value in node for text in string_values(value)]
    return []


# --- missing data is grey -----------------------------------------------------


def test_every_stat_panel_shows_missing_data_grey() -> None:
    """Catches a stat panel whose base threshold step (the one a missing value falls on) is
    still green or red: No data would be drawn as a big green tile and read as all good. The
    first coloured step must also sit at value 0, so a real 0 stays coloured and a grey base
    that simply replaced the old colour (dropping the step at 0) fails too."""
    stats: list[str] = []
    problems: list[str] = []
    for filename in DASHBOARD_FILES:
        for panel in panels_of(filename):
            if panel.get("type") != "stat":
                continue
            stats.append(f"{filename}:{panel.get('title')}")
            thresholds = defaults_of(panel).get("thresholds", {})
            steps = thresholds.get("steps", []) if isinstance(thresholds, dict) else []
            if len(steps) < 2:
                problems.append(f"{filename}:{panel.get('title')} has steps {steps}")
                continue
            if steps[0].get("color") != GREY or steps[0].get("value") is not None:
                problems.append(f"{filename}:{panel.get('title')} base step is {steps[0]}")
            if steps[1].get("value") != 0:
                problems.append(f"{filename}:{panel.get('title')} first coloured is {steps[1]}")

    assert len(stats) == 4, f"the five dashboards hold these stat panels: {stats}"
    assert problems == []


def test_the_gateway_stats_say_no_data_yet() -> None:
    """Catches a stat that still prints Grafana's default "No data" (or an empty string)
    on Circuit state or on one of the two SLO tiles. The three are found by title
    prefix, and the count is asserted so a renamed panel cannot make the loop empty."""
    stats = [panel for panel in panels_of("gateway-overview.json") if panel.get("type") == "stat"]
    prefixes = ("Circuit state", "Availability", "Latency")
    chosen = [panel for panel in stats if str(panel.get("title", "")).startswith(prefixes)]

    assert len(chosen) == 3, [panel.get("title") for panel in stats]
    for panel in chosen:
        assert no_value_of(panel) == NO_DATA_YET, panel.get("title")


# --- cluster-only panels ------------------------------------------------------


def test_the_nine_cluster_panels_say_cluster_only() -> None:
    """Catches a Kubernetes-fed chart that goes blank on a laptop with no explanation: the
    seven Platform charts plus the two Helpdesk pod charts must carry the exact empty text and
    open their description with it. Also catches the text spilling onto a panel that DOES fill
    locally (the Gateway's, Cost's, Evals' and the other Helpdesk charts), which would tell a
    viewer a working chart is broken."""
    expected = [panel for panel in panels_of("platform.json") if panel.get("type") != "text"] + [
        titled("helpdesk-agent.json", title) for title in HELPDESK_POD_TITLES
    ]
    assert len(expected) == 9, [panel.get("title") for panel in expected]

    for panel in expected:
        assert no_value_of(panel) == CLUSTER_ONLY, panel.get("title")
        assert str(panel.get("description", "")).startswith(CLUSTER_ONLY), panel.get("title")

    note_at_top("platform.json")
    expected_keys = {("platform.json", str(panel.get("title"))) for panel in expected[:7]} | {
        ("helpdesk-agent.json", title) for title in HELPDESK_POD_TITLES
    }
    for filename in DASHBOARD_FILES:
        for panel in panels_of(filename):
            if (filename, str(panel.get("title"))) in expected_keys:
                continue
            if filename == "platform.json" and panel.get("type") == "text":
                continue
            assert CLUSTER_ONLY not in json.dumps(panel), f"{filename}:{panel.get('title')}"


def test_the_platform_dashboard_opens_with_the_cluster_note() -> None:
    """Catches a Platform dashboard whose first thing is still an empty-looking chart: the note
    must be a `text` panel at y 0 saying the cluster-only sentence, full width and 2 high. The
    overlap scan catches the note being added without moving the charts down."""
    note = note_at_top("platform.json")

    assert note["type"] == "text"
    assert CLUSTER_ONLY in note_text(note)


def test_the_gateway_overview_opens_with_the_alerts_note() -> None:
    """Catches a Gateway overview that gives no hint where alerts go: the note must be a `text`
    panel at y 0 naming the server's Alertmanager and saying nothing is sent locally. The overlap
    scan catches the note added on top of the first charts instead of above them."""
    note = note_at_top("gateway-overview.json")
    text = note_text(note)

    assert note["type"] == "text"
    assert "Alertmanager" in text
    assert "nothing is sent locally" in text.lower()


# --- the alerting file --------------------------------------------------------


def test_the_alerting_file_routes_everything_to_the_server_alertmanager() -> None:
    """Catches an alerting file that keeps `grafana-default-email` as a receiver, names a
    contact point of another type (a webhook or email would look like a delivery path), points
    its URL anywhere but the server's Alertmanager, or leaves the root policy on the old
    receiver. The old name is searched across every string value in the file."""
    document = yaml.safe_load(read(ALERTING_FILE))
    assert isinstance(document, dict)

    points = document["contactPoints"]
    assert len(points) == 1, points
    assert points[0]["name"] == CONTACT_POINT
    receivers = points[0]["receivers"]
    assert len(receivers) == 1, receivers
    assert receivers[0]["type"] == CONTACT_POINT_TYPE
    assert receivers[0]["settings"]["url"] == CONTACT_POINT_URL

    policies = document["policies"]
    assert len(policies) == 1, policies
    assert policies[0]["receiver"] == CONTACT_POINT
    assert OLD_CONTACT_POINT not in string_values(document)


def compose_grafana_volumes() -> list[tuple[str, str]]:
    """`(source, target)` of every volume on the compose `grafana` service."""
    document = yaml.safe_load(read(COMPOSE_FILE))
    volumes = document["services"]["grafana"]["volumes"]
    pairs: list[tuple[str, str]] = []
    for volume in volumes:
        parts = str(volume).split(":")
        pairs.append((parts[0], parts[1]))
    return pairs


def test_both_grafanas_load_the_same_alerting_file(built: list[dict[str, Any]]) -> None:
    """Catches two copies of the alerting file drifting apart: compose must mount the one file
    under the provisioning directory, the built `grafana-alerting` ConfigMap must hold that
    file's exact text (not a pasted copy), and the Grafana pod must mount it where Grafana reads
    alerting provisioning. A cluster mount at some other path would load nothing."""
    targets = [
        target for source, target in compose_grafana_volumes() if source == ALERTING_KEY_SOURCE
    ]
    assert len(targets) == 1, compose_grafana_volumes()
    assert targets[0].startswith(COMPOSE_ALERTING_DIR)

    maps = [
        document
        for document in built
        if document.get("kind") == "ConfigMap"
        and document["metadata"]["name"] == ALERTING_CONFIGMAP
        and document["metadata"].get("namespace") == "monitoring"
    ]
    assert len(maps) == 1, f"the build has {len(maps)} ConfigMap/{ALERTING_CONFIGMAP}"
    data = maps[0]["data"]
    assert list(data.values()) == [read(ALERTING_FILE)]

    template = grafana_template(built)
    volumes = [
        volume
        for volume in template["spec"].get("volumes", [])
        if isinstance(volume.get("configMap"), dict)
        and volume["configMap"].get("name") == ALERTING_CONFIGMAP
    ]
    assert len(volumes) == 1, template["spec"].get("volumes")
    mounts = [
        mount
        for container in template["spec"]["containers"]
        for mount in container.get("volumeMounts", [])
        if mount.get("name") == volumes[0]["name"]
    ]
    assert [mount["mountPath"] for mount in mounts] == [POD_ALERTING_DIR]


def grafana_template(built: list[dict[str, Any]]) -> dict[str, Any]:
    """The Grafana Deployment's pod template."""
    found = [
        document
        for document in built
        if document.get("kind") == "Deployment"
        and document["metadata"]["name"] == "grafana"
        and document["metadata"].get("namespace") == "monitoring"
    ]
    assert len(found) == 1, f"the build has {len(found)} Deployment/grafana"
    template = found[0]["spec"]["template"]
    assert isinstance(template, dict)
    return template


def test_the_cluster_grafana_still_has_no_env(built: list[dict[str, Any]]) -> None:
    """Catches a mount change that smuggles in an `env` block (a setting passed as an
    environment variable instead of a provisioning file): the Grafana container must declare no
    `env` key at all. The alerting mount is asserted present first, so the case is read AFTER
    the new mount exists and not against a pod that never changed."""
    template = grafana_template(built)
    containers = template["spec"]["containers"]
    assert len(containers) == 1, containers
    container = containers[0]

    mounted = [mount["mountPath"] for mount in container.get("volumeMounts", [])]
    assert POD_ALERTING_DIR in mounted
    assert "env" not in container, container.get("env")


# --- every path the changed files name exists ---------------------------------


def named_paths(text: str) -> set[str]:
    """Every `infra/...` path in `text`, trailing sentence punctuation dropped."""
    return {match.rstrip(".,:;)") for match in PATH_PATTERN.findall(text)}


def test_every_path_the_changed_files_name_exists() -> None:
    """Catches a path written into the new file's comments or compose's Grafana block that does
    not exist (a typo, or a file named before it was made). Each source must name at least
    one path, so a source that names none cannot pass by having nothing to check."""
    alerting_comments = "\n".join(
        line for line in read(ALERTING_FILE).splitlines() if line.lstrip().startswith("#")
    )

    compose_lines = read(COMPOSE_FILE).splitlines()
    start = compose_lines.index("  grafana:")
    end = next(
        (
            index
            for index in range(start + 1, len(compose_lines))
            if re.match(r"  \S", compose_lines[index])
        ),
        len(compose_lines),
    )
    compose_block = "\n".join(compose_lines[start:end])

    sources = {
        "alerting file": named_paths(alerting_comments),
        "compose grafana": named_paths(compose_block),
    }
    missing: list[str] = []
    for label, paths in sources.items():
        assert paths, f"{label} names no infra/ path"
        missing.extend(
            f"{label}: {path}" for path in sorted(paths) if not (REPO_ROOT / path).exists()
        )

    assert missing == []
