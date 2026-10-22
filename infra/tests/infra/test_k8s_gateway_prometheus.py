r"""An earlier change: the gateway reads Prometheus in-cluster. One ConfigMap key, one egress rule inside
`gateway-egress`, and its ingress partner `prometheus-ingress-from-gateway` in `monitoring`
(a project rule: the two land together or the allow is a deny nobody reports).

The stream comes from the session-scoped `built` fixture in `infra/tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, shared with every other
`infra` module. Nothing here builds again. The one case that reads raw files does so because
its criterion is about what the two FILES say, not what the build emits.

The NetworkPolicy resolvers are IMPORTED from `test_k8s_netpol.py` and the `apps` object lookup
from `test_k8s_apps.py`, spelled as the package `__init__` files make them (`infra.test_k8s_x`,
a project rule). No `test_*` name is imported, so pytest collects nothing twice.

Ports are asserted RESOLVED against the Prometheus pod's own container port, never compared as a
name alone: a named port that resolves to nothing is a silent deny.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.core.config import Settings
from infra.test_k8s_apps import sole_object
from infra.test_k8s_netpol import (
    ANY_POD,
    APPS,
    GATEWAY,
    MONITORING,
    NAME_LABEL,
    NAMESPACE_NAME_LABEL,
    container_ports,
    declared_ports,
    matched_workloads,
    name_of,
    namespace_of,
    numeric_ports,
    peer_namespace,
    peer_targets,
    peers_of,
    pod_selector,
    pod_templates,
    policies,
    policy,
    policy_types,
    ports_against,
    resolved_ports,
    rules_of,
    rules_with_ip_block,
    spec_of,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GATEWAY_MANIFEST = REPO_ROOT / "infra" / "k8s" / "apps" / "gateway.yaml"
GRAFANA_MANIFEST = REPO_ROOT / "infra" / "k8s" / "monitoring" / "grafana.yaml"

PROMETHEUS = "prometheus"
PROMETHEUS_PORT = 9090
PROMETHEUS_PORT_NAME = "metrics"

GATEWAY_CONFIG_MAP = "gateway-config"
PROMETHEUS_URL_KEY = "OPSCONSOLE_PROMETHEUS_URL"
#: Written out literally, never derived from Grafana's copy: the drift the
#: equality case catches is visible only when both files and this test SAY the value.
PROMETHEUS_URL = "http://prometheus.monitoring.svc.cluster.local:9090"

SETTINGS_PREFIX = "OPSCONSOLE_"
SETTINGS_FIELD = "prometheus_url"

GRAFANA_DATASOURCES_CONFIG_MAP = "grafana-datasources"

GATEWAY_PROMETHEUS_EGRESS_PEER: dict[str, Any] = {
    "namespaceSelector": {"matchLabels": {NAMESPACE_NAME_LABEL: MONITORING}},
    "podSelector": {"matchLabels": {NAME_LABEL: PROMETHEUS}},
}

PROMETHEUS_INGRESS_POLICY = "prometheus-ingress-from-gateway"
PROMETHEUS_INGRESS_PEER: dict[str, Any] = {
    "namespaceSelector": {"matchLabels": {NAMESPACE_NAME_LABEL: APPS}},
    "podSelector": {"matchLabels": {NAME_LABEL: GATEWAY}},
}

PUBLIC_PORTS: frozenset[tuple[str, int]] = frozenset({("TCP", 443)})


def raw_config_map_data(path: Path, name: str) -> dict[str, str]:
    """The `data` of the ONE ConfigMap named `name` in a raw manifest file. Every document is
    scanned and exactly one match is required, so a second ConfigMap of the
    same name later in the file cannot be silently ignored."""
    assert path.is_file(), f"{path} does not exist"
    matches = [
        document
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if isinstance(document, dict)
        and document.get("kind") == "ConfigMap"
        and name_of(document) == name
    ]
    assert len(matches) == 1, f"{path} declares {len(matches)} ConfigMaps named {name}, not 1"
    data = matches[0].get("data", {})
    assert isinstance(data, dict), f"{path}: {name} declares no data mapping"
    return {str(key): str(value) for key, value in data.items()}


def gateway_config(built: list[dict[str, Any]]) -> dict[str, str]:
    """The built `apps/gateway-config` data, values as the strings a ConfigMap holds."""
    raw = sole_object(built, "ConfigMap", GATEWAY_CONFIG_MAP).get("data", {})
    assert isinstance(raw, dict), f"{GATEWAY_CONFIG_MAP} carries no data mapping"
    return {str(key): str(value) for key, value in raw.items()}


def prometheus_template(built: list[dict[str, Any]]) -> dict[str, Any]:
    """The Prometheus pod template in `monitoring`, the one a named port resolves against."""
    templates = pod_templates(built, MONITORING)
    assert PROMETHEUS in templates, f"{MONITORING} has no {PROMETHEUS} pod: {sorted(templates)}"
    return templates[PROMETHEUS]


def reaches_prometheus(rule: dict[str, Any], built: list[dict[str, Any]], label: str) -> bool:
    """Whether one egress rule of an `apps` policy lets its pods open TCP 9090 on Prometheus.

    A rule with no `to` allows every destination, so it counts. `ipBlock` peers are skipped: the
    only one in `apps` excludes every private range, which is where pod addresses live. A peer
    scoped to `monitoring` counts when it matches Prometheus or every pod there.
    """
    peers = peers_of(rule, "to")
    if peers:
        targets: set[tuple[str, str]] = set()
        for peer in peers:
            if "ipBlock" in peer or peer_namespace(peer, APPS) != MONITORING:
                continue
            targets |= peer_targets(built, peer, APPS)
        if not targets & {(MONITORING, PROMETHEUS), (MONITORING, ANY_POD)}:
            return False
    ports = ports_against(rule, container_ports(prometheus_template(built)), label)
    return ("TCP", PROMETHEUS_PORT) in ports


def test_the_gateway_config_names_prometheus_by_its_fully_qualified_address(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 1, by exact value. Catches the key missing (the console read routes answer
    `prometheus: error` forever), a bare `http://prometheus:9090` that resolves nowhere from
    `apps`, the short `prometheus.monitoring:9090` form, and a wrong port or scheme."""
    data = gateway_config(built)
    assert PROMETHEUS_URL_KEY in data, f"{GATEWAY_CONFIG_MAP} carries {sorted(data)}"
    assert data[PROMETHEUS_URL_KEY] == PROMETHEUS_URL, (
        f"{PROMETHEUS_URL_KEY} is {data[PROMETHEUS_URL_KEY]!r}, not {PROMETHEUS_URL!r}"
    )


def test_the_gateway_prometheus_url_equals_the_grafana_datasource_url() -> None:
    """Criterion 2 and a project rule, read off the two raw FILES. Catches the gateway's value
    injected by a kustomize patch or replacement instead of written in `apps/gateway.yaml` (the
    raw file then lacks it), and catches drift between the two readers of one Prometheus: Grafana's
    datasource moved to another address or port while the gateway keeps the old one, which the
    exact-value case above cannot see because it never reads Grafana."""
    gateway_value = raw_config_map_data(GATEWAY_MANIFEST, GATEWAY_CONFIG_MAP).get(
        PROMETHEUS_URL_KEY
    )
    assert gateway_value == PROMETHEUS_URL, (
        f"{GATEWAY_MANIFEST} writes {PROMETHEUS_URL_KEY} as {gateway_value!r}"
    )

    grafana_data = raw_config_map_data(GRAFANA_MANIFEST, GRAFANA_DATASOURCES_CONFIG_MAP)
    urls: list[str] = []
    for key, text in grafana_data.items():
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict), f"{GRAFANA_DATASOURCES_CONFIG_MAP}/{key} is not a mapping"
        declared = parsed.get("datasources", [])
        assert isinstance(declared, list), f"{key} declares datasources {declared!r}"
        urls.extend(
            str(entry.get("url"))
            for entry in declared
            if isinstance(entry, dict) and entry.get("type") == PROMETHEUS
        )
    assert len(urls) == 1, f"{GRAFANA_MANIFEST} declares {len(urls)} prometheus datasources"
    assert urls[0] == gateway_value, (
        f"Grafana reads Prometheus at {urls[0]!r}, the gateway at {gateway_value!r}"
    )


def test_the_config_key_is_the_setting_the_gateway_reads(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 3. Catches the ConfigMap key and the `Settings` field disagreeing - the field
    renamed, or the settings prefix changed - so the pod starts with the variable set and the
    gateway reads `None`, with no error anywhere."""
    assert PROMETHEUS_URL_KEY in gateway_config(built), f"{GATEWAY_CONFIG_MAP} lacks the key"
    assert Settings.model_config.get("env_prefix") == SETTINGS_PREFIX
    assert PROMETHEUS_URL_KEY.startswith(SETTINGS_PREFIX)
    field = PROMETHEUS_URL_KEY.removeprefix(SETTINGS_PREFIX).lower()
    assert field == SETTINGS_FIELD
    assert field in Settings.model_fields, f"Settings has no field {field}"


def test_the_gateway_egress_reaches_prometheus_on_exactly_9090(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 4. Catches the rule missing (the gateway's PromQL denied by `default-deny`),
    the peer written as two separate peers - a `namespaceSelector` alone is an OR that opens
    every pod in `monitoring` - a second port beside 9090, and a second rule into `monitoring`,
    since exactly one rule may name that namespace and its peers and ports are compared by
    equality."""
    document = policy(built, APPS, "gateway-egress")
    label = f"{APPS}/gateway-egress"
    into_monitoring = [
        rule
        for rule in rules_of(document, "egress")
        if any(
            "ipBlock" not in peer and peer_namespace(peer, APPS) == MONITORING
            for peer in peers_of(rule, "to")
        )
    ]
    assert len(into_monitoring) == 1, (
        f"{label} has {len(into_monitoring)} rules into {MONITORING}, not 1"
    )
    rule = into_monitoring[0]
    assert peers_of(rule, "to") == [GATEWAY_PROMETHEUS_EGRESS_PEER], (
        f"{label} reaches {MONITORING} through {peers_of(rule, 'to')}"
    )
    assert numeric_ports(rule, label) == {("TCP", PROMETHEUS_PORT)}, (
        f"{label} opens {sorted(numeric_ports(rule, label))} to {PROMETHEUS}"
    )
    targets = peer_targets(built, GATEWAY_PROMETHEUS_EGRESS_PEER, APPS)
    assert targets == {(MONITORING, PROMETHEUS)}, f"the peer resolves to {sorted(targets)}"


def test_the_gateway_egress_ip_block_rule_still_carries_only_443(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 5. Catches 9090 appended to the public `ipBlock` rule's ports instead of a rule
    of its own, which would open 9090 to every public address rather than to Prometheus."""
    document = policy(built, APPS, "gateway-egress")
    rules = rules_with_ip_block(document)
    assert len(rules) == 1, f"{len(rules)} gateway-egress rules carry an ipBlock, not 1"
    ports = numeric_ports(rules[0], f"{APPS}/gateway-egress")
    assert ports == PUBLIC_PORTS, f"the ipBlock rule opens {sorted(ports)}"


def test_prometheus_admits_only_the_gateway_from_apps_on_9090(
    built: list[dict[str, Any]],
) -> None:
    """Criteria 6 and 7. Catches the ingress partner missing (the gateway's egress allowed and
    Prometheus's side denied), the policy selecting every `monitoring` pod, an `Egress` type that
    would deny Prometheus's scrapes, the peer split into an OR that admits every `apps` pod, and a
    port name that does not resolve to 9090 on the Prometheus pod - a silent deny."""
    document = policy(built, MONITORING, PROMETHEUS_INGRESS_POLICY)
    label = f"{MONITORING}/{PROMETHEUS_INGRESS_POLICY}"
    assert pod_selector(document) == {"matchLabels": {NAME_LABEL: PROMETHEUS}}, (
        f"{label} selects {pod_selector(document)}"
    )
    assert policy_types(document) == ["Ingress"], f"{label} declares {policy_types(document)}"
    assert "egress" not in spec_of(document), f"{label} declares an egress block"

    rules = rules_of(document, "ingress")
    assert len(rules) == 1, f"{label} has {len(rules)} ingress rules, not 1"
    rule = rules[0]
    assert peers_of(rule, "from") == [PROMETHEUS_INGRESS_PEER], (
        f"{label} admits {peers_of(rule, 'from')}"
    )
    assert peer_targets(built, PROMETHEUS_INGRESS_PEER, MONITORING) == {(APPS, GATEWAY)}
    assert declared_ports(rule, label) == [("TCP", PROMETHEUS_PORT_NAME)], (
        f"{label} declares {declared_ports(rule, label)}"
    )
    assert resolved_ports(built, document, rule) == {PROMETHEUS: {("TCP", PROMETHEUS_PORT)}}, (
        f"{label} resolves to {resolved_ports(built, document, rule)}"
    )


def test_no_other_apps_pod_may_reach_prometheus(built: list[dict[str, Any]]) -> None:
    """Criterion 8, the unhappy path, scanned over EVERY `apps` policy rather than read off
    `gateway-egress` by name. Catches the Prometheus rule added to `helpdesk-egress`,
    `console-egress` or a namespace-wide policy such as `allow-dns`, which lets a pod other than
    the gateway query a store holding everything the cluster has exposed; equality also makes
    the gateway's own route missing red."""
    reaching: set[str] = set()
    for document in policies(built):
        if namespace_of(document) != APPS:
            continue
        label = f"{APPS}/{name_of(document)}"
        if any(reaches_prometheus(rule, built, label) for rule in rules_of(document, "egress")):
            reaching |= matched_workloads(built, pod_selector(document), APPS)
    assert reaching == {GATEWAY}, f"apps pods whose egress reaches {PROMETHEUS}: {sorted(reaching)}"


def test_the_monitoring_scrape_ingress_carries_no_apps_peer(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 10. Catches the gateway admitted by widening `metrics-ingress-from-monitoring`
    instead of adding its own policy - that policy selects every `monitoring` pod, so the gateway
    (or every `apps` pod) would reach both exporters and Alertmanager as well as Prometheus."""
    document = policy(built, MONITORING, "metrics-ingress-from-monitoring")
    namespaces = {
        peer_namespace(peer, MONITORING)
        for rule in rules_of(document, "ingress")
        for peer in peers_of(rule, "from")
    }
    assert APPS not in namespaces, f"metrics-ingress-from-monitoring admits from {namespaces}"
