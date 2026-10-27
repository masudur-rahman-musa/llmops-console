r"""`infra/k8s/monitoring/alerts/rules.yml`, `infra/k8s/monitoring/alertmanager.yaml` and the two
NetworkPolicy widenings an earlier change agrees, as `kubectl kustomize infra/k8s` actually emits them.

The stream comes from the session-scoped `built` fixture in `infra/tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, shared with every other
`infra/tests/infra/test_k8s_*.py` module. Nothing here builds again, and every manifest claim is
read off the BUILT stream rather than off a raw file, so a manifest written and never listed in a
`resources:` block is the same failure as a manifest never written.

Everything is asserted by EQUALITY where the plan names a whole set (the object inventory, the
`ipBlock` scan, the excluded-ranges list, the receiver's config entries): an extra allow, an extra
peer or an extra port is exactly as dangerous as a missing one (decision D of an earlier change, restated
by every sibling module in this directory).

This module does NOT own `monitoring`'s object inventory - `infra/tests/infra/test_k8s_grafana.py`'s
`MONITORING_OBJECTS` does, from an earlier change on (its own docstring says so: "23e, 23c and 23d each
edit one expected set instead of two that drift apart", moved from
`infra/tests/infra/test_k8s_monitoring.py` when 23b added nine objects). This plan's three -
`Deployment/alertmanager`, `Service/alertmanager`, `ConfigMap/alertmanager-config` - are therefore
added to THAT constant directly rather than asserted a second time here, which is also where the
CRD/`monitoring.coreos.com` prohibition already lives (`test_k8s_monitoring.py`'s
`test_the_monitoring_namespace_emits_exactly_the_objects_the_two_plans_name`, scoped over the
WHOLE build rather than one namespace). An earlier draft of this module wrote a third copy of
both checks here, which is exactly the drift the two files above already changed their own
layout to avoid - grepping `MONITORING_OBJECTS` at drafting time (the plan's own "the prior
twenty" prose undercounts it by two, which is a project rule's trap: twenty-two, not twenty)
should have found the existing convention before adding a fourth reader of the same set. The
`22` is reproduced by `grep -c` over the live file:

    python3 -c "import re; t=open('infra/tests/infra/test_k8s_grafana.py').read(); \
    m=re.search(r'MONITORING_OBJECTS.*?frozenset\(\s*\{(.*?)\}\s*\)', t, re.S); \
    print(len([l for l in m.group(1).splitlines() if l.strip()]))"
    22

Test-writer design decision, because the plan is silent on exactly where `rules.yml` lands
(section 6 test 2's own wording: "or whatever key `rules.yml` is mounted as ... you may need to
design this yourself"): `infra/k8s/monitoring/alerts/rules.yml`'s contents join the EXISTING
`prometheus-config` ConfigMap as a SECOND key, `alert-rules.yml`, rather than a new ConfigMap of
their own - "ConfigMap-mounted the same way `prometheus.yml` itself is" (the plan's own words) is
read literally, as the SAME ConfigMap, and it is also the only reading that keeps the object count
at exactly three new objects, matching this plan's own "emits exactly ... `Deployment/alertmanager`,
`Service/alertmanager`, `ConfigMap/alertmanager-config`" (a fourth ConfigMap for rules would sit
outside that list and redden `test_the_monitoring_namespace_emits_alertmanager_and_nothing_else_new`
against a manifest that did everything else right). `rule_files` therefore names the bare filename
`alert-rules.yml`, exactly as `prometheus.yml`'s own key names the file Prometheus reads it as.

Second test-writer design decision: the receiver's `email_configs`/`telegram_configs` entries read
every secret value as a `${VAR_NAME}` placeholder rather than a literal - the plan says "reading
values from env vars ... rather than literal values" but names no substitution syntax. `${VAR}` is
the one shape this scan understands; a config templated any other way (a `_file` suffixed field, a
Go template) is a different, undocumented mechanism the implementer would need to also update this
test's `ENV_PLACEHOLDER` for, rather than something this scan silently waves through.

`registered_metric_names` and the PromQL identifier scanner are COPIES of
`infra/tests/infra/test_k8s_grafana.py`'s, not imports - the convention of this suite. So are the
plain document helpers (`name_of`, `namespace_of`, `objects_of_kind`, `object_named`,
`pod_templates`, `policies`, `policy`, ...), copied from
`infra/tests/infra/test_k8s_monitoring.py` and `infra/tests/infra/test_k8s_netpol.py`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.core.metrics import registry

MONITORING = "monitoring"
APPS = "apps"
CERT_MANAGER = "cert-manager"

ALERTMANAGER = "alertmanager"
PROMETHEUS = "prometheus"
NODE_EXPORTER = "node-exporter"
KUBE_STATE_METRICS = "kube-state-metrics"
GRAFANA = "grafana"
GATEWAY = "gateway"

NETWORK_POLICY = "NetworkPolicy"

#: The one label every pod template in this build carries and the only one any selector keys off.
NAME_LABEL = "app.kubernetes.io/name"

#: Kubernetes auto-labels every namespace with its own name, which is how a cross-namespace peer
#: names the namespace it means.
NAMESPACE_NAME_LABEL = "kubernetes.io/metadata.name"

#: The peer marker for "every pod in that namespace": a peer with no `podSelector`, or an empty
#: one.
ANY_POD = "*"

#: Test-writer design decision (see the module docstring): `rules.yml`'s contents land as a
#: SECOND key on the EXISTING `prometheus-config` ConfigMap, named the way it is mounted.
PROMETHEUS_CONFIGMAP = "prometheus-config"
RULES_KEY = "alert-rules.yml"

#: Criterion 2's `alerting:` block: the Service DNS Prometheus sends firing alerts to.
ALERTMANAGER_SERVICE_DNS = "alertmanager.monitoring.svc.cluster.local:9093"

#: Criterion, by value, the exact remainder an earlier change left (its own
#: `test_grafana_fits_what_the_scrape_plane_left_and_leaves_alertmanager_its_row`): 32Mi, 50m,
#: 64Mi, no `limits.cpu`. The whole flattened mapping is compared, so an extra key is a failure.
ALERTMANAGER_RESOURCES: dict[str, str] = {
    "requests.memory": "32Mi",
    "requests.cpu": "50m",
    "limits.memory": "64Mi",
}

#: The five pod templates `monitoring` carries after this plan (23a's three, 23b's Grafana, this
#: plan's Alertmanager). Pinned so a fifth-workload count cannot be reached by a wrong pod
#: cancelling a missing Alertmanager template out.
MONITORING_WORKLOADS: frozenset[str] = frozenset(
    {PROMETHEUS, NODE_EXPORTER, KUBE_STATE_METRICS, GRAFANA, ALERTMANAGER}
)

#: Criterion, exact durations (a project rule, never a band).
GROUP_WAIT = "30s"
GROUP_INTERVAL = "5m"
REPEAT_INTERVAL = "4h"

#: Test-writer design decision (see the module docstring): the one substitution syntax this scan
#: understands for "read from an env var, not a literal".
ENV_PLACEHOLDER = re.compile(r"^\$\{(?P<name>[A-Z][A-Z0-9_]*)\}$")

#: The plain values `render.sh` writes into the template, by the env var each reads (an earlier change,
#: a design note). The password and the bot token are NOT here: they are files, see below.
EMAIL_FIELD_VARS: dict[str, str] = {
    "smarthost": "SMTP_SMARTHOST",
    "from": "SMTP_FROM",
    "auth_username": "SMTP_AUTH_USERNAME",
    "to": "SMTP_FROM",
}

#: The `telegram_configs` half of the same contract.
TELEGRAM_FIELD_VARS: dict[str, str] = {
    "chat_id": "TELEGRAM_CHAT_ID",
}

#: Where the secret keys sit as files in the `alertmanager` container.
SECRET_MOUNT = "/etc/alertmanager/secrets"
PASSWORD_FILE = f"{SECRET_MOUNT}/SMTP_AUTH_PASSWORD"
TOKEN_FILE = f"{SECRET_MOUNT}/TELEGRAM_BOT_TOKEN"

#: The init container, its script, and the two folders it reads from and writes to.
RENDER_CONFIG = "render-config"
TEMPLATE_DIR = "/etc/alertmanager/template"
RENDERED_DIR = "/etc/alertmanager-rendered"
RENDERED_FILE = f"{RENDERED_DIR}/alertmanager.yml"
RENDER_COMMAND = ["sh", f"{TEMPLATE_DIR}/render.sh"]
ALERTMANAGER_IMAGE = "prom/alertmanager:v0.27.0"

#: `render-config` asks for less than `alertmanager` on every line, so the pod's effective
#: request (the larger of the init step and the running containers) stays the 32Mi / 50m / 64Mi.
RENDER_CONFIG_RESOURCES: dict[str, str] = {
    "requests.memory": "16Mi",
    "requests.cpu": "10m",
    "limits.memory": "32Mi",
}

#: Every `render.sh` input, in the order the script takes them.
RENDER_KEYS = ("SMTP_SMARTHOST", "SMTP_FROM", "SMTP_AUTH_USERNAME", "TELEGRAM_CHAT_ID")
VALID_VALUES: dict[str, str] = {
    "SMTP_SMARTHOST": "smtp.example.com:587",
    "SMTP_FROM": "alerts@example.com",
    "SMTP_AUTH_USERNAME": "smtp-user@example.com",
    "TELEGRAM_CHAT_ID": "-1001234567890",
}

ALERTMANAGER_SECRETS = "alertmanager-secrets"

#: Copied from `infra/tests/infra/test_k8s_netpol.py`'s `PUBLIC_EXCEPT`/`PUBLIC_PORTS`:
#: `gateway-egress`'s excluded ranges and port, which `alertmanager-egress` must copy exactly
#: (a project rule - copied, not re-derived).
PUBLIC_CIDR = "0.0.0.0/0"
PUBLIC_EXCEPT: frozenset[str] = frozenset(
    {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"}
)
PUBLIC_PORTS: frozenset[tuple[str, int]] = frozenset({("TCP", 443), ("TCP", 465), ("TCP", 587)})

#: Criterion: no named-inert alert. `BackupJobFailed` left this set with row 23d's CronJobs,
#: which feed `kube_job_status_failed{job_name=~".*-backup.*"}` for real; an earlier change empties it
#: the same way, by registering `gateway_eval_*` under `ALLOWED_METRIC_PREFIXES` below instead
#: of naming `NightlyEvalRegression` an exception - the family is fed by the nightly CronJob's
#: Pushgateway push, never by `metrics.py`'s own registry, so it belongs beside `certmanager_`
#: and not on this list. An empty set still works as a membership check (`in frozenset()` is
#: always `False`), so a SECOND alert reading an unregistered, unprefixed family still fails
#: the loop below rather than being waved through silently.
INERT_ALERT_NAMES: frozenset[str] = frozenset()

#: The non-gateway metric family PREFIXES this plan's rules read, beside a family
#: `metrics.py` registers: Prometheus's own `up`, cluster infra (kube-state-metrics,
#: node-exporter), cert-manager's own metrics - not registered in this repo's code, since
#: cert-manager is helm-installed and unmanaged by this kustomize build - and, from an earlier change,
#: `gateway_eval_*`: pushed by the nightly CronJob to a Pushgateway rather than registered by
#: any in-process `prometheus_client` registry this repo owns.
ALLOWED_METRIC_PREFIXES: tuple[str, ...] = (
    "up",
    "kube_",
    "node_",
    "certmanager_",
    "gateway_eval_",
)

#: The series suffixes each `prometheus_client` family TYPE exposes beside its own name, probed
#: against the installed version, copied from
#: `infra/tests/infra/test_k8s_grafana.py`.
FAMILY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "counter": ("_total", "_created"),
    "gauge": (),
    "histogram": ("_bucket", "_sum", "_count", "_created"),
    "summary": ("_sum", "_count", "_created"),
}

#: A PromQL identifier, copied from `infra/tests/infra/test_k8s_grafana.py` verbatim, lookaround and
#: all: the `(?<![0-9A-Za-z_:])` lookbehind is what keeps the `m` of `5m` out of the scan.
PROMQL_IDENTIFIER = re.compile(
    r"(?<![0-9A-Za-z_:])[A-Za-z_:][A-Za-z0-9_:]*(?![0-9A-Za-z_:])(?!\s*\()"
)
PROMQL_KEYWORDS = frozenset({"and", "or", "unless", "bool", "offset", "start", "end", "inf", "nan"})
PROMQL_GROUPING = re.compile(
    r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)", re.IGNORECASE
)
PROMQL_STRING = re.compile(r"\"[^\"]*\"|'[^']*'")
PROMQL_SELECTOR = re.compile(r"\{[^}]*\}")


def name_of(document: dict[str, Any]) -> str:
    """`metadata.name`, as a string."""
    metadata = document.get("metadata", {})
    assert isinstance(metadata, dict)
    return str(metadata.get("name", ""))


def namespace_of(document: dict[str, Any]) -> str | None:
    """`metadata.namespace`, or None when the document declares none."""
    metadata = document.get("metadata", {})
    assert isinstance(metadata, dict)
    value = metadata.get("namespace")
    return None if value is None else str(value)


def objects_of_kind(documents: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Every built document of `kind`, in build order."""
    return [document for document in documents if document.get("kind") == kind]


def object_named(documents: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """The one `kind` named `name`, asserting the build emits exactly one."""
    found = [document for document in objects_of_kind(documents, kind) if name_of(document) == name]
    emitted = sorted(name_of(document) for document in objects_of_kind(documents, kind))
    assert len(found) == 1, f"expected one {kind}/{name}; the build has {kind}s {emitted}"
    return found[0]


def pod_owners(documents: list[dict[str, Any]], namespace: str) -> list[dict[str, Any]]:
    """Every document in `namespace` carrying a pod template, found by the PRESENCE of
    `spec.template.spec` rather than by a list of kinds."""
    owners: list[dict[str, Any]] = []
    for document in documents:
        spec = document.get("spec")
        if not isinstance(spec, dict) or namespace_of(document) != namespace:
            continue
        template = spec.get("template")
        if isinstance(template, dict) and isinstance(template.get("spec"), dict):
            owners.append(document)
    return owners


def template_labels(template: dict[str, Any]) -> dict[str, str]:
    """A pod template's labels, the ones a `podSelector` has to match."""
    metadata = template.get("metadata", {})
    labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
    assert isinstance(labels, dict)
    return {str(key): str(value) for key, value in labels.items()}


def pod_templates(documents: list[dict[str, Any]], namespace: str) -> dict[str, dict[str, Any]]:
    """Every pod template in `namespace`, keyed by its `app.kubernetes.io/name` label."""
    templates: dict[str, dict[str, Any]] = {}
    for owner in pod_owners(documents, namespace):
        template = owner["spec"]["template"]
        labels = template_labels(template)
        workload = labels.get(NAME_LABEL, "")
        assert workload, f"{name_of(owner)}'s pod template carries no {NAME_LABEL}"
        templates[workload] = template
    return templates


def containers_of(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Every container of a pod template, `initContainers` included."""
    spec = template["spec"]
    assert isinstance(spec, dict)
    collected: list[dict[str, Any]] = []
    for key in ("initContainers", "containers"):
        listed = spec.get(key, [])
        assert isinstance(listed, list), f"a pod template's {key} is not a list"
        collected.extend(container for container in listed if isinstance(container, dict))
    return collected


def named_container(template: dict[str, Any], section: str, name: str) -> dict[str, Any]:
    """The one container called `name` in `spec.<section>` (`containers` or `initContainers`)."""
    listed = template["spec"].get(section, [])
    assert isinstance(listed, list)
    found = [entry for entry in listed if isinstance(entry, dict) and entry.get("name") == name]
    assert len(found) == 1, f"{len(found)} {section} named {name!r}: {listed}"
    return found[0]


def render_script(documents: list[dict[str, Any]]) -> tuple[str, str]:
    """`ConfigMap/alertmanager-config`'s `render.sh` and `alertmanager.yml` texts."""
    data = config_map_data(documents, "alertmanager-config")
    assert set(data) == {"alertmanager.yml", "render.sh"}, (
        f"alertmanager-config holds {sorted(data)}"
    )
    return data["render.sh"], data["alertmanager.yml"]


def run_render(
    documents: list[dict[str, Any]], tmp_path: Path, values: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the built `render.sh` with `sh` in `tmp_path`, with ONLY `values` (and PATH) in the env.

    `TEMPLATE_FILE` and `OUTPUT_FILE` point inside `tmp_path`; the output file does not exist
    beforehand. Returns the finished process and the output path.
    """
    script, template = render_script(documents)
    script_path = tmp_path / "render.sh"
    script_path.write_text(script)
    template_path = tmp_path / "template.yml"
    template_path.write_text(template)
    output_path = tmp_path / "rendered.yml"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TEMPLATE_FILE": str(template_path),
        "OUTPUT_FILE": str(output_path),
        **values,
    }
    completed = subprocess.run(
        ["sh", str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    return completed, output_path


def assert_refused(
    completed: subprocess.CompletedProcess[str], tmp_path: Path, key: str, value: str
) -> None:
    """The script exited non-zero, left only its two inputs behind, named `key`, never `value`."""
    assert completed.returncode != 0, f"{key}={value!r} was accepted"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["render.sh", "template.yml"], (
        f"{key}={value!r} left files behind: {sorted(path.name for path in tmp_path.iterdir())}"
    )
    shown = completed.stdout + completed.stderr
    assert key in shown, f"{key}={value!r}: the output does not name {key}: {shown!r}"
    if value:
        assert value not in shown, f"{key}: the output repeats the value: {shown!r}"


def resource_quantities(container: dict[str, Any]) -> dict[str, str]:
    """A container's `resources` flattened to the `requests.memory`-style keys, values exactly
    as written."""
    resources = container.get("resources", {})
    if not isinstance(resources, dict):
        return {}
    flattened: dict[str, str] = {}
    for side in ("requests", "limits"):
        values = resources.get(side, {})
        if not isinstance(values, dict):
            continue
        for resource, quantity in values.items():
            flattened[f"{side}.{resource}"] = str(quantity)
    return flattened


def config_map_data(documents: list[dict[str, Any]], name: str) -> dict[str, str]:
    """One ConfigMap's `data` mapping, values as the strings a ConfigMap actually holds."""
    document = object_named(documents, "ConfigMap", name)
    data = document.get("data", {})
    assert isinstance(data, dict), f"{name} declares no data mapping"
    return {str(key): str(value) for key, value in data.items()}


def prometheus_config(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """`ConfigMap/prometheus-config`'s `prometheus.yml`, parsed."""
    data = config_map_data(documents, PROMETHEUS_CONFIGMAP)
    assert "prometheus.yml" in data, f"{PROMETHEUS_CONFIGMAP} holds keys {sorted(data)}"
    parsed = yaml.safe_load(data["prometheus.yml"])
    assert isinstance(parsed, dict), "prometheus.yml is not a YAML mapping"
    return parsed


def rule_alerts(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every alerting rule `infra/k8s/monitoring/alerts/rules.yml` carries, read off
    `ConfigMap/prometheus-config`'s `alert-rules.yml` key (see the module docstring for why it
    is a second key on that ConfigMap and not a new one)."""
    data = config_map_data(documents, PROMETHEUS_CONFIGMAP)
    assert RULES_KEY in data, f"{PROMETHEUS_CONFIGMAP} holds keys {sorted(data)}"
    parsed = yaml.safe_load(data[RULES_KEY])
    assert isinstance(parsed, dict), f"{RULES_KEY} is not a YAML mapping"
    groups = parsed.get("groups", [])
    assert isinstance(groups, list) and groups, f"{RULES_KEY} declares no rule groups"
    alerts: list[dict[str, Any]] = []
    for group in groups:
        assert isinstance(group, dict), f"{RULES_KEY} has a group that is not a mapping"
        rules = group.get("rules", [])
        assert isinstance(rules, list), f"{RULES_KEY}'s group declares no rules list"
        alerts.extend(rule for rule in rules if isinstance(rule, dict) and "alert" in rule)
    assert alerts, f"{RULES_KEY} declares no alerting rules"
    return alerts


def alertmanager_config(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """`ConfigMap/alertmanager-config`'s `alertmanager.yml`, parsed."""
    data = config_map_data(documents, "alertmanager-config")
    assert "alertmanager.yml" in data, f"alertmanager-config holds keys {sorted(data)}"
    parsed = yaml.safe_load(data["alertmanager.yml"])
    assert isinstance(parsed, dict), "alertmanager.yml is not a YAML mapping"
    return parsed


def policies(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every built NetworkPolicy, in build order."""
    return [document for document in documents if document.get("kind") == NETWORK_POLICY]


def policy(documents: list[dict[str, Any]], namespace: str, name: str) -> dict[str, Any]:
    """The one NetworkPolicy `namespace/name`, asserting the build emits exactly one."""
    found = [
        document
        for document in policies(documents)
        if namespace_of(document) == namespace and name_of(document) == name
    ]
    emitted = sorted((str(namespace_of(p)), name_of(p)) for p in policies(documents))
    assert len(found) == 1, f"expected one NetworkPolicy {namespace}/{name}, build has {emitted}"
    return found[0]


def spec_of(document: dict[str, Any]) -> dict[str, Any]:
    """`spec`, asserting it is a mapping."""
    spec = document.get("spec")
    assert isinstance(spec, dict), f"{name_of(document)} has no spec mapping"
    return spec


def pod_selector(document: dict[str, Any]) -> dict[str, Any]:
    """`spec.podSelector`, the set of pods a policy applies TO. `{}` means every pod."""
    selector = spec_of(document).get("podSelector")
    assert isinstance(selector, dict), f"{name_of(document)} declares no spec.podSelector"
    return selector


def rules_of(document: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    """`spec.ingress` or `spec.egress` as a list of rule mappings, empty when absent."""
    listed = spec_of(document).get(direction, [])
    assert isinstance(listed, list), f"{name_of(document)}.spec.{direction} is not a list"
    return [rule for rule in listed if isinstance(rule, dict)]


def peers_of(rule: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """A rule's `from` or `to` peers. An EMPTY list means the rule allows every source or
    destination, which is never what any policy here intends, so the caller asserts on it."""
    listed = rule.get(key, [])
    assert isinstance(listed, list), f"a rule's `{key}` is not a list"
    return [peer for peer in listed if isinstance(peer, dict)]


def selector_matches(selector: dict[str, Any], labels: dict[str, str]) -> bool:
    """Whether a `podSelector` matches a pod template's labels."""
    match_labels = selector.get("matchLabels", {})
    assert isinstance(match_labels, dict)
    for key, value in match_labels.items():
        if labels.get(str(key)) != str(value):
            return False
    expressions = selector.get("matchExpressions", [])
    assert isinstance(expressions, list)
    for expression in expressions:
        assert isinstance(expression, dict)
        operator = str(expression.get("operator"))
        assert operator == "In", f"this scan understands matchLabels and In, not {operator!r}"
        values = {str(value) for value in expression.get("values", [])}
        if labels.get(str(expression.get("key"))) not in values:
            return False
    return True


def matched_workloads(
    documents: list[dict[str, Any]], selector: dict[str, Any], namespace: str
) -> set[str]:
    """The workload names in `namespace` a `podSelector` matches. `{}` matches every one."""
    return {
        workload
        for workload, template in pod_templates(documents, namespace).items()
        if selector_matches(selector, template_labels(template))
    }


def peer_namespace(peer: dict[str, Any], default: str) -> str:
    """The namespace a `from`/`to` peer is scoped to: the one its `namespaceSelector` names via
    `kubernetes.io/metadata.name`, otherwise the policy's own namespace."""
    selector = peer.get("namespaceSelector")
    if not isinstance(selector, dict):
        return default
    match_labels = selector.get("matchLabels", {})
    assert isinstance(match_labels, dict), "a namespaceSelector with no matchLabels is ambiguous"
    named = match_labels.get(NAMESPACE_NAME_LABEL)
    assert named is not None, (
        f"a namespaceSelector keys off {sorted(match_labels)}, not {NAMESPACE_NAME_LABEL}"
    )
    return str(named)


def peer_targets(
    documents: list[dict[str, Any]], peer: dict[str, Any], default_namespace: str
) -> set[tuple[str, str]]:
    """A peer as `(namespace, workload)` pairs, `workload` being `ANY_POD` when the peer names no
    `podSelector` or an empty one - the shape a `namespaceSelector`-only peer has."""
    namespace = peer_namespace(peer, default_namespace)
    selector = peer.get("podSelector")
    if not isinstance(selector, dict) or selector == {}:
        return {(namespace, ANY_POD)}
    workloads = matched_workloads(documents, selector, namespace)
    assert workloads, f"podSelector {selector} matches no pod template in {namespace}"
    return {(namespace, workload) for workload in workloads}


def declared_ports(rule: dict[str, Any], label: str) -> list[tuple[str, object]]:
    """A rule's `(protocol, port-as-written)` entries, the port still a name OR a number."""
    listed = rule.get("ports", [])
    assert isinstance(listed, list) and listed, f"{label} declares no ports, so it allows all"
    entries: list[tuple[str, object]] = []
    for entry in listed:
        assert isinstance(entry, dict), f"{label} has a port entry that is not a mapping"
        entries.append((str(entry.get("protocol", "TCP")), entry.get("port")))
    return entries


def numeric_ports(rule: dict[str, Any], label: str) -> set[tuple[str, int]]:
    """A rule's ports as `(protocol, number)`, asserting every one is written as a NUMBER."""
    pairs: set[tuple[str, int]] = set()
    for protocol, port in declared_ports(rule, label):
        assert isinstance(port, int), f"{label} names port {port!r}; this rule reads in numbers"
        pairs.add((protocol, port))
    return pairs


def ports_against(rule: dict[str, Any], ports: dict[str, int], label: str) -> set[tuple[str, int]]:
    """A rule's ports resolved against ONE pod's named container ports."""
    resolved: set[tuple[str, int]] = set()
    for protocol, port in declared_ports(rule, label):
        if isinstance(port, int):
            resolved.add((protocol, port))
            continue
        number = ports.get(str(port))
        assert number is not None, f"{label} names port {port!r}; that pod has {sorted(ports)}"
        resolved.add((protocol, number))
    return resolved


def container_ports(template: dict[str, Any]) -> dict[str, int]:
    """A pod template's NAMED container ports, `name -> number`."""
    numbers: dict[str, int] = {}
    for container in containers_of(template):
        for entry in container.get("ports", []):
            if isinstance(entry, dict) and entry.get("name") is not None:
                numbers[str(entry["name"])] = int(entry["containerPort"])
    return numbers


def egress_destinations(
    documents: list[dict[str, Any]], document: dict[str, Any]
) -> set[tuple[str, int]]:
    """An egress policy's destinations as `(workload-or-namespace, resolved port)`."""
    namespace = namespace_of(document)
    assert namespace is not None, f"{name_of(document)} declares no metadata.namespace"
    label = f"{namespace}/{name_of(document)}"
    destinations: set[tuple[str, int]] = set()
    for rule in rules_of(document, "egress"):
        peers = peers_of(rule, "to")
        assert peers, f"{label} has an egress rule with no `to`, which allows every destination"
        for peer in peers:
            for target_namespace, workload in peer_targets(documents, peer, namespace):
                if workload == ANY_POD:
                    for _, port in numeric_ports(rule, label):
                        destinations.add((target_namespace, port))
                    continue
                template = pod_templates(documents, target_namespace)[workload]
                for _, port in ports_against(rule, container_ports(template), label):
                    destinations.add((workload, port))
    return destinations


def ip_blocks(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every `ipBlock` in a policy, from either direction and any peer."""
    found: list[dict[str, Any]] = []
    for direction, key in (("ingress", "from"), ("egress", "to")):
        for rule in rules_of(document, direction):
            for peer in peers_of(rule, key):
                block = peer.get("ipBlock")
                if isinstance(block, dict):
                    found.append(block)
    return found


def rules_with_ip_block(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every egress RULE of a policy carrying an `ipBlock` peer."""
    return [
        rule
        for rule in rules_of(document, "egress")
        if any(isinstance(peer.get("ipBlock"), dict) for peer in peers_of(rule, "to"))
    ]


def metric_names_in(expr: str) -> set[str]:
    r"""Every metric family name a PromQL `expr` reads. Copied from
    `infra/tests/infra/test_k8s_grafana.py`'s `metric_names_in`, verbatim."""
    stripped = PROMQL_STRING.sub(" ", expr)
    stripped = PROMQL_GROUPING.sub(" ", stripped)
    stripped = PROMQL_SELECTOR.sub(" ", stripped)
    return {
        token
        for token in PROMQL_IDENTIFIER.findall(stripped)
        if token.lower() not in PROMQL_KEYWORDS
    }


def registered_metric_names() -> set[str]:
    """Every spelling a rule `expr` may legally use for a GATEWAY family, resolved from
    `packages/opsconsole/opsconsole/core/metrics.py` AT RUN TIME. Copied from
    `infra/tests/infra/test_k8s_grafana.py`'s `registered_metric_names`, verbatim."""
    accepted: set[str] = set()
    families = list(registry().collect())
    assert families, "the gateway registry exposes no metric family at all"
    for family in families:
        assert family.type in FAMILY_SUFFIXES, (
            f"{family.name} is a {family.type}, whose exposed series this helper cannot name"
        )
        accepted.add(family.name)
        accepted.update(f"{family.name}{suffix}" for suffix in FAMILY_SUFFIXES[family.type])
    return accepted


def test_the_rules_file_is_mounted_and_every_expr_names_a_registered_family_or_a_documented_gap(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 1 and 2, over the PARSED rules file and the PARSED `prometheus.yml`: `rule_files`
    names `alert-rules.yml`, `alerting.alertmanagers` names Alertmanager's Service DNS, and every
    `expr` in every rule reads a metric family `metrics.py` registers or one under
    `up`/`kube_`/`node_`/`certmanager_`/`gateway_eval_` - an earlier change empties `INERT_ALERT_NAMES`, so
    every alert this build carries is checked here with no exception left at all.

    Catches `rules.yml` written and never referenced by `rule_files`, which builds and
    schema-validates a ConfigMap that Prometheus never reads - every alert permanently absent with
    no error anywhere. Catches Prometheus left with no `alerting:` block, which is every alert
    firing into a void: the rule evaluates, the state changes, and nothing ever notifies anyone.
    Catches an `expr` reading a family name that is a typo of a real one
    (`gateway_residual_violations_total`), which this scan cannot mistake for a documented gap
    because it matches no registered name and no allowed prefix. Catches an alert quietly marked
    inert to dodge this check - `INERT_ALERT_NAMES` is empty, so ANY name reading an unregistered,
    unprefixed family fails here rather than being waved through by a docstring nobody enforces.
    """
    parsed = prometheus_config(built)
    rule_files = [str(entry) for entry in parsed.get("rule_files", [])]
    assert RULES_KEY in rule_files, f"prometheus.yml's rule_files are {rule_files}"

    alerting = parsed.get("alerting", {})
    assert isinstance(alerting, dict), "prometheus.yml declares no alerting mapping"
    alertmanagers = alerting.get("alertmanagers", [])
    assert isinstance(alertmanagers, list) and alertmanagers, (
        "prometheus.yml's alerting block names no alertmanagers"
    )
    targets: set[str] = set()
    for entry in alertmanagers:
        assert isinstance(entry, dict)
        for static in entry.get("static_configs", []):
            assert isinstance(static, dict)
            targets |= {str(target) for target in static.get("targets", [])}
    assert targets == {ALERTMANAGER_SERVICE_DNS}, f"alertmanagers targets are {sorted(targets)}"

    registered = registered_metric_names()
    for rule in rule_alerts(built):
        alert_name = str(rule.get("alert", ""))
        assert alert_name, f"a rule carries no `alert` name: {rule}"
        if alert_name in INERT_ALERT_NAMES:
            continue
        expr = str(rule.get("expr", ""))
        assert expr, f"{alert_name} declares no expr"
        for metric in metric_names_in(expr):
            allowed = metric in registered or metric.startswith(ALLOWED_METRIC_PREFIXES)
            assert allowed, (
                f"{alert_name}'s expr reads {metric!r}, which is neither a registered gateway "
                f"family nor a documented non-gateway prefix"
            )


#: An earlier change decision A's exact expression, character for character: the
#: median rather than the average, over the same 7-day window and the same 2-point drop.
NIGHTLY_EVAL_REGRESSION_EXPR = (
    "gateway_eval_check_score - quantile_over_time(0.5, gateway_eval_check_score[7d]) < -2"
)


def test_nightly_eval_regression_reads_the_seven_day_median_and_not_the_average(
    built: list[dict[str, Any]],
) -> None:
    """Decision A, pinned by exact string. Catches the alert left on `avg_over_time` while
    every OTHER trace of "inert until row 24" is cleaned up - a rename or an annotation edit
    with the operator left unchanged would still pass the generic scan above, since
    `avg_over_time(...)` and `quantile_over_time(0.5, ...)` read the exact same metric family
    and neither is a typo of a family name that scan can catch. Only a character-for-character
    comparison of the whole expression tells "reads the average" from "reads the median" -
    the request's own wording, and the whole reason this row exists."""
    alerts = {str(rule.get("alert", "")): rule for rule in rule_alerts(built)}
    assert "NightlyEvalRegression" in alerts, sorted(alerts)
    assert str(alerts["NightlyEvalRegression"].get("expr", "")) == NIGHTLY_EVAL_REGRESSION_EXPR


def test_alertmanager_fits_the_exact_remainder_23b_left_it(
    built: list[dict[str, Any]],
) -> None:
    """The exact remainder an earlier change left, on the container named `alertmanager` (the pod also has
    an init container since an earlier change): 32Mi/50m/64Mi by VALUE (the whole flattened mapping, so
    an extra `limits.cpu` is a failure), `emptyDir` for its data volume, and no
    `PersistentVolumeClaim` named `alertmanager` anywhere in the build.

    Catches a quantity raised above the remainder, which pushes `monitoring`'s namespace over its
    unchanged `ResourceQuota` and gets the LAST-admitted pod refused with no error naming why.
    Catches a `limits.cpu` added, which throttles the one pod holding every firing alert rather
    than protecting anything. Catches decision D reversed to a
    `PersistentVolumeClaim`, which an earlier change's headroom accounting never reserved storage quota
    for and 23a's envelope never sized for either.
    """
    templates = pod_templates(built, MONITORING)
    assert ALERTMANAGER in templates, f"monitoring has pod templates {sorted(templates)}"
    container = named_container(templates[ALERTMANAGER], "containers", ALERTMANAGER)
    quantities = resource_quantities(container)
    assert quantities == ALERTMANAGER_RESOURCES, f"alertmanager declares {quantities}"

    volumes = templates[ALERTMANAGER]["spec"].get("volumes", [])
    assert isinstance(volumes, list) and volumes, "alertmanager's pod template declares no volumes"
    empty_dirs = [volume for volume in volumes if isinstance(volume, dict) and "emptyDir" in volume]
    assert empty_dirs, f"alertmanager's pod template has no emptyDir volume: {volumes}"
    pvc_backed = [
        volume
        for volume in volumes
        if isinstance(volume, dict) and "persistentVolumeClaim" in volume
    ]
    assert pvc_backed == [], f"alertmanager's pod template mounts a PVC: {pvc_backed}"

    pvcs = sorted(name_of(pvc) for pvc in objects_of_kind(built, "PersistentVolumeClaim"))
    assert ALERTMANAGER not in pvcs, f"the build emits PersistentVolumeClaims {pvcs}"


def test_alertmanager_routes_every_alert_to_both_an_email_and_a_telegram_receiver(
    built: list[dict[str, Any]],
) -> None:
    """The one route, the one receiver, exactly one `email_configs` and one `telegram_configs`
    entry; the four plain fields are `${VAR}` placeholders the init step fills in (an earlier change, ADR
    0060), and the three durations are exact strings.

    Catches a receiver with two `email_configs` entries or none, which is either a duplicate
    notification for every alert or total silence. Catches a value typed in literally
    (`smtp.gmail.com`, a real chat id) rather than left as a placeholder, which puts a real
    account detail in a committed manifest and gives `render.sh` nothing to fill. Catches
    `group_wait`/`group_interval`/`repeat_interval` left at Alertmanager's defaults. Catches a
    placeholder that names the wrong variable (`auth_username` reading `${SMTP_FROM}`), which
    reads clean here and sends mail as the wrong account.
    """
    config = alertmanager_config(built)
    route = config.get("route", {})
    assert isinstance(route, dict) and route, "alertmanager.yml declares no route"
    assert route.get("group_wait") == GROUP_WAIT, f"group_wait is {route.get('group_wait')!r}"
    assert route.get("group_interval") == GROUP_INTERVAL, (
        f"group_interval is {route.get('group_interval')!r}"
    )
    assert route.get("repeat_interval") == REPEAT_INTERVAL, (
        f"repeat_interval is {route.get('repeat_interval')!r}"
    )
    receiver_name = route.get("receiver")
    assert receiver_name, "the route names no receiver"

    receivers = config.get("receivers", [])
    assert isinstance(receivers, list) and receivers, "alertmanager.yml declares no receivers"
    matched = [
        entry
        for entry in receivers
        if isinstance(entry, dict) and entry.get("name") == receiver_name
    ]
    assert len(matched) == 1, f"{len(matched)} receivers named {receiver_name!r}: {receivers}"
    receiver = matched[0]

    emails = receiver.get("email_configs", [])
    assert isinstance(emails, list) and len(emails) == 1, (
        f"receiver {receiver_name!r} has {len(emails)} email_configs, not 1"
    )
    email = emails[0]
    assert isinstance(email, dict)
    for field, variable in EMAIL_FIELD_VARS.items():
        value = str(email.get(field, ""))
        match = ENV_PLACEHOLDER.fullmatch(value)
        assert match is not None, f"email_configs.{field} is {value!r}, not an env placeholder"
        assert match.group("name") == variable, (
            f"email_configs.{field} reads {match.group('name')!r}, not {variable!r}"
        )

    telegrams = receiver.get("telegram_configs", [])
    assert isinstance(telegrams, list) and len(telegrams) == 1, (
        f"receiver {receiver_name!r} has {len(telegrams)} telegram_configs, not 1"
    )
    telegram = telegrams[0]
    assert isinstance(telegram, dict)
    for field, variable in TELEGRAM_FIELD_VARS.items():
        value = str(telegram.get(field, ""))
        match = ENV_PLACEHOLDER.fullmatch(value)
        assert match is not None, f"telegram_configs.{field} is {value!r}, not an env placeholder"
        assert match.group("name") == variable, (
            f"telegram_configs.{field} reads {match.group('name')!r}, not {variable!r}"
        )


def test_the_config_reads_the_password_and_token_from_files(
    built: list[dict[str, Any]],
) -> None:
    """`auth_password_file` and `bot_token_file` point into the secret mount, no `auth_password`
    or `bot_token` field exists, the `alertmanager` container mounts `alertmanager-secrets` at
    that folder, and it starts on the config `render-config` wrote.

    Catches the password or token left as `auth_password: ${...}` / `bot_token: ${...}`, which
    Alertmanager reads as literal text, so mail and Telegram fail at the first alert. Catches a
    `_file` path that points outside the mount, which is a file that is never there. Catches the
    Secret mounted into `render-config` only, or under another folder, which leaves the path
    pointing at nothing. Catches `--config.file` still on the unrendered template, which is the
    config that never starts.
    """
    config = alertmanager_config(built)
    receiver = config["receivers"][0]
    email = receiver["email_configs"][0]
    telegram = receiver["telegram_configs"][0]
    assert email.get("auth_password_file") == PASSWORD_FILE, f"email: {sorted(email)}"
    assert telegram.get("bot_token_file") == TOKEN_FILE, f"telegram: {sorted(telegram)}"
    assert "auth_password" not in email, "email_configs still has an inline auth_password"
    assert "bot_token" not in telegram, "telegram_configs still has an inline bot_token"

    template = pod_templates(built, MONITORING)[ALERTMANAGER]
    running = named_container(template, "containers", ALERTMANAGER)
    volumes = template["spec"].get("volumes", [])
    secret_volumes = [
        volume["name"]
        for volume in volumes
        if isinstance(volume.get("secret"), dict)
        and volume["secret"].get("secretName") == ALERTMANAGER_SECRETS
    ]
    assert len(secret_volumes) == 1, f"volumes from {ALERTMANAGER_SECRETS}: {volumes}"
    mounts = {
        str(mount["mountPath"]): str(mount["name"]) for mount in running.get("volumeMounts", [])
    }
    assert mounts.get(SECRET_MOUNT) == secret_volumes[0], f"alertmanager mounts {mounts}"
    assert f"--config.file={RENDERED_FILE}" in running.get("args", []), (
        f"alertmanager args are {running.get('args')}"
    )


def test_render_config_writes_the_config_alertmanager_reads(
    built: list[dict[str, Any]],
) -> None:
    """The init container `render-config` runs `sh` on the script from the ConfigMap, on the same
    image as `alertmanager`, reading the template folder and writing an `emptyDir` that both
    containers mount, with the exact small resources.

    Catches an init container on another image, which is one more image to pin and scan (ADR
    0060). Catches the output folder not shared with the main container, or not writable
    (`emptyDir` is, a ConfigMap is not), which runs the script and then starts Alertmanager on no
    file. Catches the script not mounted, or run as an executable file (a ConfigMap file has no
    execute bit unless set). Catches `render-config` asking for more than `alertmanager`, which
    raises the pod's effective request over the quota sized for Alertmanager alone.
    """
    template = pod_templates(built, MONITORING)[ALERTMANAGER]
    init = named_container(template, "initContainers", RENDER_CONFIG)
    running = named_container(template, "containers", ALERTMANAGER)
    assert init.get("image") == ALERTMANAGER_IMAGE == running.get("image")
    assert init.get("command") == RENDER_COMMAND, f"command is {init.get('command')}"
    assert resource_quantities(init) == RENDER_CONFIG_RESOURCES, resource_quantities(init)

    volumes = {volume["name"]: volume for volume in template["spec"]["volumes"]}
    init_mounts = {str(mount["mountPath"]): mount for mount in init.get("volumeMounts", [])}
    assert TEMPLATE_DIR in init_mounts and RENDERED_DIR in init_mounts, sorted(init_mounts)
    template_volume = volumes[str(init_mounts[TEMPLATE_DIR]["name"])]
    assert template_volume.get("configMap", {}).get("name") == "alertmanager-config", (
        template_volume
    )
    output_volume_name = str(init_mounts[RENDERED_DIR]["name"])
    assert "emptyDir" in volumes[output_volume_name], volumes[output_volume_name]
    assert init_mounts[RENDERED_DIR].get("readOnly") is not True
    running_mounts = {
        str(mount["mountPath"]): str(mount["name"]) for mount in running["volumeMounts"]
    }
    assert running_mounts.get(RENDERED_DIR) == output_volume_name, running_mounts


def test_the_rendered_config_parses_with_a_numeric_chat_id(
    built: list[dict[str, Any]], tmp_path: Path
) -> None:
    """`render.sh` with a host and port, two different addresses and a negative chat id writes a
    config whose `chat_id` is the int `-1001234567890`, `smarthost` is the host and port, `from`
    and `auth_username` are their own addresses, `to` is the `from` address, no `${` is left, and
    the file still holds the password and token file paths.

    Catches a chat id written quoted, which Alertmanager rejects (`chat_id` must be a number).
    Catches `from` and `auth_username` swapped, or `to` left as `${SMTP_FROM}`. Catches a
    substitution that stops after the first match on a line or a file, leaving one placeholder
    behind.
    """
    completed, output = run_render(built, tmp_path, VALID_VALUES)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    text = output.read_text()
    assert "${" not in text, text
    config = yaml.safe_load(text)
    receiver = config["receivers"][0]
    email = receiver["email_configs"][0]
    telegram = receiver["telegram_configs"][0]
    assert telegram["chat_id"] == -1001234567890
    assert isinstance(telegram["chat_id"], int)
    assert email["smarthost"] == "smtp.example.com:587"
    assert email["from"] == "alerts@example.com"
    assert email["auth_username"] == "smtp-user@example.com"
    assert email["to"] == "alerts@example.com"
    assert email["auth_password_file"] == PASSWORD_FILE
    assert telegram["bot_token_file"] == TOKEN_FILE


ADDRESS_254 = "a" * 242 + "@example.com"
ADDRESS_255 = "a" * 243 + "@example.com"


def test_render_accepts_values_at_the_limits(built: list[dict[str, Any]], tmp_path: Path) -> None:
    """Port 65535, a 15-digit chat id and a 254-character address render, and so do port 1 and a
    one-digit chat id.

    Catches an off-by-one that refuses the limit itself (port below 65535, 14 digits, 253
    characters). Catches a digit count read as the whole width with the minus sign counted in
    it, which refuses `-123456789012345`.
    """
    for port in ("1", "65535"):
        values = {**VALID_VALUES, "SMTP_SMARTHOST": f"smtp.example.com:{port}"}
        completed, output = run_render(built, tmp_path, values)
        assert completed.returncode == 0, f"port {port}: {completed.stdout + completed.stderr}"
        email = yaml.safe_load(output.read_text())["receivers"][0]["email_configs"][0]
        assert email["smarthost"] == f"smtp.example.com:{port}"
        output.unlink()
    for chat in ("123456789012345", "-123456789012345", "7"):
        completed, output = run_render(built, tmp_path, {**VALID_VALUES, "TELEGRAM_CHAT_ID": chat})
        assert completed.returncode == 0, f"chat {chat}: {completed.stdout + completed.stderr}"
        telegram = yaml.safe_load(output.read_text())["receivers"][0]["telegram_configs"][0]
        assert telegram["chat_id"] == int(chat)
        output.unlink()
    values = {**VALID_VALUES, "SMTP_FROM": ADDRESS_254, "SMTP_AUTH_USERNAME": ADDRESS_254}
    completed, output = run_render(built, tmp_path, values)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    email = yaml.safe_load(output.read_text())["receivers"][0]["email_configs"][0]
    assert email["from"] == ADDRESS_254 and email["auth_username"] == ADDRESS_254


REFUSED_VALUES = [
    ("SMTP_SMARTHOST", "smtp.example.com:65536"),
    ("SMTP_SMARTHOST", "smtp.example.com:0"),
    ("SMTP_SMARTHOST", "smtp.example.com"),
    ("SMTP_SMARTHOST", "smtp.example.com:http"),
    ("SMTP_SMARTHOST", ":587"),
    ("TELEGRAM_CHAT_ID", "1234567890123456"),
    ("TELEGRAM_CHAT_ID", "-1234567890123456"),
    ("TELEGRAM_CHAT_ID", "abc"),
    ("TELEGRAM_CHAT_ID", "-"),
    ("SMTP_FROM", ADDRESS_255),
    ("SMTP_AUTH_USERNAME", ADDRESS_255),
    ("SMTP_FROM", "not-an-address"),
    ("SMTP_AUTH_USERNAME", "not-an-address"),
]


@pytest.mark.parametrize(
    ("key", "value"),
    REFUSED_VALUES,
    ids=[f"{key}-{value[:24]}" for key, value in REFUSED_VALUES],
)
def test_render_refuses_values_past_the_limits(
    built: list[dict[str, Any]], tmp_path: Path, key: str, value: str
) -> None:
    """Each value one past a limit, or in the wrong shape, exits non-zero, writes no file, prints
    the key name and never the value; the other three values are valid, so only this check can
    be what refuses it.

    Catches a limit one too wide (port 65536, a 16-digit chat id, a 255-character address).
    Catches a port 0 that passes as "digits". Catches a missing check on `SMTP_AUTH_USERNAME`
    (the same shape as `SMTP_FROM`, easy to validate once and forget). Catches an error line
    that prints the value, which puts a bad but possibly real account detail in the pod log.
    Catches a script that writes the file first and checks after.
    """
    completed, _ = run_render(built, tmp_path, {**VALID_VALUES, key: value})
    assert_refused(completed, tmp_path, key, value)


TRAILING = [("newline", "\n"), ("crlf", "\r\n"), ("space", " ")]
TRAILING_KEYS = {
    "SMTP_SMARTHOST": "smtp.example.com:587",
    "SMTP_FROM": "alerts@example.com",
    "SMTP_AUTH_USERNAME": "smtp-user@example.com",
    "TELEGRAM_CHAT_ID": "-1001234567890",
}


@pytest.mark.parametrize("key", sorted(TRAILING_KEYS))
@pytest.mark.parametrize(("label", "suffix"), TRAILING, ids=[row[0] for row in TRAILING])
def test_render_refuses_a_trailing_newline_crlf_or_space(
    built: list[dict[str, Any]], tmp_path: Path, key: str, label: str, suffix: str
) -> None:
    """A valid value with a trailing newline, CRLF or space is refused, naming the key.

    Catches a check built on `$` or a line-based `grep`/`case`, which sees the first line only and
    lets a chat id with a newline on the end through; such a value splits the rendered YAML line
    or breaks the number. Every other value is valid, so only this check can refuse it.
    """
    value = TRAILING_KEYS[key] + suffix
    completed, _ = run_render(built, tmp_path, {**VALID_VALUES, key: value})
    assert completed.returncode != 0, f"{key} with a trailing {label} was accepted"
    assert key in completed.stdout + completed.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["render.sh", "template.yml"]


@pytest.mark.parametrize("key", RENDER_KEYS)
@pytest.mark.parametrize("state", ["unset", "empty"])
def test_render_refuses_a_missing_value(
    built: list[dict[str, Any]], tmp_path: Path, key: str, state: str
) -> None:
    """With one of the four values unset, or set to the empty string, the script exits non-zero,
    names that key and writes no file.

    Catches a script that reads an unset variable as empty and writes `chat_id: ` (a null) or
    `smarthost: ` into the config. Catches a `${VAR:-}` default, and a check that only tests
    "is set" and lets the empty string through. One check per key, so a key nobody validated is
    named by the failing row.
    """
    values = {k: v for k, v in VALID_VALUES.items() if k != key}
    if state == "empty":
        values[key] = ""
    completed, _ = run_render(built, tmp_path, values)
    assert_refused(completed, tmp_path, key, "")


def test_the_secret_values_never_reach_the_pod_env(built: list[dict[str, Any]]) -> None:
    """Only `render-config` reads `alertmanager-secrets` as env; the `alertmanager` container has
    no `env` and no `envFrom` at all, and no container holds a `secretKeyRef` to it. The init
    step is in `initContainers`, Alertmanager in `containers`.

    Catches `envFrom` left on the `alertmanager` container (the old shape), which puts the SMTP
    password and bot token in the running pod's env. Catches the secret read by `env` entries
    with `secretKeyRef` instead of `envFrom`, which a scan of `envFrom` alone misses. Catches the
    init step moved into `containers`, where it would run beside Alertmanager and never before.
    """
    template = pod_templates(built, MONITORING)[ALERTMANAGER]
    init = named_container(template, "initContainers", RENDER_CONFIG)
    running = named_container(template, "containers", ALERTMANAGER)
    assert [entry["secretRef"]["name"] for entry in init.get("envFrom", [])] == [
        ALERTMANAGER_SECRETS
    ]
    assert "envFrom" not in running, f"alertmanager envFrom: {running.get('envFrom')}"
    assert "env" not in running, f"alertmanager env: {running.get('env')}"
    for container in containers_of(template):
        for entry in container.get("env", []):
            source = entry.get("valueFrom", {}).get("secretKeyRef", {})
            assert source.get("name") != ALERTMANAGER_SECRETS, (
                f"{container['name']} reads {ALERTMANAGER_SECRETS} through env {entry}"
            )


def test_alertmanager_is_the_second_and_last_pod_with_internet_egress(
    built: list[dict[str, Any]],
) -> None:
    """Scans every NetworkPolicy in the WHOLE build for an `ipBlock`, exactly as
    `infra/tests/infra/test_k8s_netpol.py`'s
    `test_the_gateway_and_alertmanager_are_the_only_pods_with_a_route_to_the_public_internet` does
    for both pods at once: the set is now `{gateway-egress, alertmanager-egress}` and nothing
    else, and `alertmanager-egress`'s excluded ranges equal `gateway-egress`'s by VALUE (standing
    rule 2 - copied, not re-derived).

    Catches a THIRD pod granted public egress anywhere in the build, which this scan sees
    build-wide rather than by opening `alertmanager-egress` by name. Catches an `except` list
    missing `169.254.0.0/16`, which reopens the `169.254.169.254` metadata route the threat model
    names under SSRF via `base_url` - the same exposure `gateway-egress` closes, now open a
    second way if the two lists are allowed to diverge. Catches a port beside 443, 465 and 587, or
    the rule hung on a `podSelector` that is not Alertmanager's own, which would open the public
    internet to every pod `monitoring` runs rather than to Alertmanager alone.
    """
    reaching = {
        (str(namespace_of(document)), name_of(document))
        for document in policies(built)
        if ip_blocks(document)
    }
    assert reaching == {
        (APPS, "gateway-egress"),
        (MONITORING, "alertmanager-egress"),
    }, f"policies with an ipBlock: {sorted(reaching)}"

    document = policy(built, MONITORING, "alertmanager-egress")
    assert pod_selector(document) == {"matchLabels": {NAME_LABEL: ALERTMANAGER}}, (
        f"alertmanager-egress selects {pod_selector(document)}"
    )

    rules = rules_with_ip_block(document)
    assert len(rules) == 1, f"{len(rules)} egress rules carry an ipBlock, not 1"
    blocks = ip_blocks(document)
    assert len(blocks) == 1, f"{len(blocks)} ipBlocks in alertmanager-egress, not 1"
    assert str(blocks[0].get("cidr")) == PUBLIC_CIDR
    excepted = {str(entry) for entry in blocks[0].get("except", [])}
    assert excepted == PUBLIC_EXCEPT, (
        f"unexpected: {sorted(excepted - PUBLIC_EXCEPT)}; "
        f"missing: {sorted(PUBLIC_EXCEPT - excepted)}"
    )
    ports = numeric_ports(rules[0], f"{MONITORING}/alertmanager-egress")
    assert ports == PUBLIC_PORTS, f"alertmanager's public egress ports are {sorted(ports)}"


def test_alertmanager_egress_allows_smtp_and_https_only(built: list[dict[str, Any]]) -> None:
    """The one `ipBlock` rule of `alertmanager-egress` lists exactly three entries, TCP 443, 465
    and 587, written as numbers: no duplicates, no UDP, no port 25 or 80.

    Catches port 587 opened without 465 (implicit-TLS mail refused) or the reverse. Catches
    plain SMTP on 25 or HTTP on 80 added, which lets the pod send mail in the clear or reach
    plain web. Catches a UDP entry, which `numeric_ports` as a set would hide behind the TCP
    ones; the list is counted here.
    """
    document = policy(built, MONITORING, "alertmanager-egress")
    rules = rules_with_ip_block(document)
    assert len(rules) == 1, f"{len(rules)} egress rules carry an ipBlock, not 1"
    declared = declared_ports(rules[0], "alertmanager-egress")
    assert sorted(declared, key=str) == [("TCP", 443), ("TCP", 465), ("TCP", 587)], declared


def test_prometheus_can_reach_alertmanager_and_cert_manager_and_nothing_new_besides(
    built: list[dict[str, Any]],
) -> None:
    """`prometheus-egress`'s destination set now resolves `(alertmanager, 9093)` against the
    built pod template - the same mechanism its other three destinations already use - and names
    a `namespaceSelector` peer for `cert-manager` with NO `podSelector`, since this build writes
    no pod template there to resolve a named port against; that peer's port is asserted by NUMBER
    (9402) rather than by name, and documented here so a reader does not expect the same
    resolution mechanism for both.

    Catches Prometheus unable to reach Alertmanager at all, which is every alert evaluating and
    firing into a `prometheus-egress` deny - a silent notification outage with no error in either
    pod's logs. Catches the cert-manager peer written WITH a `podSelector`, which
    `matched_workloads` would raise on (no pod template in that namespace) rather than silently
    resolving to nothing - the loud failure this scan exists to force instead of a quiet one.
    Catches the alertmanager and cert-manager destinations grouped into ONE rule together, which
    would cross-multiply into `(alertmanager, 9402)` and `(cert-manager, 9093)` as well - two
    destinations nobody agreed to that plain destination-set equality in
    `infra/tests/infra/test_k8s_monitoring.py`'s `PROMETHEUS_EGRESS_DESTINATIONS` already refuses,
    and which this case additionally pins to the STRUCTURAL cause (one rule, one peer shape) rather
    than only the symptom.
    """
    document = policy(built, MONITORING, "prometheus-egress")
    destinations = egress_destinations(built, document)
    assert (ALERTMANAGER, 9093) in destinations, (
        f"prometheus-egress does not resolve to (alertmanager, 9093): {sorted(destinations)}"
    )
    assert (CERT_MANAGER, 9402) in destinations, (
        f"prometheus-egress does not resolve to (cert-manager, 9402): {sorted(destinations)}"
    )

    cert_manager_peers = [
        peer
        for rule in rules_of(document, "egress")
        for peer in peers_of(rule, "to")
        if peer_namespace(peer, MONITORING) == CERT_MANAGER
    ]
    assert cert_manager_peers, "prometheus-egress names no cert-manager peer"
    with_pod_selector = [peer for peer in cert_manager_peers if peer.get("podSelector") is not None]
    assert with_pod_selector == [], (
        f"a cert-manager peer carries a podSelector, which this build has no pod template to "
        f"resolve it against: {with_pod_selector}"
    )


def test_every_podselector_still_resolves_against_a_pod_template_in_its_namespace(
    built: list[dict[str, Any]],
) -> None:
    """`monitoring` now carries exactly FIVE pod templates - 23a's three, 23b's Grafana, this
    plan's Alertmanager - which is what lets `alertmanager-egress`'s `podSelector` resolve at
    all. The build-wide claim that every `podSelector` matches at least one pod
    (`test_k8s_netpol.py`'s `test_every_selector_in_the_build_matches_a_pod_that_exists`) already
    generalises to this plan with no constant change once the fifth template exists; this case
    additionally pins the NAMED set, so a sixth pod or a renamed one is caught by identity and not
    only by "at least one selector resolves somewhere".

    Catches Alertmanager's Deployment written with `spec.template.metadata.labels` missing
    `app.kubernetes.io/name`, or carrying the wrong value - a pod that starts, is never selected
    by `alertmanager-egress`, and is denied all egress by `default-deny` with no error naming why:
    an allow that allows nothing, the exact failure shape a project rule and this suite's
    selector scan both exist to catch, pinned here to the specific five-workload identity rather
    than only to "at least one resolves".
    """
    templates = pod_templates(built, MONITORING)
    assert set(templates) == MONITORING_WORKLOADS, (
        f"monitoring has pod templates {sorted(templates)}, not {sorted(MONITORING_WORKLOADS)}"
    )
