r"""`infra/k8s/monitoring/`: Grafana, its provisioning pair and its `grafana.` Ingress.

The stream comes from the session-scoped `built` fixture in `infra/tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, shared with
`test_k8s_base.py`, `test_k8s_apps.py`, `test_k8s_platform.py`, `test_k8s_netpol.py` and
`test_k8s_monitoring.py`. Nothing here builds again, and every manifest claim is read off the
BUILT stream rather than off a file, so a manifest written and never listed in a `resources:`
block is the same failure as a manifest never written.

This module OWNS the `monitoring` namespace's object inventory from an earlier change on (its §8 entry 1).
`test_k8s_monitoring.py` keeps the scrape plane's behaviour - the scrape config, the retention
window, the RBAC and the scrape policies - and no longer asserts the namespace's
`(kind, name, namespace)` equality, because 23e, 23c and 23d each add to this namespace and two
files pinning one set is two places to edit for every later object.

Everything is asserted by EQUALITY in both directions, never by containment: an object nobody
agreed to is as dangerous as a missing one, and for the middleware list an EXTRA entry is a
router behaviour nobody reviewed.

Three scoping decisions this file inherits or makes, because the plan's criteria cannot all be
read literally at once:

* Criterion 1 names "23a's eleven plus exactly these nine" while criterion 5 keeps the
  pre-existing `ResourceQuota/monitoring-envelope` and `LimitRange/monitoring-defaults` in that
  same namespace and criterion 7 puts six NetworkPolicies in it. `MONITORING_OBJECTS` is
  therefore the eleven, plus the two envelope objects, plus this plan's nine - twenty-two rows -
  and the policies are excluded from the scan and pinned by `test_k8s_netpol.py` instead.
* `ClusterRole/kube-state-metrics` and `ClusterRoleBinding/kube-state-metrics` are cluster scoped
  and carry no `metadata.namespace`, so they are scoped into the inventory by KIND, exactly as
  an earlier change's module did - every `ClusterRole` and `ClusterRoleBinding` ANYWHERE in the build is
  expected to be one of the two.
* Criterion 1's "no CustomResource" is read as decision A of an earlier change, verbatim: no
  `CustomResourceDefinition` kind at all and no document in the `monitoring.coreos.com` group.
  `Middleware` (`traefik.io/v1alpha1`) is a custom resource the build already carries in `apps`
  and `platform`, and two more of them land in `monitoring` here; they are deliberately NOT
  forbidden, because what decision A forbids is an operator install, which is what would force
  `kubeconform -strict` into the `-ignore-missing-schemas` `test_k8s_base.py` refuses by name.

Metric names are RESOLVED against `packages/opsconsole/opsconsole/core/metrics.py` at run time
and never copied into this file. One spelling trap decides how: probed against
the installed `prometheus_client`, a Counter registered as `gateway_requests_total` reports its
family `name` as `gateway_requests` and exposes the series `gateway_requests_total` and
`gateway_requests_created`, while a Histogram exposes `_bucket`, `_sum`, `_count` and `_created`.
A dashboard `expr` is written in the SERIES spelling, so the accepted set is each family's `name`
expanded by the suffixes that family's TYPE actually exposes - which accepts both
`gateway_requests_total` and the bare family name, and rejects `gateway_request_total`.

The quantity helpers (`memory_bytes`, `cpu_millicores`, `quota_hard`, `resource_quantities`) and
the document helpers are COPIES of `test_k8s_monitoring.py`'s, not imports - the convention of
this suite. CPU is summed in whole millicores, never floats.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from app.core.metrics import registry

#: This file's own location, never the working directory - `pytest` runs from anywhere. Every
#: sibling module in this directory derives it the same way rather than importing it.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The directory the `configMapGenerator` reads (criterion 2, decision A): the dashboards live
#: where kustomize can read them, which is inside the kustomization root.
DASHBOARDS_DIR = REPO_ROOT / "infra" / "k8s" / "monitoring" / "dashboards"

#: The contract templates this repo publishes for Secrets it does NOT supply (criterion 8).
SECRETS_DIR = REPO_ROOT / "infra" / "secrets"

#: The page whose secrets table criterion 8 says the contract's keys must agree with.
K8S_README = REPO_ROOT / "infra" / "k8s" / "README.md"

MONITORING = "monitoring"
GRAFANA = "grafana"
PROMETHEUS = "prometheus"
NODE_EXPORTER = "node-exporter"
KUBE_STATE_METRICS = "kube-state-metrics"
ALERTMANAGER = "alertmanager"

NETWORK_POLICY = "NetworkPolicy"

#: The one label every pod template in this build carries and the only one any selector keys off.
NAME_LABEL = "app.kubernetes.io/name"

#: The two cluster-scoped kinds `monitoring` owns. The inventory scopes them by kind because they
#: carry no `metadata.namespace` to be found by - see the module docstring.
RBAC_KINDS = ("ClusterRole", "ClusterRoleBinding")

#: Criterion 1, as `(kind, name, namespace)`, and the equality this module now owns. The first
#: eleven rows are an earlier change's, the next two are the envelope objects criterion 5 keeps UNCHANGED,
#: the next nine are 23b's, and the last three are an earlier change's Alertmanager (its manifest's own
#: "emits exactly" wording). `None` is what a cluster-scoped object's namespace reads as.
MONITORING_OBJECTS: frozenset[tuple[str, str, str | None]] = frozenset(
    {
        ("Deployment", PROMETHEUS, MONITORING),
        ("Service", PROMETHEUS, MONITORING),
        ("ConfigMap", "prometheus-config", MONITORING),
        ("PersistentVolumeClaim", "prometheus-data", MONITORING),
        ("DaemonSet", NODE_EXPORTER, MONITORING),
        ("Service", NODE_EXPORTER, MONITORING),
        ("Deployment", KUBE_STATE_METRICS, MONITORING),
        ("Service", KUBE_STATE_METRICS, MONITORING),
        ("ServiceAccount", KUBE_STATE_METRICS, MONITORING),
        ("ClusterRole", KUBE_STATE_METRICS, None),
        ("ClusterRoleBinding", KUBE_STATE_METRICS, None),
        ("ResourceQuota", "monitoring-envelope", MONITORING),
        ("LimitRange", "monitoring-defaults", MONITORING),
        ("Deployment", GRAFANA, MONITORING),
        ("Service", GRAFANA, MONITORING),
        ("PersistentVolumeClaim", "grafana-data", MONITORING),
        ("ConfigMap", "grafana-datasources", MONITORING),
        ("ConfigMap", "grafana-dashboard-providers", MONITORING),
        ("ConfigMap", "grafana-dashboards", MONITORING),
        ("Ingress", GRAFANA, MONITORING),
        ("Middleware", "rate-limit", MONITORING),
        ("Middleware", "security-headers", MONITORING),
        # An earlier change's three (its §7 step 5 sweep, this file's own entry).
        ("Deployment", ALERTMANAGER, MONITORING),
        ("Service", ALERTMANAGER, MONITORING),
        ("ConfigMap", "alertmanager-config", MONITORING),
    }
)

#: An earlier change decision A, the half the build can actually contradict: an operator install would
#: bring this kind, and `kubeconform -strict` has no stock schema for what it declares.
FORBIDDEN_KIND = "CustomResourceDefinition"

#: An earlier change decision A's other half: the API group every `kube-prometheus-stack` custom resource
#: lives in. `traefik.io` and `cert-manager.io` are custom resources too and are deliberately NOT
#: forbidden - see the module docstring.
FORBIDDEN_API_GROUP = "monitoring.coreos.com"

#: Criterion 2 and decision E: the generated ConfigMap's name with the hash suffix disabled, and
#: therefore also the prefix every hashed spelling of it would start with.
DASHBOARDS_CONFIGMAP = "grafana-dashboards"

#: Criterion 2 and 23e criterion 1: the six dashboards this plan ships (gateway shipped in 23b,
#: the other five in 23e), and the case compares the ConfigMap's keys against the DIRECTORY as
#: well, both directions, by equality.
DASHBOARD_FILENAMES: frozenset[str] = frozenset(
    {
        "gateway-overview.json",
        "cost.json",
        "residency.json",
        "platform.json",
        "evals.json",
        "helpdesk-agent.json",
    }
)

#: Criterion 3 and criterion 4: the uid the provisioned datasource declares, which is also the
#: uid every panel target has to name. A target naming anything else renders an empty panel with
#: no error anywhere.
DATASOURCE_UID = "prometheus"

#: Criterion 4, each by exact value: the in-cluster Service DNS Grafana proxies to, the access
#: mode that keeps the query server-side (`direct` would have the BROWSER dial a ClusterIP), and
#: the default flag that makes a panel with no explicit datasource still resolve.
DATASOURCE_URL = "http://prometheus.monitoring.svc.cluster.local:9090"
DATASOURCE_ACCESS = "proxy"

#: 23e decision C: the only cluster-infra metric names platform.json's (and the Helpdesk pod
#: panels') kube-state-metrics/node-exporter panels may use, verified against
#: `infra/k8s/monitoring/exporters.yaml`'s RBAC and scrape targets - kept separate from
#: `registered_metric_names()` (which reads only the gateway's own registry) so a typo'd
#: `gateway_` name still fails against the real registry instead of being waved through here.
CLUSTER_INFRA_METRIC_NAMES: frozenset[str] = frozenset(
    {
        # An earlier change: the Helpdesk's own counters, scraped from its agent api and scheduler.
        "helpdesk_turn_endings_total",
        "helpdesk_turns_to_file_bucket",
        "helpdesk_chats_waiting",
        "kube_pod_container_status_restarts_total",
        "kube_pod_status_phase",
        "kube_node_status_condition",
        "kube_node_info",
        "kube_deployment_status_replicas",
        "kube_deployment_status_replicas_available",
        "node_cpu_seconds_total",
        "node_memory_MemAvailable_bytes",
        "node_memory_MemTotal_bytes",
    }
)
DATASOURCE_IS_DEFAULT = True

#: The two provisioning files, as the keys their ConfigMaps hold them under. Grafana reads a
#: DIRECTORY of YAML, so the key name is the file name it lands as.
DATASOURCES_CONFIGMAP = "grafana-datasources"
PROVIDERS_CONFIGMAP = "grafana-dashboard-providers"

#: Criterion 5, decision C, by value. Asserted as the WHOLE flattened mapping, so a `limits.cpu`
#: nobody agreed to is a failure rather than a silent addition (an earlier change decision C: a CPU limit
#: throttles rather than protects).
GRAFANA_RESOURCES: dict[str, str] = {
    "requests.memory": "128Mi",
    "requests.cpu": "100m",
    "limits.memory": "256Mi",
}

#: Criterion 5's totals, DERIVED in the case from the built pod templates rather than restated
#:: 256 + 32 + 64 + 128 + 32 = 512Mi, 150 + 50 + 50 + 100 + 50 = 400m,
#: 512 + 64 + 128 + 256 + 64 = 1024Mi - the fifth term each time is an earlier change's Alertmanager,
#: which fills what this namespace used to leave over.
EXPECTED_REQUEST_MEMORY_BYTES = 512 * 1024**2
EXPECTED_REQUEST_CPU_MILLICORES = 400
EXPECTED_LIMIT_MEMORY_BYTES = 1024 * 1024**2

#: Criterion 5's other half: the envelope is UNCHANGED, so these three are asserted as STRINGS
#: against the built quota - the API server compares quantities, and a test that parsed first
#: would accept `1024Mi` for a table that says `1Gi`.
MONITORING_QUOTA = {
    "requests.memory": "512Mi",
    "requests.cpu": "400m",
    "limits.memory": "1Gi",
}

#: An earlier change's Alertmanager fills what this row used to leave over, to the byte - the reason
#: `EXPECTED_REQUEST_MEMORY_BYTES` and its two siblings above already read the totals AT quota
#: (512Mi/400m/1Gi) rather than the 480Mi/350m/960Mi four pods alone would sum to: 480 + 32 =
#: 512Mi, 350 + 50 = 400m, 960 + 64 = 1024Mi. `HEADROOM_*` is now the remainder left after
#: Alertmanager too - zero on every quantity, an equality and never a `<=`.
HEADROOM_REQUEST_MEMORY_BYTES = 0
HEADROOM_REQUEST_CPU_MILLICORES = 0
HEADROOM_LIMIT_MEMORY_BYTES = 0

#: The five pod templates `monitoring` carries after an earlier change (23a's three, this plan's Grafana,
#: 23c's Alertmanager). Pinned so the totals cannot be reached by a sixth workload that happens
#: to cancel a wrong quantity out.
MONITORING_WORKLOADS: frozenset[str] = frozenset(
    {PROMETHEUS, NODE_EXPORTER, KUBE_STATE_METRICS, GRAFANA, ALERTMANAGER}
)

#: Decision D: Grafana's container port serves the UI the Ingress points at, so it is named
#: `http` and NOT `metrics` - the shared name an earlier change's ingress policy resolves against.
GRAFANA_PORT_NAME = "http"

#: Criterion 6, by value: the one public host, the one path, and the `pathType` that path is
#: worth. `Prefix` on `/` is the whole Grafana app behind one router.
GRAFANA_HOST = "grafana.example.com"
GRAFANA_PATH = "/"
PATH_TYPE_PREFIX = "Prefix"

#: Criterion 6: the Secret cert-manager WRITES from this Ingress's `tls` block. Disjoint from
#: `grafana-admin` by criterion 8 - if the two names ever collided, an issued certificate would
#: overwrite the admin credential.
GRAFANA_TLS_SECRET = "grafana-tls"

#: Criterion 6 and a design note: Traefik reads middlewares off this annotation as a comma-separated
#: list, per Ingress rather than per path.
MIDDLEWARE_ANNOTATION = "traefik.ingress.kubernetes.io/router.middlewares"

#: One entry of that annotation, `<namespace>-<name>@kubernetescrd`. The namespace is matched
#: without hyphens so the split lands after `monitoring`; an entry Traefik cannot resolve is
#: silently NO middleware rather than an error. Copied from `test_k8s_apps.py`.
MIDDLEWARE_REFERENCE = re.compile(
    r"(?P<namespace>[a-z0-9]+)-(?P<name>[a-z0-9][a-z0-9-]*)@kubernetescrd"
)

#: Criterion 6: the parsed annotation, by EQUALITY as `(namespace, name)` pairs. Both resolve to
#: `Middleware`s this build emits in `monitoring`, which the case checks against the stream.
GRAFANA_MIDDLEWARES: frozenset[tuple[str, str]] = frozenset(
    {(MONITORING, "rate-limit"), (MONITORING, "security-headers")}
)

#: Decision B: the middleware the `grafana.` host is the one public host NOT to carry. A 401 in
#: front of `/api/annotations` would break the release annotation `deploy.yml` POSTs with a
#: Bearer token for a credential Grafana is already asking for.
FORBIDDEN_MIDDLEWARE = "basic-auth"

#: Criterion 8: the fifth Secret this repo contracts for and does not supply, its template, and
#: the two keys it carries. The keys are ALSO compared against the README secrets table, which is
#: the comparison `test_secrets.py` keeps for the other four.
ADMIN_SECRET = "grafana-admin"
ADMIN_EXAMPLE_FILENAME = f"{ADMIN_SECRET}.example.yaml"
ADMIN_KEYS: frozenset[str] = frozenset({"GF_SECURITY_ADMIN_USER", "GF_SECURITY_ADMIN_PASSWORD"})

#: Criterion 8: the literal placeholder every value in a contract template carries, the same one
#: `test_secrets.py` pins for the other four - greppable, and never read as a credential.
PLACEHOLDER = "REPLACE_ME"

#: Criterion 9: the variable family that turns the login page off entirely, and the one flag that
#: hides the login form. Asserted by ABSENCE over the whole `env` list AND the contract's key set,
#: so the case cannot pass because the variable moved from one to the other.
ANONYMOUS_PREFIX = "GF_AUTH_ANONYMOUS"
LOGIN_FORM_VARIABLE = "GF_AUTH_DISABLE_LOGIN_FORM"

#: The header the secrets table in `infra/k8s/README.md` carries, cell for cell. Copied from
#: `test_secrets.py`: the table is found by this whole row and never by the first line mentioning
#: "Secret", because the page carries several tables and a lot of prose.
README_HEADER_CELLS = ["Secret", "namespace", "keys", "read by"]

#: A backticked token in a README cell, which is what turns the `keys` column into a set.
BACKTICKED = re.compile(r"`([^`]+)`")

#: Criterion 7: the two policies this plan adds, and what each is allowed to do. `grafana-ingress`
#: admits k3s's own namespace, where Traefik runs: the browser's packet reaches Grafana from the
#: ingress controller's pod, never from `monitoring`.
GRAFANA_INGRESS_POLICY = "grafana-ingress"
GRAFANA_EGRESS_POLICY = "grafana-egress"
KUBE_SYSTEM = "kube-system"

#: Criterion 7: every destination `grafana-egress` may reach, as `(workload, resolved port)`.
#: Prometheus alone, on the number 23a pinned - the DNS allow is NOT here, it lives in
#: `allow-dns`, which selects every pod in `monitoring`.
GRAFANA_EGRESS_DESTINATIONS: frozenset[tuple[str, int]] = frozenset({(PROMETHEUS, 9090)})

#: Kubernetes auto-labels every namespace with its own name, which is how a cross-namespace peer
#: names the namespace it means.
NAMESPACE_NAME_LABEL = "kubernetes.io/metadata.name"

#: The peer marker for "every pod in that namespace": a peer with no `podSelector`, or an empty
#: one. `grafana-ingress`'s `kube-system` peer is that shape - this build writes no pod template
#: there, so there is nothing to name.
ANY_POD = "*"

#: Binary and decimal suffixes a memory quantity may carry. Copied from `test_k8s_monitoring.py`
#: with the helper that reads it.
MEMORY_SUFFIXES = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "k": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
}

#: The series suffixes each `prometheus_client` family TYPE exposes beside its own name, probed
#: against the installed version rather than taken on trust. A family type
#: absent from this mapping asserts rather than silently accepting every spelling.
FAMILY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "counter": ("_total", "_created"),
    "gauge": (),
    "histogram": ("_bucket", "_sum", "_count", "_created"),
    "summary": ("_sum", "_count", "_created"),
}

#: A PromQL identifier that is NOT preceded by a word character, so the `m` of `5m` inside a
#: range selector is never read as a metric name, and NOT followed by one, which is what stops
#: the engine backtracking `sum(` into the identifier `su` when the function lookahead refuses
#: the whole word (a project rule found exactly that; it read `rat` and `su` as metrics).
PROMQL_IDENTIFIER = re.compile(
    r"(?<![0-9A-Za-z_:])[A-Za-z_:][A-Za-z0-9_:]*(?![0-9A-Za-z_:])(?!\s*\()"
)

#: PromQL's own words, which survive the grouping-clause strip and are not metric names.
PROMQL_KEYWORDS = frozenset({"and", "or", "unless", "bool", "offset", "start", "end", "inf", "nan"})

#: The grouping clauses whose parenthesised contents are LABEL names, stripped whole before any
#: identifier is read - `sum by (route)` names a label, not a family.
PROMQL_GROUPING = re.compile(
    r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)", re.IGNORECASE
)

#: String literals inside an `expr`, stripped first so a label VALUE cannot be read as a family.
PROMQL_STRING = re.compile(r"\"[^\"]*\"|'[^']*'")

#: Label selectors, stripped before identifiers are read: `le`, `route` and `tenant` live here.
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


def annotations_of(document: dict[str, Any]) -> dict[str, str]:
    """`metadata.annotations`, values as written, empty when the object declares none."""
    metadata = document.get("metadata", {})
    assert isinstance(metadata, dict)
    annotations = metadata.get("annotations", {})
    if not isinstance(annotations, dict):
        return {}
    return {str(key): str(value) for key, value in annotations.items()}


def api_group(document: dict[str, Any]) -> str:
    """The group half of `apiVersion`: `apps/v1` is `apps`, a bare `v1` is the core group."""
    api_version = str(document.get("apiVersion", ""))
    return api_version.split("/")[0] if "/" in api_version else ""


def objects_of_kind(documents: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Every built document of `kind`, in build order."""
    return [document for document in documents if document.get("kind") == kind]


def object_named(documents: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """The one `kind` named `name`, asserting the build emits exactly one.

    Fails with a message listing what the build DID emit of that kind, so a module run before
    the manifests land is red on the missing object and not on an IndexError.
    """
    found = [document for document in objects_of_kind(documents, kind) if name_of(document) == name]
    emitted = sorted(name_of(document) for document in objects_of_kind(documents, kind))
    assert len(found) == 1, f"expected one {kind}/{name}; the build has {kind}s {emitted}"
    return found[0]


def pod_owners(documents: list[dict[str, Any]], namespace: str) -> list[dict[str, Any]]:
    """Every document in `namespace` carrying a pod template, found by the PRESENCE of
    `spec.template.spec` rather than by a list of kinds - so the `node-exporter` DaemonSet is
    picked up by the same scan as the three Deployments, with no kind list to keep in step."""
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
    """Every container of a pod template, `initContainers` included: an init container with no
    memory limit is admitted against the same quota as an app container."""
    spec = template["spec"]
    assert isinstance(spec, dict)
    collected: list[dict[str, Any]] = []
    for key in ("initContainers", "containers"):
        listed = spec.get(key, [])
        assert isinstance(listed, list), f"a pod template's {key} is not a list"
        collected.extend(container for container in listed if isinstance(container, dict))
    return collected


def sole_container(template: dict[str, Any], label: str) -> dict[str, Any]:
    """The one container of a pod template, asserting there is exactly one - the budget table
    and criterion 5 both have one row per workload."""
    containers = containers_of(template)
    assert len(containers) == 1, f"{label} declares {len(containers)} containers, not 1"
    return containers[0]


def port_entries(template: dict[str, Any]) -> list[tuple[str | None, int]]:
    """Every container port of a pod template as `(name or None, number)`, across every
    container. The name is kept as None rather than skipped when a port declares none, so an
    UNNAMED port is visible to the case that asserts Grafana's port is named."""
    entries: list[tuple[str | None, int]] = []
    for container in containers_of(template):
        listed = container.get("ports", [])
        assert isinstance(listed, list), (
            f"{container.get('name')} has a ports value that is not a list"
        )
        for entry in listed:
            assert isinstance(entry, dict), "a container port entry is not a mapping"
            declared = entry.get("name")
            entries.append(
                (
                    None if declared is None else str(declared),
                    int(entry["containerPort"]),
                )
            )
    return entries


def container_ports(template: dict[str, Any]) -> dict[str, int]:
    """A pod template's NAMED container ports, `name -> number`."""
    return {name: number for name, number in port_entries(template) if name is not None}


def resource_quantities(container: dict[str, Any]) -> dict[str, str]:
    """A container's `resources` flattened to the `requests.memory`-style keys the budget table
    and the quota both use, values exactly as written."""
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


def pods_of(owner: dict[str, Any]) -> int:
    """How many pods a pod-owning object costs the quota. `spec.replicas` when it declares one,
    otherwise 1 - a DaemonSet declares none and runs one pod per node, and this cluster is the
    one-node CAX21 the envelope was sized for."""
    declared = owner["spec"].get("replicas", 1)
    return int(declared)


def memory_bytes(quantity: str) -> int:
    """`512Mi` as an integer count of bytes. Raises on a quantity this repo does not write."""
    for suffix, multiplier in MEMORY_SUFFIXES.items():
        if quantity.endswith(suffix):
            return int(quantity[: -len(suffix)]) * multiplier
    return int(quantity)


def cpu_millicores(quantity: str) -> int:
    """`100m` as 100, `2` as 2000. Whole millicores, never floats."""
    if quantity.endswith("m"):
        return int(quantity[:-1])
    return round(float(quantity) * 1000)


def quota_hard(quota: dict[str, Any]) -> dict[str, str]:
    """A ResourceQuota's `spec.hard` mapping, values as written."""
    spec = quota.get("spec", {})
    assert isinstance(spec, dict), f"{name_of(quota)} has no spec"
    hard = spec.get("hard", {})
    assert isinstance(hard, dict), f"{name_of(quota)} has no spec.hard"
    return {str(key): str(value) for key, value in hard.items()}


def config_map_data(documents: list[dict[str, Any]], name: str) -> dict[str, str]:
    """One ConfigMap's `data` mapping, values as the strings a ConfigMap actually holds."""
    document = object_named(documents, "ConfigMap", name)
    data = document.get("data", {})
    assert isinstance(data, dict), f"{name} declares no data mapping"
    return {str(key): str(value) for key, value in data.items()}


def parsed_yaml_value(text: str, label: str) -> dict[str, Any]:
    """One ConfigMap value parsed as YAML, asserting it is a mapping.

    A provisioning file is held as a STRING, so a datasource with a typo is valid YAML inside a
    valid ConfigMap; parsing here is what makes its contents assertable at all.
    """
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict), f"{label} is not a YAML mapping"
    return parsed


def grafana_template(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Grafana's built pod template, found through the namespace's templates by its own
    `app.kubernetes.io/name` label rather than by Deployment name, so a Deployment whose template
    carries the wrong label is a named failure and not a silent miss."""
    templates = pod_templates(documents, MONITORING)
    assert GRAFANA in templates, f"monitoring has pod templates {sorted(templates)}"
    return templates[GRAFANA]


def volume_named(template: dict[str, Any], name: str) -> dict[str, Any]:
    """The pod template's `volumes` entry a mount refers to by name - a mount naming a volume the
    template does not declare is a pod the API server refuses, and a mount naming the WRONG
    volume is a provisioning directory backed by something else entirely."""
    volumes = [
        volume
        for volume in template["spec"].get("volumes", [])
        if isinstance(volume, dict) and str(volume.get("name")) == name
    ]
    assert len(volumes) == 1, f"{len(volumes)} volumes named {name} on the grafana pod template"
    return volumes[0]


def mount_path_of_config_map(template: dict[str, Any], config_map: str) -> str:
    """The `mountPath` the volume backed by `config_map` is mounted at, asserting exactly one
    volume names it and exactly one mount names that volume.

    Resolved through the volume rather than matched on the path, because criterion 4 compares the
    provider's `options.path` against THIS value: a case that read the path off a literal would
    be green with the ConfigMap mounted somewhere else entirely.
    """
    named = [
        volume
        for volume in template["spec"].get("volumes", [])
        if isinstance(volume, dict)
        and isinstance(volume.get("configMap"), dict)
        and str(volume["configMap"].get("name")) == config_map
    ]
    assert len(named) == 1, (
        f"{len(named)} volumes are backed by ConfigMap/{config_map}: "
        f"{template['spec'].get('volumes')}"
    )
    volume_name = str(named[0]["name"])
    assert volume_named(template, volume_name) == named[0]
    mounts = [
        mount
        for container in containers_of(template)
        for mount in container.get("volumeMounts", [])
        if isinstance(mount, dict) and str(mount.get("name")) == volume_name
    ]
    assert len(mounts) == 1, f"{len(mounts)} mounts name volume {volume_name}"
    path = str(mounts[0].get("mountPath", ""))
    assert path, f"the mount of {volume_name} declares no mountPath"
    return path


def middleware_names(ingress: dict[str, Any]) -> set[tuple[str, str]]:
    """The `(namespace, name)` pairs of an Ingress's middleware annotation, parsed out of the
    comma-separated `<namespace>-<name>@kubernetescrd` list Traefik reads. Copied from
    `test_k8s_apps.py`, which parses the `apps` Ingresses the same way."""
    annotation = annotations_of(ingress).get(MIDDLEWARE_ANNOTATION, "")
    assert annotation, f"{name_of(ingress)} carries no {MIDDLEWARE_ANNOTATION}"
    parsed: set[tuple[str, str]] = set()
    for entry in annotation.split(","):
        match = MIDDLEWARE_REFERENCE.fullmatch(entry.strip())
        assert match is not None, (
            f"{name_of(ingress)} names `{entry.strip()}`, which is not a "
            "`<namespace>-<name>@kubernetescrd` reference Traefik can resolve"
        )
        parsed.add((match.group("namespace"), match.group("name")))
    return parsed


def resolved_container_port(target: object, container: dict[str, Any]) -> int | None:
    """The `containerPort` NUMBER a `targetPort` or a backend port resolves to, in both forms it
    takes - an int against the declared numbers, a string against the declared names - or None
    when the container declares no such port. Copied from `test_k8s_apps.py`."""
    declared = [port for port in container.get("ports", []) if isinstance(port, dict)]
    if isinstance(target, str) and not target.isdigit():
        for port in declared:
            if str(port.get("name")) == target:
                return int(port["containerPort"])
        return None
    number = int(str(target))
    return number if any(int(port["containerPort"]) == number for port in declared) else None


def service_target_port(service: dict[str, Any], backend_port: dict[str, Any]) -> object:
    """The `targetPort` of the one Service port an Ingress backend names, by name or by number.

    The chain matters: an Ingress names a SERVICE port, and the Service maps that to a
    `targetPort` on the pod. Resolving only the first hop would be green for a Service whose
    `targetPort` points at nothing.
    """
    ports = service["spec"].get("ports", [])
    assert isinstance(ports, list) and ports, f"Service/{name_of(service)} declares no ports"
    wanted_name = backend_port.get("name")
    wanted_number = backend_port.get("number")
    assert (wanted_name is None) != (wanted_number is None), (
        f"an Ingress backend names port {backend_port}, which is neither a name nor a number"
    )
    matched = [
        entry
        for entry in ports
        if isinstance(entry, dict)
        and (
            str(entry.get("name")) == str(wanted_name)
            if wanted_name is not None
            else int(entry["port"]) == int(str(wanted_number))
        )
    ]
    assert len(matched) == 1, (
        f"Service/{name_of(service)} has {len(matched)} ports matching {backend_port}: {ports}"
    )
    return matched[0].get("targetPort", matched[0]["port"])


def dashboard_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Every dashboard the generated ConfigMap carries, `key -> parsed JSON`.

    Parsed here rather than in each case because a dashboard is held as a STRING: a trailing
    comma or a truncated file is a valid ConfigMap value and an unrenderable dashboard.
    """
    parsed: dict[str, Any] = {}
    for key, text in config_map_data(documents, DASHBOARDS_CONFIGMAP).items():
        try:
            parsed[key] = json.loads(text)
        except json.JSONDecodeError as error:  # pragma: no cover - the message is the assertion
            raise AssertionError(f"{key} in {DASHBOARDS_CONFIGMAP} is not JSON: {error}") from error
    return parsed


def panels_of(dashboard: object) -> list[dict[str, Any]]:
    """Every panel of a parsed dashboard, nested `panels` included.

    A collapsed row carries its panels under its OWN `panels` key, so a flat read of the
    top-level list would skip every panel inside a collapsed row and report a clean dashboard.
    """
    found: list[dict[str, Any]] = []
    if isinstance(dashboard, dict):
        listed = dashboard.get("panels", [])
        if isinstance(listed, list):
            for panel in listed:
                if isinstance(panel, dict):
                    found.append(panel)
                    found.extend(panels_of(panel))
    return found


def datasource_uid_of(value: object) -> str:
    """The uid a `datasource` field names, in both shapes Grafana writes it: the modern mapping
    `{"type": "prometheus", "uid": "prometheus"}` and the legacy bare string."""
    if isinstance(value, dict):
        return str(value.get("uid", ""))
    return str(value)


def metric_names_in(expr: str) -> set[str]:
    r"""Every metric family name a PromQL `expr` reads.

    String literals, `{...}` label selectors and `by (...)`/`without (...)` grouping clauses are
    stripped first, because each of them carries identifiers that are LABELS and not families;
    what is left is every identifier not followed by `(` (those are functions) and not one of
    PromQL's own words. The `(?<![0-9A-Za-z_:])` lookbehind is what keeps the `m` of `5m` out.
    """
    stripped = PROMQL_STRING.sub(" ", expr)
    stripped = PROMQL_GROUPING.sub(" ", stripped)
    stripped = PROMQL_SELECTOR.sub(" ", stripped)
    return {
        token
        for token in PROMQL_IDENTIFIER.findall(stripped)
        if token.lower() not in PROMQL_KEYWORDS
    }


def registered_metric_names() -> set[str]:
    """Every spelling a dashboard `expr` may legally use, resolved from
    `packages/opsconsole/opsconsole/core/metrics.py` AT RUN TIME.

    Each family's `name` plus the suffixes its TYPE exposes - see the module docstring for the
    probe: `prometheus_client` reports a Counter's family as `gateway_requests` while the series
    reads `gateway_requests_total`, so both are accepted and `gateway_request_total` is not.
    """
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


def read(path: Path) -> str:
    """Read a file the plan owes, failing the one case that needs it when it is absent."""
    assert path.is_file(), f"{path} does not exist"
    return path.read_text(encoding="utf-8")


def table_cells(line: str) -> list[str]:
    """One markdown table row as stripped cells, or `[]` when the line is not a table row."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.split("|")[1:-1]]


def readme_secret_keys(secret: str) -> frozenset[str]:
    """The backticked keys the `infra/k8s/README.md` secrets table gives one Secret.

    The table is located by its whole HEADER ROW, and every line of the page is scanned for it
    rather than the first line mentioning "Secret" being taken: the page
    carries several tables and a lot of prose about Secrets.
    """
    lines = read(K8S_README).splitlines()
    headers = [
        index for index, line in enumerate(lines) if table_cells(line) == README_HEADER_CELLS
    ]
    assert len(headers) == 1, (
        f"{len(headers)} rows in {K8S_README} read {' | '.join(README_HEADER_CELLS)}"
    )
    rows: list[list[str]] = []
    for line in lines[headers[0] + 1 :]:
        cells = table_cells(line)
        if not cells:
            break
        if all(cell and set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    matched = [cells for cells in rows if BACKTICKED.findall(cells[0])[:1] == [secret]]
    assert len(matched) == 1, (
        f"the secrets table has {len(matched)} rows for `{secret}`; it names "
        f"{[BACKTICKED.findall(cells[0])[:1] for cells in rows]}"
    )
    return frozenset(BACKTICKED.findall(matched[0][2]))


def contract_documents() -> list[dict[str, Any]]:
    """Every YAML document in `infra/secrets/grafana-admin.example.yaml`."""
    return [
        document
        for document in yaml.safe_load_all(read(SECRETS_DIR / ADMIN_EXAMPLE_FILENAME))
        if isinstance(document, dict) and document
    ]


def contract_keys() -> frozenset[str]:
    """The key set the `grafana-admin` template contracts for, read off the one document it
    holds, across `data` and `stringData` alike."""
    documents = contract_documents()
    assert len(documents) == 1, f"{ADMIN_EXAMPLE_FILENAME} holds {len(documents)} documents, not 1"
    keys: set[str] = set()
    for mapping_key in ("data", "stringData"):
        mapping = documents[0].get(mapping_key)
        if mapping is None:
            continue
        assert isinstance(mapping, dict), f"{ADMIN_EXAMPLE_FILENAME}'s `{mapping_key}` is not a map"
        keys.update(str(key) for key in mapping)
    return frozenset(keys)


def env_names(container: dict[str, Any]) -> set[str]:
    """Every `env[*].name` a container declares."""
    listed = container.get("env", [])
    assert isinstance(listed, list), "the grafana container's env is not a list"
    names: set[str] = set()
    for entry in listed:
        assert isinstance(entry, dict), "an env entry is not a mapping"
        names.add(str(entry.get("name", "")))
    return names


def envfrom_secrets(container: dict[str, Any]) -> set[str]:
    """Every Secret a container pulls its whole environment from, by name."""
    listed = container.get("envFrom", [])
    assert isinstance(listed, list), "the grafana container's envFrom is not a list"
    found: set[str] = set()
    for entry in listed:
        assert isinstance(entry, dict), "an envFrom entry is not a mapping"
        reference = entry.get("secretRef")
        if isinstance(reference, dict):
            found.add(str(reference.get("name")))
    return found


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
    """`spec`, asserting it is a mapping - an empty policy spec is a policy with no effect."""
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
    destination, which is never what either policy here intends, so the caller asserts on it."""
    listed = rule.get(key, [])
    assert isinstance(listed, list), f"a rule's `{key}` is not a list"
    return [peer for peer in listed if isinstance(peer, dict)]


def selector_matches(selector: dict[str, Any], labels: dict[str, str]) -> bool:
    """Whether a `podSelector` matches a pod template's labels. Understands `matchLabels` and
    `matchExpressions` with `In`, the only two forms this manifest uses; any other operator
    asserts rather than silently reporting a match."""
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
    `podSelector` or an empty one - the shape `grafana-ingress`'s `kube-system` peer has, since
    this build writes no pod template in k3s's own namespace."""
    namespace = peer_namespace(peer, default_namespace)
    selector = peer.get("podSelector")
    if not isinstance(selector, dict) or selector == {}:
        return {(namespace, ANY_POD)}
    workloads = matched_workloads(documents, selector, namespace)
    assert workloads, f"podSelector {selector} matches no pod template in {namespace}"
    return {(namespace, workload) for workload in workloads}


def declared_ports(rule: dict[str, Any], label: str) -> list[tuple[str, object]]:
    """A rule's `(protocol, port-as-written)` entries, the port still a name OR a number. A rule
    with no `ports` key allows EVERY port, which neither policy here intends, so it asserts."""
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
    """A rule's ports resolved against ONE pod's named container ports: a number passes through,
    a name resolves to that pod's own number, which is what Kubernetes does per target pod."""
    resolved: set[tuple[str, int]] = set()
    for protocol, port in declared_ports(rule, label):
        if isinstance(port, int):
            resolved.add((protocol, port))
            continue
        number = ports.get(str(port))
        assert number is not None, f"{label} names port {port!r}; that pod has {sorted(ports)}"
        resolved.add((protocol, number))
    return resolved


def resolved_ports(
    documents: list[dict[str, Any]], document: dict[str, Any], rule: dict[str, Any]
) -> dict[str, set[tuple[str, int]]]:
    """An INGRESS rule's ports resolved per pod the policy's own `podSelector` protects,
    `{workload: {(protocol, number)}}` - which pins the SCOPE as well as the ports, because a
    policy that also selected Prometheus would show that pod as a second key."""
    namespace = namespace_of(document)
    assert namespace is not None, f"{name_of(document)} declares no metadata.namespace"
    selector = pod_selector(document)
    label = f"{namespace}/{name_of(document)}"
    return {
        workload: ports_against(rule, container_ports(template), label)
        for workload, template in pod_templates(documents, namespace).items()
        if selector_matches(selector, template_labels(template))
    }


def egress_destinations(
    documents: list[dict[str, Any]], document: dict[str, Any]
) -> set[tuple[str, int]]:
    """An egress policy's destinations as `(workload-or-namespace, resolved port)`.

    Every peer of a rule is crossed with every port of that rule, which is what the rule means:
    grouping two targets and two numbers into one rule allows all four, and this set says so
    rather than flattering the manifest into the pairs the criterion names.
    """
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


def test_the_monitoring_namespace_emits_grafana_and_nothing_else_new(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 1, by `(kind, name, namespace)` EQUALITY in both directions over everything the
    build files into `monitoring`, plus every `ClusterRole` and `ClusterRoleBinding` anywhere in
    the build - the twenty-two rows of `MONITORING_OBJECTS`, which this module owns from an earlier change
    on (its §8 entry 1). The CRD and `monitoring.coreos.com` scoping is an earlier change's, unchanged.

    Catches the commonest failure of a new manifest: `grafana.yaml`, `middlewares.yaml`,
    `ingress.yaml` or the `configMapGenerator` block written correctly and never added to
    `monitoring/kustomization.yaml` - valid YAML that reaches no cluster and that no schema check
    can see, and which here would be a Grafana with no datasource, or an Ingress pointing at
    middlewares Traefik cannot resolve and therefore silently does not apply. Catches an object
    that forgets its own `metadata.namespace`, which the API server would file into whatever
    namespace the apply ran against. Catches the generated dashboards ConfigMap emitted under a
    hashed name, which is decision E reversed and which no other inventory in this suite sees.
    Catches a third `Middleware` copied into `monitoring` - a `basic-auth` nobody references is
    a document that exists to be pasted into the next Ingress by mistake (§3). Catches an
    operator install sneaking in through `helmCharts:`, which would bring a
    CustomResourceDefinition and `monitoring.coreos.com` documents that force `kubeconform
    -strict` into the `-ignore-missing-schemas` infra/tests/infra/test_k8s_base.py refuses by name.
    """
    emitted = {
        (str(document.get("kind")), name_of(document), namespace_of(document))
        for document in built
        if (namespace_of(document) == MONITORING or document.get("kind") in RBAC_KINDS)
        # The `monitoring` NetworkPolicies are in this namespace too, and they are pinned by
        # equality in both directions - `(namespace, name)` and per namespace - by
        # infra/tests/infra/test_k8s_netpol.py's
        # test_the_whole_build_emits_exactly_the_eighteen_policies_and_nowhere_else and
        # test_each_namespace_emits_exactly_the_policies_the_plan_names. Listing them here as
        # well would be two files pinning one set, which drift apart.
        and document.get("kind") != NETWORK_POLICY
    }
    assert emitted == MONITORING_OBJECTS, (
        f"unexpected: {sorted(map(str, emitted - MONITORING_OBJECTS))}; "
        f"missing: {sorted(map(str, MONITORING_OBJECTS - emitted))}"
    )

    definitions = sorted(name_of(document) for document in objects_of_kind(built, FORBIDDEN_KIND))
    assert definitions == [], f"the build emits {FORBIDDEN_KIND}s: {definitions}"

    operator_documents = sorted(
        f"{document.get('kind')}/{name_of(document)}"
        for document in built
        if api_group(document) == FORBIDDEN_API_GROUP
    )
    assert operator_documents == [], (
        f"the build emits {FORBIDDEN_API_GROUP} documents: {operator_documents}"
    )


def test_the_dashboards_config_map_carries_exactly_the_files_in_the_directory(
    built: list[dict[str, Any]],
) -> None:
    r"""Criterion 2 and decision E: the generated ConfigMap's emitted name is exactly
    `grafana-dashboards` with no hash suffix, its key set equals the `*.json` file names in
    `infra/k8s/monitoring/dashboards/`, that directory holds exactly the one dashboard this plan
    ships, and every value parses as JSON.

    Catches `generatorOptions.disableNameSuffixHash` omitted: the ConfigMap would be emitted as
    `grafana-dashboards-<hash>`, which builds and applies perfectly and which every later plan's
    inventory would have to chase - the name half is asserted as "the ConfigMaps in `monitoring`
    whose name starts with `grafana-dashboards` are exactly the one", so a hashed spelling is a
    readable red rather than a missing object. Catches a dashboard JSON added to the directory
    and the `files:` entry never widened (or the reverse): the file provider polls the mounted
    directory, so a dashboard that never reaches the ConfigMap simply never appears in the UI,
    with nothing in any log. Catches a dashboard exported from Grafana's UI with a trailing comma
    or truncated on copy - Grafana logs the parse error to its own stdout and serves every other
    dashboard, which is a panel set that is silently one short.
    """
    generated = sorted(
        name_of(document)
        for document in objects_of_kind(built, "ConfigMap")
        if namespace_of(document) == MONITORING
        and name_of(document).startswith(DASHBOARDS_CONFIGMAP)
    )
    assert generated == [DASHBOARDS_CONFIGMAP], (
        f"monitoring emits ConfigMaps {generated}, not exactly [{DASHBOARDS_CONFIGMAP!r}] - a "
        "name with a suffix is disableNameSuffixHash missing"
    )

    assert DASHBOARDS_DIR.is_dir(), f"{DASHBOARDS_DIR} does not exist"
    on_disk = {path.name for path in DASHBOARDS_DIR.glob("*.json")}
    assert on_disk == DASHBOARD_FILENAMES, (
        f"unexpected: {sorted(on_disk - DASHBOARD_FILENAMES)}; "
        f"missing: {sorted(DASHBOARD_FILENAMES - on_disk)}"
    )

    parsed = dashboard_documents(built)
    assert set(parsed) == on_disk, (
        f"the ConfigMap holds {sorted(parsed)}; the directory holds {sorted(on_disk)}"
    )
    for key, document in parsed.items():
        assert isinstance(document, dict), (
            f"{key} parses as {type(document).__name__}, not an object"
        )


def test_every_dashboard_target_names_the_provisioned_datasource_and_a_registered_family(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 3: every panel target in every dashboard key names the datasource uid
    `prometheus`, and every metric name in every `expr` is either one
    `packages/opsconsole/opsconsole/core/metrics.py` registers - resolved from that module at run
    time and never from a list copied here - or one of the nine cluster-infra
    names in `CLUSTER_INFRA_METRIC_NAMES` (23e decision C).

    The accepted spellings are each family's `name` expanded by the series suffixes its TYPE
    exposes, probed against the installed `prometheus_client`: a Counter registered as
    `gateway_requests_total` reports the family `gateway_requests`, so both spellings are legal in
    an `expr` and `gateway_request_total` is neither.

    Catches a panel whose `expr` names a family the gateway never registers - a typo, or a metric
    copied from a tutorial - which renders an EMPTY panel forever with no build error, no schema
    error and no log line anywhere (§5's second risk). Catches a target left on Grafana's
    `${DS_PROMETHEUS}` export placeholder or on a uid from the Grafana instance the JSON was
    exported from, which is a panel querying a datasource that does not exist. Catches a
    dashboard whose panels are nested inside a collapsed row and whose targets a flat read would
    skip, since `panels_of` recurses. Asserts at least one target was seen, so a dashboard whose
    panels never parse into targets cannot pass this case by being empty.
    """
    accepted = registered_metric_names()
    seen = 0
    for key, dashboard in dashboard_documents(built).items():
        for panel in panels_of(dashboard):
            for target in panel.get("targets", []):
                assert isinstance(target, dict), f"{key} has a target that is not a mapping"
                seen += 1
                uid = datasource_uid_of(target.get("datasource"))
                assert uid == DATASOURCE_UID, (
                    f"{key}/{panel.get('title')!r} queries datasource uid {uid!r}, "
                    f"not {DATASOURCE_UID!r}"
                )
                expr = str(target.get("expr", ""))
                assert expr, f"{key}/{panel.get('title')!r} has a target with no expr"
                unknown = sorted(metric_names_in(expr) - accepted - CLUSTER_INFRA_METRIC_NAMES)
                assert unknown == [], (
                    f"{key}/{panel.get('title')!r} reads {unknown}, which is neither "
                    "registered by opsconsole.core.metrics nor a cluster-infra name"
                )
    assert seen, "no dashboard panel declares a target; this case asserted nothing"


def test_the_provisioning_files_point_at_prometheus_and_at_the_mounted_directory(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 4: `grafana-datasources` declares exactly ONE datasource, with the in-cluster
    Prometheus url, `access: proxy`, uid `prometheus` and `isDefault: true`; and
    `grafana-dashboard-providers` declares exactly ONE provider whose `options.path` EQUALS the
    `mountPath` of the volume the dashboards ConfigMap is mounted at - one equality against the
    built pod template, not two literals that can agree while both are wrong.

    Catches a second datasource pasted in beside the first, which makes `isDefault` ambiguous and
    a panel with no explicit datasource land on whichever Grafana provisions last. Catches
    `access: direct`, which has the BROWSER dial `prometheus.monitoring.svc.cluster.local` - a
    name that resolves nowhere outside the cluster, so every panel fails with a CORS or DNS error
    on a dashboard that renders fine from inside. Catches a url pointed at a Service name or
    namespace this build does not emit, or at the wrong port. Catches the provider's path left on
    Grafana's documented `/var/lib/grafana/dashboards` while the ConfigMap is mounted at
    `/etc/grafana/dashboards`: nothing fails, the directory is simply empty and no dashboard ever
    appears. Catches the dashboards ConfigMap mounted at a path no container reads, since the
    mount is resolved THROUGH the volume backed by that ConfigMap and not matched on the path.
    """
    datasources_data = config_map_data(built, DATASOURCES_CONFIGMAP)
    assert len(datasources_data) == 1, f"{DATASOURCES_CONFIGMAP} holds {sorted(datasources_data)}"
    datasources_file = parsed_yaml_value(
        next(iter(datasources_data.values())), DATASOURCES_CONFIGMAP
    )
    declared = datasources_file.get("datasources", [])
    assert isinstance(declared, list) and len(declared) == 1, (
        f"{DATASOURCES_CONFIGMAP} declares {declared}, not exactly one datasource"
    )
    datasource = declared[0]
    assert isinstance(datasource, dict), "the datasource entry is not a mapping"
    assert str(datasource.get("url")) == DATASOURCE_URL, (
        f"the datasource url is {datasource.get('url')!r}, not {DATASOURCE_URL!r}"
    )
    assert str(datasource.get("access")) == DATASOURCE_ACCESS, (
        f"the datasource access is {datasource.get('access')!r}, not {DATASOURCE_ACCESS!r}"
    )
    assert str(datasource.get("uid")) == DATASOURCE_UID, (
        f"the datasource uid is {datasource.get('uid')!r}, not {DATASOURCE_UID!r}"
    )
    assert datasource.get("isDefault") is DATASOURCE_IS_DEFAULT, (
        f"the datasource isDefault is {datasource.get('isDefault')!r}, not {DATASOURCE_IS_DEFAULT}"
    )

    providers_data = config_map_data(built, PROVIDERS_CONFIGMAP)
    assert len(providers_data) == 1, f"{PROVIDERS_CONFIGMAP} holds {sorted(providers_data)}"
    providers_file = parsed_yaml_value(next(iter(providers_data.values())), PROVIDERS_CONFIGMAP)
    providers = providers_file.get("providers", [])
    assert isinstance(providers, list) and len(providers) == 1, (
        f"{PROVIDERS_CONFIGMAP} declares {providers}, not exactly one provider"
    )
    provider = providers[0]
    assert isinstance(provider, dict), "the provider entry is not a mapping"
    options = provider.get("options", {})
    assert isinstance(options, dict), f"the provider declares options {options!r}"
    mounted_at = mount_path_of_config_map(grafana_template(built), DASHBOARDS_CONFIGMAP)
    assert str(options.get("path")) == mounted_at, (
        f"the provider reads {options.get('path')!r}; {DASHBOARDS_CONFIGMAP} is mounted at "
        f"{mounted_at!r}"
    )


def test_grafana_fits_what_the_scrape_plane_left_and_leaves_alertmanager_its_row(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 5 and decision C: Grafana's resources by VALUE as the whole flattened mapping,
    the three namespace totals DERIVED from the built pod templates rather than restated
    (a project rule, now summed over FIVE pod templates since an earlier change's Alertmanager fills
    what this namespace used to leave over), the `monitoring-envelope` quota still reading the
    three quantities an earlier change set, and the remainder pinned as EXACTLY zero on every quantity -
    an equality, never a `<=`.

    The remainder is the assertion with teeth even at zero. A Grafana sized at 96Mi would still
    fit the quota; what it would break is Alertmanager, admitted last, which a `<=` check could
    never say. Catches a quantity raised in isolation to make a pod start, which is the state
    where the namespace goes OVER quota and whichever pod is admitted LAST is refused, with the
    others running fine and nothing saying why. Catches an extra `limits.cpu` nobody agreed to,
    which throttles rather than protecting anything - the whole mapping is compared, so an added
    key fails. Catches a second replica, which the totals see and the per-row values cannot, and
    which would also mean two Grafanas writing one RWO sqlite volume. Catches the envelope
    quietly raised to make the arithmetic work, since the quota is compared as written strings.
    """
    templates = pod_templates(built, MONITORING)
    assert set(templates) == MONITORING_WORKLOADS, (
        f"monitoring has pod templates {sorted(templates)}, not {sorted(MONITORING_WORKLOADS)}"
    )

    container = sole_container(templates[GRAFANA], f"{MONITORING}/{GRAFANA}")
    quantities = resource_quantities(container)
    assert quantities == GRAFANA_RESOURCES, (
        f"grafana declares {quantities}, not {GRAFANA_RESOURCES}"
    )

    totals = {"requests.memory": 0, "requests.cpu": 0, "limits.memory": 0}
    for owner in pod_owners(built, MONITORING):
        pods = pods_of(owner)
        for member in containers_of(owner["spec"]["template"]):
            declared = resource_quantities(member)
            totals["requests.memory"] += pods * memory_bytes(declared["requests.memory"])
            totals["requests.cpu"] += pods * cpu_millicores(declared["requests.cpu"])
            totals["limits.memory"] += pods * memory_bytes(declared["limits.memory"])

    assert totals["requests.memory"] == EXPECTED_REQUEST_MEMORY_BYTES, (
        f"monitoring requests {totals['requests.memory']} bytes, not "
        f"{EXPECTED_REQUEST_MEMORY_BYTES}"
    )
    assert totals["requests.cpu"] == EXPECTED_REQUEST_CPU_MILLICORES, (
        f"monitoring requests {totals['requests.cpu']}m, not {EXPECTED_REQUEST_CPU_MILLICORES}m"
    )
    assert totals["limits.memory"] == EXPECTED_LIMIT_MEMORY_BYTES, (
        f"monitoring limits {totals['limits.memory']} bytes, not {EXPECTED_LIMIT_MEMORY_BYTES}"
    )

    hard = quota_hard(object_named(built, "ResourceQuota", "monitoring-envelope"))
    assert hard == MONITORING_QUOTA, f"the monitoring quota reads {hard}, not {MONITORING_QUOTA}"

    assert memory_bytes(hard["requests.memory"]) - totals["requests.memory"] == (
        HEADROOM_REQUEST_MEMORY_BYTES
    ), (
        f"{memory_bytes(hard['requests.memory']) - totals['requests.memory']} bytes of memory "
        f"requests are left, not {HEADROOM_REQUEST_MEMORY_BYTES}"
    )
    assert cpu_millicores(hard["requests.cpu"]) - totals["requests.cpu"] == (
        HEADROOM_REQUEST_CPU_MILLICORES
    ), (
        f"{cpu_millicores(hard['requests.cpu']) - totals['requests.cpu']}m of cpu requests are "
        f"left, not {HEADROOM_REQUEST_CPU_MILLICORES}m"
    )
    assert memory_bytes(hard["limits.memory"]) - totals["limits.memory"] == (
        HEADROOM_LIMIT_MEMORY_BYTES
    ), (
        f"{memory_bytes(hard['limits.memory']) - totals['limits.memory']} bytes of memory limits "
        f"are left, not {HEADROOM_LIMIT_MEMORY_BYTES}"
    )


def test_the_grafana_ingress_publishes_one_host_behind_two_middlewares_and_no_basic_auth(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 6 and decision B: one host, one `/` Prefix path, TLS secret `grafana-tls`, a
    backend resolved all the way from the Ingress's port reference through `Service/grafana`'s
    `targetPort` to the container port Grafana itself declares (named `http`, decision D), and a
    middleware annotation parsing to EXACTLY the two namespace-qualified references, each of
    which resolves to a `Middleware` this build emits in `monitoring`.

    Catches a middleware reference Traefik cannot resolve - a bare `rate-limit`, an `apps-`
    prefix copied from the gateway Ingress, a missing `@kubernetescrd` - which Traefik treats as
    NO middleware rather than as an error, so the host is served with no rate limit and no
    security headers and nothing reports it. Catches `basic-auth` added "for consistency" with
    the `apps` hosts, which is decision B's rejected alternative and which puts a Traefik 401 in
    front of the `/api/annotations` POST `deploy.yml` makes with a Bearer token. Catches a backend
    port number that drifts from the port the Service targets, which resolves to no endpoint and
    serves a 503 on the whole host. Catches the `tls` block naming `gateway-tls`, which would have
    cert-manager reissue another host's certificate into this namespace. Catches a second path or
    a second host on the router, since paths and hosts are compared by set equality.
    """
    ingress = object_named(built, "Ingress", GRAFANA)
    assert namespace_of(ingress) == MONITORING, f"the grafana Ingress is in {namespace_of(ingress)}"

    rules = ingress["spec"]["rules"]
    assert isinstance(rules, list), "the grafana Ingress declares no rules list"
    hosts = {str(rule["host"]) for rule in rules}
    assert hosts == {GRAFANA_HOST}, f"the grafana Ingress serves {sorted(hosts)}"

    service = object_named(built, "Service", GRAFANA)
    container = sole_container(grafana_template(built), f"{MONITORING}/{GRAFANA}")
    declared_ports = container_ports(grafana_template(built))
    assert set(declared_ports) == {GRAFANA_PORT_NAME}, (
        f"the grafana container declares ports {sorted(declared_ports)}, not "
        f"[{GRAFANA_PORT_NAME!r}]"
    )

    paths: set[tuple[str, str]] = set()
    for rule in rules:
        for entry in rule["http"]["paths"]:
            paths.add((str(entry["path"]), str(entry.get("pathType"))))
            backend = entry["backend"]["service"]
            assert str(backend["name"]) == GRAFANA, (
                f"the grafana Ingress routes to Service/{backend['name']}"
            )
            port = backend.get("port", {})
            assert isinstance(port, dict), f"the grafana Ingress names backend port {port}"
            resolved = resolved_container_port(service_target_port(service, port), container)
            assert resolved == declared_ports[GRAFANA_PORT_NAME], (
                f"the grafana Ingress backend port {port} resolves to {resolved}, and the "
                f"container publishes {declared_ports[GRAFANA_PORT_NAME]}"
            )
    assert paths == {(GRAFANA_PATH, PATH_TYPE_PREFIX)}, f"the grafana Ingress serves {paths}"

    tls = ingress["spec"]["tls"]
    assert isinstance(tls, list) and len(tls) == 1, f"the grafana Ingress declares {tls} for TLS"
    assert [str(host) for host in tls[0]["hosts"]] == [GRAFANA_HOST], (
        f"the grafana tls block covers {tls[0]['hosts']}"
    )
    assert str(tls[0].get("secretName")) == GRAFANA_TLS_SECRET, (
        f"the grafana tls block names {tls[0].get('secretName')!r}, not {GRAFANA_TLS_SECRET!r}"
    )

    parsed = middleware_names(ingress)
    assert parsed == set(GRAFANA_MIDDLEWARES), (
        f"unexpected: {sorted(parsed - GRAFANA_MIDDLEWARES)}; "
        f"missing: {sorted(GRAFANA_MIDDLEWARES - parsed)}"
    )
    assert FORBIDDEN_MIDDLEWARE not in {name for _, name in parsed}, (
        f"the grafana host is behind {FORBIDDEN_MIDDLEWARE}, which decision B refuses"
    )
    emitted = {
        (str(namespace_of(document)), name_of(document))
        for document in objects_of_kind(built, "Middleware")
    }
    assert parsed <= emitted, (
        f"the grafana Ingress names middlewares the build does not emit: {sorted(parsed - emitted)}"
    )


def test_the_only_way_into_grafana_is_the_secret_this_repo_does_not_supply(
    built: list[dict[str, Any]],
) -> None:
    r"""Criteria 8 and 9: the Grafana container's `envFrom` names exactly the `grafana-admin`
    Secret, that contract's template carries exactly `GF_SECURITY_ADMIN_USER` and
    `GF_SECURITY_ADMIN_PASSWORD` with every value `REPLACE_ME`, the same two keys as the
    `infra/k8s/README.md` secrets table row, the Secret name is disjoint from `grafana-tls`, and
    no `GF_AUTH_ANONYMOUS_*` variable and no `GF_AUTH_DISABLE_LOGIN_FORM` appears in EITHER the
    container's `env` list or the contract's key set.

    The absence is asserted over both surfaces on purpose: an anonymous-access variable moved
    from `env` into the Secret template would leave a case that only read `env` green, and it is
    the Secret the cluster actually reads. Catches `GF_AUTH_ANONYMOUS_ENABLED=true` added to get
    a dashboard onto a screen without logging in, which publishes every panel - including the
    residency and cost views - to anyone who resolves the public host. Catches
    `GF_AUTH_DISABLE_LOGIN_FORM`, which is decision B's rejected alternative and which leaves no
    way to mint the service-account token `deploy.yml`'s annotation step needs. Catches a real
    password committed into the template instead of the placeholder, which is the one thing that
    would put a working credential for a public host into a public repo. Catches the contract
    drifting from the README row a human reads before the apply - the keys are compared against
    the table, which is how the other four contracts are held.
    """
    container = sole_container(grafana_template(built), f"{MONITORING}/{GRAFANA}")
    assert envfrom_secrets(container) == {ADMIN_SECRET}, (
        f"grafana reads envFrom {sorted(envfrom_secrets(container))}, not [{ADMIN_SECRET!r}]"
    )
    assert ADMIN_SECRET != GRAFANA_TLS_SECRET, (
        "the admin credential and the cert-manager certificate share a name, so an issued "
        "certificate would overwrite the credential"
    )

    keys = contract_keys()
    assert keys == ADMIN_KEYS, f"{ADMIN_EXAMPLE_FILENAME} contracts for {sorted(keys)}"
    assert keys == readme_secret_keys(ADMIN_SECRET), (
        f"{ADMIN_EXAMPLE_FILENAME} holds {sorted(keys)}; the README row names "
        f"{sorted(readme_secret_keys(ADMIN_SECRET))}"
    )

    document = contract_documents()[0]
    assert document.get("kind") == "Secret", f"{ADMIN_EXAMPLE_FILENAME} declares a non-Secret"
    assert name_of(document) == ADMIN_SECRET, f"{ADMIN_EXAMPLE_FILENAME} names {name_of(document)}"
    assert namespace_of(document) == MONITORING, (
        f"{ADMIN_EXAMPLE_FILENAME} files the Secret into {namespace_of(document)}"
    )
    for mapping_key in ("data", "stringData"):
        mapping = document.get(mapping_key, {})
        if not isinstance(mapping, dict):
            continue
        values = sorted({str(value) for value in mapping.values()})
        assert values in ([], [PLACEHOLDER]), (
            f"{ADMIN_EXAMPLE_FILENAME}'s `{mapping_key}` carries values {values}, not "
            f"{PLACEHOLDER!r}"
        )

    declared = env_names(container) | set(keys)
    anonymous = sorted(name for name in declared if name.startswith(ANONYMOUS_PREFIX))
    assert anonymous == [], f"grafana declares {anonymous}, which opens the UI to anyone"
    assert LOGIN_FORM_VARIABLE not in declared, (
        f"grafana declares {LOGIN_FORM_VARIABLE}, which removes the login the credential is for"
    )


def test_grafana_is_reachable_only_from_the_router_and_reaches_only_prometheus(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 7's second half, by EQUALITY in both directions: `grafana-ingress` selects
    Grafana alone, admits `kube-system` and no other peer, and opens exactly ONE port - Grafana's
    own serving port, resolved against the built pod template rather than compared to a literal;
    `grafana-egress` reaches exactly `prometheus` on 9090 in `monitoring` and carries no
    `ipBlock`; and neither policy declares a block in the direction it is not for.

    Equality and not containment, because Kubernetes unions allows: an extra `from` peer or an
    extra port is an opening no containment check can see, and a missing one is a page that hangs.

    Catches `grafana-ingress` written with `monitoring` as its peer instead of `kube-system`,
    which admits the scrape plane and denies the router - the host answers nothing and
    `default-deny` is what reports it, silently. Catches a second port on that rule, an admin or
    debug surface published through the same policy that admits the browser. Catches the
    `podSelector` left empty, which is a policy that admits `kube-system` to Prometheus and the
    two exporters as well. Catches `grafana-egress` widened to every pod in `monitoring`, which
    would let the UI dial the exporters directly, and catches an `ipBlock` added so a panel can
    reach a hosted datasource - the exfiltration path infra/tests/infra/test_k8s_netpol.py's
    test_the_gateway_and_alertmanager_are_the_only_pods_with_a_route_to_the_public_internet keeps
    to those two pods alone. Catches an `egress` block added to `grafana-ingress`, which under
    `default-deny`'s `Egress` type would quietly become the only thing deciding where Grafana
    may dial.
    """
    ingress = policy(built, MONITORING, GRAFANA_INGRESS_POLICY)
    assert matched_workloads(built, pod_selector(ingress), MONITORING) == {GRAFANA}, (
        f"{GRAFANA_INGRESS_POLICY} selects "
        f"{sorted(matched_workloads(built, pod_selector(ingress), MONITORING))}"
    )
    assert "egress" not in spec_of(ingress), f"{GRAFANA_INGRESS_POLICY} declares an egress block"

    serving_port = container_ports(grafana_template(built))[GRAFANA_PORT_NAME]
    expected_ports = {GRAFANA: {("TCP", serving_port)}}
    rules = rules_of(ingress, "ingress")
    assert rules, f"{GRAFANA_INGRESS_POLICY} declares no ingress rule"
    peers: set[tuple[str, str]] = set()
    for rule in rules:
        listed = peers_of(rule, "from")
        assert listed, f"{GRAFANA_INGRESS_POLICY} has a rule with no `from`, admitting everyone"
        for peer in listed:
            peers |= peer_targets(built, peer, MONITORING)
        assert resolved_ports(built, ingress, rule) == expected_ports, (
            f"{GRAFANA_INGRESS_POLICY} resolves to {resolved_ports(built, ingress, rule)}, "
            f"not {expected_ports}"
        )
    assert peers == {(KUBE_SYSTEM, ANY_POD)}, f"{GRAFANA_INGRESS_POLICY} admits {sorted(peers)}"

    egress = policy(built, MONITORING, GRAFANA_EGRESS_POLICY)
    assert matched_workloads(built, pod_selector(egress), MONITORING) == {GRAFANA}, (
        f"{GRAFANA_EGRESS_POLICY} selects "
        f"{sorted(matched_workloads(built, pod_selector(egress), MONITORING))}"
    )
    assert "ingress" not in spec_of(egress), f"{GRAFANA_EGRESS_POLICY} declares an ingress block"

    # The `ipBlock` scan runs BEFORE the destination equality on purpose: a CIDR peer carries no
    # `namespaceSelector`, so `peer_targets` would read it as the policy's own namespace and the
    # destination set would report the CIDR as a pod peer. Asserting absence first is what makes
    # this claim red on its own line rather than through the equality next to it.
    assert ip_blocks(egress) == [], f"{GRAFANA_EGRESS_POLICY} carries ipBlocks: {ip_blocks(egress)}"
    assert ip_blocks(ingress) == [], (
        f"{GRAFANA_INGRESS_POLICY} carries ipBlocks: {ip_blocks(ingress)}"
    )

    destinations = egress_destinations(built, egress)
    assert destinations == set(GRAFANA_EGRESS_DESTINATIONS), (
        f"unexpected: {sorted(destinations - GRAFANA_EGRESS_DESTINATIONS)}; "
        f"missing: {sorted(GRAFANA_EGRESS_DESTINATIONS - destinations)}"
    )
