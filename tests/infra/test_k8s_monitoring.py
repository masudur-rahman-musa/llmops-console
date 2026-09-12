r"""`infra/k8s/monitoring/`: the scrape plane an earlier change agree, as kustomize emits it.

The stream comes from the session-scoped `built` fixture in `tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, shared with
`test_k8s_base.py`, `test_k8s_apps.py`, `test_k8s_platform.py`, `test_k8s_netpol.py` and
`test_k8s_grafana.py`. Nothing here builds again and nothing here reads a raw manifest file, so a
manifest written and never listed in a `resources:` block is the same failure as a manifest never
written.

This file no longer owns the namespace's OBJECT INVENTORY. An earlier change's §8 entry 1 moved that
equality to `tests/infra/test_k8s_grafana.py`, which pins every object `monitoring` carries -
23a's eleven, the two envelope objects and 23b's nine - by `(kind, name, namespace)` in both
directions. What stays here is the scrape plane's BEHAVIOUR: the scrape config, the retention
window, the envelope arithmetic, the RBAC and the scrape policies. Two files asserting one
equality would be two places to edit for every object 23e, 23c and 23d add.

Everything is asserted by EQUALITY in both directions. For the resource rows that is because an
extra `limits.cpu` nobody agreed to is as dangerous as a missing request; for the policies it is
because Kubernetes unions allows, so an extra `to` peer or an extra port is an opening no
containment check can see.

Three scoping decisions this file had to make, because the plans' criteria cannot all be read
literally at once:

* 23a's criterion 1 names ELEVEN objects "in `monitoring`, nothing else", while its criterion 5
  requires the pre-existing `ResourceQuota/monitoring-envelope` and `LimitRange/monitoring-defaults`
  to stay in that same namespace unchanged, and its criterion 6 puts NetworkPolicies in it. A
  literal eleven would therefore be red against a correct build, and 23b adds nine more objects
  on top. That whole reconciliation now lives with the equality, in `test_k8s_grafana.py`.
* `ClusterRole/kube-state-metrics` and `ClusterRoleBinding/kube-state-metrics` are cluster
  scoped and carry no `metadata.namespace`, so they cannot be found by namespace. They are
  scoped by KIND instead - every `ClusterRole` and `ClusterRoleBinding` ANYWHERE in the build is
  expected to be one of the two, which is what
  `test_kube_state_metrics_is_the_only_subject_with_rbac_and_reads_only` asserts here and what
  the moved inventory asserts there.
* Criterion 1 says "no CustomResource of ANY kind", and the build already emits
  `cert-manager.io/v1` `ClusterIssuer` and `traefik.io/v1alpha1` `Middleware`, which are custom
  resources. What the criterion means is decision A - no operator - so it is asserted as the two
  things decision A actually forbids: no `CustomResourceDefinition` kind at all, and no document
  in the `monitoring.coreos.com` group (`ServiceMonitor`, `PrometheusRule`, `Prometheus`). Those
  are what would force `kubeconform -strict` into `-ignore-missing-schemas`, which
  `test_k8s_base.py` forbids by name.

Three cases here pin NetworkPolicy behaviour, which normally lives in `test_k8s_netpol.py`. They
are here because 23a's §8 authorised exactly four edits to that file - three constants and one
renamed case - and adding a case to it was not among them; 23b's §8 widens the same constants to
the seventeen-policy NAME set and touches no case there either. What those edits carry is the
policy NAME set, `gateway-ingress`'s peer and the selector scan; what they do not reach is
`prometheus-egress`'s destination set and the two scrape ingress policies' ports, and
`test_k8s_netpol.py`'s ingress parametrization resolves inside `apps` only, so neither policy
could be a row in it. 23a's criteria 7 and 8 would otherwise be pinned by nothing.

Named ports are why the ingress case resolves rather than compares. One NetworkPolicy has one
`podSelector` and its `ports` apply to every pod that selector matches, and `monitoring`'s
`metrics-ingress-from-monitoring` selects EVERY pod in the namespace while naming one port
(`port: metrics`), which Kubernetes resolves against each target pod's own container port. So
`resolved_ports` reads in numbers while the manifest reads in names, and the three scrape targets
on 9090, 9100 and 8080 are covered by one rule. That shared name is the whole mechanism 23a's
criterion 2 exists to protect, and it is exactly what an earlier change's decision D takes away from
Grafana: a fourth pod in this namespace publishing `http` is not covered by a named-port allow at
all, which is why 23b gives it `grafana-ingress` of its own.

`resource_quantities` and the policy helpers are COPIES of `test_k8s_base.py`'s,
`test_k8s_platform.py`'s and `test_k8s_netpol.py`'s, not imports - the convention of this suite.
The quantity ARITHMETIC that used to live here (`memory_bytes`, `cpu_millicores`, `quota_hard`,
`pods_of` and the namespace totals) went with the envelope case to `test_k8s_grafana.py`, which
sums all four pod templates; what is left here is one row per scrape workload, compared as
written strings.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

MONITORING = "monitoring"
PLATFORM = "platform"
APPS = "apps"

PROMETHEUS = "prometheus"
NODE_EXPORTER = "node-exporter"
KUBE_STATE_METRICS = "kube-state-metrics"
GATEWAY = "gateway"
QDRANT = "qdrant"

NETWORK_POLICY = "NetworkPolicy"

#: The one label every pod template in this build carries and the only one any selector keys
#: off. A workload's value here is also its Deployment name and its Service name.
NAME_LABEL = "app.kubernetes.io/name"

#: Kubernetes auto-labels every namespace with its own name, which is how a cross-namespace peer
#: names the namespace it means.
NAMESPACE_NAME_LABEL = "kubernetes.io/metadata.name"

#: The peer marker for "every pod in that namespace": a peer with no `podSelector`, or with an
#: empty one.
ANY_POD = "*"

#: Decision A, the half the build can actually contradict: an operator install would bring this
#: kind, and `kubeconform -strict` has no stock schema for what it declares.
FORBIDDEN_KIND = "CustomResourceDefinition"

#: Decision A's other half: the API group every `kube-prometheus-stack` custom resource lives in
#: (`ServiceMonitor`, `PrometheusRule`, `Prometheus`, `Alertmanager`). `cert-manager.io` and
#: `traefik.io` are custom resources too and are deliberately NOT forbidden - they are already in
#: the build and have nothing to do with decision A.
FORBIDDEN_API_GROUP = "monitoring.coreos.com"

#: Criterion 2: one shared port NAME across the three SCRAPE TARGETS, which is what lets ONE
#: ingress policy cover three pods listening on three different numbers.
METRICS_PORT_NAME = "metrics"

#: Criterion 2's three numbers, by workload. Pinned separately from the name so a renamed port
#: cannot silently move one.
METRICS_PORT_NUMBERS = {PROMETHEUS: 9090, NODE_EXPORTER: 9100, KUBE_STATE_METRICS: 8080}

#: An earlier change decision D, the narrowing this file pays for it: Grafana is the fourth pod template in
#: `monitoring` and it is NOT scraped, so its port serves the UI its Ingress points at and is
#: named `http`. The scrape case asserts that spelling rather than dropping Grafana from the scan,
#: so a fifth pod renaming its way into the scrape allow is still visible here.
GRAFANA = "grafana"
GRAFANA_PORT_NAME = "http"

#: Criterion 3 and decision C: the five static targets, as Service DNS. Every one is a Service
#: this build emits, in the namespace it emits it in.
SCRAPE_TARGETS: frozenset[str] = frozenset(
    {
        "gateway.apps.svc.cluster.local:8080",
        "qdrant.platform.svc.cluster.local:6333",
        "kube-state-metrics.monitoring.svc.cluster.local:8080",
        "node-exporter.monitoring.svc.cluster.local:9100",
        "prometheus.monitoring.svc.cluster.local:9090",
    }
)

#: Criterion 3: the one path every job scrapes.
METRICS_PATH = "/metrics"

#: Decision C: any key of this shape is service discovery, which is exactly what static targets
#: were chosen INSTEAD of - and what would need the cluster-wide API read Prometheus does not get.
DISCOVERY_KEY_SUFFIX = "_sd_configs"

#: Criterion 4, each by exact value, never a range.
SCRAPE_INTERVAL = "30s"
RETENTION_FLAG = "--storage.tsdb.retention.time"
RETENTION_ARG = f"{RETENTION_FLAG}=15d"
PVC_STORAGE = "4Gi"

#: Criterion 5, one row per workload, flattened the way the budget table and the quota both read.
#: Asserted as the WHOLE mapping, so a `limits.cpu` nobody agreed to is a failure rather than a
#: silent addition (decision C of an earlier change: a CPU limit throttles rather than kills).
WORKLOAD_RESOURCES: dict[str, dict[str, str]] = {
    PROMETHEUS: {"requests.memory": "256Mi", "requests.cpu": "150m", "limits.memory": "512Mi"},
    NODE_EXPORTER: {"requests.memory": "32Mi", "requests.cpu": "50m", "limits.memory": "64Mi"},
    KUBE_STATE_METRICS: {
        "requests.memory": "64Mi",
        "requests.cpu": "50m",
        "limits.memory": "128Mi",
    },
}

#: Decision C: the read-only verb set kube-state-metrics gets, and the ceiling every rule's
#: `verbs` list must sit inside. `create`, `patch` or `*` here is a scraper that can mutate.
READ_ONLY_VERBS = frozenset({"get", "list", "watch"})

#: Criterion 7: every destination `prometheus-egress` may reach, as `(workload, resolved port)`.
#: The DNS allow is NOT here - it lives in `allow-dns`, which selects every pod in `monitoring`.
PROMETHEUS_EGRESS_DESTINATIONS: frozenset[tuple[str, int]] = frozenset(
    {
        (GATEWAY, 8080),
        (QDRANT, 6333),
        (NODE_EXPORTER, 9100),
        (KUBE_STATE_METRICS, 8080),
    }
)

#: Criterion 7: the only `monitoring` policies that may declare an `egress` block at all.
#: `default-deny` names `Egress` in its `policyTypes` and carries no block, which is what makes
#: it a deny.
#:
#: `grafana-egress` joined with an earlier change. This set is what stops an outbound allow appearing in
#: this namespace unnoticed, so it is deliberately a NAME set rather than a count and it moves in
#: the same change as the policy; what that policy may actually reach is asserted in
#: `tests/infra/test_k8s_grafana.py`, by
#: `test_grafana_is_reachable_only_from_the_router_and_reaches_only_prometheus`, including its
#: own absent `ipBlock`.
MONITORING_EGRESS_POLICIES: frozenset[str] = frozenset(
    {"allow-dns", "prometheus-egress", "grafana-egress"}
)


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
    picked up by the same scan as the two Deployments, with no kind list to keep in step."""
    owners: list[dict[str, Any]] = []
    for document in documents:
        spec = document.get("spec")
        if not isinstance(spec, dict) or namespace_of(document) != namespace:
            continue
        template = spec.get("template")
        if isinstance(template, dict) and isinstance(template.get("spec"), dict):
            owners.append(document)
    return owners


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


def template_labels(template: dict[str, Any]) -> dict[str, str]:
    """A pod template's labels, the ones a `podSelector` has to match."""
    metadata = template.get("metadata", {})
    labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
    assert isinstance(labels, dict)
    return {str(key): str(value) for key, value in labels.items()}


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
    UNNAMED port is visible to the case that asserts every port here is named."""
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
                (None if declared is None else str(declared), int(entry["containerPort"]))
            )
    return entries


def container_ports(template: dict[str, Any]) -> dict[str, int]:
    """A pod template's NAMED container ports, `name -> number`, which is what a NetworkPolicy
    naming a port resolves against."""
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


def scrape_config(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """`ConfigMap/prometheus-config`'s `prometheus.yml`, parsed.

    The ConfigMap holds the file as a STRING, so a scrape config with a typo is valid YAML
    inside a valid ConfigMap; parsing here is what makes its contents assertable at all.
    """
    configmap = object_named(documents, "ConfigMap", "prometheus-config")
    data = configmap.get("data", {})
    assert isinstance(data, dict), "prometheus-config declares no data mapping"
    assert "prometheus.yml" in data, f"prometheus-config holds keys {sorted(data)}"
    parsed = yaml.safe_load(str(data["prometheus.yml"]))
    assert isinstance(parsed, dict), "prometheus.yml is not a YAML mapping"
    return parsed


def every_key(value: object) -> list[str]:
    """Every mapping key anywhere in a parsed structure, at any depth.

    A whole-document walk rather than a look at the five jobs: service discovery can be declared
    beside `static_configs` in a job, under `global`, or inside a `relabel_configs` entry, and a
    scan of the jobs alone would report a discovered target as absent.
    """
    if isinstance(value, dict):
        found: list[str] = []
        for key, nested in value.items():
            found.append(str(key))
            found.extend(every_key(nested))
        return found
    if isinstance(value, list):
        return [key for entry in value for key in every_key(entry)]
    return []


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


def policy_types(document: dict[str, Any]) -> list[str]:
    """`spec.policyTypes` as written, in order."""
    listed = spec_of(document).get("policyTypes", [])
    assert isinstance(listed, list), f"{name_of(document)}.spec.policyTypes is not a list"
    return [str(entry) for entry in listed]


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
    """Whether a `podSelector` matches a pod template's labels.

    Understands `matchLabels` and `matchExpressions` with `In`, the only two forms this manifest
    uses; any other operator asserts rather than silently reporting a match.
    """
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
    `podSelector` or an empty one - namespace-wide, the shape criterion 8 states for
    `metrics-ingress-from-monitoring`'s own `from` peer."""
    namespace = peer_namespace(peer, default_namespace)
    selector = peer.get("podSelector")
    if not isinstance(selector, dict) or selector == {}:
        return {(namespace, ANY_POD)}
    workloads = matched_workloads(documents, selector, namespace)
    assert workloads, f"podSelector {selector} matches no pod template in {namespace}"
    return {(namespace, workload) for workload in workloads}


def declared_ports(rule: dict[str, Any], label: str) -> list[tuple[str, object]]:
    """A rule's `(protocol, port-as-written)` entries, the port still a name OR a number.

    A rule with no `ports` key allows EVERY port, which no policy here intends, so it asserts.
    """
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


def selected_workloads(
    documents: list[dict[str, Any]], document: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """The pod templates a policy's own `podSelector` protects, keyed by workload."""
    namespace = namespace_of(document)
    assert namespace is not None, f"{name_of(document)} declares no metadata.namespace"
    selector = pod_selector(document)
    return {
        workload: template
        for workload, template in pod_templates(documents, namespace).items()
        if selector_matches(selector, template_labels(template))
    }


def unreachable_workloads(
    documents: list[dict[str, Any]], document: dict[str, Any], rule: dict[str, Any]
) -> set[str]:
    """The workloads a policy selects that a named port in `rule` cannot resolve against, because
    none of their container ports carries that name.

    Kubernetes resolves a named port per target pod, so such a pod is selected by the policy and
    reached by none of its allows - the allow simply does not apply to it. Naming them is what
    lets the case ASSERT the exclusion instead of a
    resolution helper raising on the first pod that does not publish `metrics`.
    """
    unreachable: set[str] = set()
    for workload, template in selected_workloads(documents, document).items():
        ports = container_ports(template)
        names = {port for _, port in declared_ports(rule, workload) if isinstance(port, str)}
        if names and not names & set(ports):
            unreachable.add(workload)
    return unreachable


def resolved_ports(
    documents: list[dict[str, Any]], document: dict[str, Any], rule: dict[str, Any]
) -> dict[str, set[tuple[str, int]]]:
    """An INGRESS rule's ports resolved per pod the policy's own `podSelector` protects AND can
    reach, `{workload: {(protocol, number)}}` - the shape one rule saying `port: metrics` needs
    when it protects three pods listening on 9090, 9100 and 8080. A pod the named port cannot
    resolve against is left out here and asserted by `unreachable_workloads` instead."""
    namespace = namespace_of(document)
    assert namespace is not None, f"{name_of(document)} declares no metadata.namespace"
    label = f"{namespace}/{name_of(document)}"
    unreachable = unreachable_workloads(documents, document, rule)
    return {
        workload: ports_against(rule, container_ports(template), label)
        for workload, template in selected_workloads(documents, document).items()
        if workload not in unreachable
    }


def egress_destinations(
    documents: list[dict[str, Any]], document: dict[str, Any]
) -> set[tuple[str, int]]:
    """An egress policy's destinations as `(workload-or-namespace, resolved port)`.

    Every peer of a rule is crossed with every port of that rule, which is what the rule means:
    grouping four targets and four numbers into one rule allows all sixteen, and this set says so
    rather than flattering the manifest into the four the criterion names.
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


def test_the_monitoring_namespace_emits_exactly_the_objects_the_two_plans_name(
    built: list[dict[str, Any]],
) -> None:
    """An earlier change's criterion 1, the half that is a PROHIBITION rather than an inventory: no
    `CustomResourceDefinition` anywhere in the build and no document in the
    `monitoring.coreos.com` group, which is what decision A actually forbids (see the module
    docstring for why a `Middleware` and a `ClusterIssuer` are custom resources this build keeps).

    The `(kind, name, namespace)` EQUALITY that used to stand here moved to
    tests/infra/test_k8s_grafana.py's
    test_the_monitoring_namespace_emits_grafana_and_nothing_else_new when an earlier change added nine
    objects to this namespace (its §8 entry 1): that file owns the inventory from here on, so
    23e, 23c and 23d each edit one expected set instead of two that drift apart. What is left
    here is scanned over the WHOLE build rather than over one namespace, so it is not an
    inventory at all.

    Catches an operator install sneaking in through `helmCharts:` - it would bring a
    CustomResourceDefinition and `monitoring.coreos.com` custom resources (`ServiceMonitor`,
    `PrometheusRule`, `Prometheus`, `Alertmanager`), which decision A forbids because
    `kubeconform -strict` has no stock schema for them and tests/infra/test_k8s_base.py's
    test_infra_check_runs_the_kustomize_half_for_real_now_that_the_file_exists refuses the flag
    that would skip them. Catches such a document landing in ANY namespace, not only this one,
    which is the scope the claim is made at and which no per-namespace inventory can see.
    """
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


def test_every_scraped_container_port_is_named_metrics_on_its_agreed_number(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 2, narrowed by an earlier change decision D to the THREE SCRAPE TARGETS (its §8 entry 2):
    the shared NAME by set equality across their pod templates, the three NUMBERS by workload,
    and Grafana's own port asserted to be named `http` rather than excluded from the scan.

    The name is the mechanism 23a's criterion 8 depends on - one ingress policy has one
    `podSelector` and its `ports` apply to every pod it matches, so three pods on 9090, 9100 and
    8080 can only be covered by one rule if all three call the port the same thing (standing rule
    50). Grafana is the fourth pod in this namespace and is NOT scraped: its 3000 serves the UI
    `Ingress/grafana` points at, and calling it `metrics` would make the Ingress backend read
    `port: name: metrics` for a browser, which is why 23b gives it `grafana-ingress` instead.

    Catches an exporter whose port is written `http`, or left unnamed, which builds and validates
    perfectly and then silently drops that pod out of the ingress allow: Prometheus connects to
    nothing and the target simply reads DOWN. Catches a number transposed between the two
    exporters (9100 for kube-state-metrics), which the shared name alone cannot see because the
    name would still resolve. Catches a second container port added beside `metrics` - an admin
    or pprof surface the policy would then also open, since the allow is written by name. Catches
    Grafana's port renamed to `metrics` to make one assertion whole again, which is the plan
    bending to the test and which would put a UI port inside the scrape allow.
    """
    templates = pod_templates(built, MONITORING)
    scraped = sorted(METRICS_PORT_NUMBERS)
    assert sorted(templates) == sorted([*scraped, GRAFANA]), (
        f"monitoring has pod templates {sorted(templates)}"
    )

    names = {name for workload in scraped for name, _ in port_entries(templates[workload])}
    # An unnamed port reads as None here rather than being dropped, so it is printed as `None`
    # and the set comparison sees it.
    printable = sorted(str(name) for name in names)
    assert names == {METRICS_PORT_NAME}, f"scraped container port names are {printable}"

    numbers = {
        workload: [number for _, number in port_entries(templates[workload])]
        for workload in scraped
    }
    expected = {workload: [number] for workload, number in METRICS_PORT_NUMBERS.items()}
    assert numbers == expected, f"container ports are {numbers}, not {expected}"

    grafana_names = sorted(str(name) for name, _ in port_entries(templates[GRAFANA]))
    assert grafana_names == [GRAFANA_PORT_NAME], (
        f"grafana publishes ports named {grafana_names}, not [{GRAFANA_PORT_NAME!r}]"
    )


def test_the_scrape_config_carries_exactly_the_five_static_targets(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 3 and decision C, over the PARSED `prometheus.yml` the ConfigMap holds as a
    string: five jobs, each with exactly one `static_configs` entry, the targets by set equality,
    every `metrics_path` `/metrics`, and no key ending `_sd_configs` anywhere in the document.

    Catches a sixth job nobody agreed to and a fifth that never landed - a missing job is a
    dashboard panel that is empty forever with no error anywhere, since a scrape config
    that omits a target is as valid as one that names it. Catches a target written against a
    Service name or a namespace this build does not emit (`qdrant.apps...`,
    `node-exporter.platform...`), which resolves to nothing and reads as a permanently DOWN
    target. Catches a port that disagrees with the container port the job scrapes. Catches a job
    left on Prometheus's default `/metrics`-less path or pointed at `/prometheus/metrics`.
    Catches `kubernetes_sd_configs` added later: discovery is exactly what decision C traded away
    to keep this scraper unable to read the API server at all, and it would need the ClusterRole
    that `test_kube_state_metrics_is_the_only_subject_with_rbac_and_reads_only` refuses it.
    """
    parsed = scrape_config(built)
    jobs = parsed.get("scrape_configs", [])
    assert isinstance(jobs, list), "prometheus.yml declares no scrape_configs list"
    assert len(jobs) == len(SCRAPE_TARGETS), (
        f"{len(jobs)} scrape_configs, not {len(SCRAPE_TARGETS)}: "
        f"{[job.get('job_name') for job in jobs]}"
    )

    targets: set[str] = set()
    paths: dict[str, str] = {}
    for job in jobs:
        assert isinstance(job, dict), "a scrape_configs entry is not a mapping"
        job_name = str(job.get("job_name", ""))
        assert job_name, f"a scrape_configs entry carries no job_name: {job}"
        statics = job.get("static_configs", [])
        assert isinstance(statics, list) and len(statics) == 1, (
            f"{job_name} declares {statics}, not exactly one static_configs entry"
        )
        entry = statics[0]
        assert isinstance(entry, dict), f"{job_name}'s static_configs entry is not a mapping"
        listed = entry.get("targets", [])
        assert isinstance(listed, list), f"{job_name} declares no targets list"
        targets |= {str(target) for target in listed}
        paths[job_name] = str(job.get("metrics_path", ""))

    assert targets == SCRAPE_TARGETS, (
        f"unexpected: {sorted(targets - SCRAPE_TARGETS)}; "
        f"missing: {sorted(SCRAPE_TARGETS - targets)}"
    )
    assert paths == dict.fromkeys(paths, METRICS_PATH), f"metrics paths are {paths}"

    discovery = sorted({key for key in every_key(parsed) if key.endswith(DISCOVERY_KEY_SUFFIX)})
    assert discovery == [], f"prometheus.yml declares service discovery: {discovery}"


def test_the_retention_window_the_scrape_interval_and_the_volume_are_the_agreed_values(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 4 and decision D, each as an EXACT equality and never a range:
    `global.scrape_interval` is `30s`, the Prometheus container carries exactly one
    `--storage.tsdb.retention.time` argument and it reads `15d`, and the PVC requests `4Gi`.

    These three are one budget: 30s over five targets for 15 days is what 4Gi was sized for, so a
    wrong value anywhere shows up as a disk that fills silently and a TSDB that starts dropping
    the oldest blocks - the series the "evals over time" panel and row 24's 7-day median read.
    Catches `15d` written on a second `--storage.tsdb.retention.time` flag beside a first, where
    Prometheus takes one of them and the manifest reads as if it takes the other. Catches the
    retention flag dropped entirely, which leaves Prometheus's own 15d default in place today and
    changes underneath the repo the day the image is bumped. Catches a scrape interval tightened
    to 15s, which doubles every series' sample count against an unchanged volume. Catches the PVC
    given `4G` where the plan says `4Gi` - 7% less disk, which the API server does not treat as
    equal either.
    """
    parsed = scrape_config(built)
    globals_block = parsed.get("global", {})
    assert isinstance(globals_block, dict), "prometheus.yml declares no global mapping"
    assert str(globals_block.get("scrape_interval")) == SCRAPE_INTERVAL, (
        f"scrape_interval is {globals_block.get('scrape_interval')!r}, not {SCRAPE_INTERVAL!r}"
    )

    deployment = object_named(built, "Deployment", PROMETHEUS)
    container = sole_container(deployment["spec"]["template"], f"{MONITORING}/{PROMETHEUS}")
    args = [str(argument) for argument in container.get("args", [])]
    retention = [argument for argument in args if argument.startswith(f"{RETENTION_FLAG}=")]
    assert retention == [RETENTION_ARG], f"retention arguments are {retention}; args are {args}"

    claim = object_named(built, "PersistentVolumeClaim", "prometheus-data")
    requests = claim["spec"].get("resources", {}).get("requests", {})
    assert isinstance(requests, dict), "prometheus-data declares no resources.requests"
    assert str(requests.get("storage")) == PVC_STORAGE, (
        f"prometheus-data requests {requests.get('storage')!r}, not {PVC_STORAGE!r}"
    )


def test_each_scrape_workload_declares_the_quantities_the_budget_prices(
    built: list[dict[str, Any]],
) -> None:
    """An earlier change criterion 5, narrowed by an earlier change to the THREE SCRAPE WORKLOADS' rows, one at a
    time by VALUE. The namespace TOTALS and the `monitoring-envelope` quota moved with the
    inventory to tests/infra/test_k8s_grafana.py's
    test_grafana_fits_what_the_scrape_plane_left_and_leaves_alertmanager_its_row, which sums
    every pod template in `monitoring` - Grafana's included - and pins the remainder as exactly
    32Mi, 50m and 64Mi. A totals assertion written for three workloads is red against a correct
    four-workload build, and two files summing one namespace would drift.

    Catches a quantity raised in isolation to make a pod start - the state where the namespace is
    over its quota and whichever pod is admitted LAST is refused, with the others running fine
    and nothing saying why the exporter never appeared. Catches Prometheus given a limit under
    its request. Catches an extra `limits.cpu` nobody agreed to, which throttles the scraper
    rather than protecting anything - the whole flattened mapping is
    compared, so an added key is a failure. Catches a row silently dropped, since every workload
    in `WORKLOAD_RESOURCES` must have a pod template to read.
    """
    templates = pod_templates(built, MONITORING)
    for workload, expected in WORKLOAD_RESOURCES.items():
        assert workload in templates, f"monitoring has no pod template for {workload}"
        container = sole_container(templates[workload], f"{MONITORING}/{workload}")
        quantities = resource_quantities(container)
        assert quantities == expected, f"{workload} declares {quantities}, not {expected}"


def test_kube_state_metrics_is_the_only_subject_with_rbac_and_reads_only(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 1 and decision C: exactly one ClusterRole and one ClusterRoleBinding in the WHOLE
    build, both `kube-state-metrics`, the binding pointing at that role and at exactly the
    `kube-state-metrics` ServiceAccount in `monitoring`, every rule's verbs inside
    `{get, list, watch}`, and no `serviceAccountName` on the Prometheus pod spec at all.

    Scanned over the whole build rather than read off the two objects by name, because the claim
    is that nothing ELSE in this repo grants cluster-wide API access; a case that opened them by
    name could not see a second ClusterRole added anywhere. Catches decision C reversed quietly -
    `kubernetes_sd_configs` needs Prometheus to list Services, Endpoints and Pods cluster-wide,
    and the first sign of it is a ServiceAccount on the Prometheus pod, which is why that is
    asserted here rather than left to the scrape-config case. Catches `create`, `patch`, `delete`
    or `*` in a rule, which turns a read-only exporter into a cluster-wide mutator holding the
    most valuable credential on the node. Catches a binding whose `subjects` names the `default`
    ServiceAccount or one in another namespace, which grants that whole namespace the role.
    Catches a `roleRef` left pointing at some other role, which is a silent privilege swap that
    schema-validates perfectly.
    """
    roles = objects_of_kind(built, "ClusterRole")
    bindings = objects_of_kind(built, "ClusterRoleBinding")
    assert [name_of(role) for role in roles] == [KUBE_STATE_METRICS], (
        f"the build emits ClusterRoles {sorted(name_of(role) for role in roles)}"
    )
    assert [name_of(binding) for binding in bindings] == [KUBE_STATE_METRICS], (
        f"the build emits ClusterRoleBindings {sorted(name_of(b) for b in bindings)}"
    )

    subjects = bindings[0].get("subjects", [])
    assert isinstance(subjects, list) and len(subjects) == 1, f"the binding names {subjects}"
    bound = {
        (
            str(subject.get("kind")),
            str(subject.get("name")),
            str(subject.get("namespace")),
        )
        for subject in subjects
    }
    expected_subject = {("ServiceAccount", KUBE_STATE_METRICS, MONITORING)}
    assert bound == expected_subject, f"the binding's subjects are {sorted(bound)}"

    role_ref = bindings[0].get("roleRef", {})
    assert isinstance(role_ref, dict), "the ClusterRoleBinding declares no roleRef"
    assert (str(role_ref.get("kind")), str(role_ref.get("name"))) == (
        "ClusterRole",
        KUBE_STATE_METRICS,
    ), f"the binding's roleRef is {role_ref}"

    granted: set[str] = set()
    rules = roles[0].get("rules", [])
    assert isinstance(rules, list) and rules, "the ClusterRole grants no rules"
    for rule in rules:
        assert isinstance(rule, dict), "a ClusterRole rule is not a mapping"
        verbs = rule.get("verbs", [])
        assert isinstance(verbs, list) and verbs, f"a rule declares no verbs: {rule}"
        granted |= {str(verb) for verb in verbs}
    assert granted <= READ_ONLY_VERBS, f"the ClusterRole grants {sorted(granted - READ_ONLY_VERBS)}"

    prometheus_spec = object_named(built, "Deployment", PROMETHEUS)["spec"]["template"]["spec"]
    assert "serviceAccountName" not in prometheus_spec, (
        f"the Prometheus pod names a ServiceAccount: {prometheus_spec.get('serviceAccountName')!r}"
    )


def test_prometheus_egress_reaches_exactly_the_four_scrape_targets_and_nothing_off_cluster(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 7, by EQUALITY of `(destination, resolved port)` pairs, plus the two absences it
    names: no `ipBlock` on `prometheus-egress`, and no `monitoring` policy other than `allow-dns`
    and `prometheus-egress` declaring an `egress` block at all.

    It lives here and not in `test_k8s_netpol.py` because that file's §8 authorises three
    constant changes and one rename, and no new case - see the module docstring.

    Catches the destination set widened to the whole `apps` namespace, which is a pod that can
    dial every workload in the cluster holding a 15-day store of everything it has seen. Catches
    a destination missing: a scrape target that is allowed by the ingress side and denied by the
    egress side reads as DOWN with no error in either pod's logs, which is the single most
    expensive failure shape in this namespace. Catches an `ipBlock` added so Prometheus can reach
    a hosted endpoint, which is the exfiltration path tests/infra/test_k8s_netpol.py's
    test_the_gateway_is_the_only_pod_with_a_route_to_the_public_internet keeps to the gateway
    alone - that case sees the CIDR build-wide and this one sees it on the policy, so neither
    depends on the other being written. Catches an egress block added to
    `metrics-ingress-from-monitoring` or to `default-deny`, which would make the deny an allow.
    """
    document = policy(built, MONITORING, "prometheus-egress")
    assert pod_selector(document) == {"matchLabels": {NAME_LABEL: PROMETHEUS}}, (
        f"prometheus-egress selects {pod_selector(document)}"
    )

    destinations = egress_destinations(built, document)
    assert destinations == PROMETHEUS_EGRESS_DESTINATIONS, (
        f"unexpected: {sorted(destinations - PROMETHEUS_EGRESS_DESTINATIONS)}; "
        f"missing: {sorted(PROMETHEUS_EGRESS_DESTINATIONS - destinations)}"
    )

    blocks = ip_blocks(document)
    assert blocks == [], f"prometheus-egress carries ipBlocks: {blocks}"

    with_egress = {
        name_of(candidate)
        for candidate in policies(built)
        if namespace_of(candidate) == MONITORING and "egress" in spec_of(candidate)
    }
    assert with_egress == MONITORING_EGRESS_POLICIES, (
        f"unexpected: {sorted(with_egress - MONITORING_EGRESS_POLICIES)}; "
        f"missing: {sorted(MONITORING_EGRESS_POLICIES - with_egress)}"
    )

    default_deny = policy(built, MONITORING, "default-deny")
    assert pod_selector(default_deny) == {}, "monitoring/default-deny must select every pod"
    assert policy_types(default_deny) == ["Ingress", "Egress"], (
        f"monitoring/default-deny declares {policy_types(default_deny)}"
    )


@pytest.mark.parametrize(
    ("namespace", "expected_peers", "expected_ports", "expected_unreachable"),
    [
        pytest.param(
            MONITORING,
            {(MONITORING, ANY_POD)},
            {
                PROMETHEUS: {("TCP", 9090)},
                NODE_EXPORTER: {("TCP", 9100)},
                KUBE_STATE_METRICS: {("TCP", 8080)},
            },
            # Grafana is selected by this policy (it selects every pod) and reached by none of
            # it: an earlier change decision D names its port `http`, not `metrics`, so the named-port
            # allow resolves against nothing on that pod - which is exactly why 23b gives it
            # `grafana-ingress` of its own. Excluded here by NAME so a fourth scrape target that
            # quietly loses its `metrics` port is still a failure and not a silent exemption.
            {GRAFANA},
            id="monitoring-every-pod-by-port-name",
        ),
        pytest.param(
            PLATFORM,
            {(MONITORING, ANY_POD)},
            {QDRANT: {("TCP", 6333)}},
            # `platform`'s policy selects `qdrant` alone, so there is no pod for the name to miss.
            frozenset(),
            id="platform-qdrant-alone",
        ),
    ],
)
def test_each_metrics_ingress_policy_admits_monitoring_on_exactly_its_own_scrape_port(
    built: list[dict[str, Any]],
    namespace: str,
    expected_peers: set[tuple[str, str]],
    expected_ports: dict[str, set[tuple[str, int]]],
    expected_unreachable: set[str],
) -> None:
    """Criteria 8 and 9, for both `metrics-ingress-from-monitoring` policies, with every port
    RESOLVED against the target pod's own container port rather than compared to a literal. The
    resolved mapping is keyed by the pods the policy's own `podSelector` matches AND the named
    port can reach, so comparing it by equality pins the SCOPE as well as the ports: three pods
    in `monitoring`, `qdrant` alone in `platform`. The pods the name cannot reach are compared by
    equality too, against `expected_unreachable` - an earlier change's Grafana is the only one, for the
    reason written on that parametrize row, and an exemption nobody agreed to is as red as a
    missing allow.

    Both rows live here rather than in `test_k8s_netpol.py` because that file's §8 authorises no
    new case and its ingress parametrization resolves inside `apps` only - see the module
    docstring.

    Catches `monitoring`'s policy written with a numeric port, which can only be right for one of
    the three pods and silently denies the other two: two scrape targets permanently DOWN behind
    a manifest that looks correct, which is the trap a project rule names. Catches its
    `podSelector` narrowed to `prometheus`, which protects the scraper and leaves the two
    exporters closed - and would equally break 23b's Grafana and 23c's Alertmanager reaching
    Prometheus, the reason the policy selects every pod. Catches the `platform` row widened to
    Postgres and Redis or to Qdrant's gRPC 6334, a second protocol surface nothing scrapes.
    Catches either `from` peer written as a bare `podSelector` with no `namespaceSelector`, which
    scopes it to the policy's OWN namespace and so denies every packet Prometheus sends. Catches
    a second port smuggled into either rule, since each rule's resolved ports are compared by
    equality. Catches Grafana's port renamed to `metrics`, which would put a UI port inside the
    scrape allow and which shows up here as an empty unreachable set rather than as an extra row.
    """
    document = policy(built, namespace, "metrics-ingress-from-monitoring")
    rules = rules_of(document, "ingress")
    assert rules, f"{namespace}/metrics-ingress-from-monitoring declares no ingress rule"
    assert "egress" not in spec_of(document), (
        f"{namespace}/metrics-ingress-from-monitoring declares an egress block"
    )

    peers: set[tuple[str, str]] = set()
    for rule in rules:
        listed = peers_of(rule, "from")
        assert listed, f"{namespace}/metrics-ingress has a rule with no `from`, admitting everyone"
        for peer in listed:
            peers |= peer_targets(built, peer, namespace)
        unreachable = unreachable_workloads(built, document, rule)
        assert unreachable == set(expected_unreachable), (
            f"{namespace}/metrics-ingress-from-monitoring does not reach "
            f"{sorted(unreachable)}, and the plans agree on {sorted(expected_unreachable)}"
        )
        assert resolved_ports(built, document, rule) == expected_ports, (
            f"{namespace}/metrics-ingress-from-monitoring resolves to "
            f"{resolved_ports(built, document, rule)}"
        )
    assert peers == expected_peers, (
        f"unexpected: {sorted(peers - expected_peers)}; missing: {sorted(expected_peers - peers)}"
    )
