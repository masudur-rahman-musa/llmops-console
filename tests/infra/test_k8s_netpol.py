r"""`infra/k8s/base/networkpolicies.yaml`: the eighteen NetworkPolicies an earlier change and
23c agree, read off what kustomize actually emits rather than off the file, so a policy written
and never listed in `base/kustomization.yaml`'s `resources` is the same failure as a policy never
written.

The stream comes from the session-scoped `built` fixture in `tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, shared with
`test_k8s_base.py`, `test_k8s_apps.py` and `test_k8s_platform.py`. Nothing here builds again and
nothing here reads a raw file.

Everything is asserted by EQUALITY, never by containment (decision D): Kubernetes unions allows,
so an extra `to` peer, an extra port or an extra `ipBlock` is exactly as dangerous as a missing
one, and only equality sees both. The guarantee #54 asks for is a property of what is ABSENT.

Named ports are the reason this file resolves rather than compares. One NetworkPolicy has one
`podSelector`, and its `ports` apply to every pod that selector matches - but `llama-ingress`
protects two pods listening on different numbers (llama 8082, llama-embed 8083) and
`helpdesk-api-ingress` protects two more (8000, 8001). The manifest resolves that by NAMING the
port (`port: http`), which Kubernetes resolves against each target pod's own container port. So
`resolved_ports` looks the name up in the pod templates the same build emitted and the
assertions read in numbers, while the manifest reads in names. A literal number asserted against
those rules would either be wrong for one of the two pods or would force the manifest into a
shape that cannot express criterion 6 at all. `platform`'s `data-ingress-from-apps` is the
exception and is asserted numerically: its three targets name their ports differently
(`postgres`, `redis`, `http`), so there is no shared name to resolve.

`helpdesk-api-ingress` is pinned by name in `POLICY_NAMES` and by the selector scan, but its
`from` peers are not pinned here: criterion 6 states peers for `llama-ingress`,
`gateway-ingress` and `helpdesk-web-ingress` only, and a peer set invented for the fourth would
be red against a correct manifest rather than against a wrong one.

The selector scan is the unhappy path and the only case here that can see a typo: a selector
matching no pod builds, validates and schema-checks identically to a correct one, and is an
allow that allows nothing - a deny nobody reports. It skips peers pointing at `kube-system`,
because this build writes no pod templates there (Traefik and kube-dns are k3s's own), so a scan
that did not skip would be red against a correct manifest.

The root kustomization's `labels:` block sets `includeSelectors: false`, so
`app.kubernetes.io/part-of` and `app.kubernetes.io/managed-by` reach each policy's
`metadata.labels` and never its `spec.podSelector`. Nothing here asserts on them.
"""

from __future__ import annotations

from typing import Any

import pytest

#: The kind this whole module reads. Every filter starts here.
NETWORK_POLICY = "NetworkPolicy"

APPS = "apps"
PLATFORM = "platform"

#: The scrape plane's namespace: Prometheus and the two exporters live here, which is
#: why it joins `NAMESPACES_WITH_POD_TEMPLATES` below.
MONITORING = "monitoring"

#: k3s's own namespace: Traefik (the ingress path into `gateway` and `helpdesk-web`) and
#: kube-dns live here. This build emits no pod template in it, which is why the selector scan
#: skips peers scoped to it.
KUBE_SYSTEM = "kube-system"

#: The namespaces this build writes pod templates for, and therefore the only ones the selector
#: scan can resolve a peer against.
NAMESPACES_WITH_POD_TEMPLATES = frozenset({APPS, PLATFORM, MONITORING})

#: The one label every pod template in this build carries and the only one any selector keys
#: off. A workload's value here is also its Deployment name and its Service name.
NAME_LABEL = "app.kubernetes.io/name"

#: Kubernetes auto-labels every namespace with its own name, which is how a cross-namespace peer
#: names the namespace it means.
NAMESPACE_NAME_LABEL = "kubernetes.io/metadata.name"

GATEWAY = "gateway"
#: An earlier change's own pod, the second and last one in the build with a route to the public internet.
ALERTMANAGER = "alertmanager"
LLAMA = "llama"
LLAMA_EMBED = "llama-embed"
HELPDESK_AGENT_API = "helpdesk-agent-api"
HELPDESK_MCP_SERVER = "helpdesk-mcp-server"
HELPDESK_SCHEDULER = "helpdesk-scheduler"
HELPDESK_WEB = "helpdesk-web"
POSTGRES = "postgres"
REDIS = "redis"
QDRANT = "qdrant"

#: Criterion 5's four workloads: the policies that select any of them are the ones the
#: no-`ipBlock` scan runs over. `helpdesk-scheduler` serves no port and is here because it is
#: still a Helpdesk pod whose EGRESS criterion 5 constrains.
HELPDESK_WORKLOADS = frozenset(
    {HELPDESK_AGENT_API, HELPDESK_MCP_SERVER, HELPDESK_SCHEDULER, HELPDESK_WEB}
)

#: Criterion 1, the `apps` half: eight policies, by name.
APPS_POLICIES = frozenset(
    {
        "default-deny",
        "allow-dns",
        "gateway-ingress",
        "gateway-egress",
        "helpdesk-web-ingress",
        "helpdesk-api-ingress",
        "helpdesk-egress",
        "llama-ingress",
    }
)

#: Criterion 1, the `platform` half: three policies, by name. `metrics-ingress-from-monitoring`
#: is an earlier change's - a NEW policy for Qdrant rather than a widening of `data-ingress-from-apps`, so
#: that policy's `apps`-only guarantee is untouched.
PLATFORM_POLICIES = frozenset(
    {"default-deny-ingress", "data-ingress-from-apps", "metrics-ingress-from-monitoring"}
)

#: An earlier change criterion 6, an earlier change criterion 7 and an earlier change's own criterion, the `monitoring`
#: half: SEVEN policies, by name. Its own catch-all in both directions, DNS, the named-port
#: scrape ingress every pod in the namespace carries, Prometheus's egress to what it scrapes plus
#: Alertmanager, Grafana's own pair - 23b decision D leaves Grafana publishing `http` rather than
#: `metrics`, so it is the first pod here that the namespace-wide named-port allow cannot cover
#: and it gets an ingress and an egress policy of its own - and 23c's `alertmanager-egress`,
#: scoped to the one pod that needs a route to the public internet (decision B). Alertmanager
#: needs no ingress policy of its own: `metrics-ingress-from-monitoring` already covers it, since
#: its container port is named `metrics` like the three scrape targets.
MONITORING_POLICIES = frozenset(
    {
        "default-deny",
        "allow-dns",
        "metrics-ingress-from-monitoring",
        "prometheus-egress",
        "grafana-ingress",
        "grafana-egress",
        "alertmanager-egress",
    }
)

#: Criterion 1 as `(namespace, name)` pairs, which is the identity a NetworkPolicy actually has:
#: two policies of the same name in different namespaces are two different objects - which is
#: exactly the shape `metrics-ingress-from-monitoring` has, one in `platform` and one in
#: `monitoring`.
POLICY_NAMES: frozenset[tuple[str, str]] = frozenset(
    {(APPS, name) for name in APPS_POLICIES}
    | {(PLATFORM, name) for name in PLATFORM_POLICIES}
    | {(MONITORING, name) for name in MONITORING_POLICIES}
)

#: Criterion 2: `apps`'s catch-all closes both directions, `platform`'s closes ingress only
#: (decision A - `platform` has no egress policy at all, so declaring `Egress` there would deny
#: every outbound connection the data pods make).
APPS_DEFAULT_DENY_POLICY_TYPES = frozenset({"Ingress", "Egress"})
PLATFORM_DEFAULT_DENY_POLICY_TYPES = frozenset({"Ingress"})

#: Criterion 3: kube-dns answers on both protocols on the same number, and a policy allowing
#: only UDP breaks every response over 512 bytes - intermittently, and only under load.
DNS_PORTS: frozenset[tuple[str, int]] = frozenset({("UDP", 53), ("TCP", 53)})

#: Criterion 4 and decision E: the public destination, its four carve-outs and its one port.
#: `169.254.0.0/16` is the one that closes the `169.254.169.254` metadata route the threat model
#: names under SSRF via `base_url`; the other three keep the gateway out of the cluster's private
#: space except where a named policy lets it back in.
PUBLIC_CIDR = "0.0.0.0/0"
PUBLIC_EXCEPT: frozenset[str] = frozenset(
    {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"}
)
PUBLIC_PORTS: frozenset[tuple[str, int]] = frozenset({("TCP", 443)})

#: Criterion 5: every destination the four Helpdesk workloads may reach, as
#: `(workload, resolved port)`. The DNS allow is NOT here - it lives in `allow-dns`, which
#: selects every pod in `apps` rather than the Helpdesk pods specifically.
HELPDESK_EGRESS_DESTINATIONS: frozenset[tuple[str, int]] = frozenset(
    {
        (GATEWAY, 8080),
        (HELPDESK_AGENT_API, 8000),
        (HELPDESK_MCP_SERVER, 8001),
        (POSTGRES, 5432),
        (REDIS, 6379),
        (QDRANT, 6333),
    }
)

#: Criterion 7: the three data ports `platform` admits from `apps`, and nothing else. Qdrant's
#: 6334 (gRPC) is absent on purpose - nothing in `apps` speaks it.
DATA_PORTS: frozenset[int] = frozenset({5432, 6379, 6333})

#: The peer marker for "every pod in that namespace": a peer with no `podSelector`, or with an
#: empty one. Kept as a marker rather than expanded into the namespace's workload list so that a
#: later row adding a pod to `apps` cannot redden a policy it did not touch.
ANY_POD = "*"


def policies(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every built NetworkPolicy, in build order."""
    return [document for document in documents if document.get("kind") == NETWORK_POLICY]


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


def policy(documents: list[dict[str, Any]], namespace: str, name: str) -> dict[str, Any]:
    """The one NetworkPolicy `namespace/name`, asserting the build emits exactly one.

    Fails with a readable message listing what the build DID emit, rather than raising, so a
    module run before the manifest lands is red on the missing policy and not on a KeyError.
    """
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


def pod_templates(documents: list[dict[str, Any]], namespace: str) -> dict[str, dict[str, Any]]:
    """Every pod template in `namespace`, keyed by its `app.kubernetes.io/name` label.

    Found by the PRESENCE of `spec.template.spec` rather than by a list of kinds, so a Job or a
    DaemonSet a later row adds is scanned without editing this file.
    """
    templates: dict[str, dict[str, Any]] = {}
    for document in documents:
        spec = document.get("spec")
        if not isinstance(spec, dict) or namespace_of(document) != namespace:
            continue
        template = spec.get("template")
        if not isinstance(template, dict) or not isinstance(template.get("spec"), dict):
            continue
        metadata = template.get("metadata", {})
        labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
        assert isinstance(labels, dict), f"{name_of(document)}'s pod template has no labels"
        workload = str(labels.get(NAME_LABEL, ""))
        assert workload, f"{name_of(document)}'s pod template carries no {NAME_LABEL}"
        templates[workload] = template
    return templates


def template_labels(template: dict[str, Any]) -> dict[str, str]:
    """A pod template's labels, the ones a `podSelector` has to match."""
    metadata = template.get("metadata", {})
    labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
    assert isinstance(labels, dict)
    return {str(key): str(value) for key, value in labels.items()}


def container_ports(template: dict[str, Any]) -> dict[str, int]:
    """A pod template's NAMED container ports, `name -> number`, across every container."""
    spec = template["spec"]
    assert isinstance(spec, dict)
    numbers: dict[str, int] = {}
    for key in ("initContainers", "containers"):
        listed = spec.get(key, [])
        if not isinstance(listed, list):
            continue
        for container in listed:
            if not isinstance(container, dict):
                continue
            ports = container.get("ports", [])
            if not isinstance(ports, list):
                continue
            for entry in ports:
                if isinstance(entry, dict) and entry.get("name") is not None:
                    numbers[str(entry["name"])] = int(entry["containerPort"])
    return numbers


def selector_matches(selector: dict[str, Any], labels: dict[str, str]) -> bool:
    """Whether a `podSelector` matches a pod template's labels.

    Understands `matchLabels` and `matchExpressions` with `In`, the only two forms this manifest
    uses; any other operator asserts rather than silently reporting a match, because a scan that
    quietly ignored a form would report every selector written in it as fine.
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
    templates = pod_templates(documents, namespace)
    return {
        workload
        for workload, template in templates.items()
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
    """A peer as `(namespace, workload)` pairs, `workload` being `ANY_POD` when the peer names
    no `podSelector` or an empty one - namespace-wide, which is a shape criterion 6 states for
    the `kube-system` and in-`apps` peers of `gateway-ingress`."""
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


def resolved_ports(
    documents: list[dict[str, Any]], document: dict[str, Any], rule: dict[str, Any]
) -> dict[str, set[tuple[str, int]]]:
    """An INGRESS rule's ports resolved per pod the policy's own `podSelector` protects.

    `{workload: {(protocol, number)}}`. This is the shape criterion 6 needs: one `llama-ingress`
    rule saying `port: http` is 8082 on `llama` and 8083 on `llama-embed`, and only a per-pod
    resolution can assert both from the one rule.
    """
    namespace = namespace_of(document)
    assert namespace is not None, f"{name_of(document)} declares no metadata.namespace"
    templates = pod_templates(documents, namespace)
    selector = pod_selector(document)
    label = f"{namespace}/{name_of(document)}"
    return {
        workload: ports_against(rule, container_ports(template), label)
        for workload, template in templates.items()
        if selector_matches(selector, template_labels(template))
    }


def egress_destinations(
    documents: list[dict[str, Any]], document: dict[str, Any]
) -> set[tuple[str, int]]:
    """An egress policy's destinations as `(workload-or-namespace, resolved port)`.

    Every peer of a rule is crossed with every port of that rule, which is what the rule means:
    grouping three data services and three numbers into one rule allows all nine, and this set
    says so rather than flattering the manifest into the six the criterion names.
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


def rules_with_ip_block(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every RULE of a policy carrying an `ipBlock` peer - the rule, so its ports are readable
    beside the CIDR they apply to."""
    return [
        rule
        for direction, key in (("ingress", "from"), ("egress", "to"))
        for rule in rules_of(document, direction)
        if any(isinstance(peer.get("ipBlock"), dict) for peer in peers_of(rule, key))
    ]


def selector_scope(
    document: dict[str, Any],
) -> list[tuple[str, dict[str, Any], str]]:
    """Every `podSelector` a policy declares, as `(namespace it resolves in, selector, label)`.

    Covers the policy's own top-level selector and every one nested in a `from` or a `to`. The
    nested ones resolve in the namespace their `namespaceSelector` names, otherwise in the
    policy's own - the scoping rule that makes a cross-namespace peer checkable at all.
    """
    namespace = namespace_of(document)
    assert namespace is not None, f"{name_of(document)} declares no metadata.namespace"
    label = f"{namespace}/{name_of(document)}"
    scoped: list[tuple[str, dict[str, Any], str]] = [
        (namespace, pod_selector(document), f"{label} spec.podSelector")
    ]
    for direction, key in (("ingress", "from"), ("egress", "to")):
        for index, rule in enumerate(rules_of(document, direction)):
            for peer in peers_of(rule, key):
                selector = peer.get("podSelector")
                if not isinstance(selector, dict):
                    continue
                scoped.append(
                    (
                        peer_namespace(peer, namespace),
                        selector,
                        f"{label} {direction}[{index}].{key} podSelector",
                    )
                )
    return scoped


@pytest.mark.parametrize(
    ("namespace", "expected"),
    [
        pytest.param(APPS, APPS_POLICIES, id="apps-eight"),
        pytest.param(PLATFORM, PLATFORM_POLICIES, id="platform-three"),
        pytest.param(MONITORING, MONITORING_POLICIES, id="monitoring-seven"),
    ],
)
def test_each_namespace_emits_exactly_the_policies_the_plan_names(
    built: list[dict[str, Any]], namespace: str, expected: frozenset[str]
) -> None:
    """Criterion 1, by SET EQUALITY of NetworkPolicy names within one namespace. Catches the
    policies written into `base/networkpolicies.yaml` and the file never added to
    `base/kustomization.yaml`'s `resources` - valid YAML that reaches no cluster and that no
    schema check can see. Catches a policy that keeps `namespace: apps` when it belongs in
    `platform`, since a policy only constrains pods in its own namespace and one filed in the
    wrong one leaves its real target open. Catches an extra policy nobody agreed to, which
    equality makes as red as a missing one because Kubernetes unions allows."""
    emitted = {
        name_of(document) for document in policies(built) if namespace_of(document) == namespace
    }
    assert emitted == expected, (
        f"unexpected: {sorted(emitted - expected)}; missing: {sorted(expected - emitted)}"
    )


def test_the_whole_build_emits_exactly_the_eighteen_policies_and_nowhere_else(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 1 over the WHOLE stream as `(namespace, name)`. Catches a nineteenth policy
    landing in a fourth namespace, which the three per-namespace rows above cannot see because each
    filters to its own namespace first; and catches a policy emitted with no
    `metadata.namespace` at all, which the API server would file into whatever namespace the
    apply ran against rather than the one the plan names."""
    emitted = {(str(namespace_of(document)), name_of(document)) for document in policies(built)}
    assert emitted == POLICY_NAMES, (
        f"unexpected: {sorted(emitted - POLICY_NAMES)}; missing: {sorted(POLICY_NAMES - emitted)}"
    )


def test_the_default_deny_policies_close_the_directions_the_plan_names(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 2 and decision A. Catches an `apps` catch-all written with `policyTypes:
    [Ingress]` only, which leaves every pod's EGRESS unrestricted and makes criterion 4's whole
    `ipBlock` argument decorative - the gateway would not be the only pod that can reach the
    internet, every pod could. Catches a `podSelector` on the catch-all that names a workload
    instead of being empty, which denies one pod and leaves the rest open. Catches a `platform`
    policy that declares an `egress` block or an `Egress` policyType, which would deny every
    outbound connection Postgres, Redis and Qdrant make the moment it is applied."""
    apps_default_deny = policy(built, APPS, "default-deny")
    assert pod_selector(apps_default_deny) == {}, "apps/default-deny must select every pod"
    assert set(policy_types(apps_default_deny)) == APPS_DEFAULT_DENY_POLICY_TYPES
    assert len(policy_types(apps_default_deny)) == 2, "policyTypes repeats an entry"

    platform_default_deny = policy(built, PLATFORM, "default-deny-ingress")
    assert pod_selector(platform_default_deny) == {}, "platform's catch-all must select every pod"
    assert set(policy_types(platform_default_deny)) == PLATFORM_DEFAULT_DENY_POLICY_TYPES
    assert len(policy_types(platform_default_deny)) == 1, "policyTypes repeats an entry"

    with_egress = sorted(
        name_of(document)
        for document in policies(built)
        if namespace_of(document) == PLATFORM
        and ("egress" in spec_of(document) or "Egress" in policy_types(document))
    )
    assert with_egress == [], f"platform policies declaring egress: {with_egress}"


def test_the_dns_policy_opens_both_protocols_on_53_for_every_pod_in_apps(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 3. Catches a DNS allow written with a `podSelector` naming one workload, which
    leaves every other pod in `apps` unable to resolve anything once `default-deny` is applied -
    a failure that looks like a provider outage, not like a policy. Catches UDP 53 alone, which
    works until a response exceeds 512 bytes and the resolver retries over TCP; and catches an
    extra port smuggled into the same rule, since equality makes an added allow as red as a
    missing one."""
    allow_dns = policy(built, APPS, "allow-dns")
    assert pod_selector(allow_dns) == {}, "allow-dns must select every pod in apps"
    rules = rules_of(allow_dns, "egress")
    assert rules, "allow-dns declares no egress rule"
    opened: set[tuple[str, int]] = set()
    for rule in rules:
        opened |= numeric_ports(rule, f"{APPS}/allow-dns")
    assert opened == DNS_PORTS, f"DNS ports are {sorted(opened)}, not {sorted(DNS_PORTS)}"


def test_the_gateway_and_alertmanager_are_the_only_pods_with_a_route_to_the_public_internet(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 4 and decision E, widened by an earlier change decision B: scanned over EVERY policy in
    the build rather than read off `gateway-egress` and `alertmanager-egress` by name - the claim
    is that no OTHER policy has one, and a case that opened the two by name could not see a third
    `ipBlock` added anywhere else. Both are asserted to share the SAME excluded-ranges list, by
    VALUE (a project rule - copied, not re-derived): `alertmanager-egress`'s own shape is pinned
    a second time, more narrowly, by `tests/infra/test_k8s_alerting.py`'s
    test_alertmanager_is_the_second_and_last_pod_with_internet_egress.

    Catches a later row granting the Helpdesk or llama a public egress. Catches an `except` list
    missing `169.254.0.0/16` on EITHER policy, which reopens the `169.254.169.254` metadata route
    SSRF via `base_url` reaches. Catches port 80 or an unported rule beside 443 on either, and
    catches a rule hung on a `podSelector` that is not the workload's own."""
    reaching = {
        (str(namespace_of(document)), name_of(document))
        for document in policies(built)
        if ip_blocks(document)
    }
    assert reaching == {(APPS, "gateway-egress"), (MONITORING, "alertmanager-egress")}, (
        f"policies with an ipBlock: {sorted(reaching)}"
    )

    gateway_egress = policy(built, APPS, "gateway-egress")
    assert pod_selector(gateway_egress) == {"matchLabels": {NAME_LABEL: GATEWAY}}
    alertmanager_egress = policy(built, MONITORING, "alertmanager-egress")
    assert pod_selector(alertmanager_egress) == {"matchLabels": {NAME_LABEL: ALERTMANAGER}}

    for label, document in (
        (f"{APPS}/gateway-egress", gateway_egress),
        (f"{MONITORING}/alertmanager-egress", alertmanager_egress),
    ):
        rules = rules_with_ip_block(document)
        assert len(rules) == 1, f"{label}: {len(rules)} rules carry an ipBlock, not 1"
        blocks = ip_blocks(document)
        assert len(blocks) == 1, f"{label}: {len(blocks)} ipBlocks, not 1"
        assert str(blocks[0].get("cidr")) == PUBLIC_CIDR, (
            f"{label}: cidr is {blocks[0].get('cidr')!r}"
        )
        excepted = {str(entry) for entry in blocks[0].get("except", [])}
        assert excepted == PUBLIC_EXCEPT, (
            f"{label} unexpected: {sorted(excepted - PUBLIC_EXCEPT)}; "
            f"missing: {sorted(PUBLIC_EXCEPT - excepted)}"
        )
        ports = numeric_ports(rules[0], label)
        assert ports == PUBLIC_PORTS, f"{label}: public egress ports are {sorted(ports)}"


def test_the_helpdesk_may_reach_exactly_six_destinations_and_no_ip_block(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 5 and decision B, by EQUALITY of `(destination, resolved port)` pairs. Catches
    a `to` peer left in that lets the Helpdesk reach a service the plan does not list - the
    provider-key path #54 exists to close, since a Helpdesk pod that can dial a provider
    directly bypasses every ledger, redaction and budget control in the gateway. Catches a
    destination missing, which is an outage. Catches an `ipBlock` on any policy selecting a
    Helpdesk workload, which is the same bypass written a second way and which the destination
    set alone would not see, because an `ipBlock` peer contributes no workload pair."""
    destinations = egress_destinations(built, policy(built, APPS, "helpdesk-egress"))
    assert destinations == HELPDESK_EGRESS_DESTINATIONS, (
        f"unexpected: {sorted(destinations - HELPDESK_EGRESS_DESTINATIONS)}; "
        f"missing: {sorted(HELPDESK_EGRESS_DESTINATIONS - destinations)}"
    )

    offending = sorted(
        f"{namespace_of(document)}/{name_of(document)}"
        for document in policies(built)
        if namespace_of(document) is not None
        and matched_workloads(built, pod_selector(document), str(namespace_of(document)))
        & HELPDESK_WORKLOADS
        and ip_blocks(document)
    )
    assert offending == [], (
        f"policies selecting a Helpdesk pod and carrying an ipBlock: {offending}"
    )


@pytest.mark.parametrize(
    ("name", "expected_peers", "expected_ports"),
    [
        pytest.param(
            "llama-ingress",
            {(APPS, GATEWAY)},
            {LLAMA: {("TCP", 8082)}, LLAMA_EMBED: {("TCP", 8083)}},
            id="llama-from-gateway-only",
        ),
        pytest.param(
            "gateway-ingress",
            {(KUBE_SYSTEM, ANY_POD), (APPS, ANY_POD), (MONITORING, ANY_POD)},
            {GATEWAY: {("TCP", 8080)}},
            id="gateway-from-traefik-apps-and-monitoring",
        ),
        pytest.param(
            "helpdesk-web-ingress",
            {(KUBE_SYSTEM, ANY_POD)},
            {HELPDESK_WEB: {("TCP", 3000)}},
            id="helpdesk-web-from-traefik",
        ),
    ],
)
def test_each_ingress_policy_admits_exactly_its_named_sources_on_its_own_port(
    built: list[dict[str, Any]],
    name: str,
    expected_peers: set[tuple[str, str]],
    expected_ports: dict[str, set[tuple[str, int]]],
) -> None:
    """Criterion 6, with every port RESOLVED against the target pod's own container port rather
    than compared to a literal. Catches `llama-ingress` written with a numeric port, which can
    only be right for one of the two llama pods and silently denies the other - the embedding
    endpoint answering nothing while the manifest looks correct. Catches a `from` peer opening
    llama to every pod in `apps` instead of to the gateway alone, which is the hop that lets a
    Helpdesk pod use the local model without a trace or a budget. Catches `gateway-ingress`
    losing its `kube-system` peer, which is Traefik denied and the public host dead, or losing
    the in-namespace peer, which is every internal caller denied. Catches a second port added to
    any of the three, since each rule's resolved ports are compared by equality."""
    document = policy(built, APPS, name)
    rules = rules_of(document, "ingress")
    assert rules, f"{APPS}/{name} declares no ingress rule"
    peers: set[tuple[str, str]] = set()
    for rule in rules:
        listed = peers_of(rule, "from")
        assert listed, f"{APPS}/{name} has an ingress rule with no `from`, which admits everyone"
        for peer in listed:
            peers |= peer_targets(built, peer, APPS)
        assert resolved_ports(built, document, rule) == expected_ports, (
            f"{APPS}/{name} resolves to {resolved_ports(built, document, rule)}"
        )
    assert peers == expected_peers, (
        f"unexpected: {sorted(peers - expected_peers)}; missing: {sorted(expected_peers - peers)}"
    )


def test_the_data_namespace_admits_apps_on_exactly_the_three_data_ports(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 7, asserted NUMERICALLY because the three data pods name their ports
    differently (`postgres`, `redis`, `http`) and there is no shared name to resolve. Catches
    Qdrant's gRPC 6334 admitted beside 6333, which is a second protocol surface nothing in
    `apps` speaks. Catches the peer written as a bare `podSelector` with no `namespaceSelector`,
    which under `platform`'s scoping means pods in `platform` and denies every caller in `apps`
    - the whole point of the policy. Catches a peer naming a namespace this build does not
    write, and catches a missing port, which is one data service unreachable."""
    document = policy(built, PLATFORM, "data-ingress-from-apps")
    rules = rules_of(document, "ingress")
    assert rules, f"{PLATFORM}/data-ingress-from-apps declares no ingress rule"
    peers: set[tuple[str, str]] = set()
    pairs: set[tuple[str, int]] = set()
    for rule in rules:
        listed = peers_of(rule, "from")
        assert listed, "data-ingress-from-apps has an ingress rule with no `from`"
        for peer in listed:
            peers |= peer_targets(built, peer, PLATFORM)
        pairs |= numeric_ports(rule, f"{PLATFORM}/data-ingress-from-apps")
    assert peers == {(APPS, ANY_POD)}, f"admitted sources are {sorted(peers)}"
    assert {port for _, port in pairs} == DATA_PORTS, f"admitted ports are {sorted(pairs)}"
    assert {protocol for protocol, _ in pairs} == {"TCP"}, f"admitted protocols: {sorted(pairs)}"


def test_every_selector_in_the_build_matches_a_pod_that_exists(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 8, the unhappy path: every `podSelector` in the build - a policy's own and
    every one nested in a `from` or a `to` - matches at least one pod template in the namespace
    it resolves in. Catches a typo'd label (`app.kubernetes/name`, `helpdesk-api` for
    `helpdesk-agent-api`), which builds, validates and schema-checks exactly like a correct
    selector: as a policy target it protects nothing, and as a peer it is an allow that allows
    nothing, so the traffic it was written to permit is denied by `default-deny` with no error
    anywhere. Catches a cross-namespace peer whose `namespaceSelector` names one namespace while
    its `podSelector` names a workload living in another. Peers scoped to `kube-system` are
    skipped, with the reason in the code: this build writes no pod template there, so a scan
    that did not skip would be red against a correct manifest."""
    empty_checked: list[str] = []
    for document in policies(built):
        for namespace, selector, label in selector_scope(document):
            # kube-system carries k3s's own Traefik and kube-dns; this build emits no pod
            # template in it, so there is nothing here to resolve a selector against.
            if namespace not in NAMESPACES_WITH_POD_TEMPLATES:
                assert namespace == KUBE_SYSTEM, f"{label} names unknown namespace {namespace}"
                continue
            templates = pod_templates(built, namespace)
            if selector == {}:
                assert templates, f"{label} is empty and {namespace} has no pod template"
                empty_checked.append(label)
                continue
            matched = matched_workloads(built, selector, namespace)
            assert matched, f"{label} matches no pod in {namespace}: {sorted(templates)}"
    assert empty_checked, "no empty podSelector was scanned; default-deny should have one"
