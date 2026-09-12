r"""`infra/k8s/platform`: the three data services, the two ClusterIssuers and the three
Traefik middlewares an earlier change agrees, read off what kustomize actually emits.

The stream comes from the session-scoped `built` fixture in `tests/infra/conftest.py`
(decision G): one `kubectl kustomize infra/k8s` for the whole session, parsed once, shared
with `test_k8s_base.py`. Nothing here reads the raw files under `platform/` except the
credential scan, which is a scan of what is WRITTEN rather than of what is built - a password
removed by a later patch would still be in git, and a build that never emits the file would
hide it entirely.

Quantities are compared as STRINGS where §1 states them (`512Mi`, `200m`, `1Gi`): the API
server compares quantities, so a test that parsed first would accept `524288Ki` for a table
that says `512Mi`, and the budget table is prose a reader checks by eye. Parsing happens in
exactly one case, the subtotals, which DERIVES its three sums from the built pod templates
plus `CERT_MANAGER_RESERVE` and compares them to the `platform` `ResourceQuota` read from the
SAME build. The totals `1376Mi / 530m / 2688Mi` appear nowhere in this file:
restating them beside the per-workload rows would let the two disagree. CPU is summed in whole
millicores, never floats - `0.2 + 0.1 + 0.2 + 0.03` is not `0.53` in binary floating point.

The parsing helpers below (`read`, `objects_of_kind`, `name_of`, `namespace_of`,
`memory_bytes`, `cpu_millicores`, `quota_hard`) are copies of `test_k8s_base.py`'s, not
imports: a test module importing another test module makes the second one's collection a
dependency of the first's, and this repo already copies `trigger()` across three files for the
same reason (a project rule sends a shared helper where the imports already run, and nothing
here imports anything there). `conftest.py` holds the fixture and nothing else, so a reader
opening either module sees every rule it applies.

§2's kustomize mechanic is what the selector case exists for: the root's `labels:` block sets
`includeSelectors: false` and does not set `includeTemplates`, so the two common labels reach
each object's `metadata.labels` and NOT its pod template. Each workload therefore writes its
own `app.kubernetes.io/name` on `spec.selector.matchLabels`, on the Service's `spec.selector`
and on `spec.template.metadata.labels`, and a mismatch is a workload that rolls out zero
matching pods while reporting nothing wrong.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_ROOT = REPO_ROOT / "infra" / "k8s"
PLATFORM_DIR = K8S_ROOT / "platform"
K8S_README = K8S_ROOT / "README.md"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

#: The namespace every object this plan adds belongs to, except the two cluster-scoped issuers.
PLATFORM = "platform"

#: The kinds an earlier change's `base/` already emits. Criterion 1 is asserted by set equality over
#: everything the build contains that is NOT one of these, so it covers what `platform/` adds
#: without hardcoding a total object count that every later row would have to edit.
BASE_KINDS = frozenset({"Namespace", "ResourceQuota", "LimitRange"})

#: Criterion 1, as `(kind, name, namespace)`. `None` is a cluster-scoped object. Set EQUALITY:
#: a file left out of `platform/kustomization.yaml`'s `resources` and an object nobody agreed
#: to are the same failure, and only equality sees both. Scoped to what `platform/` puts in
#: the build - its own namespace plus the two cluster-scoped issuers - rather than to
#: "everything that is not a base kind", which was the same set only while `platform/` was the
#: last directory in the root's `resources`.
PLATFORM_OBJECTS: frozenset[tuple[str, str, str | None]] = frozenset(
    {
        ("StatefulSet", "postgres", PLATFORM),
        ("Service", "postgres", PLATFORM),
        ("Deployment", "redis", PLATFORM),
        ("Service", "redis", PLATFORM),
        ("StatefulSet", "qdrant", PLATFORM),
        ("Service", "qdrant", PLATFORM),
        ("Middleware", "basic-auth", PLATFORM),
        ("Middleware", "rate-limit", PLATFORM),
        ("Middleware", "security-headers", PLATFORM),
        ("ClusterIssuer", "letsencrypt-staging", None),
        ("ClusterIssuer", "letsencrypt-prod", None),
        # Row 21a, from `base/networkpolicies.yaml`: they are written in `base/` and land in
        # this namespace, and this case scopes by namespace rather than by directory.
        # `tests/infra/test_k8s_netpol.py` owns what they contain; these are the two names.
        ("NetworkPolicy", "default-deny-ingress", PLATFORM),
        ("NetworkPolicy", "data-ingress-from-apps", PLATFORM),
        # Row 23a, the same way: Qdrant's scrape is a NEW policy in this namespace rather than a
        # widening of `data-ingress-from-apps`, so that policy's `apps`-only guarantee is
        # untouched and this is a third name here.
        ("NetworkPolicy", "metrics-ingress-from-monitoring", PLATFORM),
        # Row 23a's RBAC, cluster-scoped and therefore inside this case's `None` half although it
        # belongs to `monitoring/`. `tests/infra/test_k8s_monitoring.py` owns what it grants and
        # that there is no third ClusterRole in the build; these are the two names.
        ("ClusterRole", "kube-state-metrics", None),
        ("ClusterRoleBinding", "kube-state-metrics", None),
    }
)

#: Criterion 3's budget table, by value and per workload. Compared as
#: strings; the subtotals case is the only place these are parsed.
WORKLOAD_RESOURCES: dict[str, dict[str, str]] = {
    "postgres": {"requests.memory": "512Mi", "requests.cpu": "200m", "limits.memory": "1Gi"},
    "redis": {"requests.memory": "256Mi", "requests.cpu": "100m", "limits.memory": "512Mi"},
    "qdrant": {"requests.memory": "512Mi", "requests.cpu": "200m", "limits.memory": "768Mi"},
}

#: Decision A: cert-manager is installed by helm into `platform` with these per-pod values, so
#: this repo writes no manifest for it and holds its share of the quota as a named reserve.
#: Three pods - controller, webhook, cainjector - each with the same three quantities.
CERT_MANAGER_RESERVE: dict[str, str | int] = {
    "pods": 3,
    "requests.memory": "32Mi",
    "requests.cpu": "10m",
    "limits.memory": "128Mi",
}

#: Decision C: a CPU limit throttles rather than kills, so no container declares one. Criterion
#: 2 asserts its ABSENCE over every pod template, which is the manifest half of 20a's quota.
FORBIDDEN_CONTAINER_KEY = "limits.cpu"

#: Criterion 2: the three a container must declare, as `resources` sub-paths.
REQUIRED_CONTAINER_KEYS = ("requests.memory", "requests.cpu", "limits.memory")

#: Decision E: the workload's k8s image must equal the image `docker-compose.yml` runs under
#: the same service name. Both sides are READ; neither tag is written here, so the case cannot
#: pass by agreeing with itself.
COMPOSE_SERVICES = {"postgres": "postgres", "redis": "redis", "qdrant": "qdrant"}

#: Criterion 6: the ports each Service publishes, pinned as ints because a Service port is a
#: number and not a quantity. Qdrant serves HTTP on 6333 and gRPC on 6334.
SERVICE_PORTS = {"postgres": {5432}, "redis": {6379}, "qdrant": {6333, 6334}}

#: Criterion 7, Traefik v3's group (k3s v1.31.5+k3s1 ships v3, so never `traefik.containo.us`).
MIDDLEWARE_API_VERSION = "traefik.io/v1alpha1"

#: Criterion 7 and decision D: the basic-auth middleware names a Secret this repo never writes.
BASIC_AUTH_SECRET = "traefik-basic-auth"

#: Criterion 7's rate-limit triple, by value. `average` and `burst` are ints, `period` a
#: quantity string - Traefik reads `50` and `"50"` differently in one of the two positions.
RATE_LIMIT = {"average": 50, "burst": 100, "period": "1s"}

#: Criterion 7's five header values, each pinned. `stsSeconds` is a year in seconds.
SECURITY_HEADERS = {
    "stsSeconds": 31536000,
    "stsIncludeSubdomains": True,
    "contentTypeNosniff": True,
    "frameDeny": True,
    "referrerPolicy": "strict-origin-when-cross-origin",
}

#: Criterion 8, decision F: cert-manager's API group, and the two exact ACME directory URLs.
#: A staging issuer pointed at the production endpoint burns real rate limits on a rehearsal,
#: and a prod issuer pointed at staging issues certificates no browser trusts - both of which
#: are one character of difference in a URL nothing else checks.
ISSUER_API_VERSION = "cert-manager.io/v1"
ISSUERS = {
    "letsencrypt-staging": {
        "server": "https://acme-staging-v02.api.letsencrypt.org/directory",
        "privateKeySecretRef": "letsencrypt-staging-account-key",
    },
    "letsencrypt-prod": {
        "server": "https://acme-v02.api.letsencrypt.org/directory",
        "privateKeySecretRef": "letsencrypt-prod-account-key",
    },
}

#: Decision F: a domain address, never a personal one, and a design note's HTTP-01 solver on the
#: ingress class k3s runs.
ACME_EMAIL = "ops@example.com"
ACME_SOLVERS = [{"http01": {"ingress": {"class": "traefik"}}}]

#: Decision D: the only Secret names anything in the build may reference, checked build-wide
#: on purpose - "no Secret is emitted anywhere" is the claim worth keeping whole. All four are
#: supplied SOPS-encrypted before the first apply; until then an apply fails loudly on a
#: missing Secret, which is the intended direction. The TLS secrets the Ingresses name are
#: cert-manager's to write, so they are excluded where that comparison is made
#: (`tests/infra/test_k8s_apps.py`) rather than added here. `helpdesk-secrets` joined the list
#: with row 20f's four workloads, which name it and supply it nowhere; this constant is
#: build-wide, so it has to move in the same change or a correct manifest reddens here.
#: `grafana-admin` joined with an earlier change's Grafana, which names it by `envFrom` in `monitoring`,
#: for exactly the reason the sentence above gives: this constant is build-wide, so it moves in
#: the same change as the manifest or a correct manifest reddens here.
PINNED_SECRET_NAMES = frozenset(
    {
        "postgres-credentials",
        BASIC_AUTH_SECRET,
        "gateway-secrets",
        "helpdesk-secrets",
        "grafana-admin",
    }
)

#: Criterion 9's file scan, first shape: a credential KEY with a scalar value on the same
#: line. Anchored on the key, so `passwordSecretRef:` and a `secretRef:` reference do not
#: match, and an empty value (`password:`) is a mapping header rather than a credential.
INLINE_CREDENTIAL = re.compile(
    r"^\s*-?\s*(password|passwd|postgres_password|token|apikey|api_key)\s*:\s*(\S.*)$",
    re.IGNORECASE,
)

#: Second shape, and the one a Kubernetes manifest actually takes: an `env` entry whose
#: `name` is a credential and whose `value` sits on a following line, where the key the first
#: pattern looks for is the word `value`. Found by mutating the throwaway stub - the pattern
#: above passed a `- name: POSTGRES_PASSWORD` / `value: hunter2` pair untouched.
CREDENTIAL_ENV_NAME = re.compile(
    r"^\s*-?\s*name\s*:\s*[\"']?(\w*(password|passwd|token|apikey|api_key|secret)\w*)[\"']?\s*$",
    re.IGNORECASE,
)

#: The literal value of such an entry, on one of the lines that follow its name.
ENV_VALUE = re.compile(r"^\s*value\s*:\s*(\S.*)$")

#: How far after a credential `name:` its `value:` may sit before the entry is a different
#: one. An env entry is `name`, then `value` or `valueFrom` with its own two lines.
ENV_VALUE_LOOKAHEAD = 3

#: Binary and decimal suffixes a `resources.requests.memory` may carry. Copied from
#: `test_k8s_base.py` with the helper that reads it.
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


def read(path: Path) -> str:
    """Read a file the plan owes, failing the one case that needs it when it is absent."""
    assert path.is_file(), f"{path} does not exist"
    return path.read_text(encoding="utf-8")


def objects_of_kind(documents: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Every built document of `kind`, in build order."""
    return [document for document in documents if document.get("kind") == kind]


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


def memory_bytes(quantity: str) -> int:
    """`512Mi` as an integer count of bytes. Raises on a quantity this repo does not write."""
    for suffix, multiplier in MEMORY_SUFFIXES.items():
        if quantity.endswith(suffix):
            return int(quantity[: -len(suffix)]) * multiplier
    return int(quantity)


def cpu_millicores(quantity: str) -> int:
    """`200m` as 200, `2` as 2000. Whole millicores, never floats - see the module docstring."""
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


def pod_owners(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every built document carrying a pod template, found by the PRESENCE of
    `spec.template.spec` rather than by a list of kinds - a Job, a CronJob or a DaemonSet
    added by a later row is covered without editing this file."""
    owners: list[dict[str, Any]] = []
    for document in documents:
        spec = document.get("spec")
        if not isinstance(spec, dict):
            continue
        template = spec.get("template")
        if isinstance(template, dict) and isinstance(template.get("spec"), dict):
            owners.append(document)
    return owners


def pod_spec(owner: dict[str, Any]) -> dict[str, Any]:
    """The `spec.template.spec` of a document `pod_owners` selected."""
    spec = owner["spec"]["template"]["spec"]
    assert isinstance(spec, dict)
    return dict(spec)


def template_labels(owner: dict[str, Any]) -> dict[str, str]:
    """`spec.template.metadata.labels`, the labels a selector has to match."""
    metadata = owner["spec"]["template"].get("metadata", {})
    assert isinstance(metadata, dict), f"{name_of(owner)}'s pod template has no metadata"
    labels = metadata.get("labels", {})
    assert isinstance(labels, dict), f"{name_of(owner)}'s pod template has no labels"
    return {str(key): str(value) for key, value in labels.items()}


def all_containers(owner: dict[str, Any]) -> list[dict[str, Any]]:
    """Every container of a pod template, `initContainers` included: an init container with no
    memory limit is admitted against the same quota as an app container."""
    spec = pod_spec(owner)
    collected: list[dict[str, Any]] = []
    for key in ("initContainers", "containers"):
        listed = spec.get(key, [])
        assert isinstance(listed, list), f"{name_of(owner)}.{key} is not a list"
        collected.extend(container for container in listed if isinstance(container, dict))
    return collected


def workload(documents: list[dict[str, Any]], workload_name: str) -> dict[str, Any]:
    """The single pod-owning object named `workload_name` in namespace `platform`."""
    found = [
        owner
        for owner in pod_owners(documents)
        if name_of(owner) == workload_name and namespace_of(owner) == PLATFORM
    ]
    assert len(found) == 1, f"{len(found)} pod-owning objects named {workload_name} in platform"
    return found[0]


def sole_container(owner: dict[str, Any]) -> dict[str, Any]:
    """The one app container of a workload, asserting the template declares exactly one."""
    containers = pod_spec(owner).get("containers", [])
    assert isinstance(containers, list) and len(containers) == 1, (
        f"{name_of(owner)} declares {len(containers) if isinstance(containers, list) else 0} "
        "containers, and the budget table has one row per workload"
    )
    container = containers[0]
    assert isinstance(container, dict)
    return container


def resource_quantities(container: dict[str, Any]) -> dict[str, str]:
    """A container's `resources` flattened to the `requests.memory`-style keys the budget
    table and the quota both use, values exactly as written."""
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


def replicas_of(owner: dict[str, Any]) -> int:
    """`spec.replicas`, defaulting to Kubernetes' own default of 1. The quota counts pods, so
    a workload scaled to two costs twice its row."""
    declared = owner["spec"].get("replicas", 1)
    return int(declared)


def compose_images() -> dict[str, str]:
    """The image each named service runs in `docker-compose.yml`, read from the file."""
    document = yaml.safe_load(read(COMPOSE_FILE))
    assert isinstance(document, dict), f"{COMPOSE_FILE} is not a YAML mapping"
    services = document.get("services", {})
    assert isinstance(services, dict), f"{COMPOSE_FILE} declares no services"
    images: dict[str, str] = {}
    for workload_name, service_name in COMPOSE_SERVICES.items():
        service = services.get(service_name, {})
        assert isinstance(service, dict), f"compose declares no `{service_name}` service"
        images[workload_name] = str(service.get("image", ""))
    return images


def secret_references(value: object, found: set[str]) -> set[str]:
    """Every Secret NAME reachable anywhere in the built stream, collected by walking the
    parsed documents rather than by grepping text: `secretRef`, `secretKeyRef` and Traefik's
    `basicAuth.secret` are three different shapes for the same reference, and a fourth added
    later under a mapping key nobody grepped would be invisible to a pattern."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("secretRef", "secretKeyRef") and isinstance(child, dict):
                found.add(str(child.get("name", "")))
            elif key in ("secret", "secretName") and isinstance(child, str):
                found.add(child)
            else:
                secret_references(child, found)
    elif isinstance(value, list):
        for child in value:
            secret_references(child, found)
    return found


def tls_secret_names(documents: list[dict[str, Any]]) -> set[str]:
    """Every `spec.tls[].secretName` the built Ingresses name. cert-manager WRITES these, so
    they are not names anyone supplies and they cannot sit inside `PINNED_SECRET_NAMES`."""
    names: set[str] = set()
    for ingress in objects_of_kind(documents, "Ingress"):
        for entry in ingress.get("spec", {}).get("tls", []):
            if isinstance(entry, dict) and entry.get("secretName"):
                names.add(str(entry["secretName"]))
    return names


def inline_credentials(label: str, lines: list[str]) -> list[str]:
    """Every line of a manifest that writes a credential VALUE, in both shapes it takes: a
    credential key with a scalar on the same line, and an `env` entry whose `name` is a
    credential and whose `value` follows within `ENV_VALUE_LOOKAHEAD` lines. A `valueFrom`
    or a `secretRef` is a reference, not a value, and matches neither."""
    offenders: list[str] = []
    for index, line in enumerate(lines):
        if INLINE_CREDENTIAL.match(line):
            offenders.append(f"{label}:{index + 1}: {line.strip()}")
            continue
        if not CREDENTIAL_ENV_NAME.match(line):
            continue
        for offset, following in enumerate(lines[index + 1 : index + 1 + ENV_VALUE_LOOKAHEAD]):
            if ENV_VALUE.match(following):
                offenders.append(
                    f"{label}:{index + offset + 2}: {line.strip()} -> {following.strip()}"
                )
                break
    return offenders


def test_the_platform_directory_emits_the_objects_the_plan_names(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 1, by SET EQUALITY over `(kind, name, namespace)` for everything the build
    emits that is not one of 20a's three base kinds. Catches an implementation that writes the
    seven YAML files but forgets one in `platform/kustomization.yaml`'s `resources` - the files
    are perfectly valid and the object is simply not in the stream, which is exactly what
    decision D chose "build and parse" to see. Catches Redis shipped as a StatefulSet
    (decision C says Deployment over `emptyDir`, because a disk image of the redaction map is
    personal data at rest), catches a Service written without its workload or vice versa, and
    catches a middleware or an issuer given a namespace it should not have: a namespaced
    `ClusterIssuer` is silently ignored by cert-manager, and a `platform` middleware moved to
    `default` is an Ingress referencing nothing. The base kinds are excluded rather than
    enumerated alongside, so this case cannot pass or fail on anything `test_k8s_base.py`
    already owns; the namespace filter is what keeps it from failing on objects a LATER
    directory adds, which is a claim this case never made - `tests/infra/test_k8s_apps.py`
    asserts the same equality over `apps`, and the cluster-scoped half stays here because a
    `ClusterIssuer` belongs to no namespace at all."""
    added = {
        (str(document.get("kind")), name_of(document), namespace_of(document))
        for document in built
        if str(document.get("kind")) not in BASE_KINDS
        and namespace_of(document) in (PLATFORM, None)
    }
    assert added == PLATFORM_OBJECTS, (
        f"unexpected: {sorted(added - PLATFORM_OBJECTS)}; "
        f"missing: {sorted(PLATFORM_OBJECTS - added)}"
    )


def test_every_container_in_the_build_declares_requests_and_a_memory_limit(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 2, over every pod template in the WHOLE build and every container in it,
    `initContainers` included. Catches a container added with no `resources` block at all -
    which 20a's LimitRange would silently default rather than reject, so nothing else in this
    repo ever says the manifest was incomplete - and catches one of the three quantities
    dropped while the other two stay, which no per-workload row would see for a workload added
    by a later plan. Catches `limits.cpu` creeping in, which decision C refuses because a CPU
    limit throttles the gateway's tail latency rather than protecting the node. Pod owners are
    found by the PRESENCE of `spec.template.spec`, so a Job or a DaemonSet a later row adds is
    covered without this case being edited."""
    owners = pod_owners(built)
    assert owners, "the build emits no pod templates at all"

    missing: list[str] = []
    forbidden: list[str] = []
    for owner in owners:
        label = f"{owner.get('kind')}/{name_of(owner)}"
        for container in all_containers(owner):
            quantities = resource_quantities(container)
            where = f"{label}:{container.get('name')}"
            missing.extend(
                f"{where} has no {key}" for key in REQUIRED_CONTAINER_KEYS if key not in quantities
            )
            if FORBIDDEN_CONTAINER_KEY in quantities:
                forbidden.append(where)

    assert missing == [], f"containers missing resources: {missing}"
    assert forbidden == [], f"containers declaring {FORBIDDEN_CONTAINER_KEY}: {forbidden}"


@pytest.mark.parametrize(("workload_name", "expected"), sorted(WORKLOAD_RESOURCES.items()))
def test_each_platform_workload_carries_the_agreed_quantities(
    workload_name: str, expected: dict[str, str], built: list[dict[str, Any]]
) -> None:
    """Criterion 3, one row per workload, by VALUE and not by presence - the case above proves
    the three keys exist, so the only thing that can flip this one is a wrong number. Catches
    Postgres given Redis's 256Mi because the file was copied, catches a limit written as `1G`
    where the table says `1Gi` (a 7% difference the API server does not treat as equal, and
    the direction that gets a database OOM-killed under load), and catches a quantity raised
    in isolation to make a pod fit, which the subtotals case then catches a second way.
    Equality on the whole flattened mapping, so an extra `limits.cpu` or a `requests.storage`
    nobody agreed to is a failure rather than a silent addition."""
    quantities = resource_quantities(sole_container(workload(built, workload_name)))
    assert quantities == expected, f"{workload_name} declares {quantities}"


def test_the_platform_subtotals_fit_inside_the_namespace_quota(built: list[dict[str, Any]]) -> None:
    """Criterion 4, derived and never restated: the three sums are parsed off
    the BUILT pod templates in `platform`, multiplied by each workload's replica count, added
    to `CERT_MANAGER_RESERVE`, and compared to the `platform` `ResourceQuota` read from the
    SAME build. Catches the arithmetic failing even though every individual row above is
    correct - a workload raised to fit, a second replica, or the quota lowered by 20a's table
    - which is the state where `helm install cert-manager` is refused at admission with three
    pods' worth of budget already spent, and nothing in this repo said so first. Millicores as
    ints: `0.2 + 0.1 + 0.2 + 0.03` is not `0.53` in binary floating point, so a float
    comparison would redden a correct manifest. Compared with `<=`, because the reserve is a
    floor on what cert-manager may take and the headroom above it is deliberate."""
    quotas = [
        quota
        for quota in objects_of_kind(built, "ResourceQuota")
        if namespace_of(quota) == PLATFORM
    ]
    assert len(quotas) == 1, f"{len(quotas)} ResourceQuotas in {PLATFORM}"
    hard = quota_hard(quotas[0])

    memory_request = memory_bytes(str(CERT_MANAGER_RESERVE["requests.memory"]))
    cpu_request = cpu_millicores(str(CERT_MANAGER_RESERVE["requests.cpu"]))
    memory_limit = memory_bytes(str(CERT_MANAGER_RESERVE["limits.memory"]))
    pods = int(str(CERT_MANAGER_RESERVE["pods"]))
    totals = {
        "requests.memory": pods * memory_request,
        "requests.cpu": pods * cpu_request,
        "limits.memory": pods * memory_limit,
    }

    owners = [owner for owner in pod_owners(built) if namespace_of(owner) == PLATFORM]
    assert len(owners) == len(WORKLOAD_RESOURCES), f"{len(owners)} platform workloads built"
    for owner in owners:
        replicas = replicas_of(owner)
        for container in all_containers(owner):
            quantities = resource_quantities(container)
            totals["requests.memory"] += replicas * memory_bytes(quantities["requests.memory"])
            totals["requests.cpu"] += replicas * cpu_millicores(quantities["requests.cpu"])
            totals["limits.memory"] += replicas * memory_bytes(quantities["limits.memory"])

    assert totals["requests.memory"] <= memory_bytes(hard["requests.memory"]), (
        f"platform requests {totals['requests.memory']} bytes against {hard['requests.memory']}"
    )
    assert totals["requests.cpu"] <= cpu_millicores(hard["requests.cpu"]), (
        f"platform requests {totals['requests.cpu']}m against {hard['requests.cpu']}"
    )
    assert totals["limits.memory"] <= memory_bytes(hard["limits.memory"]), (
        f"platform limits {totals['limits.memory']} bytes against {hard['limits.memory']}"
    )


def test_the_readme_budget_row_for_cert_manager_matches_the_reserve_constant() -> None:
    """Criterion 10's half that keeps prose and arithmetic from drifting: the README's
    cert-manager row must state the multiplier and the three subtotals this file's constant
    produces, and all four strings are COMPUTED from `CERT_MANAGER_RESERVE` here rather than
    typed, so raising the reserve reddens the table until the table is rewritten. Catches the
    §5 risk being handled in the test alone - the reserve raised because the chart's real
    footprint is bigger, with the README still promising 96Mi to whoever reads the budget
    before running `helm install`. All four values are required TOGETHER on one line (standing
    rule 44): the page names `32Mi` and `3` in the helm command a few lines above, so a
    per-value scan would pass on a table with no cert-manager row at all."""
    pods = int(str(CERT_MANAGER_RESERVE["pods"]))
    memory = f"{pods * memory_bytes(str(CERT_MANAGER_RESERVE['requests.memory'])) // 1024**2}Mi"
    cpu = f"{pods * cpu_millicores(str(CERT_MANAGER_RESERVE['requests.cpu']))}m"
    limit = f"{pods * memory_bytes(str(CERT_MANAGER_RESERVE['limits.memory'])) // 1024**2}Mi"

    rows = [
        line
        for line in read(K8S_README).splitlines()
        if "cert-manager" in line
        and all(value in line for value in (f"{pods} pods", memory, cpu, limit))
    ]
    assert len(rows) == 1, (
        f"{len(rows)} README rows state the cert-manager reserve as "
        f"{pods} pods / {memory} / {cpu} / {limit}"
    )


def test_the_images_match_docker_compose(built: list[dict[str, Any]]) -> None:
    """Criterion 5 and decision E, read from BOTH files and compared as a mapping, so a drift
    in either direction reddens and neither side can be the test's own expectation. Catches the
    cluster pinned tighter than the local stack (`postgres:16.6` against compose's
    `postgres:16`), which is decision E's named temptation and makes a developer debug a
    version they are not running; catches compose being bumped without the manifests; and
    catches a Qdrant tag drifting off `v1.12.4`, where the on-disk collection format is what
    changes underneath a StatefulSet's PVC. The mapping compare, rather than three
    assertions, is what makes a missing workload visible here too."""
    from_k8s = {
        workload_name: str(sole_container(workload(built, workload_name)).get("image", ""))
        for workload_name in COMPOSE_SERVICES
    }
    assert from_k8s == compose_images()


@pytest.mark.parametrize("workload_name", sorted(WORKLOAD_RESOURCES))
def test_each_workload_has_both_probes_and_a_service_that_resolves(
    workload_name: str, built: list[dict[str, Any]]
) -> None:
    """Criterion 6, three things a manifest can get wrong without any tool saying so. Catches
    a readiness probe with no liveness probe (a wedged Postgres that answers nothing and is
    never restarted) and the reverse (traffic routed to a pod that has not finished starting).
    Catches a `targetPort` naming a port the pod template never declares - `5433`, or a name
    that was renamed on one side - which is a Service with no endpoints and a connection
    refused at runtime, not at build. And catches §2's kustomize mechanic going wrong: the
    root sets `includeSelectors: false`, so the common labels do NOT reach pod templates and
    each workload writes its own `app.kubernetes.io/name` on all three of the Service's
    selector, the workload's `matchLabels` and the template's labels - a selector matching no
    pod is a rollout of zero replicas that reports nothing wrong. The `targetPort` check
    resolves both forms it may take, an int against `containerPort` and a string against the
    port's `name`, so a correct named port is not a red."""
    owner = workload(built, workload_name)
    container = sole_container(owner)
    for probe in ("readinessProbe", "livenessProbe"):
        assert probe in container, f"{workload_name} declares no {probe}"

    labels = template_labels(owner)
    match_labels = owner["spec"]["selector"]["matchLabels"]
    assert isinstance(match_labels, dict) and match_labels, f"{workload_name} selects nothing"
    unmatched = {key: value for key, value in match_labels.items() if labels.get(key) != value}
    assert unmatched == {}, f"{workload_name}'s selector {unmatched} matches no pod: {labels}"

    declared = container.get("ports", [])
    assert isinstance(declared, list) and declared, f"{workload_name} declares no container ports"
    numbers = {int(port["containerPort"]) for port in declared if "containerPort" in port}
    names = {str(port["name"]) for port in declared if port.get("name")}

    services = [
        document
        for document in objects_of_kind(built, "Service")
        if name_of(document) == workload_name and namespace_of(document) == PLATFORM
    ]
    assert len(services) == 1, f"{len(services)} Services named {workload_name} in {PLATFORM}"
    service_spec = services[0]["spec"]

    selector = service_spec.get("selector", {})
    assert isinstance(selector, dict) and selector, f"{workload_name}'s Service selects nothing"
    stray = {key: value for key, value in selector.items() if labels.get(key) != value}
    assert stray == {}, f"{workload_name}'s Service selector {stray} matches no pod: {labels}"

    ports = service_spec.get("ports", [])
    assert isinstance(ports, list) and ports, f"{workload_name}'s Service publishes no ports"
    assert {int(port["port"]) for port in ports} == SERVICE_PORTS[workload_name]

    unresolved = [
        port["targetPort"]
        for port in ports
        if (port["targetPort"] not in names if isinstance(port["targetPort"], str) else True)
        and (
            int(port["targetPort"]) not in numbers
            if not isinstance(port["targetPort"], str)
            else True
        )
    ]
    assert unresolved == [], (
        f"{workload_name}'s Service targets {unresolved}, declared by nothing in its own "
        f"template (containerPorts {sorted(numbers)}, names {sorted(names)})"
    )


def test_the_middlewares_carry_the_agreed_values(built: list[dict[str, Any]]) -> None:
    """Criterion 7, all three by value. Catches `basic-auth` naming a Secret other than the one
    `infra/secrets/traefik-basic-auth.example.yaml` contracts, which is an Ingress with
    authentication configured and no credentials behind it. Catches the rate-limit triple
    written with `average` and `burst` swapped - 100
    sustained with a burst of 50 is a limit that never bites the way it was agreed - and
    catches `period` written as the int `1`, where Traefik reads a bare number as nanoseconds
    and the limiter becomes effectively unbounded. Catches any of the five security headers
    dropped or weakened: `stsSeconds` shortened, `frameDeny` flipped to false, the referrer
    policy widened to `unsafe-url` which leaks full URLs cross-origin. The headers are compared
    as the five keys RESTRICTED out of `spec.headers` and then checked for nothing else being
    there, so an additional key is reported rather than silently accepted - a sixth header can
    contradict one of these (`stsPreload` absent is fine; `customResponseHeaders` overwriting
    `X-Frame-Options` is not)."""
    middlewares = {name_of(document): document for document in objects_of_kind(built, "Middleware")}
    assert set(middlewares) == {"basic-auth", "rate-limit", "security-headers"}
    for name, document in middlewares.items():
        assert document.get("apiVersion") == MIDDLEWARE_API_VERSION, (
            f"{name} is {document.get('apiVersion')}, not Traefik v3's {MIDDLEWARE_API_VERSION}"
        )

    assert middlewares["basic-auth"]["spec"]["basicAuth"]["secret"] == BASIC_AUTH_SECRET
    assert middlewares["rate-limit"]["spec"]["rateLimit"] == RATE_LIMIT

    headers = middlewares["security-headers"]["spec"]["headers"]
    assert isinstance(headers, dict)
    assert {key: headers.get(key) for key in SECURITY_HEADERS} == SECURITY_HEADERS
    extra = sorted(set(headers) - set(SECURITY_HEADERS))
    assert extra == [], f"security-headers also sets {extra}, which nothing here checks"


def test_both_cluster_issuers_use_the_exact_acme_urls_and_the_traefik_solver(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 8 and decision F. Catches the two ACME endpoints swapped, which is the one
    mistake nothing else in this repo can see: a `letsencrypt-prod` pointed at staging issues
    certificates no browser trusts, and a `letsencrypt-staging` pointed at production burns
    real Let's Encrypt rate limits on the rehearsal decision F created it to absorb. Catches a
    shared `privateKeySecretRef`, where the staging account key is reused for production and
    the first issuer to reconcile decides which account both are - asserted as exact names AND
    as a distinctness check, so a copy-paste that renames neither is red twice. Catches the
    HTTP-01 solver replaced or its ingress class changed off `traefik`, which is the only path
    a design note's DNS-only records leave open for a challenge, and catches a personal account
    email replacing the domain one."""
    issuers = {name_of(document): document for document in objects_of_kind(built, "ClusterIssuer")}
    assert set(issuers) == set(ISSUERS)

    keys: list[str] = []
    for name, expected in ISSUERS.items():
        document = issuers[name]
        assert document.get("apiVersion") == ISSUER_API_VERSION, f"{name} is not cert-manager's"
        assert namespace_of(document) is None, f"{name} is namespaced and cert-manager ignores it"
        acme = document["spec"]["acme"]
        assert acme["server"] == expected["server"], f"{name} points at {acme['server']}"
        assert acme["email"] == ACME_EMAIL
        assert acme["solvers"] == ACME_SOLVERS, f"{name} solves with {acme['solvers']}"
        assert acme["privateKeySecretRef"]["name"] == expected["privateKeySecretRef"]
        keys.append(str(acme["privateKeySecretRef"]["name"]))

    assert len(set(keys)) == len(keys), f"both issuers share an account key: {keys}"


def test_no_secret_is_emitted_and_no_credential_is_written_inline(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 9 and decision D, in both halves the decision names. The build half: no
    `Secret` kind at all, and every Secret NAME the stream references is one of the three
    supplied SOPS-encrypted - catches the convenient repair of the apply failing on a
    missing Secret, which is a placeholder `Secret` with a password in git that nothing ever
    fails to mention again, and catches a reference to a Secret nobody has agreed to supply.
    An Ingress's `tls[].secretName` is subtracted first: cert-manager WRITES those, so they
    are not names anyone supplies, and the check that still bites on them - that they are
    disjoint from the supplied three, or a certificate lands on top of the provider keys -
    is asserted in `tests/infra/test_k8s_apps.py` beside the Ingresses that name them.
    The file half is a scan of what is WRITTEN under `platform/`, not of what is built: a
    literal that a later patch strips out of the output is still in the history, and hard
    rule 5 is about the repo. The
    scan asserts the directory exists first, so a missing `platform/` is a loud red and not a
    vacuous pass over zero files - the trap this whole case would otherwise fall into. It
    reads BOTH shapes a written credential takes: `password: hunter2` on one line, and the
    `- name: POSTGRES_PASSWORD` / `value: hunter2` env pair a Kubernetes manifest actually
    uses, which a key-anchored pattern alone walks straight past (found by mutating the
    throwaway stub, a project rule)."""
    emitted = objects_of_kind(built, "Secret")
    assert emitted == [], f"the build emits Secrets: {[name_of(document) for document in emitted]}"

    referenced = {name for name in secret_references(built, set()) if name} - tls_secret_names(
        built
    )
    assert referenced <= PINNED_SECRET_NAMES, (
        f"the build references {sorted(referenced - PINNED_SECRET_NAMES)}, which nothing supplies"
    )

    assert PLATFORM_DIR.is_dir(), f"{PLATFORM_DIR} does not exist, so this scan reads no files"
    files = sorted(PLATFORM_DIR.glob("*.yaml"))
    assert files, f"{PLATFORM_DIR} holds no YAML at all"
    offenders: list[str] = []
    for path in files:
        offenders.extend(inline_credentials(path.name, read(path).splitlines()))
    assert offenders == [], f"inline credentials under {PLATFORM_DIR.name}/: {offenders}"
