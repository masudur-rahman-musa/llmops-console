r"""`infra/k8s/apps`: the gateway Deployment, its Service, HPA, PDB and ConfigMap, the three
copied Traefik middlewares and the two `api.` Ingresses an earlier change agrees, plus the llama.cpp
Deployment, its Service and the `llama-models` claim an earlier change adds, read off what kustomize
actually emits.

Row 20d's cases read four SHIPPED product files as well - `routing.yaml` for the address the
gateway dials and the model ids it resolves, `prices.yaml`, `docker-compose.yml` and
`.env.example` - and compare them to the built manifest and to each other. The model id itself
is written down nowhere here: it is the basename of the Deployment's `-m` argument, so the one
thing five files have to agree on has exactly one source in this file too.

The stream comes from the session-scoped `built` fixture in `tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, parsed once, shared with
`test_k8s_base.py` and `test_k8s_platform.py`. Nothing here reads the raw files under `apps/`
except the credential scan, which is a scan of what is WRITTEN rather than of what is built -
a key removed by a later patch would still be in git, and a build that never emits the file
would hide it entirely.

Quantities are compared as STRINGS where §1 states them (`256Mi`, `150m`, `7Gi`): the API
server compares quantities, so a test that parsed first would accept `262144Ki` for a table
that says `256Mi`, and the budget table is prose a reader checks by eye. Parsing happens in
exactly one case, the subtotals, which DERIVES its three sums from the built pod templates in
`apps` plus the three named reserves and compares them to the `apps` `ResourceQuota` read from
the SAME build. The totals criterion 2's table states appear nowhere in this file (standing
rule 6): restating them beside the per-workload rows would let the two disagree. The gateway's
multiplier is the HPA's `maxReplicas` READ FROM THE BUILD rather than written down a second
time, because a quota sized to `spec.replicas` turns the first scale-up into a pod rejected at
admission - which is the failure decision F's max exists to price. CPU is summed in whole
millicores, never floats.

The parsing helpers below (`read`, `objects_of_kind`, `name_of`, `namespace_of`,
`memory_bytes`, `cpu_millicores`, `quota_hard`, `pod_owners`, `pod_spec`, `template_labels`,
`all_containers`, `sole_container`, `resource_quantities`, `replicas_of`, `secret_references`,
`inline_credentials`) are copies of `test_k8s_platform.py`'s, not imports: a test module
importing another test module makes the second one's collection a dependency of the first's. `conftest.py` holds the fixture and nothing else, so a reader opening any
one of the three modules sees every rule it applies.

§2's kustomize mechanic is what the selector half of the probe case exists for: the root's
`labels:` block sets `includeSelectors: false`, so the two common labels reach each object's
`metadata.labels` and NOT its pod template. The gateway therefore writes its own
`app.kubernetes.io/name` on `spec.selector.matchLabels`, on `spec.template.metadata.labels`
and on the Service's `spec.selector`, and a mismatch is a Deployment that rolls out zero
matching pods while reporting nothing wrong.

The middleware annotation is parsed rather than string-compared: Traefik reads
`traefik.ingress.kubernetes.io/router.middlewares` as a comma-separated list of
`<namespace>-<name>@kubernetescrd` entries, and a name that resolves to nothing is silently
no middleware at all - on the `/` Ingress that is unauthenticated traffic, which is the
failure decision B paid a copy to remove.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_ROOT = REPO_ROOT / "infra" / "k8s"
APPS_DIR = K8S_ROOT / "apps"
K8S_README = K8S_ROOT / "README.md"
DOCKERFILE = REPO_ROOT / "Dockerfile"

#: The namespace every object this plan adds belongs to. Criterion 1 says nothing this plan
#: writes is cluster-scoped and nothing it writes lands anywhere else.
APPS = "apps"

#: The namespace 20b's middlewares live in, and the side this plan's copies are compared to.
PLATFORM = "platform"

#: The kinds an earlier change's `base/` emits. `apps` already carries a ResourceQuota and a LimitRange
#: from that row, so criterion 1's set equality runs over everything in `apps` that is NOT one
#: of these - it covers what `apps/` adds without asserting anything `test_k8s_base.py` owns.
BASE_KINDS = frozenset({"Namespace", "ResourceQuota", "LimitRange"})

#: Row 20d's workload: the llama.cpp server, its Service and the claim its weights live on.
LLAMA = "llama"
LLAMA_PVC = "llama-models"

GATEWAY = "gateway"
GATEWAY_CONFIG_MAP = "gateway-config"
GATEWAY_SECRET = "gateway-secrets"
ROOT_INGRESS = "gateway"
HEALTHZ_INGRESS = "gateway-healthz"

#: Criterion 1, as `(kind, name, namespace)`. Set EQUALITY: a file left out of
#: `apps/kustomization.yaml`'s `resources` and an object nobody agreed to are the same failure,
#: and only equality sees both.
APPS_OBJECTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("Deployment", GATEWAY, APPS),
        ("Service", GATEWAY, APPS),
        ("HorizontalPodAutoscaler", GATEWAY, APPS),
        ("PodDisruptionBudget", GATEWAY, APPS),
        ("ConfigMap", GATEWAY_CONFIG_MAP, APPS),
        ("Middleware", "basic-auth", APPS),
        ("Middleware", "rate-limit", APPS),
        ("Middleware", "security-headers", APPS),
        ("Ingress", ROOT_INGRESS, APPS),
        ("Ingress", HEALTHZ_INGRESS, APPS),
        ("PersistentVolumeClaim", LLAMA_PVC, APPS),
        ("Deployment", LLAMA, APPS),
        ("Service", LLAMA, APPS),
    }
)

#: Criterion 1's second half: "nothing cluster-scoped is added". The two 20b issuers are the
#: only non-base cluster-scoped objects the build may contain, so this set is what 20c must
#: leave untouched rather than a claim about what 20b writes.
CLUSTER_SCOPED_OBJECTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ClusterIssuer", "letsencrypt-staging"),
        ("ClusterIssuer", "letsencrypt-prod"),
    }
)

#: Decision D: the repository the `images:` transformer sets, and the honest placeholder tag.
#: Row 22's deploy replaces the tag with a SHA; until then an apply fails on a pull, which is
#: the intended direction.
GATEWAY_IMAGE_REPOSITORY = "ghcr.io/developermusa/ai-operations-console/gateway"
GATEWAY_IMAGE_TAG = "0.0.0-unbuilt"

#: Decision D's refusal list. A floating tag in git is a cluster whose running code no commit
#: names, and these are the two spellings of it this repo would plausibly write.
FLOATING_TAGS = frozenset({"latest", "main"})

#: Criterion 3's gateway row, by value, per pod. Compared as strings; the
#: subtotals case is the only place these are parsed. Decision C's `limits.cpu` is absent on
#: purpose, and equality on the whole mapping is what keeps it absent.
GATEWAY_RESOURCES: dict[str, str] = {
    "requests.memory": "256Mi",
    "requests.cpu": "150m",
    "limits.memory": "512Mi",
}

#: Row 20d's llama.cpp, one pod. It was a named reserve until 20d built it; it is now the ROW
#: the built manifest is held to (`LLAMA_RESOURCES` below derives from it), and it is no longer
#: summed as a reserve. the design names 3 GiB for it and 20c's decision A raised the quota's
#: limits column to hold exactly that.
LLAMA_RESERVE: dict[str, str | int] = {
    "pods": 1,
    "requests.memory": "1024Mi",
    "requests.cpu": "300m",
    "limits.memory": "3072Mi",
}

#: Row 20f's four Helpdesk services, priced as three pods of the same size. Row 20 split
#: after 20c was written: 20d built llama.cpp, 20e adds the embedding server and 20f the
#: Helpdesk workloads, so the row this comment named before the split no longer does it.
HELPDESK_RESERVE: dict[str, str | int] = {
    "pods": 3,
    "requests.memory": "192Mi",
    "requests.cpu": "100m",
    "limits.memory": "384Mi",
}

#: Row 27's console, one pod.
CONSOLE_RESERVE: dict[str, str | int] = {
    "pods": 1,
    "requests.memory": "256Mi",
    "requests.cpu": "100m",
    "limits.memory": "512Mi",
}

#: The reserves still held as constants. `LLAMA_RESERVE` is NOT one of them any more: row 20d
#: builds that pod, so the subtotals case reads its quantities off the built template the way it
#: already reads the gateway's, and summing the reserve as well would price the one workload
#: twice. The reserve stays in this module as the row
#: `test_llama_declares_the_quantities_the_budget_prices` holds the manifest to.
RESERVES = (HELPDESK_RESERVE, CONSOLE_RESERVE)

#: Decision A: the `apps` envelope after the limits column is raised. The requests column does
#: not move - it is the 6Gi cluster-wide total #55 and the design pin.
APPS_QUOTA: dict[str, str] = {
    "requests.memory": "3Gi",
    "requests.cpu": "1500m",
    "limits.memory": "7Gi",
}

#: Criterion 1: the claim's size and access mode, and the `storageClassName` it must NOT
#: declare - k3s's local-path provisioner is the cluster default and naming a class here is a
#: claim that binds to nothing, the same reason 20b's Postgres omits it.
LLAMA_PVC_STORAGE = "5Gi"
LLAMA_PVC_ACCESS_MODES = ["ReadWriteOnce"]

#: Criterion 2: one replica, and the only update strategy a `ReadWriteOnce` volume survives.
LLAMA_REPLICAS = 1
LLAMA_STRATEGY = "Recreate"

#: Criterion 5: where the weights are mounted, and the suffix the upstream Q4_K_M artefact
#: carries - the model ID is the file name with this taken off, which is how criterion 8's
#: case derives an id from the manifest instead of writing one down.
LLAMA_MOUNT_PATH = "/models"
GGUF_SUFFIX = "-q4_k_m.gguf"

#: Criterion 5, by value and in order: the design's context and parallelism, the address the
#: server binds, and the one file it loads. The port reads the same numeral criterion 3 derives
#: from `routing.yaml`; it is written here because the args are pinned as a
#: LIST and a list assembled from three sources cannot state an order.
LLAMA_MODEL_FILE = f"qwen2.5-3b-instruct{GGUF_SUFFIX}"
LLAMA_ARGS: list[str] = [
    "--host",
    "0.0.0.0",
    "--port",
    "8082",
    "--ctx-size",
    "4096",
    "--parallel",
    "2",
    "-m",
    f"{LLAMA_MOUNT_PATH}/{LLAMA_MODEL_FILE}",
]

#: Criterion 6 and decision E: the path llama.cpp serves, and the 600-second ceiling on a cold
#: load of a multi-gigabyte file - `10` seconds times `60` attempts, both exact (standing
#: rule 2). A failure threshold of 30 is a pod killed halfway through its first load.
LLAMA_PROBE_PATH = "/health"
LLAMA_STARTUP_PERIOD_SECONDS = 10
LLAMA_STARTUP_FAILURE_THRESHOLD = 60

#: The three probes criterion 6 names, in the order a pod runs them.
LLAMA_PROBES = ("startupProbe", "readinessProbe", "livenessProbe")

#: Decision C: the upstream repository the `images:` transformer points `llama` at, and the
#: exact tag resolved from GHCR while this plan was implemented. Compose runs the same one, so
#: the two stacks run the same server.
LLAMA_IMAGE_REPOSITORY = "ghcr.io/ggml-org/llama.cpp"
LLAMA_IMAGE_TAG = "server-b10902"

#: Decision C's refusal list. `server` is upstream's own moving tag - the one compose used
#: before this plan - so it joins the two spellings decision D already refuses for the gateway.
#: A separate constant, because the gateway's case refuses exactly two and must keep doing so.
LLAMA_FLOATING_TAGS = FLOATING_TAGS | {"server"}

#: Criterion 4's llama row, DERIVED from the reserve the apps budget already holds rather than
#: written a second time: a manifest that disagrees with the reserve moves
#: the subtotals, and the two cases would then have to be corrected together. `pods` is the
#: reserve's own multiplier and not a container quantity. `limits.cpu` is absent on purpose and
#: equality on the whole mapping is what keeps it absent.
LLAMA_RESOURCES: dict[str, str] = {
    key: str(value) for key, value in LLAMA_RESERVE.items() if key != "pods"
}

#: The shipped files criterion 8 compares to one another. None of them is this file's own
#: expectation: the id comes off the built `-m` argument and every file is read from the repo.
ROUTING_POLICY = REPO_ROOT / "routing.yaml"
PRICES = REPO_ROOT / "prices.yaml"
COMPOSE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

#: The provider name in `routing.yaml` this plan's pod answers for, and the path its `base_url`
#: carries. Criterion 3 derives the Service's name and port from that URL and asserts the path
#: separately, because a base URL missing `/v1` is an OpenAI-compatible client 404ing.
LOCAL_PROVIDER = "local"
LOCAL_BASE_PATH = "/v1"

#: Criterion 8: how many `local` CHAT candidates the shipped policy names - the `agent-default`
#: alias's and nordlicht's override, both of which decision A renames. Exact, so renaming one
#: line and not the other is red rather than half-green.
LOCAL_CHAT_CANDIDATES = 2

#: The id decision A replaces. Its `prices.yaml` key is KEPT (a project rule - a historical ledger
#: row is priced against it), and `routing.yaml` must stop naming it.
PREVIOUS_LOCAL_MODEL = "qwen2.5-7b-instruct"

#: The GGUF file name compose interpolates, and the same default in `.env.example`. Every
#: matching line is scanned and the values required to agree: `llama` and
#: `models` appear on several lines of both files, so a first-hit scan would read the wrong one.
COMPOSE_MODEL_DEFAULT = re.compile(r"\$\{LLAMA_MODEL_FILE:-(?P<file>[^}]+)\}")
ENV_MODEL_DEFAULT = re.compile(r"^LLAMA_MODEL_FILE=(?P<file>\S+)$")

#: Decision F, by value: two replicas at rest, an HPA from 2 to 3 on 70% CPU, and a PDB that
#: still lets a single-node drain proceed.
GATEWAY_REPLICAS = 2
HPA_MIN_REPLICAS = 2
HPA_MAX_REPLICAS = 3
HPA_CPU_UTILIZATION = 70
PDB_MIN_AVAILABLE = 1

#: Criterion 6: the two probe paths the gateway already serves, each on the endpoint that
#: means what the probe means - `/readyz` is dependencies reachable, `/healthz` is process up.
READINESS_PATH = "/readyz"
LIVENESS_PATH = "/healthz"

#: The port the container publishes, by name and by number. The Service may target either.
CONTAINER_PORT_NAME = "http"
CONTAINER_PORT = 8080

#: Decision E: configuration arrives as two whole objects, so adding a variable is a change to
#: one of them and never to the Deployment. `gateway-secrets` is NAMED here and supplied by
#: row 21 - the ConfigMap is written by this plan.
ENV_FROM_SOURCES: dict[str, str] = {
    "configMapRef": GATEWAY_CONFIG_MAP,
    "secretRef": GATEWAY_SECRET,
}

#: §2's kustomize mechanic: the label the gateway writes itself, because the root's `labels:`
#: block carries `includeSelectors: false` and never reaches a pod template.
SELECTOR_LABEL = "app.kubernetes.io/name"

#: Criterion 7. `infra/terraform/hetzner/variables.tf` fixes the subdomain; no new choice is
#: made here.
INGRESS_HOST = "api.example.com"

#: Criterion 7: the production issuer, on both Ingresses. A staging annotation here is a
#: certificate no browser trusts on the public host.
CLUSTER_ISSUER_ANNOTATION = "cert-manager.io/cluster-issuer"
CLUSTER_ISSUER = "letsencrypt-prod"

#: Criterion 7 and decision C: Traefik middleware annotations are per Ingress, not per path,
#: which is why there are two Ingresses on one host.
MIDDLEWARE_ANNOTATION = "traefik.ingress.kubernetes.io/router.middlewares"

#: The middleware list each Ingress carries, by name. `/healthz` is the one public path that
#: is not behind basic auth (the design); everything under `/`, `/metrics` and `/admin/*`
#: included, is.
INGRESS_MIDDLEWARES: dict[str, frozenset[str]] = {
    ROOT_INGRESS: frozenset({"basic-auth", "rate-limit", "security-headers"}),
    HEALTHZ_INGRESS: frozenset({"rate-limit", "security-headers"}),
}

#: Decision B's three copies, by name. Their VALUES are never written here: the equality case
#: reads both sides off the same build.
MIDDLEWARE_NAMES = frozenset({"basic-auth", "rate-limit", "security-headers"})

#: One entry of the middleware annotation, `<namespace>-<name>@kubernetescrd`. The namespace is
#: matched without hyphens so the split lands after `apps`, which is the only namespace this
#: plan's Ingresses may name; an entry Traefik cannot resolve is silently no middleware at all.
MIDDLEWARE_REFERENCE = re.compile(
    r"(?P<namespace>[a-z0-9]+)-(?P<name>[a-z0-9][a-z0-9-]*)@kubernetescrd"
)

#: Decision E and row 21: the only three Secret names anything in the build may reference. The
#: emitted-Secret half of the same case is build-wide and stays that way.
PINNED_SECRET_NAMES = frozenset({"postgres-credentials", "traefik-basic-auth", GATEWAY_SECRET})

#: Criterion 9's file scan, first shape: a credential KEY with a scalar value on the same line.
#: Anchored on the key, so `passwordSecretRef:` and a `secretRef:` reference do not match, and
#: an empty value (`password:`) is a mapping header rather than a credential.
INLINE_CREDENTIAL = re.compile(
    r"^\s*-?\s*(password|passwd|postgres_password|token|apikey|api_key)\s*:\s*(\S.*)$",
    re.IGNORECASE,
)

#: Second shape, and the one a Kubernetes manifest actually takes: an `env` entry whose `name`
#: is a credential and whose `value` sits on a following line, where the key the first pattern
#: looks for is the word `value`.
CREDENTIAL_ENV_NAME = re.compile(
    r"^\s*-?\s*name\s*:\s*[\"']?(\w*(password|passwd|token|apikey|api_key|secret)\w*)[\"']?\s*$",
    re.IGNORECASE,
)

#: The literal value of such an entry, on one of the lines that follow its name.
ENV_VALUE = re.compile(r"^\s*value\s*:\s*(\S.*)$")

#: How far after a credential `name:` its `value:` may sit before the entry is a different one.
ENV_VALUE_LOOKAHEAD = 3

#: Binary and decimal suffixes a `resources.requests.memory` may carry.
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

#: Decision G: the pair the image already carries for `routing.yaml`, and the pair it owes for
#: `prices.yaml`. `PriceStore.__init__` loads the file eagerly and `create_app` builds the
#: store, so an image missing the second pair cannot reach its first request.
DOCKERFILE_DATA_FILES: dict[str, str] = {
    "routing.yaml": "OPSCONSOLE_ROUTING_POLICY_PATH",
    "prices.yaml": "OPSCONSOLE_PRICES_PATH",
}

#: Where the image puts those files, and therefore what the environment variables must read.
IMAGE_WORKDIR = "/app"

#: A `COPY <file> ./` line of the build stage, and an `OPSCONSOLE_*` assignment anywhere in the
#: runtime stage's `ENV` block - the block is written as backslash-continued lines, so the scan
#: is per line rather than per instruction.
DOCKERFILE_COPY = re.compile(r"^\s*COPY\s+(?P<files>[^\n]*?)\s+\./\s*$")
DOCKERFILE_ENV_ASSIGNMENT = re.compile(r"(?P<name>OPSCONSOLE_[A-Z0-9_]+)=(?P<value>\S+)")


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


def annotations_of(document: dict[str, Any]) -> dict[str, str]:
    """`metadata.annotations`, values as written, empty when the object declares none."""
    metadata = document.get("metadata", {})
    assert isinstance(metadata, dict)
    annotations = metadata.get("annotations", {})
    if not isinstance(annotations, dict):
        return {}
    return {str(key): str(value) for key, value in annotations.items()}


def memory_bytes(quantity: str) -> int:
    """`512Mi` as an integer count of bytes. Raises on a quantity this repo does not write."""
    for suffix, multiplier in MEMORY_SUFFIXES.items():
        if quantity.endswith(suffix):
            return int(quantity[: -len(suffix)]) * multiplier
    return int(quantity)


def cpu_millicores(quantity: str) -> int:
    """`150m` as 150, `2` as 2000. Whole millicores, never floats - see the module docstring."""
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
    `spec.template.spec` rather than by a list of kinds - a Job, a CronJob or a DaemonSet added
    by a later row is covered without editing this file."""
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


def replicas_of(owner: dict[str, Any]) -> int:
    """`spec.replicas`, defaulting to Kubernetes' own default of 1. The quota counts pods, so a
    workload scaled to two costs twice its row."""
    declared = owner["spec"].get("replicas", 1)
    return int(declared)


def secret_references(value: object, found: set[str]) -> set[str]:
    """Every Secret NAME reachable anywhere in the built stream, collected by walking the parsed
    documents rather than by grepping text: `secretRef`, `secretKeyRef`, Traefik's
    `basicAuth.secret` and an Ingress's `tls[].secretName` are four shapes for the same
    reference, and a fifth added later under a key nobody grepped would be invisible to a
    pattern."""
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
    """Every `spec.tls[].secretName` the built Ingresses name. These are Secrets cert-manager
    WRITES, not ones row 21 supplies, which is why the pinned-name check subtracts them."""
    names: set[str] = set()
    for ingress in objects_of_kind(documents, "Ingress"):
        for entry in ingress.get("spec", {}).get("tls", []):
            if isinstance(entry, dict) and entry.get("secretName"):
                names.add(str(entry["secretName"]))
    return names


def inline_credentials(label: str, lines: list[str]) -> list[str]:
    """Every line of a manifest that writes a credential VALUE, in both shapes it takes: a
    credential key with a scalar on the same line, and an `env` entry whose `name` is a
    credential and whose `value` follows within `ENV_VALUE_LOOKAHEAD` lines. A `valueFrom` or a
    `secretRef` is a reference, not a value, and matches neither."""
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


def in_apps(documents: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    """Every built document of `kind` in namespace `apps`, keyed by name."""
    return {
        name_of(document): document
        for document in objects_of_kind(documents, kind)
        if namespace_of(document) == APPS
    }


def sole_object(documents: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """The single `kind` named `name` in namespace `apps`, failing loudly when it is absent -
    which is what every case owes while `apps/` does not exist yet."""
    found = in_apps(documents, kind)
    assert name in found, f"the build emits no {kind}/{name} in {APPS}: {sorted(found)}"
    return found[name]


def gateway_pod_owner(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """The gateway Deployment, selected by name and namespace and asserted to own a pod."""
    owner = sole_object(documents, "Deployment", GATEWAY)
    assert owner in pod_owners(documents), f"{GATEWAY} carries no pod template"
    return owner


def cpu_utilization_target(hpa: dict[str, Any]) -> int | None:
    """The `averageUtilization` of the HPA's CPU Resource metric, or None when it declares
    none - an HPA scaling on something other than CPU is decision F changed, not satisfied."""
    metrics = hpa["spec"].get("metrics", [])
    if not isinstance(metrics, list):
        return None
    for metric in metrics:
        if not isinstance(metric, dict) or metric.get("type") != "Resource":
            continue
        resource = metric.get("resource", {})
        if not isinstance(resource, dict) or resource.get("name") != "cpu":
            continue
        target = resource.get("target", {})
        if isinstance(target, dict) and "averageUtilization" in target:
            return int(target["averageUtilization"])
    return None


def scale_targets(documents: list[dict[str, Any]]) -> dict[tuple[str, str, str | None], int]:
    """Every `(kind, name, namespace)` an HPA scales, mapped to that HPA's `maxReplicas`. The
    budget prices a scaled workload at its CEILING, and reading the ceiling off the built HPA
    is what keeps decision F's max from being written down twice."""
    targets: dict[tuple[str, str, str | None], int] = {}
    for hpa in objects_of_kind(documents, "HorizontalPodAutoscaler"):
        reference = hpa["spec"]["scaleTargetRef"]
        assert isinstance(reference, dict), f"{name_of(hpa)} scales nothing"
        key = (str(reference.get("kind")), str(reference.get("name")), namespace_of(hpa))
        targets[key] = int(hpa["spec"]["maxReplicas"])
    return targets


def reserve_totals() -> dict[str, int]:
    """The named reserves, summed as pods times their per-pod quantities. Memory in
    bytes, CPU in whole millicores."""
    totals = {"requests.memory": 0, "requests.cpu": 0, "limits.memory": 0}
    for reserve in RESERVES:
        pods = int(str(reserve["pods"]))
        totals["requests.memory"] += pods * memory_bytes(str(reserve["requests.memory"]))
        totals["requests.cpu"] += pods * cpu_millicores(str(reserve["requests.cpu"]))
        totals["limits.memory"] += pods * memory_bytes(str(reserve["limits.memory"]))
    return totals


def middleware_names(ingress: dict[str, Any]) -> set[tuple[str, str]]:
    """The `(namespace, name)` pairs of an Ingress's middleware annotation, parsed out of the
    comma-separated `<namespace>-<name>@kubernetescrd` list Traefik reads."""
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


def dockerfile_copied_files(text: str) -> set[str]:
    """Every file a `COPY <files> ./` line of the Dockerfile puts in the workdir."""
    copied: set[str] = set()
    for line in text.splitlines():
        match = DOCKERFILE_COPY.match(line)
        if match is None:
            continue
        copied.update(part for part in match.group("files").split() if not part.startswith("--"))
    return copied


def dockerfile_environment(text: str) -> dict[str, str]:
    """Every `OPSCONSOLE_*` assignment in the Dockerfile, scanned per line because the runtime
    stage's `ENV` block is one instruction written as backslash-continued lines."""
    assignments: dict[str, str] = {}
    for line in text.splitlines():
        for match in DOCKERFILE_ENV_ASSIGNMENT.finditer(line):
            assignments[match.group("name")] = match.group("value").rstrip("\\").strip("\"'")
    return assignments


def llama_pod_owner(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """The llama Deployment, selected by name and namespace and asserted to own a pod."""
    owner = sole_object(documents, "Deployment", LLAMA)
    assert owner in pod_owners(documents), f"{LLAMA} carries no pod template"
    return owner


def local_base_url() -> SplitResult:
    """`providers.local.base_url` out of the SHIPPED `routing.yaml`, split into its parts. This
    is the address the gateway dials in the cluster, and the only source criterion 3's Service
    name and port are read from."""
    policy = yaml.safe_load(read(ROUTING_POLICY))
    assert isinstance(policy, dict), f"{ROUTING_POLICY.name} is not a mapping"
    providers = policy.get("providers", {})
    assert isinstance(providers, dict) and LOCAL_PROVIDER in providers, (
        f"{ROUTING_POLICY.name} names no `{LOCAL_PROVIDER}` provider"
    )
    declared = providers[LOCAL_PROVIDER].get("base_url", "")
    assert declared, f"{ROUTING_POLICY.name}'s `{LOCAL_PROVIDER}` provider declares no base_url"
    return urlsplit(str(declared))


def local_candidate_models() -> list[str]:
    """Every model id the shipped policy names for the `local` provider, alias candidates and
    tenant overrides alike, WITH duplicates: the count is the claim (criterion 8 renames two
    lines), so collapsing them to a set would let one renamed line pass for two."""
    policy = yaml.safe_load(read(ROUTING_POLICY))
    assert isinstance(policy, dict), f"{ROUTING_POLICY.name} is not a mapping"
    groups: list[Any] = [
        alias.get("candidates", []) for alias in policy.get("aliases", {}).values()
    ]
    for tenant in policy.get("tenants", {}).values():
        groups.extend(tenant.get("overrides", {}).values())
    return [
        str(candidate["model"])
        for group in groups
        if isinstance(group, list)
        for candidate in group
        if isinstance(candidate, dict) and candidate.get("provider") == LOCAL_PROVIDER
    ]


def sole_arg_value(container: dict[str, Any], flag: str) -> str:
    """The value following `flag` in a container's `args`, asserting the flag appears EXACTLY
    once. A second `-m` is a second model file, and a scan that took the first occurrence would
    report the one that is not loaded."""
    args = [str(arg) for arg in container.get("args", [])]
    positions = [index for index, arg in enumerate(args) if arg == flag]
    assert len(positions) == 1, f"{flag} appears {len(positions)} times in {args}"
    index = positions[0]
    assert index + 1 < len(args), f"{flag} is the last argument of {args}"
    return args[index + 1]


def mount_at(container: dict[str, Any], path: str) -> dict[str, Any]:
    """The one `volumeMounts` entry at `path`, asserting the container declares exactly one."""
    mounts = [
        mount
        for mount in container.get("volumeMounts", [])
        if isinstance(mount, dict) and str(mount.get("mountPath")) == path
    ]
    assert len(mounts) == 1, f"{len(mounts)} mounts at {path}: {container.get('volumeMounts')}"
    return mounts[0]


def volume_named(owner: dict[str, Any], name: str) -> dict[str, Any]:
    """The pod template's `volumes` entry a mount refers to by name - a mount naming a volume
    the template does not declare is a pod the API server refuses, and a mount naming the WRONG
    volume is a model directory backed by something else entirely."""
    volumes = [
        volume
        for volume in pod_spec(owner).get("volumes", [])
        if isinstance(volume, dict) and str(volume.get("name")) == name
    ]
    assert len(volumes) == 1, f"{len(volumes)} volumes named {name} on {name_of(owner)}"
    return volumes[0]


def resolved_container_port(target: object, container: dict[str, Any]) -> int | None:
    """The `containerPort` NUMBER a `targetPort` or a probe port resolves to, in both forms it
    takes - an int against the declared numbers, a string against the declared names - or None
    when the container declares no such port. A name that resolves to nothing is a Service with
    no endpoints and a probe that never succeeds, neither of which fails at build time."""
    declared = [port for port in container.get("ports", []) if isinstance(port, dict)]
    if isinstance(target, str) and not target.isdigit():
        for port in declared:
            if str(port.get("name")) == target:
                return int(port["containerPort"])
        return None
    number = int(str(target))
    return number if any(int(port["containerPort"]) == number for port in declared) else None


def model_id_from(file_name: str) -> str:
    """The model id inside a GGUF file name: `qwen2.5-3b-instruct-q4_k_m.gguf` is
    `qwen2.5-3b-instruct`. The suffix is asserted rather than stripped conditionally, so a file
    in another quantisation is a named failure and not an id with a tail on it."""
    assert file_name.endswith(GGUF_SUFFIX), f"{file_name} is not a {GGUF_SUFFIX} artefact"
    return file_name[: -len(GGUF_SUFFIX)]


def compose_model_files() -> set[str]:
    """Every GGUF file name a `${LLAMA_MODEL_FILE:-...}` default in `docker-compose.yml` falls
    back to. All matching lines, never the first."""
    return {match.group("file") for match in COMPOSE_MODEL_DEFAULT.finditer(read(COMPOSE))}


def env_example_model_files() -> set[str]:
    """Every `LLAMA_MODEL_FILE=` value written in `.env.example`, all matching lines scanned."""
    files: set[str] = set()
    for line in read(ENV_EXAMPLE).splitlines():
        match = ENV_MODEL_DEFAULT.fullmatch(line.strip())
        if match is not None:
            files.add(match.group("file"))
    return files


def priced_keys() -> dict[str, Any]:
    """The `prices` mapping of the shipped `prices.yaml`, keyed `provider/model`."""
    table = yaml.safe_load(read(PRICES))
    assert isinstance(table, dict), f"{PRICES.name} is not a mapping"
    prices = table.get("prices", {})
    assert isinstance(prices, dict) and prices, f"{PRICES.name} prices nothing"
    return prices


def test_the_apps_directory_emits_the_objects_the_plan_names(built: list[dict[str, Any]]) -> None:
    """Criterion 1, by SET EQUALITY over `(kind, name, namespace)` for everything the build
    emits in `apps` that is not one of 20a's three base kinds, plus the claim that nothing
    cluster-scoped is added. Catches an implementation that writes all four YAML files but
    leaves one out of `apps/kustomization.yaml`'s `resources`, or leaves `apps` out of the
    root's - the files are perfectly valid and the objects are simply not in the stream, which
    no linter and no schema check can see. Catches a middleware copy that keeps its
    `namespace: platform` from the file it was copied out of (decision B), which is an Ingress
    in `apps` referencing nothing and therefore serving unauthenticated traffic. Catches the
    HPA, the PDB or the ConfigMap written and never listed, and catches the two Ingresses
    collapsed into one, which decision C says cannot exempt `/healthz` from basic auth. The
    20a kinds are excluded rather than enumerated, so this case cannot fail on anything
    `test_k8s_base.py` already owns."""
    added = {
        (str(document.get("kind")), name_of(document), APPS)
        for document in built
        if namespace_of(document) == APPS and str(document.get("kind")) not in BASE_KINDS
    }
    assert added == APPS_OBJECTS, (
        f"unexpected: {sorted(added - APPS_OBJECTS)}; missing: {sorted(APPS_OBJECTS - added)}"
    )

    cluster_scoped = {
        (str(document.get("kind")), name_of(document))
        for document in built
        if namespace_of(document) is None and str(document.get("kind")) not in BASE_KINDS
    }
    assert cluster_scoped == CLUSTER_SCOPED_OBJECTS, (
        f"cluster-scoped objects changed: {sorted(cluster_scoped ^ CLUSTER_SCOPED_OBJECTS)}"
    )


def test_the_gateway_image_is_pinned_and_not_a_floating_tag(built: list[dict[str, Any]]) -> None:
    """Criterion 5 and decision D, read off the BUILT container rather than off
    `apps/kustomization.yaml`, so an `images:` transformer that names a container the Deployment
    does not have - a typo in `name:`, which kustomize accepts silently - leaves the original
    reference in the stream and reddens here. Catches a reference with no tag at all, which
    Kubernetes resolves as `latest`; catches `latest` or `main` written out, which is a cluster
    whose running code no commit names and a rollout that changes meaning without a diff; and
    catches the repository pointed at another owner or another image name, which is an apply
    that pulls somebody else's code. The tag is asserted as an exact string as well as against
    the refusal list, because `0.0.0-unbuilt` is the honest placeholder row 22 replaces."""
    image = str(sole_container(gateway_pod_owner(built)).get("image", ""))
    assert image, f"{GATEWAY} declares no image"
    assert "@" not in image, f"{GATEWAY} is pinned by digest ({image}); decision D names a tag"
    assert ":" in image.rsplit("/", 1)[-1], f"{GATEWAY} image `{image}` carries no tag at all"

    repository, _, tag = image.rpartition(":")
    assert repository == GATEWAY_IMAGE_REPOSITORY, f"{GATEWAY} runs from `{repository}`"
    assert tag not in FLOATING_TAGS, f"{GATEWAY} runs a floating tag `{tag}`"
    assert tag == GATEWAY_IMAGE_TAG, f"{GATEWAY} is tagged `{tag}`"


def test_the_gateway_carries_both_probes_and_a_service_that_resolves(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 6, plus decision E's two `envFrom` sources and decision F's three bounds -
    every one of which is a thing a manifest gets wrong with no tool saying so. Catches a
    readiness probe with no liveness probe (a wedged gateway that answers nothing and is never
    restarted) and the reverse (traffic routed to a pod whose dependencies are unreachable),
    and catches the two paths SWAPPED, which is the sharper failure: `/readyz` as the liveness
    probe restarts the pod every time Redis blips, and `/healthz` as the readiness probe sends
    traffic to a process that has no database. Catches a `targetPort` naming a port the pod
    template never declares - a number nothing listens on, or a name renamed on one side -
    which is a Service with no endpoints and a connection refused at runtime, not at build; both
    forms are resolved, an int against `containerPort` and a string against the port's `name`,
    so a correct named port is not a red. Catches §2's kustomize mechanic going wrong: the root
    sets `includeSelectors: false`, so the gateway writes its own `app.kubernetes.io/name` on
    all three of `matchLabels`, the template's labels and the Service's selector, and a
    mismatch is a rollout of zero pods reporting nothing wrong. Catches `envFrom` losing either
    source, which is a container starting with no provider keys or no cluster URLs. And catches
    decision F's bounds drifting: an HPA whose min is 1 (a single pod behind a PDB of 1 is a
    drain that can never proceed), whose max is 4 (a pod rejected at admission against the
    `apps` quota), or scaling on something other than 70% CPU."""
    owner = gateway_pod_owner(built)
    container = sole_container(owner)

    assert replicas_of(owner) == GATEWAY_REPLICAS, f"{GATEWAY} runs {replicas_of(owner)} replicas"
    assert container["readinessProbe"]["httpGet"]["path"] == READINESS_PATH
    assert container["livenessProbe"]["httpGet"]["path"] == LIVENESS_PATH

    sources = {
        key: str(entry[key].get("name", ""))
        for entry in container.get("envFrom", [])
        for key in entry
        if isinstance(entry.get(key), dict)
    }
    assert sources == ENV_FROM_SOURCES, f"{GATEWAY} takes its environment from {sources}"

    labels = template_labels(owner)
    assert labels.get(SELECTOR_LABEL) == GATEWAY, f"the pod template is labelled {labels}"
    match_labels = owner["spec"]["selector"]["matchLabels"]
    assert isinstance(match_labels, dict) and match_labels, f"{GATEWAY} selects nothing"
    unmatched = {key: value for key, value in match_labels.items() if labels.get(key) != value}
    assert unmatched == {}, f"{GATEWAY}'s selector {unmatched} matches no pod: {labels}"

    declared = container.get("ports", [])
    assert isinstance(declared, list) and declared, f"{GATEWAY} declares no container ports"
    numbers = {int(port["containerPort"]) for port in declared if "containerPort" in port}
    names = {str(port["name"]) for port in declared if port.get("name")}
    assert CONTAINER_PORT in numbers, f"{GATEWAY} listens on {sorted(numbers)}"
    assert CONTAINER_PORT_NAME in names, f"{GATEWAY}'s ports are named {sorted(names)}"

    service_spec = sole_object(built, "Service", GATEWAY)["spec"]
    selector = service_spec.get("selector", {})
    assert isinstance(selector, dict) and selector, f"{GATEWAY}'s Service selects nothing"
    stray = {key: value for key, value in selector.items() if labels.get(key) != value}
    assert stray == {}, f"{GATEWAY}'s Service selector {stray} matches no pod: {labels}"

    ports = service_spec.get("ports", [])
    assert isinstance(ports, list) and ports, f"{GATEWAY}'s Service publishes no ports"
    unresolved: list[object] = []
    for port in ports:
        target = port.get("targetPort", port["port"])
        if isinstance(target, str):
            if target not in names:
                unresolved.append(target)
        elif int(target) not in numbers:
            unresolved.append(target)
    assert unresolved == [], (
        f"{GATEWAY}'s Service targets {unresolved}, declared by nothing in its own template "
        f"(containerPorts {sorted(numbers)}, names {sorted(names)})"
    )

    hpa = sole_object(built, "HorizontalPodAutoscaler", GATEWAY)
    assert hpa["spec"]["minReplicas"] == HPA_MIN_REPLICAS
    assert hpa["spec"]["maxReplicas"] == HPA_MAX_REPLICAS
    assert cpu_utilization_target(hpa) == HPA_CPU_UTILIZATION, (
        f"{GATEWAY}'s HPA targets {cpu_utilization_target(hpa)}% CPU utilization"
    )
    reference = hpa["spec"]["scaleTargetRef"]
    assert (reference.get("kind"), reference.get("name")) == ("Deployment", GATEWAY)

    budget = sole_object(built, "PodDisruptionBudget", GATEWAY)
    assert budget["spec"].get("minAvailable") == PDB_MIN_AVAILABLE
    budget_selector = budget["spec"]["selector"]["matchLabels"]
    assert isinstance(budget_selector, dict) and budget_selector, "the PDB selects nothing"
    missed = {key: value for key, value in budget_selector.items() if labels.get(key) != value}
    assert missed == {}, f"the PDB's selector {missed} matches no gateway pod: {labels}"


def test_the_apps_subtotals_fit_inside_the_namespace_quota(built: list[dict[str, Any]]) -> None:
    """20c's criterion 3, derived and never restated: the three sums are
    parsed off the BUILT pod templates in `apps`, multiplied by each workload's replica count -
    or by the `maxReplicas` of the HPA that scales it, read from the same build - added to the
    two remaining named reserves, and compared to the `apps` `ResourceQuota` read from the same
    build. Two workloads are now read off the build rather than reserved: the gateway, and
    llama.cpp since row 20d wrote its manifest - `LLAMA_RESERVE` left `RESERVES` in the same
    change, because a pod both summed as a reserve and read off the build is priced twice and
    overflows a quota that is correct. The subtotals must NOT move across that change, which is
    what proves the manifest was written to the reserve rather than the reserve rewritten to
    the manifest. Both templates are asserted present before the sums, so a workload left out
    of `apps/kustomization.yaml`'s `resources` is a loud red here and not a subtotal that
    quietly fits. Catches the arithmetic failing even though every individual quantity is
    correct: the gateway raised to fit, a third replica, a reserve revised upward, or the quota
    lowered by a later edit to 20a's table. The gateway is priced at the HPA's CEILING, which
    catches the quota being sized to `spec.replicas` - the state where the cluster is admitted
    happily at
    rest and the first scale-up is rejected at admission, which is precisely the event an
    autoscaler exists to serve. Millicores as ints, because `0.15 + 0.3 + 0.1` is not `0.55` in
    binary floating point and a float comparison would redden a correct manifest. Compared with
    `<=`: the reserves are floors on what rows 20f and 27 may take, and the headroom above them
    is deliberate."""
    quotas = [
        quota for quota in objects_of_kind(built, "ResourceQuota") if namespace_of(quota) == APPS
    ]
    assert len(quotas) == 1, f"{len(quotas)} ResourceQuotas in {APPS}"
    hard = quota_hard(quotas[0])

    totals = reserve_totals()
    ceilings = scale_targets(built)
    owners = [owner for owner in pod_owners(built) if namespace_of(owner) == APPS]
    built_names = {name_of(owner) for owner in owners}
    missing = {GATEWAY, LLAMA} - built_names
    assert missing == set(), (
        f"the build emits no pod template in {APPS} for {sorted(missing)}: {sorted(built_names)}"
    )

    for owner in owners:
        key = (str(owner.get("kind")), name_of(owner), namespace_of(owner))
        pods = ceilings.get(key, replicas_of(owner))
        for container in all_containers(owner):
            quantities = resource_quantities(container)
            totals["requests.memory"] += pods * memory_bytes(quantities["requests.memory"])
            totals["requests.cpu"] += pods * cpu_millicores(quantities["requests.cpu"])
            totals["limits.memory"] += pods * memory_bytes(quantities["limits.memory"])

    assert totals["requests.memory"] <= memory_bytes(hard["requests.memory"]), (
        f"apps requests {totals['requests.memory']} bytes against {hard['requests.memory']}"
    )
    assert totals["requests.cpu"] <= cpu_millicores(hard["requests.cpu"]), (
        f"apps requests {totals['requests.cpu']}m against {hard['requests.cpu']}"
    )
    assert totals["limits.memory"] <= memory_bytes(hard["limits.memory"]), (
        f"apps limits {totals['limits.memory']} bytes against {hard['limits.memory']}"
    )


def test_the_gateway_declares_the_quantities_the_budget_prices(built: list[dict[str, Any]]) -> None:
    """Criterion 3's per-pod row, by VALUE rather than by presence, because the subtotals case
    above passes for any set of numbers that happens to fit. Catches the gateway given another
    workload's quantities because the file was copied, catches a limit written as `512M` where
    the table says `512Mi` (a 7% difference the API server does not treat as equal, and the
    direction that gets the gateway OOM-killed under load), and catches decision C's forbidden
    `limits.cpu` appearing - a CPU limit throttles the gateway's tail latency rather than
    protecting the node. Equality on the whole flattened mapping is what makes both the missing
    key and the extra one visible."""
    quantities = resource_quantities(sole_container(gateway_pod_owner(built)))
    assert quantities == GATEWAY_RESOURCES, f"{GATEWAY} declares {quantities}"


def test_the_apps_quota_row_reads_seven_gibibytes_of_limits(built: list[dict[str, Any]]) -> None:
    """Criterion 4 and decision A, in both places the number lives. The build half asserts the
    `apps` quota's whole `spec.hard` by equality, which catches the limits column left at the
    `5Gi` 20a wrote - the value the four reserves overflow by 1152Mi, so the llama.cpp pod row
    20d schedules would be rejected at admission - and catches the REQUESTS column being moved
    to make room, which is the 6Gi cluster-wide total #55 and the design pin and the one number
    decision A promises not to touch. The README half catches the manifest being raised while
    the table a human reads before planning a workload still promises 5Gi. All three values are
    required TOGETHER on one line: `3Gi` and `1500m` appear in the page's
    other rows and its prose, so a per-value scan would pass on a table with no `apps` row at
    all, and every matching line is counted rather than the first one taken."""
    quotas = [
        quota for quota in objects_of_kind(built, "ResourceQuota") if namespace_of(quota) == APPS
    ]
    assert len(quotas) == 1, f"{len(quotas)} ResourceQuotas in {APPS}"
    assert quota_hard(quotas[0]) == APPS_QUOTA, f"the {APPS} quota reads {quota_hard(quotas[0])}"

    rows = [
        line
        for line in read(K8S_README).splitlines()
        if f"`{APPS}`" in line and all(value in line for value in APPS_QUOTA.values())
    ]
    assert len(rows) == 1, (
        f"{len(rows)} README envelope rows state {APPS} as "
        f"{APPS_QUOTA['requests.memory']} / {APPS_QUOTA['requests.cpu']} / "
        f"{APPS_QUOTA['limits.memory']}"
    )


def test_both_ingresses_publish_the_host_with_tls_and_the_production_issuer(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 7's host, TLS and issuer half, asserted on BOTH Ingresses because decision C
    splits them and a value corrected on one is the failure this case exists for. Catches the
    `letsencrypt-staging` annotation left on a public Ingress after the rehearsal 20b's second
    issuer exists for, which is a certificate no browser trusts; catches the annotation dropped
    entirely, where cert-manager issues nothing at all and Traefik serves its own default
    certificate with no error anywhere; catches a `tls` block whose `hosts` omit the host it
    serves, which is a certificate issued for nothing the router matches; and catches one of
    the two Ingresses published on a different subdomain, which `infra/terraform/hetzner`
    publishes no A record for."""
    for ingress_name in INGRESS_MIDDLEWARES:
        ingress = sole_object(built, "Ingress", ingress_name)
        annotations = annotations_of(ingress)
        assert annotations.get(CLUSTER_ISSUER_ANNOTATION) == CLUSTER_ISSUER, (
            f"{ingress_name} is issued by {annotations.get(CLUSTER_ISSUER_ANNOTATION)}"
        )

        rules = ingress["spec"]["rules"]
        assert isinstance(rules, list) and rules, f"{ingress_name} routes nothing"
        assert {str(rule.get("host")) for rule in rules} == {INGRESS_HOST}, (
            f"{ingress_name} publishes {[rule.get('host') for rule in rules]}"
        )

        tls = ingress["spec"]["tls"]
        assert isinstance(tls, list) and len(tls) == 1, f"{ingress_name} declares {tls} for TLS"
        assert [str(host) for host in tls[0]["hosts"]] == [INGRESS_HOST]
        assert str(tls[0].get("secretName", "")), f"{ingress_name}'s tls block names no secret"


def test_only_the_root_ingress_carries_basic_auth(built: list[dict[str, Any]]) -> None:
    """Criterion 7's middleware half and decision C, by set equality per Ingress. Catches
    basic-auth listed on the `/healthz` Ingress, which puts a 401 in front of the endpoint an
    uptime monitor and a load balancer poll and makes the service look down while it is up;
    catches basic-auth dropped from the `/` one, which publishes `/metrics` and `/admin/*` to
    the internet unauthenticated; and catches rate-limit or security-headers missing from
    either. Every name parsed out of either annotation is then resolved against the
    `Middleware`s the build actually emits in `apps` - which is decision B's whole point, since
    Traefik treats an unresolvable reference as no middleware rather than as an error, so a
    `platform-basic-auth@kubernetescrd` left over from the cross-namespace shape, or a name
    misspelt, is an Ingress that silently serves unauthenticated traffic. Resolution is against
    the built set rather than against a written list, so a middleware renamed on both sides
    stays green and a middleware renamed on one side does not."""
    emitted = {(APPS, name) for name in in_apps(built, "Middleware")}
    for ingress_name, expected in INGRESS_MIDDLEWARES.items():
        parsed = middleware_names(sole_object(built, "Ingress", ingress_name))
        assert {name for _, name in parsed} == set(expected), (
            f"{ingress_name} names {sorted(name for _, name in parsed)}"
        )
        unresolved = sorted(parsed - emitted)
        assert unresolved == [], (
            f"{ingress_name} references {unresolved}, which the build emits nowhere: "
            f"{sorted(emitted)}"
        )


def test_each_apps_middleware_matches_its_platform_original(built: list[dict[str, Any]]) -> None:
    """Criterion 8 and decision B, with BOTH sides read off the same build so neither can be
    this file's own expectation and no value from 20b is restated here. Catches the copy
    drifting from its original in either direction: a rate limit doubled in `apps` while the
    README and 20b's tests still describe the platform one, an `stsSeconds` shortened on one
    side, a `basicAuth.secret` renamed in `platform` when row 21 lands and not carried across -
    which is authentication configured against a Secret that does not exist. Catches a copy
    made with an EXTRA field as well as a missing one, because the comparison is whole-`spec`
    equality rather than a per-key check. Both sides are asserted present first, so a copy
    deleted and a platform original deleted are separate, named failures rather than one
    vacuous pass over an empty intersection."""
    copies = in_apps(built, "Middleware")
    originals = {
        name_of(document): document
        for document in objects_of_kind(built, "Middleware")
        if namespace_of(document) == PLATFORM
    }
    assert set(copies) == MIDDLEWARE_NAMES, f"{APPS} holds middlewares {sorted(copies)}"
    assert set(originals) == MIDDLEWARE_NAMES, f"{PLATFORM} holds middlewares {sorted(originals)}"

    for name in sorted(MIDDLEWARE_NAMES):
        assert copies[name]["apiVersion"] == originals[name]["apiVersion"], (
            f"{name} is {copies[name]['apiVersion']} in {APPS} and "
            f"{originals[name]['apiVersion']} in {PLATFORM}"
        )
        assert copies[name]["spec"] == originals[name]["spec"], (
            f"{APPS}/{name} is {copies[name]['spec']} against {PLATFORM}/{name}'s "
            f"{originals[name]['spec']}"
        )


def test_no_secret_is_emitted_and_no_credential_is_written_inline_under_apps(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 9, in both halves. The build half: no `Secret` kind anywhere, and every Secret
    NAME the stream references is one of the three row 21 owes - catches the convenient repair
    of an apply failing on a missing Secret, which is a placeholder `Secret` carrying the
    provider keys in git that nothing ever fails to mention again, and catches a fourth Secret
    name nobody has agreed to supply. The `tls[].secretName` of an Ingress is excluded from
    that comparison and checked separately: it is a Secret cert-manager CREATES rather than
    one row 21 supplies, so criterion 7 (every Ingress names one) and criterion 9 (every
    referenced name is one of the three) cannot both be read literally. The separate check is
    the half that still bites - a `tls` secret named `gateway-secrets` would have cert-manager
    overwrite the provider keys with a certificate, so the two sets are asserted disjoint. The
    file half is a scan of what is WRITTEN under `apps/`, not of what
    is built: a literal a later patch strips out of the output is still in the history, and
    a project rule is about the repo. The scan asserts the directory exists and holds YAML first,
    so a missing `apps/` is a loud red and not a vacuous pass over zero files - the trap this
    whole case would otherwise fall into. It reads both shapes a written credential takes:
    `api_key: sk-...` on one line, and the `- name: OPENAI_API_KEY` / `value: sk-...` env pair
    a Kubernetes manifest actually uses, which a key-anchored pattern alone walks past."""
    emitted = objects_of_kind(built, "Secret")
    assert emitted == [], f"the build emits Secrets: {[name_of(document) for document in emitted]}"

    issued = tls_secret_names(built)
    assert issued, "no Ingress in the build names a tls secret for cert-manager to fill"
    assert issued.isdisjoint(PINNED_SECRET_NAMES), (
        f"cert-manager is told to write its certificate into {sorted(issued & PINNED_SECRET_NAMES)}"
    )

    referenced = {name for name in secret_references(built, set()) if name} - issued
    assert referenced <= PINNED_SECRET_NAMES, (
        f"the build references {sorted(referenced - PINNED_SECRET_NAMES)}, which row 21 does "
        "not supply"
    )

    assert APPS_DIR.is_dir(), f"{APPS_DIR} does not exist, so this scan reads no files"
    files = sorted(APPS_DIR.glob("*.yaml"))
    assert files, f"{APPS_DIR} holds no YAML at all"
    offenders: list[str] = []
    for path in files:
        offenders.extend(inline_credentials(path.name, read(path).splitlines()))
    assert offenders == [], f"inline credentials under {APPS_DIR.name}/: {offenders}"


def test_the_image_carries_the_price_table_it_loads_at_startup() -> None:
    """Criterion 10 and decision G, read off the Dockerfile. `PriceStore.__init__` loads
    `prices.yaml` eagerly and `create_app` builds the store, so an image that copies
    `routing.yaml` and not `prices.yaml` is a container this plan's Deployment schedules and
    that cannot answer its first request - a CrashLoopBackOff whose cause is a file, visible
    nowhere in the manifests. Catches the `COPY` added with no `ENV`, where the default path
    resolves outside the workdir and the failure is identical; catches the `ENV` added with no
    `COPY`, which is the same failure pointed at a path that does not exist; and catches either
    pointing somewhere other than the workdir the build stage is copied into. The routing pair
    that already exists is asserted alongside, so a sweep that removes it while adding the new
    one is red rather than a one-for-one trade nobody notices."""
    text = read(DOCKERFILE)
    copied = dockerfile_copied_files(text)
    environment = dockerfile_environment(text)

    for data_file, variable in DOCKERFILE_DATA_FILES.items():
        assert data_file in copied, f"the image copies {sorted(copied)}, not {data_file}"
        assert environment.get(variable) == f"{IMAGE_WORKDIR}/{data_file}", (
            f"{variable} reads {environment.get(variable)}"
        )


@pytest.mark.parametrize("ingress_name", sorted(INGRESS_MIDDLEWARES))
def test_each_ingress_routes_its_own_path_to_the_gateway_service(
    ingress_name: str, built: list[dict[str, Any]]
) -> None:
    """Criterion 1's routing half, which decision C's two-Ingress split is only worth anything
    with: each Ingress must send its path to the gateway `Service` this plan emits. Catches the
    `/healthz` Ingress pointing at a backend Service that does not exist, which is a 503 on the
    one path that is meant to answer without credentials, and catches the two Ingresses given
    the same path - where the router's basic-auth-free rule would match `/` as well and the
    split decision C made would be undone without any value in this file changing."""
    ingress = sole_object(built, "Ingress", ingress_name)
    services = {name_of(document) for document in objects_of_kind(built, "Service")}
    paths: set[str] = set()
    for rule in ingress["spec"]["rules"]:
        for path in rule["http"]["paths"]:
            paths.add(str(path["path"]))
            backend = path["backend"]["service"]
            assert str(backend["name"]) in services, (
                f"{ingress_name} routes {path['path']} to Service/{backend['name']}, "
                f"which the build does not emit"
            )
            assert str(backend["name"]) == GATEWAY, f"{ingress_name} routes away from {GATEWAY}"
    expected = LIVENESS_PATH if ingress_name == HEALTHZ_INGRESS else "/"
    assert paths == {expected}, f"{ingress_name} routes {sorted(paths)}"


def test_the_llama_service_is_where_the_shipped_routing_policy_expects_it(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 3, with BOTH sides of the address derived rather than written down twice: the host and the port come out of the shipped `routing.yaml`'s
    `providers.local.base_url`, which is a BARE name and therefore resolves only to a Service of
    that name in this pod's own namespace. Catches the Service named anything but `llama` and
    the Service published on any port but the one the policy dials - both of which build, lint
    and kubeconform cleanly and are a connection refused from the gateway at runtime, on the one
    candidate an EU `pii` chat is allowed to use. Catches a `targetPort` that resolves to
    nothing in the pod's own template (a name renamed on one side, a number nothing listens on),
    which is a Service with no endpoints; and catches it resolving to a DIFFERENT container port
    than the URL names, where the Service answers on 8082 and forwards to a port the server is
    not bound to. Catches a `type: NodePort` or `LoadBalancer`, which publishes an unauthenticated
    model server on every node - the `/` Ingress's basic auth is in front of the gateway, not
    this. And catches the base URL losing its `/v1` prefix, which is every OpenAI-compatible
    call 404ing while the pod is perfectly healthy. Nothing about the address is stated here."""
    url = local_base_url()
    host = url.hostname
    assert host, f"{ROUTING_POLICY.name}'s local base_url {url.geturl()} names no host"
    assert url.port is not None, f"local base_url {url.geturl()} names no port"
    assert url.path == LOCAL_BASE_PATH, f"local base_url {url.geturl()} is rooted at {url.path}"

    services = in_apps(built, "Service")
    assert host in services, f"the build emits no Service/{host} in {APPS}: {sorted(services)}"
    service = services[host]
    assert name_of(service) == host
    assert service["spec"].get("type", "ClusterIP") == "ClusterIP", (
        f"Service/{host} is a {service['spec'].get('type')}, reachable from outside the cluster"
    )

    published = [port for port in service["spec"]["ports"] if int(port["port"]) == url.port]
    assert len(published) == 1, (
        f"Service/{host} publishes {[port.get('port') for port in service['spec']['ports']]}, "
        f"and the shipped policy dials {url.port}"
    )

    container = sole_container(llama_pod_owner(built))
    target = published[0].get("targetPort", published[0]["port"])
    assert resolved_container_port(target, container) == url.port, (
        f"Service/{host} targets {target!r}, which resolves to "
        f"{resolved_container_port(target, container)} against the container's "
        f"{container.get('ports')}"
    )


def test_llama_declares_the_quantities_the_budget_prices(built: list[dict[str, Any]]) -> None:
    """Criterion 4's per-pod row, by VALUE on the whole flattened mapping, against the reserve
    the apps budget already holds (`LLAMA_RESERVE`, minus its pod multiplier) rather than three
    quantities restated here. The subtotals case passes for any set of numbers that happens to
    fit, so this is the one that says WHICH numbers. Catches a request raised to what a 7B needs
    (the reserve is what 20c priced, and raising it is a pod the `apps` quota rejects at
    admission with the cluster reporting nothing wrong until the next rollout), catches `3072Mi`
    written as `3G` - 7% smaller, which the API server does not treat as equal and which
    OOM-kills the server mid-load - and catches a `limits.cpu` appearing, which throttles token
    generation on the one workload whose latency is the product. Both the missing key and the
    extra one are visible because the comparison is equality."""
    quantities = resource_quantities(sole_container(llama_pod_owner(built)))
    assert quantities == LLAMA_RESOURCES, f"{LLAMA} declares {quantities}"


def test_llama_serves_one_model_from_a_read_only_volume(built: list[dict[str, Any]]) -> None:
    """Criteria 1 and 2, which are one failure seen from three sides. Catches `replicas: 2` and
    catches the default `RollingUpdate` strategy: a `ReadWriteOnce` claim is attached to one node
    at a time, so a second pod - a scale-up or the new pod of a rolling update - waits for a
    volume the old pod will not release until it is deleted, which is a rollout wedged forever
    with both pods reporting exactly what they are waiting for and nothing acting on it. Catches
    a claim sized for the weights it cannot hold, or a `storageClassName` naming a class this
    cluster does not define, which is a claim stuck Pending and a pod that never schedules.
    Catches the `/models` mount losing `readOnly: true`, which is a server process able to
    truncate the weights a human copied in by hand (decision B) and no way to tell that it did.
    And catches the mount pointed at another volume - an `emptyDir` left over from a draft is a
    pod that starts, finds no weights, and fails its startup probe with the claim bound to
    nothing - because the mount's NAME is resolved against the template's `volumes`."""
    owner = llama_pod_owner(built)
    assert replicas_of(owner) == LLAMA_REPLICAS, f"{LLAMA} runs {replicas_of(owner)} replicas"
    strategy = owner["spec"].get("strategy", {})
    assert strategy.get("type") == LLAMA_STRATEGY, f"{LLAMA} updates with {strategy}"

    claim = sole_object(built, "PersistentVolumeClaim", LLAMA_PVC)
    assert [str(mode) for mode in claim["spec"]["accessModes"]] == LLAMA_PVC_ACCESS_MODES
    assert str(claim["spec"]["resources"]["requests"]["storage"]) == LLAMA_PVC_STORAGE, (
        f"{LLAMA_PVC} claims {claim['spec']['resources']['requests']['storage']}"
    )
    assert "storageClassName" not in claim["spec"], (
        f"{LLAMA_PVC} names storage class {claim['spec']['storageClassName']!r}; the cluster's "
        "default provisioner is what binds it"
    )

    mount = mount_at(sole_container(owner), LLAMA_MOUNT_PATH)
    assert mount.get("readOnly") is True, f"{LLAMA_MOUNT_PATH} is mounted {mount}"
    volume = volume_named(owner, str(mount["name"]))
    assert volume.get("persistentVolumeClaim", {}).get("claimName") == LLAMA_PVC, (
        f"{LLAMA_MOUNT_PATH} is backed by {volume}"
    )


def test_llama_runs_with_the_context_and_parallelism_the_spec_names(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 5, as the whole `args` LIST by value and in order, because every one of these
    is a flag whose wrong value produces a server that starts happily and is wrong at runtime.
    Catches `--host` left off or set to `127.0.0.1`, where the Service has endpoints and every
    connection is refused; catches a `--port` that disagrees with the container port and the
    policy's URL; catches the design's `--ctx-size 4096` replaced by compose's developer-machine
    `8192`, which is twice the KV cache on a pod with a 3Gi ceiling and an OOM kill under the
    second concurrent request; catches `--parallel` missing, where llama.cpp serves one slot and
    the second caller queues behind the first with no error to show for it; and catches `-m`
    naming a file the PVC does not hold. Order is part of the assertion: a value orphaned from
    its flag by an insertion is a different command line, and equality is what sees it."""
    args = [str(arg) for arg in sole_container(llama_pod_owner(built)).get("args", [])]
    assert args == LLAMA_ARGS, f"{LLAMA} runs with {args}"


def test_llama_waits_out_a_cold_model_load_before_liveness_can_restart_it(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 6 and decision E, exactly: `10` seconds times `60` attempts
    is the 600-second ceiling on the first load of a multi-gigabyte file off a cold disk.
    Catches the startup probe dropped altogether, which is the failure the whole decision is
    about - without one, Kubernetes runs the liveness probe from the start, `/health` answers
    nothing while the weights load, and the pod is killed and restarted before it has ever
    finished loading, forever, with the event log showing only `Liveness probe failed`. Catches
    the threshold lowered to 30 (300 seconds, which is inside the range a cold 2.5GB read can
    take on this box) and the period stretched, which changes the same ceiling from the other
    side. Catches any of the three probes pointed at a path llama.cpp does not serve - a 404 is
    a failed probe, so a readiness probe on `/healthz` is a Service with no endpoints and a
    liveness probe on one is a permanent crash loop. The startup probe's port is resolved
    against the container's own declared ports, so a probe aimed at a port nothing listens on -
    which fails in exactly the same way and looks like a slow model - is red here too."""
    container = sole_container(llama_pod_owner(built))
    for probe_name in LLAMA_PROBES:
        probe = container.get(probe_name)
        assert isinstance(probe, dict), f"{LLAMA} declares no {probe_name}"
        assert probe["httpGet"]["path"] == LLAMA_PROBE_PATH, (
            f"{LLAMA}'s {probe_name} polls {probe['httpGet']['path']}"
        )

    startup = container["startupProbe"]
    assert startup["periodSeconds"] == LLAMA_STARTUP_PERIOD_SECONDS, (
        f"the startup probe polls every {startup['periodSeconds']}s"
    )
    assert startup["failureThreshold"] == LLAMA_STARTUP_FAILURE_THRESHOLD, (
        f"the startup probe gives up after {startup['failureThreshold']} attempts"
    )
    assert resolved_container_port(startup["httpGet"]["port"], container) is not None, (
        f"the startup probe polls port {startup['httpGet']['port']!r}, which the container's "
        f"{container.get('ports')} does not declare"
    )


def test_the_local_model_id_reads_the_same_in_every_file_that_names_it(
    built: list[dict[str, Any]],
) -> None:
    """Criteria 8 and 9 and decision A, with the id read OFF THE BUILD - the basename of the
    `-m` argument, minus its quantisation suffix - and every file compared to that and to each
    other, never to a literal repeated five times. Catches the rename half-done, which is the
    only way this fails in practice: the manifest serving a 3B while `routing.yaml` still
    resolves `agent-default` to the 7B id puts a model that never ran on every ledger row, in
    every `X-Gateway-Model` header and in the eval report, with the gateway answering normally
    throughout. Catches the alias line renamed and nordlicht's OVERRIDE left behind, which is
    the same failure for the one tenant whose traffic never leaves the box - the count of
    `local` chat candidates is asserted exactly, so one of two is not half a pass. Catches
    `prices.yaml` missing a row for the id that actually runs, which books every local request
    `unpriced`; catches a row written with no `effective_from`, which cannot be selected
    against an instant. Catches compose and `.env.example` still defaulting to a file the image
    now has no reason to hold, where the local stack and the cluster serve different models
    under one name. And asserts the `local/{PREVIOUS_LOCAL_MODEL}` key is still priced (hard
    rule 7): a ledger row booked last month is priced against the entry it was charged with, so
    the old key is KEPT beside the new one and never edited into it."""
    file_name = Path(sole_arg_value(sole_container(llama_pod_owner(built)), "-m")).name
    model_id = model_id_from(file_name)

    candidates = local_candidate_models()
    named = [model for model in candidates if model == model_id]
    assert len(named) == LOCAL_CHAT_CANDIDATES, (
        f"{ROUTING_POLICY.name} gives `{LOCAL_PROVIDER}` the candidates {candidates}, of which "
        f"{len(named)} name the model the manifest loads ({model_id})"
    )
    assert PREVIOUS_LOCAL_MODEL not in candidates, (
        f"{ROUTING_POLICY.name} still routes to {PREVIOUS_LOCAL_MODEL}, which no pod serves"
    )

    prices = priced_keys()
    key = f"{LOCAL_PROVIDER}/{model_id}"
    assert key in prices, f"{PRICES.name} prices {sorted(prices)}, and the pod serves {key}"
    entries = prices[key]
    assert isinstance(entries, list) and entries, f"{key} carries no price entries"
    assert all(entry.get("effective_from") for entry in entries), f"{key} reads {entries}"
    assert f"{LOCAL_PROVIDER}/{PREVIOUS_LOCAL_MODEL}" in prices, (
        f"{PRICES.name} dropped the {PREVIOUS_LOCAL_MODEL} row that prices already-booked rows"
    )

    defaults = {
        COMPOSE.name: compose_model_files(),
        ENV_EXAMPLE.name: env_example_model_files(),
    }
    for source, written in defaults.items():
        assert written == {file_name}, f"{source} defaults the weights file to {sorted(written)}"


def test_the_llama_image_is_pinned_and_not_a_floating_tag(built: list[dict[str, Any]]) -> None:
    """Criterion 7 and decision C, the gateway's case applied to the one image this repo does
    not build, and read off the BUILT container so an `images:` transformer whose `name:` does
    not match the Deployment's container - which kustomize accepts in silence - leaves the bare
    `llama` reference in the stream and reddens here. Catches the tag left at upstream's own
    moving `server`, which compose used before this plan: the same manifest applied twice pulls
    two different servers, so a regression in generation quality or in token throughput cannot
    be attributed to any commit. Catches `latest` and `main` for the same reason, catches a
    reference with no tag at all (which Kubernetes resolves as `latest`), and catches the
    repository pointed somewhere else, which is an apply that runs an image nobody reviewed on
    the pod holding the model weights. The exact tag is asserted as well as the refusal list,
    because the pinned tag IS the claim - a refusal list alone passes for any tag at all."""
    image = str(sole_container(llama_pod_owner(built)).get("image", ""))
    assert image, f"{LLAMA} declares no image"
    assert "@" not in image, f"{LLAMA} is pinned by digest ({image}); decision C names a tag"
    assert ":" in image.rsplit("/", 1)[-1], f"{LLAMA} image `{image}` carries no tag at all"

    repository, _, tag = image.rpartition(":")
    assert repository == LLAMA_IMAGE_REPOSITORY, f"{LLAMA} runs from `{repository}`"
    assert tag not in LLAMA_FLOATING_TAGS, f"{LLAMA} runs a floating tag `{tag}`"
    assert tag == LLAMA_IMAGE_TAG, f"{LLAMA} is tagged `{tag}`"
