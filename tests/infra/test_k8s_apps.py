r"""`infra/k8s/apps`: the gateway Deployment, its Service, HPA, PDB and ConfigMap, the three
copied Traefik middlewares and the two `api.` Ingresses an earlier change agrees, the two `helpdesk.`
Ingresses row 20g adds over the same three middleware copies, plus the llama.cpp
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

import functools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit

import pytest
import yaml

from opsconsole.gateway.app import create_app
from opsconsole.gateway.routes.admin import ADMIN_ROUTING, ADMIN_ROUTING_RELOAD
from opsconsole.gateway.routes.directory import ADMIN_DIRECTORY
from opsconsole.gateway.routes.metrics import METRICS

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

#: Row 20e's second llama.cpp process: one server serves one model, so the embedding endpoint
#: is its own Deployment, its own Service and its own claim. The names are here rather than in
#: the embedding block below because `APPS_OBJECTS` names them first.
LLAMA_EMBED = "llama-embed"
LLAMA_EMBED_PVC = "llama-embed-models"

#: Row 20f's four Helpdesk workloads, their shared ConfigMap and the Secret this repo names
#: and never supplies. The names are here rather than in the Helpdesk block below because
#: `HELPDESK_OBJECTS` names them first, and every one of them is also the value of that
#: workload's `app.kubernetes.io/name` label and of its Service.
HELPDESK_AGENT_API = "helpdesk-agent-api"
HELPDESK_MCP_SERVER = "helpdesk-mcp-server"
HELPDESK_SCHEDULER = "helpdesk-scheduler"
HELPDESK_WEB = "helpdesk-web"
HELPDESK_CONFIG_MAP = "helpdesk-config"
HELPDESK_SECRET = "helpdesk-secrets"

GATEWAY = "gateway"
GATEWAY_CONFIG_MAP = "gateway-config"
GATEWAY_SECRET = "gateway-secrets"
ROOT_INGRESS = "gateway"
HEALTHZ_INGRESS = "gateway-healthz"

#: Row 20g criterion 1: the two public routers for the Helpdesk web host, named for the pair
#: they mirror (decision D). They are here rather than in the Helpdesk block below because
#: `APPS_OBJECTS` names them first.
HELPDESK_INGRESS = "helpdesk"
HELPDESK_HEALTHZ_INGRESS = "helpdesk-healthz"

#: Row 20f criterion 1, as `(kind, name, namespace)`: the eight objects the Helpdesk adds.
#: `Service/helpdesk-scheduler` is absent on purpose (criterion 2) and its absence is asserted
#: by name in its own case rather than left to this set. Written as its own frozenset because
#: two cases read it - `APPS_OBJECTS` unions it below, so the eight are named once.
HELPDESK_OBJECTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("ConfigMap", HELPDESK_CONFIG_MAP, APPS),
        ("Deployment", HELPDESK_AGENT_API, APPS),
        ("Deployment", HELPDESK_MCP_SERVER, APPS),
        ("Deployment", HELPDESK_SCHEDULER, APPS),
        ("Deployment", HELPDESK_WEB, APPS),
        ("Service", HELPDESK_AGENT_API, APPS),
        ("Service", HELPDESK_MCP_SERVER, APPS),
        ("Service", HELPDESK_WEB, APPS),
    }
)

#: Row 21a's eight `apps` NetworkPolicies, by name. They come from `base/networkpolicies.yaml`
#: rather than from `apps/`, but they land in this namespace, and this case scopes by namespace
#: rather than by the directory a document was written in - so a policy added there and not here
#: is red. `tests/infra/test_k8s_netpol.py` owns everything about their CONTENT; this set is
#: only the eight names, in the one place the `apps` object list is asserted whole.
APPS_NETWORK_POLICIES = frozenset(
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

#: Criterion 1, as `(kind, name, namespace)`. Set EQUALITY: a file left out of
#: `apps/kustomization.yaml`'s `resources` and an object nobody agreed to are the same failure,
#: and only equality sees both. Row 20f's eight are unioned in rather than listed again, so the
#: object list and the Helpdesk cases cannot disagree about what the build owes.
APPS_OBJECTS: frozenset[tuple[str, str, str]] = HELPDESK_OBJECTS | frozenset(
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
        ("Ingress", HELPDESK_INGRESS, APPS),
        ("Ingress", HELPDESK_HEALTHZ_INGRESS, APPS),
        ("PersistentVolumeClaim", LLAMA_PVC, APPS),
        ("Deployment", LLAMA, APPS),
        ("Service", LLAMA, APPS),
        ("PersistentVolumeClaim", LLAMA_EMBED_PVC, APPS),
        ("Deployment", LLAMA_EMBED, APPS),
        ("Service", LLAMA_EMBED, APPS),
    }
    | {("NetworkPolicy", name, APPS) for name in APPS_NETWORK_POLICIES}
)

#: Criterion 1's second half: "nothing cluster-scoped is added". The two 20b issuers are the
#: only non-base cluster-scoped objects the build may contain, so this set is what 20c must
#: leave untouched rather than a claim about what 20b writes.
CLUSTER_SCOPED_OBJECTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ClusterIssuer", "letsencrypt-staging"),
        ("ClusterIssuer", "letsencrypt-prod"),
        # An earlier change, from `monitoring/exporters.yaml`: the one subject in this repo with
        # cluster-wide RBAC. They are listed here because this case asserts the build's
        # cluster-scoped objects by EQUALITY, so an addition anywhere has to be agreed here;
        # `tests/infra/test_k8s_monitoring.py`'s
        # `test_kube_state_metrics_is_the_only_subject_with_rbac_and_reads_only` owns what they
        # grant and that there is no third.
        ("ClusterRole", "kube-state-metrics"),
        ("ClusterRoleBinding", "kube-state-metrics"),
    }
)

#: Decision D: the repository the `images:` transformer sets, and the honest placeholder tag.
#: `deploy.yml` rewrites the tag to the deployed SHA in the runner and commits nothing back,
#: so this value is permanent here and an apply from a checkout fails on the pull, which is the
#: intended direction.
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

#: The envelope the apps budget held for the Helpdesk before row 20f built it: three pods of
#: one size, which 20f divides into FOUR built workloads without moving the three totals
#: (decision A). It is no longer a reserve - `RESERVES` lost it in the same change that added
#: the four built rows, because a pod both summed here and read off the built template is
#: priced twice - and it stays in this module as the envelope
#: `test_the_four_helpdesk_pods_fit_the_envelope_the_reserve_held` holds the four manifests to,
#: the same move 20d made with `LLAMA_RESERVE`.
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

#: The reserve still held as a constant - row 27's console, the one workload in `apps` that no
#: plan has built yet. `LLAMA_RESERVE` and `HELPDESK_RESERVE` are NOT in it any more: rows 20d
#: and 20f build those pods, so the subtotals case reads their quantities off the built
#: templates the way it already reads the gateway's, and summing a reserve as well would price
#: the same workload twice and overflow a quota that is correct. Both stay in this module as
#: the rows their own cases hold the manifests to
#: (`test_llama_declares_the_quantities_the_budget_prices`,
#: `test_the_four_helpdesk_pods_fit_the_envelope_the_reserve_held`).
RESERVES = (CONSOLE_RESERVE,)

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

#: Row 20e criterion 4: the provider row the `embed` alias resolves to after the split. One
#: provider carries one `base_url`, so the second server is a second provider and not a second
#: alias - and this key becomes the ledger's `provider` value for every embedding row.
LOCAL_EMBED_PROVIDER = "local_embed"

#: Criterion 4: the alias whose only candidate moves, and the model that candidate names. Every
#: CHAT alias keeps `local`, which the same case asserts.
EMBED_ALIAS = "embed"
EMBED_MODEL = "bge-small-en-v1.5"

#: Decision B: the upstream Q8_0 artefact, ~130 MB of a 33M-parameter encoder where quantising
#: harder buys nothing. The suffix is what criterion 10's case takes off the built `-m` file
#: name to get the model id, rather than writing the id down a second time.
Q8_SUFFIX = "-q8_0.gguf"
LLAMA_EMBED_MODEL_FILE = f"{EMBED_MODEL}{Q8_SUFFIX}"

#: Criterion 1: the embedding claim's size and access mode, and the `storageClassName` it must
#: NOT declare - k3s's local-path provisioner is the default, the same reason 20b's Postgres
#: and 20d's claim omit it. `1Gi` for a ~130 MB file, exactly.
LLAMA_EMBED_PVC_STORAGE = "1Gi"
LLAMA_EMBED_PVC_ACCESS_MODES = ["ReadWriteOnce"]

#: Criterion 2: one replica, and the only update strategy a `ReadWriteOnce` volume survives -
#: 20d's reason, unchanged by the file being smaller.
LLAMA_EMBED_REPLICAS = 1
LLAMA_EMBED_STRATEGY = "Recreate"

#: Criterion 6, by value and in order: decision D's BERT-encoder flags (`--pooling cls` written
#: out because a mean-pooled vector is a different vector space that still returns plausible
#: numbers, and `n_ubatch >= n_ctx` because llama.cpp refuses a non-causal model otherwise), the
#: address it binds and the one file it loads. The port reads the same numeral criterion 3
#: derives from `routing.yaml`; it is written here because the args are pinned
#: as a LIST, and a list assembled from three sources cannot state an order.
LLAMA_EMBED_ARGS: list[str] = [
    "--embedding",
    "--pooling",
    "cls",
    "--ctx-size",
    "512",
    "--ubatch-size",
    "512",
    "--host",
    "0.0.0.0",
    "--port",
    "8083",
    "-m",
    f"{LLAMA_MOUNT_PATH}/{LLAMA_EMBED_MODEL_FILE}",
]

#: Criterion 7, exactly: `5` seconds times `24` attempts is the 120-second
#: ceiling on a cold load of a ~130 MB file - not 20d's 600, which was sized for a
#: multi-gigabyte read. `LLAMA_PROBE_PATH` and `LLAMA_PROBES` are reused rather than restated:
#: this is the same llama.cpp server, so the path it serves and the three probes a pod runs are
#: the same fact, and writing them twice would let the two drift.
LLAMA_EMBED_STARTUP_PERIOD_SECONDS = 5
LLAMA_EMBED_STARTUP_FAILURE_THRESHOLD = 24

#: Criterion 9's embedding row, by value, per pod. Decision E takes it from
#: the headroom rather than from llama.cpp's row, and `limits.cpu` is absent on purpose -
#: equality on the whole mapping is what keeps it absent.
LLAMA_EMBED_RESOURCES: dict[str, str] = {
    "requests.memory": "256Mi",
    "requests.cpu": "100m",
    "limits.memory": "512Mi",
}

#: Criterion 10's compose and `.env.example` defaults. Every matching line is scanned and the
#: values required to agree. Neither pattern can match the chat server's
#: `LLAMA_MODEL_FILE` lines and neither of those can match these: `${LLAMA_MODEL_FILE` is not a
#: substring of `${LLAMA_EMBED_MODEL_FILE`, and the `.env.example` pattern is anchored whole.
COMPOSE_EMBED_MODEL_DEFAULT = re.compile(r"\$\{LLAMA_EMBED_MODEL_FILE:-(?P<file>[^}]+)\}")
ENV_EMBED_MODEL_DEFAULT = re.compile(r"^LLAMA_EMBED_MODEL_FILE=(?P<file>\S+)$")

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
#: nothing in the build: row 21c contracts it in `infra/secrets/gateway-secrets.example.yaml`
#: and the ciphertext is applied out of band. The ConfigMap is written by this plan.
ENV_FROM_SOURCES: dict[str, str] = {
    "configMapRef": GATEWAY_CONFIG_MAP,
    "secretRef": GATEWAY_SECRET,
}

#: §2's kustomize mechanic: the label the gateway writes itself, because the root's `labels:`
#: block carries `includeSelectors: false` and never reaches a pod template.
SELECTOR_LABEL = "app.kubernetes.io/name"

#: Row 20g criterion 5's second host. `infra/terraform/hetzner/variables.tf` publishes `api`,
#: `console`, `helpdesk` and `grafana` and validates that all four are present, so no new
#: choice is made here either.
HELPDESK_HOST = "helpdesk.example.com"

#: Row 20g criterion 5, PER INGRESS rather than as one shared constant: two routers on
#: `api.example.com` and two on `helpdesk.example.com`. A router copied from its
#: neighbour and left on the neighbour's host is what a single shared value cannot see.
INGRESS_HOSTS: dict[str, str] = {
    ROOT_INGRESS: "api.example.com",
    HEALTHZ_INGRESS: "api.example.com",
    HELPDESK_INGRESS: HELPDESK_HOST,
    HELPDESK_HEALTHZ_INGRESS: HELPDESK_HOST,
}

#: Row 20g criteria 4 and 5: the backend `Service` each router sends its path to. An earlier change
#: decision B leaves `helpdesk-web` as the only public Helpdesk surface, so no entry here names
#: the agent-api or the MCP server and no other Ingress may either (criterion 7).
INGRESS_BACKENDS: dict[str, str] = {
    ROOT_INGRESS: GATEWAY,
    HEALTHZ_INGRESS: GATEWAY,
    HELPDESK_INGRESS: HELPDESK_WEB,
    HELPDESK_HEALTHZ_INGRESS: HELPDESK_WEB,
}

#: Row 21b criterion 1: the one path the public `gateway` router serves. `/v1` is the whole
#: OpenAI-compatible surface; `/metrics`, `/admin/*` and `/readyz` are served by the same pod
#: and left with no public router at all (decisions A and B), which is what "cluster-internal"
#: means once a NetworkPolicy has been shown unable to see a URL path.
VERSIONED_PREFIX = "/v1"

#: Row 21b criteria 1 and 7, PER INGRESS beside `INGRESS_HOSTS` and `INGRESS_BACKENDS` rather
#: than as one rule for every non-healthz router: only `gateway` narrows to `/v1`, and
#: `helpdesk` keeps `/` because narrowing the browser host takes the Helpdesk off the internet.
INGRESS_PATHS: dict[str, str] = {
    ROOT_INGRESS: VERSIONED_PREFIX,
    HEALTHZ_INGRESS: LIVENESS_PATH,
    HELPDESK_INGRESS: "/",
    HELPDESK_HEALTHZ_INGRESS: LIVENESS_PATH,
}

#: Row 21b criterion 2: the four paths the gateway serves that no public router may match,
#: IMPORTED from the three route modules that mount them rather than retyped here, so a route
#: moved to a new path moves this list with it instead of leaving a stale string behind.
CLOSED_PUBLIC_PATHS: tuple[str, ...] = (
    ADMIN_ROUTING,
    ADMIN_ROUTING_RELOAD,
    ADMIN_DIRECTORY,
    METRICS,
)

#: Row 21b criterion 5: the prefix every admin route mounts under, WITH the trailing slash that
#: makes it a `/`-segment boundary - `/admin` alone also reads as a prefix of `/admin-keys`,
#: which is a path `CLOSED_PUBLIC_PATHS` does not name.
ADMIN_PREFIX = "/admin/"

#: What `create_app` reads provider keys from in this file: nothing (#66). `tests/gateway`
#: passes the same empty mapping, so a developer with a real key exported cannot have these
#: cases build an app holding a live-keyed client.
NO_ENVIRON: Mapping[str, str] = {}

#: Row 20g criterion 3: the Secret cert-manager WRITES for each router, by name. An earlier change
#: criterion 10 named the two Helpdesk ones before this row existed. All four are asserted
#: distinct from one another and disjoint from `PINNED_SECRET_NAMES`, because ingress-shim owns
#: one `Certificate` per `secretName` through the Ingress that asked for it.
INGRESS_TLS_SECRETS: dict[str, str] = {
    ROOT_INGRESS: "gateway-tls",
    HEALTHZ_INGRESS: "gateway-healthz-tls",
    HELPDESK_INGRESS: "helpdesk-tls",
    HELPDESK_HEALTHZ_INGRESS: "helpdesk-healthz-tls",
}

#: The two routers the design exempts from basic auth, one per host. Named rather than derived
#: from a `-healthz` suffix: a router called `helpdesk-healthz-web` would pass a suffix test
#: while being neither of these.
HEALTHZ_INGRESSES = frozenset({HEALTHZ_INGRESS, HELPDESK_HEALTHZ_INGRESS})

#: Row 20g criterion 4's `pathType`s. `Exact` on the exempt router because Traefik orders
#: routers by rule length: an exact `/healthz` wins over the other router's `PathPrefix(/)`,
#: and `Prefix` there instead would pull every path under it out from behind basic auth.
PATH_TYPE_PREFIX = "Prefix"
PATH_TYPE_EXACT = "Exact"

#: Row 21b's boundary rows, both sides of the one rule in this file with an edge (§6's
#: "Boundaries" line). `(path, pathType, request path, matches)`: a `Prefix` matches an equal
#: path and one continuing at a `/` boundary and NOT one continuing mid-segment, an `Exact`
#: matches only itself, and a `/` Prefix - the router this plan removes - matches everything,
#: which is why the closed paths are public until it goes.
MATCHER_BOUNDARY_ROWS: tuple[tuple[str, str, str, bool], ...] = (
    (VERSIONED_PREFIX, PATH_TYPE_PREFIX, VERSIONED_PREFIX, True),
    (VERSIONED_PREFIX, PATH_TYPE_PREFIX, "/v1/chat/completions", True),
    (VERSIONED_PREFIX, PATH_TYPE_PREFIX, "/v1beta/x", False),
    (VERSIONED_PREFIX, PATH_TYPE_PREFIX, ADMIN_ROUTING, False),
    ("/", PATH_TYPE_PREFIX, ADMIN_ROUTING, True),
    ("/", PATH_TYPE_PREFIX, "/", True),
    (LIVENESS_PATH, PATH_TYPE_EXACT, LIVENESS_PATH, True),
    (LIVENESS_PATH, PATH_TYPE_EXACT, "/healthz/deeper", False),
    (LIVENESS_PATH, PATH_TYPE_EXACT, READINESS_PATH, False),
)

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
    HELPDESK_INGRESS: frozenset({"basic-auth", "rate-limit", "security-headers"}),
    HELPDESK_HEALTHZ_INGRESS: frozenset({"rate-limit", "security-headers"}),
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

#: Row 20f criterion 1: the four workloads in the order the budget table prices them, and the
#: three of them that serve traffic. The scheduler is in the first tuple and not the second -
#: it holds a Redis leader lock and runs jobs, so it has no port, no Service (criterion 2) and
#: no probes (criterion 7), while everything the other three are held to it is held to as well.
HELPDESK_WORKLOADS = (HELPDESK_AGENT_API, HELPDESK_MCP_SERVER, HELPDESK_SCHEDULER, HELPDESK_WEB)
HELPDESK_SERVING = (HELPDESK_AGENT_API, HELPDESK_MCP_SERVER, HELPDESK_WEB)

#: Criterion 1, by value: the port each serving workload's container publishes, which is also
#: the port its Service publishes. Three different numbers on purpose - a Service copied from
#: its neighbour and left on the neighbour's port is endpoints pointing at a closed socket,
#: which nothing in a build reports.
HELPDESK_CONTAINER_PORTS: dict[str, int] = {
    HELPDESK_AGENT_API: 8000,
    HELPDESK_MCP_SERVER: 8001,
    HELPDESK_WEB: 3000,
}

#: Criterion 1: the Service type all three carry. Decision B leaves one public surface and the
#: `helpdesk` Ingress pair publishes it, so a `LoadBalancer` or a `NodePort` here is a Helpdesk
#: service reachable from outside the cluster without any router in front of it.
HELPDESK_SERVICE_TYPE = "ClusterIP"

#: Criterion 3, by value, per pod, compared as strings - the subtotals and
#: the envelope cases are the only places these are parsed. Decision A divides the envelope
#: `HELPDESK_RESERVE` held for three pods across four workloads: the agent-api keeps the
#: reserve's own row and the other three take 128Mi. `limits.cpu` is absent from every row on
#: purpose, and equality on the whole flattened mapping is what keeps it absent - a CPU limit
#: throttles the agent-api's tail latency, which the design puts an SLO on.
HELPDESK_RESOURCES: dict[str, dict[str, str]] = {
    HELPDESK_AGENT_API: {
        "requests.memory": "192Mi",
        "requests.cpu": "100m",
        "limits.memory": "384Mi",
    },
    HELPDESK_MCP_SERVER: {
        "requests.memory": "128Mi",
        "requests.cpu": "75m",
        "limits.memory": "256Mi",
    },
    HELPDESK_SCHEDULER: {
        "requests.memory": "128Mi",
        "requests.cpu": "50m",
        "limits.memory": "256Mi",
    },
    HELPDESK_WEB: {
        "requests.memory": "128Mi",
        "requests.cpu": "75m",
        "limits.memory": "256Mi",
    },
}

#: Criterion 7: the scheduler's replica count and the only strategy that keeps exactly one copy
#: through a rollout - `RollingUpdate` starts the new pod before the old one stops, which is two
#: schedulers for the length of a deploy.
HELPDESK_SCHEDULER_REPLICAS = 1
HELPDESK_SCHEDULER_STRATEGY = "Recreate"

#: Criterion 7's third and fourth halves: the three probe keys a container may declare, all of
#: which the scheduler must declare none of (decision F). A liveness probe against a command
#: nobody has written restarts the pod forever.
PROBE_KEYS = ("readinessProbe", "livenessProbe", "startupProbe")

#: Criterion 8, by value: the whole `helpdesk-config` mapping, asserted by equality so a fifth
#: key nobody agreed to is as visible as a missing one. Every host is fully qualified because
#: these pods are in `apps` and two of the three addresses are in `platform`, where a bare
#: `redis` resolves to nothing. Redis database `1`, not the gateway's `0`: one Redis, two
#: tenants of it, and a shared index is one `FLUSHDB` away from taking out the other repo.
HELPDESK_CONFIG: dict[str, str] = {
    "HELPDESK_GATEWAY_URL": "http://gateway.apps.svc.cluster.local:8080/v1",
    "HELPDESK_REDIS_URL": "redis://redis.platform.svc.cluster.local:6379/1",
    "HELPDESK_QDRANT_URL": "http://qdrant.platform.svc.cluster.local:6333",
    "HELPDESK_ENV": "production",
}

#: Criterion 8's in-cluster claim, checked on every URL value of the ConfigMap rather than on
#: the three keys named above, so a fourth address added later is covered by the same rule.
IN_CLUSTER_SUFFIX = ".svc.cluster.local"

#: Criterion 8: the Redis database index each side owns, as the path of its own URL. The
#: gateway's is READ OFF THE BUILD from the key below rather than restated;
#: the constant is the key's name, not its value.
HELPDESK_REDIS_URL_KEY = "HELPDESK_REDIS_URL"
HELPDESK_REDIS_DATABASE_PATH = "/1"
GATEWAY_REDIS_URL_KEY = "OPSCONSOLE_REDIS_URL"

#: Criterion 9: the per-container variable that names a process in a trace viewer, and the
#: gateway's key for the same thing. The Helpdesk's is per-container `env` and NOT a key of
#: `helpdesk-config`, because a shared ConfigMap would give all four pods one name and collapse
#: them into one service in the viewer - the failure a trace test would not catch.
HELPDESK_TRACE_NAME_KEY = "HELPDESK_OTEL_SERVICE_NAME"
GATEWAY_TRACE_NAME_KEY = "OPSCONSOLE_OTEL_SERVICE_NAME"

#: Criterion 9's four values, by value. They read the same as the workload
#: names, and they are written out rather than derived: the claim is that the trace viewer
#: shows these four strings, and a comprehension over the names would assert only that two
#: things this file computes agree.
HELPDESK_TRACE_NAMES: dict[str, str] = {
    HELPDESK_AGENT_API: "helpdesk-agent-api",
    HELPDESK_MCP_SERVER: "helpdesk-mcp-server",
    HELPDESK_SCHEDULER: "helpdesk-scheduler",
    HELPDESK_WEB: "helpdesk-web",
}

#: Criterion 8 and decision E: configuration arrives as two whole objects, the same shape
#: decision E gave the gateway. `helpdesk-secrets` is NAMED here and supplied nowhere in this
#: repo (criterion 10), so an apply made before it exists fails loudly rather than starting a
#: pod with no gateway key.
HELPDESK_ENV_FROM_SOURCES: dict[str, str] = {
    "configMapRef": HELPDESK_CONFIG_MAP,
    "secretRef": HELPDESK_SECRET,
}

#: Criterion 11 and decision D: the companion repo's registry namespace and the same honest
#: placeholder tag the gateway carries. Each workload's repository is that prefix plus its own
#: name, so an `images:` entry pointing two workloads at one image is visible.
HELPDESK_IMAGE_REPOSITORY_PREFIX = "ghcr.io/developermusa/ai-helpdesk-governed-access"
HELPDESK_IMAGE_TAG = "0.0.0-unbuilt"

#: Criterion 12: the two labels the ROOT kustomization stamps on every object's
#: `metadata.labels`. Its `labels:` block carries `includeSelectors: false`, so neither may
#: reach a `matchLabels` or a Service selector - a label that reaches a selector is immutable
#: after the first apply, and changing one then means deleting the Deployment.
KUSTOMIZE_WIDE_LABELS = frozenset({"app.kubernetes.io/part-of", "app.kubernetes.io/managed-by"})

#: Decision E and row 20f criterion 10: the only Secret names anything in the build may
#: reference. Four, and row 21c contracts all four in `infra/secrets/*.example.yaml` -
#: `helpdesk-secrets` is the gateway API key the Helpdesk calls with, its database URL and the
#: Langfuse pair, supplied SOPS-encrypted and never here. The emitted-Secret half of the
#: same case is build-wide and stays that way.
PINNED_SECRET_NAMES = frozenset(
    {"postgres-credentials", "traefik-basic-auth", GATEWAY_SECRET, HELPDESK_SECRET}
)

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
    WRITES, not ones `infra/secrets/` contracts, which is why the pinned-name check
    subtracts them."""
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


@functools.lru_cache(maxsize=1)
def mounted_paths() -> frozenset[str]:
    """Every path the shipped gateway app mounts, read off the app's own router (row 21b
    criteria 3 and 5) and never written down here.

    `create_app` is given an EMPTY environ (#66): a developer with a real `OPENAI_API_KEY`
    exported would otherwise have this file build an app holding a live-keyed client, and the
    route table is the same either way.

    The walk is recursive because `app.routes` is NOT the route table in this FastAPI version:
    `include_router` leaves one `_IncludedRouter` object per router, whose own routes hang off
    `original_router.routes`, and a flat `getattr(route, "path", None)` over `app.routes` sees
    only Starlette's `/docs`, `/redoc` and `/openapi.json`. A flat read would therefore make
    both derived cases pass over an EMPTY served set - exactly the vacuum they exist to
    prevent - which is why each caller asserts the paths it expects to find are present.
    """
    collected: set[str] = set()

    def walk(routes: list[Any]) -> None:
        for route in routes:
            path = getattr(route, "path", None)
            if isinstance(path, str):
                collected.add(path)
            nested = getattr(route, "routes", None)
            if nested is None:
                included = getattr(route, "original_router", None)
                nested = getattr(included, "routes", None)
            if nested:
                walk(list(nested))

    walk(list(create_app(environ=NO_ENVIRON).routes))
    return frozenset(collected)


def router_matches(route_path: str, path_type: str, request_path: str) -> bool:
    """Traefik's matching rule, the one piece of logic in this file, in ONE place so no two
    cases can disagree about what "matched" means.

    `Exact` matches an equal path and nothing else. `Prefix` matches an equal path or one
    continuing at a `/`-segment boundary, so `/v1` matches `/v1/chat/completions` and not
    `/v1beta/x`, and a `/` Prefix matches every path there is. Pinned both sides by
    `MATCHER_BOUNDARY_ROWS` inside `test_no_public_router_matches_an_admin_path_or_the_scrape`.
    """
    if path_type == PATH_TYPE_EXACT:
        return request_path == route_path
    assert path_type == PATH_TYPE_PREFIX, (
        f"a router serves {route_path} as `{path_type}`, which is neither "
        f"{PATH_TYPE_EXACT} nor {PATH_TYPE_PREFIX}"
    )
    if request_path == route_path:
        return True
    boundary = route_path if route_path.endswith("/") else f"{route_path}/"
    return request_path.startswith(boundary)


def gateway_routers(documents: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Every `(ingress, path, pathType)` triple in the build whose backend is `Service/gateway`.

    Scoped by BACKEND rather than over every Ingress in the build, because the `helpdesk`
    router keeps `/` (criterion 7) and a `/`-wide read would make criterion 2 unsatisfiable on
    a path the gateway never sees: `helpdesk.example.com/admin/routing` is forwarded to
    `Service/helpdesk-web`, which serves no admin API. What criterion 2 forbids is a router
    that hands one of the closed paths to the gateway.
    """
    routers: list[tuple[str, str, str]] = []
    for ingress in objects_of_kind(documents, "Ingress"):
        ingress_name = name_of(ingress)
        for rule in ingress["spec"]["rules"]:
            for path in rule["http"]["paths"]:
                backend = path["backend"]["service"]
                if str(backend["name"]) != GATEWAY:
                    continue
                routers.append((ingress_name, str(path["path"]), str(path.get("pathType"))))
    return routers


def routers_matching(documents: list[dict[str, Any]], request_path: str) -> list[str]:
    """The gateway-backed routers that would serve `request_path`, rendered for a message."""
    return [
        f"{ingress_name} {route_path} ({path_type})"
        for ingress_name, route_path, path_type in gateway_routers(documents)
        if router_matches(route_path, path_type, request_path)
    ]


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


def helpdesk_pod_owner(documents: list[dict[str, Any]], workload_name: str) -> dict[str, Any]:
    """The Helpdesk Deployment named `workload_name`, selected by name and namespace and
    asserted to own a pod - four near-identical workloads are exactly where one file copied
    from another loses its `spec.template`."""
    owner = sole_object(documents, "Deployment", workload_name)
    assert owner in pod_owners(documents), f"{workload_name} carries no pod template"
    return owner


def envelope_totals(reserve: dict[str, str | int]) -> dict[str, int]:
    """One reserve row, multiplied out the way `reserve_totals` multiplies the ones still
    held: memory in bytes, CPU in whole millicores. `HELPDESK_RESERVE` is no longer summed as a
    reserve, so the envelope case parses it here instead."""
    pods = int(str(reserve["pods"]))
    return {
        "requests.memory": pods * memory_bytes(str(reserve["requests.memory"])),
        "requests.cpu": pods * cpu_millicores(str(reserve["requests.cpu"])),
        "limits.memory": pods * memory_bytes(str(reserve["limits.memory"])),
    }


def container_image_parts(container: dict[str, Any], label: str) -> tuple[str, str]:
    """A container's image split into `(repository, tag)`, asserting it carries an explicit tag
    and is not pinned by digest. A reference with no tag is resolved by Kubernetes as `latest`,
    so the absence of a tag and a floating tag are one failure and this helper refuses the
    first half of it."""
    image = str(container.get("image", ""))
    assert image, f"{label} declares no image"
    assert "@" not in image, f"{label} is pinned by digest ({image}); decision D names a tag"
    assert ":" in image.rsplit("/", 1)[-1], f"{label}'s image `{image}` carries no tag at all"
    repository, _, tag = image.rpartition(":")
    return repository, tag


def llama_embed_pod_owner(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """The embedding Deployment, selected by name and namespace and asserted to own a pod."""
    owner = sole_object(documents, "Deployment", LLAMA_EMBED)
    assert owner in pod_owners(documents), f"{LLAMA_EMBED} carries no pod template"
    return owner


def llama_embed_base_url() -> SplitResult:
    """`providers.local_embed.base_url` out of the SHIPPED `routing.yaml`, split into its parts.
    This is the address the gateway dials for an embedding after the split, and the only source
    criterion 3's Service name and port are read from."""
    policy = yaml.safe_load(read(ROUTING_POLICY))
    assert isinstance(policy, dict), f"{ROUTING_POLICY.name} is not a mapping"
    providers = policy.get("providers", {})
    assert isinstance(providers, dict) and LOCAL_EMBED_PROVIDER in providers, (
        f"{ROUTING_POLICY.name} names no `{LOCAL_EMBED_PROVIDER}` provider: "
        f"{sorted(providers) if isinstance(providers, dict) else providers}"
    )
    declared = providers[LOCAL_EMBED_PROVIDER].get("base_url", "")
    assert declared, (
        f"{ROUTING_POLICY.name}'s `{LOCAL_EMBED_PROVIDER}` provider declares no base_url"
    )
    return urlsplit(str(declared))


def alias_candidates(alias: str) -> list[dict[str, str]]:
    """The shipped policy's candidate list for one alias, as `{provider, model}` mappings."""
    policy = yaml.safe_load(read(ROUTING_POLICY))
    assert isinstance(policy, dict), f"{ROUTING_POLICY.name} is not a mapping"
    aliases = policy.get("aliases", {})
    assert isinstance(aliases, dict) and alias in aliases, (
        f"{ROUTING_POLICY.name} names no `{alias}` alias"
    )
    candidates = aliases[alias].get("candidates", [])
    assert isinstance(candidates, list), f"`{alias}` names no candidate list"
    return [
        {str(key): str(value) for key, value in candidate.items()}
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def providers_named_outside(alias: str) -> list[str]:
    """Every provider the shipped policy names ANYWHERE except in `alias`'s own candidate list -
    the other aliases' candidates and every tenant override, WITH duplicates, so a chat line
    repointed at the embedding server is visible however many other lines are correct."""
    policy = yaml.safe_load(read(ROUTING_POLICY))
    assert isinstance(policy, dict), f"{ROUTING_POLICY.name} is not a mapping"
    groups: list[Any] = [
        entry.get("candidates", [])
        for name, entry in policy.get("aliases", {}).items()
        if name != alias
    ]
    for tenant in policy.get("tenants", {}).values():
        groups.extend(tenant.get("overrides", {}).values())
    return [
        str(candidate["provider"])
        for group in groups
        if isinstance(group, list)
        for candidate in group
        if isinstance(candidate, dict)
    ]


def embed_model_id_from(file_name: str) -> str:
    """The model id inside the embedding GGUF file name: `bge-small-en-v1.5-q8_0.gguf` is
    `bge-small-en-v1.5`. The suffix is asserted rather than stripped conditionally, so a file in
    another quantisation is a named failure and not an id with a tail on it."""
    assert file_name.endswith(Q8_SUFFIX), f"{file_name} is not a {Q8_SUFFIX} artefact"
    return file_name[: -len(Q8_SUFFIX)]


def compose_embed_model_files() -> set[str]:
    """Every GGUF file name a `${LLAMA_EMBED_MODEL_FILE:-...}` default in `docker-compose.yml`
    falls back to. All matching lines, never the first."""
    return {match.group("file") for match in COMPOSE_EMBED_MODEL_DEFAULT.finditer(read(COMPOSE))}


def env_example_embed_model_files() -> set[str]:
    """Every `LLAMA_EMBED_MODEL_FILE=` value in `.env.example`, all matching lines scanned."""
    files: set[str] = set()
    for line in read(ENV_EXAMPLE).splitlines():
        match = ENV_EMBED_MODEL_DEFAULT.fullmatch(line.strip())
        if match is not None:
            files.add(match.group("file"))
    return files


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
    the refusal list, because `0.0.0-unbuilt` is the placeholder the tree keeps: `build.yml` publishes the image this tag does not name, and `deploy.yml` chooses one only
    inside the runner, committing nothing back (its ADR is 0014)."""
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
    one remaining named reserve, and compared to the `apps` `ResourceQuota` read from the same
    build. Seven workloads are now read off the build rather than reserved: the gateway,
    llama.cpp since row 20d wrote its manifest - `LLAMA_RESERVE` left `RESERVES` in the same
    change, because a pod both summed as a reserve and read off the build is priced twice and
    overflows a quota that is correct - row 20e's embedding server, the one workload no
    reserve was EVER held for (decision E), so unlike 20d's change the three sums genuinely
    move there and the README table moves with them, and row 20f's four Helpdesk workloads,
    which `HELPDESK_RESERVE` left `RESERVES` for on the same terms. `CONSOLE_RESERVE` is the
    only reserve left. All seven templates are asserted present
    before the sums, so a workload left out of `apps/kustomization.yaml`'s `resources` is a
    loud red here and not a build that quietly sums less and fits. Catches the arithmetic
    failing even though every individual quantity is correct: the gateway raised to fit, a
    third replica, a reserve revised upward, or the quota lowered by a later edit to 20a's
    table. The gateway is priced at the HPA's CEILING, which
    catches the quota being sized to `spec.replicas` - the state where the cluster is admitted
    happily at
    rest and the first scale-up is rejected at admission, which is precisely the event an
    autoscaler exists to serve. Millicores as ints, because `0.15 + 0.3 + 0.1` is not `0.55` in
    binary floating point and a float comparison would redden a correct manifest. Compared with
    `<=`: the reserve is a floor on what row 27 may take, and the headroom above it is
    deliberate."""
    quotas = [
        quota for quota in objects_of_kind(built, "ResourceQuota") if namespace_of(quota) == APPS
    ]
    assert len(quotas) == 1, f"{len(quotas)} ResourceQuotas in {APPS}"
    hard = quota_hard(quotas[0])

    totals = reserve_totals()
    ceilings = scale_targets(built)
    owners = [owner for owner in pod_owners(built) if namespace_of(owner) == APPS]
    built_names = {name_of(owner) for owner in owners}
    missing = {GATEWAY, LLAMA, LLAMA_EMBED, *HELPDESK_WORKLOADS} - built_names
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


def test_every_ingress_publishes_its_own_host_with_tls_and_the_production_issuer(
    built: list[dict[str, Any]],
) -> None:
    """An earlier change's criterion 7 and row 20g's criteria 2 and 5, asserted on EVERY Ingress out of
    `INGRESS_HOSTS` rather than against one shared host, because two hosts now carry two
    routers each and a value corrected on one of the four is the failure this case exists for.
    Catches the `letsencrypt-staging` annotation left on a public Ingress after the rehearsal
    20b's second issuer exists for, which is a certificate no browser trusts; catches the
    annotation dropped entirely, where cert-manager issues nothing at all and Traefik serves
    its own default certificate with no error anywhere; catches a `tls` block whose `hosts`
    omit the host it serves, which is a certificate issued for nothing the router matches; and
    catches a Helpdesk router copied from the gateway pair and left on `api.example.com`,
    or a gateway router moved to `helpdesk.example.com` - each of which a single shared
    `INGRESS_HOST` accepted silently."""
    for ingress_name, expected_host in INGRESS_HOSTS.items():
        ingress = sole_object(built, "Ingress", ingress_name)
        annotations = annotations_of(ingress)
        assert annotations.get(CLUSTER_ISSUER_ANNOTATION) == CLUSTER_ISSUER, (
            f"{ingress_name} is issued by {annotations.get(CLUSTER_ISSUER_ANNOTATION)}"
        )

        rules = ingress["spec"]["rules"]
        assert isinstance(rules, list) and rules, f"{ingress_name} routes nothing"
        assert {str(rule.get("host")) for rule in rules} == {expected_host}, (
            f"{ingress_name} publishes {[rule.get('host') for rule in rules]}"
        )

        tls = ingress["spec"]["tls"]
        assert isinstance(tls, list) and len(tls) == 1, f"{ingress_name} declares {tls} for TLS"
        assert [str(host) for host in tls[0]["hosts"]] == [expected_host]
        assert str(tls[0].get("secretName", "")), f"{ingress_name}'s tls block names no secret"


def test_only_the_healthz_ingresses_are_exempt_from_basic_auth(built: list[dict[str, Any]]) -> None:
    """An earlier change's criterion 7 middleware half and row 20g's criterion 4, by set equality per
    Ingress over all four routers - the name says "the healthz ingresses" because the exempt
    router is now one of two hosts' pairs rather than "the root ingress" of one. Catches
    basic-auth listed on the `/healthz` Ingress, which puts a 401 in front of the endpoint an
    uptime monitor and a load balancer poll and makes the service look down while it is up;
    catches basic-auth dropped from the `/v1` one, which publishes the OpenAI-compatible API
    to the internet unauthenticated - `/metrics` and `/admin/*` are no longer part of what that
    costs, because row 21b leaves them with no public router at all; and catches rate-limit or
    security-headers missing from either. Every name parsed out of either annotation is then
    resolved against the
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


def test_every_tls_secret_name_is_distinct_and_supplied_by_nobody(
    built: list[dict[str, Any]],
) -> None:
    """Row 20g criterion 3, over every Ingress the build emits. The kill that lives ONLY here
    is two Ingresses pointing at one `secretName`: cert-manager's ingress-shim creates one
    `Certificate` per `secretName` and owns it through the Ingress that asked, so a Helpdesk
    router copied from the gateway pair with `gateway-tls` left in place is two owners for one
    certificate, each re-issuing over the other, and no other case in this file can see it -
    `tls_secret_names` and
    `test_no_secret_is_emitted_and_no_credential_is_written_inline_under_apps` both collapse
    the names into a SET, where a shared name is indistinguishable from a missing one. It also
    pins each router's secret by name against `INGRESS_TLS_SECRETS`, which catches
    `helpdesk-healthz-tls` and `helpdesk-tls` swapped between the two routers - four distinct
    names stay four distinct names under a swap. The disjointness assertion at the end repeats
    a build-wide check that case already makes; it is kept because criterion 3 states it on the
    four names specifically, and a `tls.secretName` of `helpdesk-secrets` would have
    cert-manager overwrite the Helpdesk's credentials with a certificate."""
    issued: list[str] = []
    for ingress in objects_of_kind(built, "Ingress"):
        ingress_name = name_of(ingress)
        assert ingress_name in INGRESS_TLS_SECRETS, (
            f"the build emits Ingress/{ingress_name}, which row 20g's four routers do not name"
        )
        entries = ingress["spec"]["tls"]
        assert isinstance(entries, list) and len(entries) == 1, (
            f"{ingress_name} declares {entries} for TLS"
        )
        secret_name = str(entries[0].get("secretName", ""))
        assert secret_name == INGRESS_TLS_SECRETS[ingress_name], (
            f"{ingress_name} asks cert-manager to write `{secret_name}`"
        )
        issued.append(secret_name)

    assert len(issued) == len(INGRESS_TLS_SECRETS), (
        f"the build emits {len(issued)} Ingress TLS blocks, not {len(INGRESS_TLS_SECRETS)}"
    )
    assert len(set(issued)) == len(issued), f"two Ingresses claim one certificate: {sorted(issued)}"
    assert tls_secret_names(built).isdisjoint(PINNED_SECRET_NAMES), (
        "cert-manager is told to write a certificate into "
        f"{sorted(tls_secret_names(built) & PINNED_SECRET_NAMES)}, which this repo supplies"
    )


def test_no_ingress_publishes_the_agent_api_or_the_mcp_server(built: list[dict[str, Any]]) -> None:
    """Row 20g criterion 7, by ABSENCE across every Ingress backend in the build rather than
    across the four routers `INGRESS_BACKENDS` names. An earlier change decision B leaves `helpdesk-web`
    as the only public Helpdesk surface - the agent-api is the shape a project rule rejects for
    the console and the MCP server is a tool endpoint the design gives no host - and what this
    case catches is a FIFTH Ingress added later that publishes one of them. Its neighbour
    `test_each_ingress_routes_its_own_path_to_its_own_backend_service` parametrises over
    `INGRESS_BACKENDS`, so a router nobody wrote a mapping entry for is never visited by it at
    all; only a build-wide scan sees one."""
    forbidden = {HELPDESK_AGENT_API, HELPDESK_MCP_SERVER}
    published: dict[str, set[str]] = {}
    for ingress in objects_of_kind(built, "Ingress"):
        backends: set[str] = set()
        for rule in ingress["spec"]["rules"]:
            for path in rule["http"]["paths"]:
                backends.add(str(path["backend"]["service"]["name"]))
        published[name_of(ingress)] = backends

    offenders = {
        ingress_name: sorted(backends & forbidden)
        for ingress_name, backends in published.items()
        if backends & forbidden
    }
    assert offenders == {}, (
        f"{offenders} publish a Helpdesk surface plan 20f decision B keeps private"
    )


def test_each_apps_middleware_matches_its_platform_original(built: list[dict[str, Any]]) -> None:
    """Criterion 8 and decision B, with BOTH sides read off the same build so neither can be
    this file's own expectation and no value from 20b is restated here. Catches the copy
    drifting from its original in either direction: a rate limit doubled in `apps` while the
    README and 20b's tests still describe the platform one, an `stsSeconds` shortened on one
    side, a `basicAuth.secret` renamed in `platform` and not carried across -
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
    NAME the stream references is one of the four `infra/secrets/` contracts - catches the repair
    of an apply failing on a missing Secret, which is a placeholder `Secret` carrying the
    provider keys in git that nothing ever fails to mention again, and catches a fourth Secret
    name nobody has agreed to supply. The `tls[].secretName` of an Ingress is excluded from
    that comparison and checked separately: it is a Secret cert-manager CREATES rather than
    one `infra/secrets/` contracts, so criterion 7 (every Ingress names one) and criterion 9
    (every referenced name is one of the four) cannot both be read literally. The separate check is
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
        f"the build references {sorted(referenced - PINNED_SECRET_NAMES)}, which "
        "`infra/secrets/` does not contract"
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
def test_each_ingress_routes_its_own_path_to_its_own_backend_service(
    ingress_name: str, built: list[dict[str, Any]]
) -> None:
    """An earlier change's criterion 1 routing half and row 20g's criteria 4 and 5: each router sends
    its own path to ITS OWN backend out of `INGRESS_BACKENDS`, on the port NAME the pod
    publishes, with the `pathType` the path is worth. Catches a Helpdesk router copied from the
    gateway pair and left pointing at `Service/gateway`, which serves the OpenAI-compatible API
    on the browser host - the `== GATEWAY` form this case replaced asserted exactly that wrong
    thing. Catches a router pointing at a backend Service the build does not emit, which is a
    503 on the one path meant to answer without credentials. Catches a backend port given as a
    number that drifts from the `http` the Service targets, which resolves to no endpoint.
    Catches the two routers of one host given the same path, where the basic-auth-free rule
    matches `/` as well. And catches `pathType: Prefix` on the exempt `/healthz` router, which
    puts everything under `/healthz/...` outside basic auth - the one thing decision B's
    `Exact` buys and the only assertion here that sees it."""
    ingress = sole_object(built, "Ingress", ingress_name)
    services = {name_of(document) for document in objects_of_kind(built, "Service")}
    expected_backend = INGRESS_BACKENDS[ingress_name]
    is_healthz = ingress_name in HEALTHZ_INGRESSES
    expected_path = INGRESS_PATHS[ingress_name]
    expected_path_type = PATH_TYPE_EXACT if is_healthz else PATH_TYPE_PREFIX
    paths: set[str] = set()
    for rule in ingress["spec"]["rules"]:
        for path in rule["http"]["paths"]:
            paths.add(str(path["path"]))
            assert str(path.get("pathType")) == expected_path_type, (
                f"{ingress_name} serves {path['path']} as {path.get('pathType')}"
            )
            backend = path["backend"]["service"]
            assert str(backend["name"]) in services, (
                f"{ingress_name} routes {path['path']} to Service/{backend['name']}, "
                f"which the build does not emit"
            )
            assert str(backend["name"]) == expected_backend, (
                f"{ingress_name} routes to Service/{backend['name']} and not {expected_backend}"
            )
            port = backend.get("port", {})
            assert isinstance(port, dict), f"{ingress_name} names backend port {port}"
            assert str(port.get("name")) == CONTAINER_PORT_NAME, (
                f"{ingress_name} names backend port {port} and not `{CONTAINER_PORT_NAME}`"
            )
    assert paths == {expected_path}, f"{ingress_name} routes {sorted(paths)}"


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


def test_the_embedding_service_is_where_the_shipped_routing_policy_expects_it(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 3, with BOTH halves of the address derived and never written down twice: the host and the port come out of the shipped `routing.yaml`'s
    `providers.local_embed.base_url`, which is a BARE name and therefore resolves only to a
    Service of that name in this pod's own namespace. Catches an implementation that publishes
    the embedding server under any other Service name, or on any port but the one the policy
    dials - both build, lint and kubeconform cleanly and are a connection refused on every
    `/v1/embeddings` call and every semantic-cache probe, with the pod perfectly healthy.
    Catches a `targetPort` that resolves to nothing in this pod's own template, which is a
    Service with no endpoints, and catches it resolving to a DIFFERENT container port than the
    URL names, where the Service answers on 8083 and forwards where nothing is bound. Catches a
    `type: NodePort` or `LoadBalancer`, which publishes an unauthenticated model server on every
    node. And catches the base URL losing its `/v1` prefix, which is every OpenAI-compatible
    call 404ing. The chat Service is a NEIGHBOURING guard and cannot produce this pass: the
    lookup is by the embedding provider's own host, so a build emitting only `llama` reddens on
    the missing Service rather than resolving against it."""
    url = llama_embed_base_url()
    host = url.hostname
    assert host, f"{ROUTING_POLICY.name}'s {LOCAL_EMBED_PROVIDER} base_url names no host"
    assert url.port is not None, f"{LOCAL_EMBED_PROVIDER} base_url {url.geturl()} names no port"
    assert url.path == LOCAL_BASE_PATH, (
        f"{LOCAL_EMBED_PROVIDER} base_url {url.geturl()} is rooted at {url.path}"
    )

    services = in_apps(built, "Service")
    assert host in services, f"the build emits no Service/{host} in {APPS}: {sorted(services)}"
    service = services[host]
    assert service["spec"].get("type", "ClusterIP") == "ClusterIP", (
        f"Service/{host} is a {service['spec'].get('type')}, reachable from outside the cluster"
    )

    published = [port for port in service["spec"]["ports"] if int(port["port"]) == url.port]
    assert len(published) == 1, (
        f"Service/{host} publishes {[port.get('port') for port in service['spec']['ports']]}, "
        f"and the shipped policy dials {url.port}"
    )

    container = sole_container(llama_embed_pod_owner(built))
    target = published[0].get("targetPort", published[0]["port"])
    assert resolved_container_port(target, container) == url.port, (
        f"Service/{host} targets {target!r}, which resolves to "
        f"{resolved_container_port(target, container)} against the container's "
        f"{container.get('ports')}"
    )


def test_the_embed_alias_resolves_to_the_embedding_server_and_not_the_chat_one() -> None:
    """Criterion 4, off the shipped policy alone - no build is involved, so this is the case
    that still speaks when `kubectl` is absent. Catches the split half-done in either
    direction: the `local_embed` provider row added and the `embed` alias left pointing at
    `local`, which is every embedding request sent to a chat server that answers 501 for
    `/v1/embeddings`; and the alias repointed while `local`'s own `base_url` is edited to the
    embedding address, which silently moves every chat completion - including the one candidate
    an EU `pii` tenant is allowed - onto a 384-dimension encoder. The chat port is read out of
    `LLAMA_ARGS` rather than written again, so the two servers cannot be
    collapsed onto one address without this reddening. And it catches a CHAT alias or a tenant
    override naming `local_embed`, which no other case here can see: `providers_named_outside`
    keeps duplicates, so one bad line among correct ones is still a failure."""
    candidates = alias_candidates(EMBED_ALIAS)
    assert candidates == [{"provider": LOCAL_EMBED_PROVIDER, "model": EMBED_MODEL}], (
        f"`{EMBED_ALIAS}` resolves to {candidates}"
    )

    chat = local_base_url()
    assert chat.hostname == LLAMA, f"`{LOCAL_PROVIDER}` now dials {chat.geturl()}"
    assert chat.port == int(LLAMA_ARGS[LLAMA_ARGS.index("--port") + 1]), (
        f"`{LOCAL_PROVIDER}` dials port {chat.port}, and the chat pod binds "
        f"{LLAMA_ARGS[LLAMA_ARGS.index('--port') + 1]}"
    )

    elsewhere = providers_named_outside(EMBED_ALIAS)
    assert LOCAL_EMBED_PROVIDER not in elsewhere, (
        f"{ROUTING_POLICY.name} routes non-embedding traffic to {LOCAL_EMBED_PROVIDER}: {elsewhere}"
    )


def test_the_embedding_pod_declares_the_quantities_the_budget_prices(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 9's per-pod row, by VALUE on the whole flattened mapping.
    The subtotals case passes for any set of numbers that happens to fit inside the quota, so
    this is the one that says WHICH numbers - decision E's `256Mi` / `100m` / `512Mi`, taken
    from the headroom rather than out of llama.cpp's row. Catches a request copied from the
    chat server's 1024Mi, which is a pod the `apps` quota rejects at admission with nothing
    else reporting a problem; catches `512Mi` written as `500M` - 5% smaller and not equal to
    the API server either, which is an OOM kill under load; and catches a `limits.cpu`
    appearing, which throttles the encoder that every semantic-cache probe waits on. Both a
    missing key and an extra one are visible because the comparison is equality."""
    quantities = resource_quantities(sole_container(llama_embed_pod_owner(built)))
    assert quantities == LLAMA_EMBED_RESOURCES, f"{LLAMA_EMBED} declares {quantities}"


def test_the_embedding_pod_serves_one_model_from_a_read_only_volume(
    built: list[dict[str, Any]],
) -> None:
    """Criteria 1 and 2, which are one failure seen from four sides. Catches `replicas: 2` and
    catches the default `RollingUpdate` strategy: a `ReadWriteOnce` claim attaches to one node
    at a time, so the new pod of a rolling update waits for a volume the old pod will not
    release until it is deleted - a rollout wedged forever with both pods reporting exactly
    what they wait on and nothing acting on it. Catches the claim sized wrong or given a
    `storageClassName` this cluster does not define, which is a claim stuck Pending and a pod
    that never schedules. Catches the `/models` mount losing `readOnly: true`, which is a
    server process able to truncate weights a human copied in by hand (decision B). And catches
    the mount pointed at the CHAT claim or at an `emptyDir` left over from a draft, because the
    mount's NAME is resolved against this template's own `volumes` and the claim name is
    asserted - decision B's whole argument is that sharing 20d's `llama-models` pins both pods
    to one node forever."""
    owner = llama_embed_pod_owner(built)
    assert replicas_of(owner) == LLAMA_EMBED_REPLICAS, (
        f"{LLAMA_EMBED} runs {replicas_of(owner)} replicas"
    )
    strategy = owner["spec"].get("strategy", {})
    assert strategy.get("type") == LLAMA_EMBED_STRATEGY, f"{LLAMA_EMBED} updates with {strategy}"

    claim = sole_object(built, "PersistentVolumeClaim", LLAMA_EMBED_PVC)
    assert [str(mode) for mode in claim["spec"]["accessModes"]] == LLAMA_EMBED_PVC_ACCESS_MODES
    assert str(claim["spec"]["resources"]["requests"]["storage"]) == LLAMA_EMBED_PVC_STORAGE, (
        f"{LLAMA_EMBED_PVC} claims {claim['spec']['resources']['requests']['storage']}"
    )
    assert "storageClassName" not in claim["spec"], (
        f"{LLAMA_EMBED_PVC} names storage class {claim['spec']['storageClassName']!r}; the "
        "cluster's default provisioner is what binds it"
    )

    mount = mount_at(sole_container(owner), LLAMA_MOUNT_PATH)
    assert mount.get("readOnly") is True, f"{LLAMA_MOUNT_PATH} is mounted {mount}"
    volume = volume_named(owner, str(mount["name"]))
    assert volume.get("persistentVolumeClaim", {}).get("claimName") == LLAMA_EMBED_PVC, (
        f"{LLAMA_MOUNT_PATH} is backed by {volume}"
    )


def test_the_embedding_server_runs_with_the_flags_a_bert_encoder_needs(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 6 and decision D, as the whole `args` LIST by value and in order, because every
    one of these is a flag whose absence produces a server that starts happily and is wrong at
    runtime. Catches `--embedding` dropped, which is a llama.cpp process that serves completions
    and refuses `/v1/embeddings` - the exact 502 this plan exists to remove. Catches `--pooling
    cls` dropped, which is the sharpest one: the GGUF's own metadata decides, a mean-pooled
    vector is a DIFFERENT vector space, every number it returns is plausible, and the only
    symptom is retrieval quality nobody can attribute. Catches `--ubatch-size` below
    `--ctx-size`, which llama.cpp refuses at request time for a non-causal model while the pod
    stays Ready; catches a `--ctx-size` above 512, which buys an error at position 513 instead
    of a truncation; catches `--host` left off or set to `127.0.0.1`, where the Service has
    endpoints and every connection is refused; catches a `--port` that disagrees with the
    address the policy dials; and catches `-m` naming a file this claim does not hold. Order is
    part of the assertion: a value orphaned from its flag by an insertion is a different command
    line, and only equality sees it."""
    args = [str(arg) for arg in sole_container(llama_embed_pod_owner(built)).get("args", [])]
    assert args == LLAMA_EMBED_ARGS, f"{LLAMA_EMBED} runs with {args}"


def test_the_embedding_pod_waits_out_its_model_load_before_liveness_can_restart_it(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 7, exactly: `5` seconds times `24` attempts is the
    120-second ceiling this file's ~130 MB earns, and not 20d's 600. Catches the startup probe
    dropped altogether, which is the failure the whole decision is about - without one,
    Kubernetes runs the liveness probe from the start, `/health` answers nothing while the
    weights load, and the pod is killed before it has ever finished loading, forever, with the
    event log showing only `Liveness probe failed`. Catches the threshold lowered, which is the
    same crash loop with a longer fuse, and catches the period stretched, which moves the same
    ceiling from the other side. Catches any of the three probes pointed at a path llama.cpp
    does not serve - a 404 is a failed probe, so a readiness probe on `/healthz` is a Service
    with no endpoints and a liveness probe on one is a permanent crash loop. The startup
    probe's port is resolved against this container's OWN declared ports, so a probe aimed at
    the chat server's 8082 - which fails identically and looks like a slow model - is red
    here too."""
    container = sole_container(llama_embed_pod_owner(built))
    for probe_name in LLAMA_PROBES:
        probe = container.get(probe_name)
        assert isinstance(probe, dict), f"{LLAMA_EMBED} declares no {probe_name}"
        assert probe["httpGet"]["path"] == LLAMA_PROBE_PATH, (
            f"{LLAMA_EMBED}'s {probe_name} polls {probe['httpGet']['path']}"
        )

    startup = container["startupProbe"]
    assert startup["periodSeconds"] == LLAMA_EMBED_STARTUP_PERIOD_SECONDS, (
        f"the startup probe polls every {startup['periodSeconds']}s"
    )
    assert startup["failureThreshold"] == LLAMA_EMBED_STARTUP_FAILURE_THRESHOLD, (
        f"the startup probe gives up after {startup['failureThreshold']} attempts"
    )
    assert resolved_container_port(startup["httpGet"]["port"], container) is not None, (
        f"the startup probe polls port {startup['httpGet']['port']!r}, which the container's "
        f"{container.get('ports')} does not declare"
    )


def test_the_embedding_image_is_the_same_pinned_build_the_chat_server_runs(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 6's first clause and decision C, asserted as EQUALITY between the two built
    containers rather than against a repository and a tag written down a second time: one
    `images:` entry rewrites both bare `llama` references, so the claim is that the two agree
    and not that either matches a literal. Catches a second `images:` entry added for
    `llama-embed`, which is a capability nobody chose - two llama.cpp builds on one box, with a
    regression in either server attributable to no commit. Catches a container image name the
    transformer does not match (a typo in `name:`, which kustomize accepts in silence), which
    leaves a bare `llama-embed` in the stream that no registry can pull. Catches the tag left
    at upstream's floating `server`, or `latest`, or `main`, or absent altogether - which
    Kubernetes resolves as `latest` - each of which is a cluster whose running code no commit
    names. The floating-tag list is applied to the embedding container itself, so the check
    does not depend on the chat case having run."""
    image = str(sole_container(llama_embed_pod_owner(built)).get("image", ""))
    chat_image = str(sole_container(llama_pod_owner(built)).get("image", ""))
    assert image, f"{LLAMA_EMBED} declares no image"
    assert ":" in image.rsplit("/", 1)[-1], f"{LLAMA_EMBED} image `{image}` carries no tag"
    assert image == chat_image, (
        f"{LLAMA_EMBED} runs `{image}` while {LLAMA} runs `{chat_image}`; decision C says one "
        "build serves both"
    )

    tag = image.rpartition(":")[2]
    assert tag not in LLAMA_FLOATING_TAGS, f"{LLAMA_EMBED} runs a floating tag `{tag}`"


def test_the_embedding_model_file_reads_the_same_in_every_file_that_names_it(
    built: list[dict[str, Any]],
) -> None:
    """Criteria 5 and 10, with the file read OFF THE BUILD - the basename of the `-m` argument -
    and compose, `.env.example` and `prices.yaml` compared to that and to each other, never to a
    literal repeated four times. Catches the cluster and the local stack serving different
    weights under one alias, which is a developer reproducing nothing and a CI compose run that
    proves nothing about the pod. Catches `prices.yaml` missing a row for the id the pod
    actually serves under its NEW provider key: decision A renames the first half of every
    embedding price key, so the row that priced `local/bge-small-en-v1.5` no longer matches and
    every embedding books `unpriced` while the gateway answers normally. Catches such a row
    written with no `effective_from`, which cannot be selected against an instant. And asserts
    the old `local/` key is still priced: a ledger row already booked under it is
    priced against the entry it was charged with, so the old key is KEPT beside the new one and
    never edited into it. The compose and `.env.example` scans read EVERY matching line - both files name `llama` and `models` on several - and neither pattern
    can match the chat server's `LLAMA_MODEL_FILE` lines."""
    file_name = Path(sole_arg_value(sole_container(llama_embed_pod_owner(built)), "-m")).name
    model_id = embed_model_id_from(file_name)

    defaults = {
        COMPOSE.name: compose_embed_model_files(),
        ENV_EXAMPLE.name: env_example_embed_model_files(),
    }
    for source, written in defaults.items():
        assert written == {file_name}, f"{source} defaults the weights file to {sorted(written)}"

    prices = priced_keys()
    key = f"{LOCAL_EMBED_PROVIDER}/{model_id}"
    assert key in prices, f"{PRICES.name} prices {sorted(prices)}, and the pod serves {key}"
    entries = prices[key]
    assert isinstance(entries, list) and entries, f"{key} carries no price entries"
    assert all(entry.get("effective_from") for entry in entries), f"{key} reads {entries}"
    assert f"{LOCAL_PROVIDER}/{model_id}" in prices, (
        f"{PRICES.name} dropped the {LOCAL_PROVIDER}/{model_id} row that prices rows already "
        "booked under the provider name before the split"
    )


def test_the_build_emits_every_helpdesk_object_the_plan_names(built: list[dict[str, Any]]) -> None:
    """Row 20f criterion 1, one object at a time so the red names the object rather than a set
    difference, plus the half no other case holds: each Service's TYPE and PORT, and the
    `targetPort` resolved against its own workload's container ports. Catches
    `Service/helpdesk-web` copied from `helpdesk-agent-api` and left on 8000, which is a
    Service whose endpoints point at a socket nothing listens on - a correct Deployment, a
    correct Service, and a connection refused at runtime that no build and no schema check
    sees. Catches a `targetPort: http` naming a port the workload's own template never
    declares, in both forms the field takes. Catches a `LoadBalancer` or a `NodePort`, which is
    a Helpdesk service reachable from outside the cluster with no router in front of it
    (decision B leaves one public surface, and the `helpdesk` Ingress pair publishes it). The
    cluster-scoped equality is criterion 1's second half; the set is shared with
    `test_the_apps_directory_emits_the_objects_the_plan_names`, which is the case that owns
    the `apps` set equality, so what is this case's own is the ports and the types."""
    emitted = {
        (str(document.get("kind")), name_of(document), namespace_of(document)) for document in built
    }
    missing = sorted(HELPDESK_OBJECTS - emitted)
    assert missing == [], f"the build emits no {missing}"

    for workload_name, expected_port in sorted(HELPDESK_CONTAINER_PORTS.items()):
        service_spec = sole_object(built, "Service", workload_name)["spec"]
        assert str(service_spec.get("type")) == HELPDESK_SERVICE_TYPE, (
            f"{workload_name}'s Service is a {service_spec.get('type')}"
        )
        ports = service_spec.get("ports", [])
        assert isinstance(ports, list) and len(ports) == 1, (
            f"{workload_name}'s Service publishes {ports}"
        )
        assert int(ports[0]["port"]) == expected_port, (
            f"{workload_name}'s Service publishes port {ports[0]['port']}"
        )
        target = ports[0].get("targetPort", ports[0]["port"])
        container = sole_container(helpdesk_pod_owner(built, workload_name))
        assert resolved_container_port(target, container) == expected_port, (
            f"{workload_name}'s Service targets {target}, which resolves to "
            f"{resolved_container_port(target, container)} against its own container ports "
            f"{container.get('ports')}"
        )

    cluster_scoped = {
        (str(document.get("kind")), name_of(document))
        for document in built
        if namespace_of(document) is None and str(document.get("kind")) not in BASE_KINDS
    }
    assert cluster_scoped == CLUSTER_SCOPED_OBJECTS, (
        f"cluster-scoped objects changed: {sorted(cluster_scoped ^ CLUSTER_SCOPED_OBJECTS)}"
    )


def test_the_scheduler_has_no_service(built: list[dict[str, Any]]) -> None:
    """Criterion 2, by NAME absence across every Service in `apps`, with the three serving
    Services asserted present first so the case cannot pass vacuously over a namespace that
    holds no Helpdesk at all - the trap an absence assertion falls into on the day the
    manifests do not exist yet. Catches the fourth Service a copy-paste of the three produces:
    the scheduler holds a Redis leader lock and serves nothing, so a Service in front of it is
    a cluster address with no listener behind it, published to every pod in the cluster and
    silently failing whatever later dials it."""
    services = in_apps(built, "Service")
    absent = [name for name in HELPDESK_SERVING if name not in services]
    assert absent == [], f"the build emits no Service for {absent}: {sorted(services)}"
    assert HELPDESK_SCHEDULER not in services, (
        f"the build emits Service/{HELPDESK_SCHEDULER}, which fronts a process that listens on "
        "nothing"
    )


@pytest.mark.parametrize(("workload_name", "expected"), sorted(HELPDESK_RESOURCES.items()))
def test_each_helpdesk_workload_declares_the_quantities_the_budget_prices(
    workload_name: str, expected: dict[str, str], built: list[dict[str, Any]]
) -> None:
    """Criterion 3's per-pod row, by VALUE rather than by presence, because the subtotals and
    the envelope cases both pass for any set of numbers that happens to add up. Catches a row
    written as `256M` where decision A's table says `256Mi` - a 7% difference the API server
    does not treat as equal, and the direction that gets a Next.js server OOM-killed - and
    catches one workload given its neighbour's quantities, which four near-identical Deployments
    in one file invite. Catches `limits.cpu` appearing on any of the four: equality on the whole
    flattened mapping is what keeps it absent, and a CPU limit throttles the agent-api's tail
    latency rather than protecting the node."""
    quantities = resource_quantities(sole_container(helpdesk_pod_owner(built, workload_name)))
    assert quantities == expected, f"{workload_name} declares {quantities}"


def test_the_four_helpdesk_pods_fit_the_envelope_the_reserve_held(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 4, as an EQUALITY between the four BUILT templates and `HELPDESK_RESERVE`
    multiplied out - the envelope the apps budget held for three pods before this row divided
    it across four. The totals themselves are never written here: both sides
    are derived. Catches decision A being abandoned quietly - a workload raised from 128Mi to
    192Mi because it looked tight, which still fits the quota today and takes the headroom row
    27's console has to land in - and catches it in the other direction too, a row shaved so the
    README table and the manifests no longer describe the same cluster. An HPA over any of the
    four would make this sum a floor rather than the cost, so the absence of one is asserted
    here: the quota admits pods at their ceiling, not at `spec.replicas`."""
    scaled = [
        key for key in scale_targets(built) if key[1] in HELPDESK_WORKLOADS and key[2] == APPS
    ]
    assert scaled == [], f"an HPA scales {scaled}, so these four cost more than one pod each"

    totals = {"requests.memory": 0, "requests.cpu": 0, "limits.memory": 0}
    for workload_name in HELPDESK_WORKLOADS:
        owner = helpdesk_pod_owner(built, workload_name)
        pods = replicas_of(owner)
        for container in all_containers(owner):
            quantities = resource_quantities(container)
            totals["requests.memory"] += pods * memory_bytes(quantities["requests.memory"])
            totals["requests.cpu"] += pods * cpu_millicores(quantities["requests.cpu"])
            totals["limits.memory"] += pods * memory_bytes(quantities["limits.memory"])

    envelope = envelope_totals(HELPDESK_RESERVE)
    assert totals == envelope, (
        f"the four Helpdesk pods cost {totals} against the envelope {envelope} the apps budget "
        "still prices them at"
    )


def test_each_serving_helpdesk_workload_probes_readyz_and_healthz(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 6, on the three workloads that serve. Catches the two paths SWAPPED, which is
    the failure `gateway.yaml` records beside the same pair and the one a reader skims past:
    `/readyz` as the liveness probe restarts the pod every time Redis blips, and `/healthz` as
    the readiness probe sends traffic to a process with no database. Catches a readiness probe
    with no liveness probe (a wedged process nothing restarts) and the reverse. Catches a probe
    port copied from a neighbour - 8080 on a container that listens on 3000, or the name `http`
    on a template that never declares it - which is a probe that never succeeds, so the pod
    either never becomes ready or is restarted forever; both forms are resolved against the
    workload's OWN declared ports, an int against `containerPort` and a string against the
    port's name."""
    for workload_name in HELPDESK_SERVING:
        container = sole_container(helpdesk_pod_owner(built, workload_name))
        expected_port = HELPDESK_CONTAINER_PORTS[workload_name]
        declared = container.get("ports", [])
        assert isinstance(declared, list) and declared, f"{workload_name} declares no ports"
        numbers = {int(port["containerPort"]) for port in declared if "containerPort" in port}
        names = {str(port["name"]) for port in declared if port.get("name")}
        assert numbers == {expected_port}, f"{workload_name} listens on {sorted(numbers)}"
        assert CONTAINER_PORT_NAME in names, f"{workload_name}'s ports are named {sorted(names)}"

        for probe_key, path in (
            ("readinessProbe", READINESS_PATH),
            ("livenessProbe", LIVENESS_PATH),
        ):
            probe = container.get(probe_key)
            assert isinstance(probe, dict), f"{workload_name} declares no {probe_key}"
            request = probe.get("httpGet", {})
            assert isinstance(request, dict), f"{workload_name}'s {probe_key} is {probe}"
            assert str(request.get("path")) == path, (
                f"{workload_name}'s {probe_key} polls {request.get('path')}"
            )
            resolved = resolved_container_port(request.get("port", ""), container)
            assert resolved == expected_port, (
                f"{workload_name}'s {probe_key} polls port {request.get('port')}, which resolves "
                f"to {resolved} against its own ports {declared}"
            )


def test_the_scheduler_runs_one_replica_recreate_with_no_ports_and_no_probes(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 7 and decision F, all four halves asserted positively so the omissions read as
    a decision rather than as an oversight. Catches `replicas: 2` or a `RollingUpdate` strategy,
    which are the two ways two schedulers run at once - `RollingUpdate` starts the new pod
    before the old one stops, and the Redis leader lock the design names is a safety net for a
    brief overlap, not a licence to schedule one on every deploy. Catches a probe copied in
    from one of the three serving workloads: an HTTP probe against a process that binds no port
    fails every time, so a liveness probe here is a CrashLoopBackOff on a healthy scheduler and
    a readiness probe is a pod that never reports ready. Catches a `ports` block copied with
    it, which is what a Service in front of it would then look justified by."""
    owner = helpdesk_pod_owner(built, HELPDESK_SCHEDULER)
    assert replicas_of(owner) == HELPDESK_SCHEDULER_REPLICAS, (
        f"{HELPDESK_SCHEDULER} runs {replicas_of(owner)} replicas"
    )
    strategy = owner["spec"].get("strategy", {})
    assert isinstance(strategy, dict), f"{HELPDESK_SCHEDULER} declares strategy {strategy}"
    assert str(strategy.get("type")) == HELPDESK_SCHEDULER_STRATEGY, (
        f"{HELPDESK_SCHEDULER} rolls out with {strategy.get('type')}"
    )

    container = sole_container(owner)
    assert container.get("ports", []) == [], f"{HELPDESK_SCHEDULER} declares {container['ports']}"
    probes = [key for key in PROBE_KEYS if key in container]
    assert probes == [], f"{HELPDESK_SCHEDULER} declares {probes} against a process with no port"


def test_the_helpdesk_config_carries_only_fully_qualified_in_cluster_addresses(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 8. The whole `data` mapping by equality, so a key added without agreement is as
    visible as one missing, and every URL value's HOST checked against `.svc.cluster.local` -
    catches a bare `redis:6379` or a `localhost` carried over from a compose file, which these
    pods cannot resolve at all: they run in `apps` and two of the three addresses live in
    `platform`, so the failure is a connection error at startup rather than anything a build
    reports. Catches the gateway address written without `/v1`, which is an OpenAI-compatible
    client 404ing on every call. Catches the Redis database index left at the gateway's `0`,
    which is one Redis shared by two repos with no partition: a `FLUSHDB` from either takes out
    the other's state, and nothing about that looks wrong until it happens. The gateway's index
    is read off ITS built ConfigMap rather than written down here, so the claim is that the two
    differ rather than that either is a number this file remembers."""
    raw = sole_object(built, "ConfigMap", HELPDESK_CONFIG_MAP).get("data", {})
    assert isinstance(raw, dict), f"{HELPDESK_CONFIG_MAP} carries no data mapping"
    data = {str(key): str(value) for key, value in raw.items()}
    assert data == HELPDESK_CONFIG, f"{HELPDESK_CONFIG_MAP} carries {data}"

    external = {
        key: value
        for key, value in data.items()
        if "://" in value and not (urlsplit(value).hostname or "").endswith(IN_CLUSTER_SUFFIX)
    }
    assert external == {}, (
        f"{HELPDESK_CONFIG_MAP} names addresses no pod in {APPS} resolves: {external}"
    )

    assert urlsplit(data[HELPDESK_REDIS_URL_KEY]).path == HELPDESK_REDIS_DATABASE_PATH, (
        f"the Helpdesk reads Redis database {urlsplit(data[HELPDESK_REDIS_URL_KEY]).path}"
    )
    gateway_raw = sole_object(built, "ConfigMap", GATEWAY_CONFIG_MAP).get("data", {})
    gateway_redis = str(gateway_raw[GATEWAY_REDIS_URL_KEY])
    assert urlsplit(gateway_redis).path != HELPDESK_REDIS_DATABASE_PATH, (
        f"the gateway reads the same Redis database as the Helpdesk ({gateway_redis})"
    )


def test_each_helpdesk_workload_carries_its_own_trace_service_name(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 9. Catches the convenient version of this variable: one
    `HELPDESK_OTEL_SERVICE_NAME` in the shared ConfigMap, which gives all four pods one name and
    collapses them into a single service in the trace viewer - every span correctly recorded,
    the trace correctly propagated, and no way to tell the scheduler's work from the web's. That
    is the failure a tracing test cannot see, because the spans are all there. Catches two of
    the four given the same value by a copy, and catches any of them given the gateway's
    `opsconsole-gateway`, which #44 and #67 need distinguishable because one trace carries spans
    from both repos. The gateway's name is read off its built ConfigMap, so the five-way
    distinctness is a claim about the build and not about this file's memory."""
    raw = sole_object(built, "ConfigMap", HELPDESK_CONFIG_MAP).get("data", {})
    assert isinstance(raw, dict), f"{HELPDESK_CONFIG_MAP} carries no data mapping"
    assert HELPDESK_TRACE_NAME_KEY not in raw, (
        f"{HELPDESK_CONFIG_MAP} sets {HELPDESK_TRACE_NAME_KEY}, so all four pods share one name"
    )

    names: dict[str, str] = {}
    for workload_name in HELPDESK_WORKLOADS:
        container = sole_container(helpdesk_pod_owner(built, workload_name))
        environment = {
            str(entry["name"]): str(entry.get("value", ""))
            for entry in container.get("env", [])
            if isinstance(entry, dict) and "name" in entry
        }
        assert HELPDESK_TRACE_NAME_KEY in environment, (
            f"{workload_name} sets {sorted(environment)} and not {HELPDESK_TRACE_NAME_KEY}"
        )
        assert environment[HELPDESK_TRACE_NAME_KEY] == HELPDESK_TRACE_NAMES[workload_name], (
            f"{workload_name} traces as {environment[HELPDESK_TRACE_NAME_KEY]}"
        )
        names[workload_name] = environment[HELPDESK_TRACE_NAME_KEY]

    gateway_raw = sole_object(built, "ConfigMap", GATEWAY_CONFIG_MAP).get("data", {})
    gateway_name = str(gateway_raw[GATEWAY_TRACE_NAME_KEY])
    distinct = set(names.values()) | {gateway_name}
    assert len(distinct) == len(names) + 1, (
        f"the build traces {sorted(names.values())} and the gateway as {gateway_name}, which is "
        f"{len(distinct)} distinct services rather than {len(names) + 1}"
    )


def test_every_helpdesk_workload_reads_the_secret_this_repo_does_not_supply(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 10 and decision E's two-whole-objects shape, on all four Deployments. Catches
    `secretRef` dropped from one of them while the other three keep it - the scheduler is the
    likely one, since it looks like the workload with nothing to call - which is a process that
    starts, finds no gateway key and fails on its first request rather than at admission.
    Catches the configMapRef dropped the same way, and catches a third source nobody agreed to.
    The second half catches the convenient repair of an apply that fails on a missing Secret: a
    placeholder `Secret/helpdesk-secrets` committed here, which is a project rule broken in git
    forever and an apply that never mentions the omission again."""
    for workload_name in HELPDESK_WORKLOADS:
        container = sole_container(helpdesk_pod_owner(built, workload_name))
        sources = {
            key: str(entry[key].get("name", ""))
            for entry in container.get("envFrom", [])
            for key in entry
            if isinstance(entry.get(key), dict)
        }
        assert sources == HELPDESK_ENV_FROM_SOURCES, (
            f"{workload_name} takes its environment from {sources}"
        )

    supplied = [name_of(document) for document in objects_of_kind(built, "Secret")]
    assert HELPDESK_SECRET not in supplied, (
        f"the build emits {HELPDESK_SECRET}, which is credentials in git rather than an apply "
        "that fails loudly until they are supplied out of band"
    )


def test_no_image_in_the_build_carries_a_floating_tag(built: list[dict[str, Any]]) -> None:
    """Criterion 11 and decision D, build-wide over every container of every pod template in
    every namespace, `initContainers` included. Catches a reference with no tag at all, which
    Kubernetes resolves as `latest`; catches `latest` or `main` written out - a cluster whose
    running code no commit names, and a rollout that changes meaning with no diff to show for
    it. llama.cpp's pinned `server-b10902` is not a floating tag and is unaffected: the refusal
    is the two spellings in `FLOATING_TAGS`, not any tag with a word in it. The second half
    catches the Helpdesk half of an `images:` block that silently did nothing: kustomize accepts
    an entry whose `name:` matches no container, so a typo leaves the bare `helpdesk-web` in the
    stream, and it catches two workloads pointed at one repository, which is a Deployment
    running another service's code."""
    owners = pod_owners(built)
    assert owners, "the build emits no pod templates at all"
    floating: list[str] = []
    for owner in owners:
        label = f"{owner.get('kind')}/{name_of(owner)}"
        for container in all_containers(owner):
            where = f"{label}:{container.get('name')}"
            _, tag = container_image_parts(container, where)
            if tag in FLOATING_TAGS:
                floating.append(f"{where} runs `{tag}`")
    assert floating == [], f"floating tags in the build: {floating}"

    for workload_name in HELPDESK_WORKLOADS:
        container = sole_container(helpdesk_pod_owner(built, workload_name))
        repository, tag = container_image_parts(container, workload_name)
        assert repository == f"{HELPDESK_IMAGE_REPOSITORY_PREFIX}/{workload_name}", (
            f"{workload_name} runs from `{repository}`"
        )
        assert tag == HELPDESK_IMAGE_TAG, f"{workload_name} is tagged `{tag}`"


def test_no_helpdesk_selector_carries_a_kustomize_wide_label(built: list[dict[str, Any]]) -> None:
    """Criterion 12. The root kustomization's `labels:` block carries `includeSelectors: false`
    precisely so `app.kubernetes.io/part-of` and `app.kubernetes.io/managed-by` reach
    `metadata.labels` and nothing else; a workload that writes either into its OWN
    `spec.selector.matchLabels` undoes that per object, and a label in a selector is immutable
    after the first apply - changing it later means deleting the Deployment. Catches that, by
    equality on the whole selector rather than by the absence of two names, which also catches
    the other half of §2's kustomize mechanic: `app.kubernetes.io/name` omitted from a pod
    template or written with a neighbour's value, which is a Deployment rolling out zero
    matching pods while reporting nothing wrong, and a Service selecting another Helpdesk
    workload's pods entirely."""
    for workload_name in HELPDESK_WORKLOADS:
        owner = helpdesk_pod_owner(built, workload_name)
        declared = owner["spec"]["selector"]["matchLabels"]
        assert isinstance(declared, dict) and declared, f"{workload_name} selects nothing"
        match_labels = {str(key): str(value) for key, value in declared.items()}
        reached = sorted(KUSTOMIZE_WIDE_LABELS & set(match_labels))
        assert reached == [], f"{workload_name}'s selector carries {reached}, immutable on apply"
        assert match_labels == {SELECTOR_LABEL: workload_name}, (
            f"{workload_name} selects {match_labels}"
        )
        labels = template_labels(owner)
        assert labels.get(SELECTOR_LABEL) == workload_name, (
            f"{workload_name}'s pod template is labelled {labels}"
        )

    for workload_name in HELPDESK_SERVING:
        declared = sole_object(built, "Service", workload_name)["spec"].get("selector", {})
        assert isinstance(declared, dict) and declared, f"{workload_name}'s Service selects nothing"
        selector = {str(key): str(value) for key, value in declared.items()}
        reached = sorted(KUSTOMIZE_WIDE_LABELS & set(selector))
        assert reached == [], f"{workload_name}'s Service selector carries {reached}"
        assert selector == {SELECTOR_LABEL: workload_name}, (
            f"{workload_name}'s Service selects {selector}"
        )


def test_no_helpdesk_workload_has_a_disruption_budget(built: list[dict[str, Any]]) -> None:
    """Criterion 13, as absence, with the four templates read first so the case cannot pass
    vacuously over a namespace that holds no Helpdesk yet. Catches a PDB copied from the
    gateway's onto a single-replica workload: `minAvailable: 1` against one replica blocks a
    node drain forever, and on this one-node cluster that is every drain there is - an upgrade
    that hangs with nothing reporting an error. The gateway's own PDB is legitimate because it
    runs two replicas, so the check is not "no PDB in `apps`" but "no PDB SELECTS a Helpdesk
    pod": a budget named something else whose `matchLabels` match one of these templates is the
    same outage, and a name-only check would walk past it."""
    owners = {name: helpdesk_pod_owner(built, name) for name in HELPDESK_WORKLOADS}
    budgets = in_apps(built, "PodDisruptionBudget")

    named = sorted(set(budgets) & set(HELPDESK_WORKLOADS))
    assert named == [], f"the build emits PodDisruptionBudgets for {named}"

    selecting: list[str] = []
    for budget_name, budget in sorted(budgets.items()):
        declared = budget["spec"]["selector"]["matchLabels"]
        assert isinstance(declared, dict) and declared, f"{budget_name} selects nothing"
        match_labels = {str(key): str(value) for key, value in declared.items()}
        for workload_name, owner in owners.items():
            labels = template_labels(owner)
            if all(labels.get(key) == value for key, value in match_labels.items()):
                selecting.append(f"{budget_name} selects {workload_name}")
    assert selecting == [], f"disruption budgets over single-replica Helpdesk pods: {selecting}"


def test_the_gateway_router_serves_only_the_versioned_api(built: list[dict[str, Any]]) -> None:
    """Row 21b criterion 1: the `gateway` Ingress's path set is exactly `{"/v1"}`, `Prefix`, to
    `Service/gateway` on port name `http`, with `gateway-healthz` left on `/healthz`, `Exact`.
    Catches the router this plan narrows left at `/`, which is the whole point of the row: with
    `/` served, `/metrics` and every `/admin/*` route the pod mounts are published on the public
    host behind nothing but basic auth, and a NetworkPolicy cannot see a path to stop it.
    Catches `/metrics` or an `/admin` path ADDED back beside `/v1` as a second entry, which set
    equality sees and a `startswith` or a "some path is /v1" check would not. Catches `/v1`
    given `pathType: Exact`, where `/v1/chat/completions` - every request the gateway exists to
    serve - stops matching and 404s while the manifest still names the right prefix. And catches
    the narrowed router repointed at another Service or given a numeric backend port that drifts
    from the `http` the Service targets, which resolves to no endpoint. The `/healthz` half is
    asserted alongside so a sweep that narrows both routers, taking the unauthenticated probe
    off the internet, is red here rather than discovered by an uptime monitor."""
    ingress = sole_object(built, "Ingress", ROOT_INGRESS)
    paths: set[str] = set()
    for rule in ingress["spec"]["rules"]:
        for path in rule["http"]["paths"]:
            paths.add(str(path["path"]))
            assert str(path.get("pathType")) == PATH_TYPE_PREFIX, (
                f"{ROOT_INGRESS} serves {path['path']} as {path.get('pathType')}"
            )
            backend = path["backend"]["service"]
            assert str(backend["name"]) == GATEWAY, (
                f"{ROOT_INGRESS} routes {path['path']} to Service/{backend['name']}"
            )
            port = backend.get("port", {})
            assert isinstance(port, dict) and str(port.get("name")) == CONTAINER_PORT_NAME, (
                f"{ROOT_INGRESS} names backend port {port} and not `{CONTAINER_PORT_NAME}`"
            )
    assert paths == {VERSIONED_PREFIX}, f"{ROOT_INGRESS} routes {sorted(paths)}"

    healthz = sole_object(built, "Ingress", HEALTHZ_INGRESS)
    healthz_paths = {
        (str(path["path"]), str(path.get("pathType")))
        for rule in healthz["spec"]["rules"]
        for path in rule["http"]["paths"]
    }
    assert healthz_paths == {(LIVENESS_PATH, PATH_TYPE_EXACT)}, (
        f"{HEALTHZ_INGRESS} routes {sorted(healthz_paths)}"
    )


def test_no_public_router_matches_an_admin_path_or_the_scrape(built: list[dict[str, Any]]) -> None:
    """Row 21b criterion 2: none of the four closed paths - `/admin/routing`,
    `/admin/routing/reload`, `/admin/directory`, `/metrics` - is matched by any router in the
    build that forwards to `Service/gateway`, under Traefik's own matching rule. The four are
    IMPORTED from the route modules that mount them, so a route moved to a new path moves this
    case with it instead of leaving it green against a string nothing serves any more. Catches
    the `/` router this plan removes, which matches all four and is what makes this case red
    today; catches a later router added back for one of them, an `ipAllowList`-guarded
    `/admin` Ingress being exactly the shape decision A rejected because k3s's servicelb can
    hide the real client IP; and catches a narrowing written as `/` plus a deny middleware,
    which leaves the path matched and the closure a middleware's job again. `MATCHER_BOUNDARY_ROWS`
    pins both sides of the one edge in the matcher first, so a matcher that answered False for
    everything - which would make this case pass over any build at all - is red before the
    build is read."""
    for route_path, path_type, request_path, expected in MATCHER_BOUNDARY_ROWS:
        assert router_matches(route_path, path_type, request_path) is expected, (
            f"`{route_path}` ({path_type}) against `{request_path}` is not {expected}"
        )

    for closed in CLOSED_PUBLIC_PATHS:
        matched = routers_matching(built, closed)
        assert matched == [], f"the build publishes {closed} through {matched}"


def test_every_served_versioned_path_is_still_publicly_routable(
    built: list[dict[str, Any]],
) -> None:
    """Row 21b criterion 3, the other half of criterion 2 and the reason the narrowing is not
    just "route less": every path the app mounts under `/v1`, plus `/healthz`, is matched by
    EXACTLY one gateway-backed router. The served set is read off the app's own router
    (`mounted_paths`), so a `/v2` or a fourth `/v1` route added later fails here rather than
    404ing in production against a manifest nobody thought to widen. Catches a narrowing that
    overshoots - `/v1/chat` in place of `/v1`, or `Exact` where `Prefix` was meant - which
    silently drops served paths; catches `/healthz` narrowed away with the rest, which is the
    one public path with no credential and a real consumer. And catches an OVERLAP rather than
    a gap: two routers matching one path is the state the build is in today, where `/healthz`
    is served both by `gateway`'s `/` and by `gateway-healthz`'s `Exact` rule and Traefik picks
    between them by rule length - a tie decided by a mechanic no manifest states, and on that
    path the difference between the two routers is basic auth."""
    served = sorted(
        {path for path in mounted_paths() if path.startswith(f"{VERSIONED_PREFIX}/")}
        | {VERSIONED_PREFIX, LIVENESS_PATH}
    )
    assert len(served) > 2, f"the app mounts nothing under {VERSIONED_PREFIX}: {served}"

    for path in served:
        matched = routers_matching(built, path)
        assert len(matched) == 1, f"{path} is matched by {len(matched)} routers: {matched}"


def test_readyz_is_reachable_only_in_cluster(built: list[dict[str, Any]]) -> None:
    """Row 21b criterion 4 and decision B: `/readyz` is matched by no gateway-backed router.
    It reports WHICH dependencies are reachable, which is the shape of an internal disclosure,
    and it has no public consumer - the kubelet dials the pod and an in-cluster caller uses the
    Service. Catches the `/` router that matches it today; catches `/readyz` added to the
    `gateway-healthz` router alongside `/healthz`, which is the tempting edit because the two
    probes look alike and is the one decision B rules out; and catches `gateway-healthz`
    widened from `Exact` to `Prefix`, where `/readyz` stays unmatched but the assertion on
    `/healthz` below fixes the matcher's `Exact` behaviour so the widening cannot be mistaken
    for a pass. The same `router_matches` as criterion 2 decides it, so the two cases cannot
    disagree about what "matched" means; and `/healthz` is asserted matched in the same breath,
    so a build that routes nothing at all - which would satisfy this case trivially - is red."""
    assert routers_matching(built, READINESS_PATH) == [], (
        f"the build publishes {READINESS_PATH} through {routers_matching(built, READINESS_PATH)}"
    )
    assert routers_matching(built, LIVENESS_PATH) != [], (
        f"the build routes {READINESS_PATH} nowhere and {LIVENESS_PATH} nowhere either"
    )


def test_every_admin_path_the_app_mounts_is_under_the_prefix_the_build_closes(
    built: list[dict[str, Any]],
) -> None:
    """Row 21b criterion 5: every path the app mounts that begins `/admin` is inside `/admin/`
    AND is matched by no gateway-backed router - which is what makes criterion 2's four-path
    list a rule rather than a snapshot of today's three routes. Derived from the app, so a
    fourth admin route added later is covered without anyone remembering to add it to
    `CLOSED_PUBLIC_PATHS`. Catches an admin route mounted just outside the prefix - `/admin`
    bare, or `/admin-keys`, both of which start `/admin`, neither of which any of the four
    constants names - and catches an admin route that IS inside the prefix but reachable,
    which is the state the `/` router leaves every one of them in today. The three known admin
    paths are asserted present first: `mounted_paths` walks a nested router structure, and an
    empty derivation would otherwise let both halves of this case pass over nothing."""
    admin_paths = sorted(path for path in mounted_paths() if path.startswith("/admin"))
    missing = sorted(
        path
        for path in CLOSED_PUBLIC_PATHS
        if path.startswith("/admin") and path not in admin_paths
    )
    assert missing == [], f"the app mounts {admin_paths}, which does not include {missing}"

    outside = [path for path in admin_paths if not path.startswith(ADMIN_PREFIX)]
    assert outside == [], f"admin routes outside `{ADMIN_PREFIX}`: {outside}"

    published = {path: routers_matching(built, path) for path in admin_paths}
    assert all(matched == [] for matched in published.values()), (
        f"the build publishes admin routes: { ({k: v for k, v in published.items() if v}) }"
    )
