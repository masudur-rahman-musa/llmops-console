r"""An earlier change: the console in the cluster, read off what kustomize actually emits, plus the three
files that build its image (`web/Dockerfile`, `.dockerignore`, `next.config.ts`).

The stream comes from the session-scoped `built` fixture in `infra/tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, shared with every other
`infra` module. Nothing here builds again.

The console's names, the parsing helpers and the Traefik matcher are IMPORTED from
`test_k8s_apps.py` and the NetworkPolicy resolvers from `test_k8s_netpol.py`, spelled as the
package `__init__` files make them (`infra.test_k8s_apps`, a project rule). One copy of
`router_matches` means this file and the gateway's cases cannot disagree about what a router
matches. No `test_*` name is imported, so pytest collects nothing twice.

The three image files are read as TEXT and parsed only as far as each criterion needs, so a
Dockerfile or ignore file written in another legal shape is not red here.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from infra.test_k8s_apps import (
    APPS,
    CLUSTER_ISSUER,
    CLUSTER_ISSUER_ANNOTATION,
    CONSOLE,
    CONSOLE_CONFIG_MAP,
    CONSOLE_HEALTHZ_INGRESS,
    CONSOLE_HOST,
    CONSOLE_INGRESS,
    CONSOLE_LOGIN_INGRESS,
    CONSOLE_LOGIN_RATE_LIMIT,
    CONSOLE_RESERVE,
    CONSOLE_SECRET,
    CONTAINER_PORT_NAME,
    GATEWAY,
    K8S_README,
    LIVENESS_PATH,
    PATH_TYPE_EXACT,
    PATH_TYPE_PREFIX,
    READINESS_PATH,
    SELECTOR_LABEL,
    in_apps,
    middleware_names,
    name_of,
    namespace_of,
    objects_of_kind,
    pod_owners,
    read,
    replicas_of,
    resource_quantities,
    router_matches,
    sole_container,
    sole_object,
    template_labels,
)
from infra.test_k8s_netpol import (
    ANY_POD,
    KUBE_SYSTEM,
    MONITORING,
    declared_ports,
    egress_destinations,
    ip_blocks,
    matched_workloads,
    peer_targets,
    peers_of,
    pod_selector,
    policies,
    policy,
    resolved_ports,
    rules_of,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
APPS_DIR = REPO_ROOT / "infra" / "k8s" / "apps"
APPS_KUSTOMIZATION = APPS_DIR / "kustomization.yaml"
CONSOLE_MANIFEST = APPS_DIR / "console.yaml"
CONSOLE_DIR = REPO_ROOT / "web"
CONSOLE_DOCKERFILE = CONSOLE_DIR / "Dockerfile"
CONSOLE_DOCKERIGNORE = CONSOLE_DIR / ".dockerignore"
NEXT_CONFIG = CONSOLE_DIR / "next.config.ts"
CONSOLE_SECRETS_EXAMPLE = REPO_ROOT / "infra" / "secrets" / "console-secrets.example.yaml"

#: The six objects `apps/console.yaml` emits, each carrying `app.kubernetes.io/name: console`.
CONSOLE_LABELLED: frozenset[tuple[str, str]] = frozenset(
    {
        ("ConfigMap", CONSOLE_CONFIG_MAP),
        ("Deployment", CONSOLE),
        ("Service", CONSOLE),
        ("Ingress", CONSOLE_INGRESS),
        ("Ingress", CONSOLE_HEALTHZ_INGRESS),
        ("Ingress", CONSOLE_LOGIN_INGRESS),
    }
)

#: Every object in the build NAMED for the console: the six above, the login router's own
#: middleware and the two NetworkPolicies. All of them in `apps`.
CONSOLE_NAMED: frozenset[tuple[str, str, str]] = frozenset(
    {(kind, name, APPS) for kind, name in CONSOLE_LABELLED}
    | {
        ("Middleware", CONSOLE_LOGIN_RATE_LIMIT, APPS),
        ("NetworkPolicy", "console-ingress", APPS),
        ("NetworkPolicy", "console-egress", APPS),
    }
)

#: The image transformer entry, and the bare name the Deployment writes.
CONSOLE_IMAGE_REPOSITORY = "ghcr.io/masudur-rahman-musa/ai-operations-console/console"
CONSOLE_IMAGE_TAG = "0.0.0-unbuilt"

#: The one port, by number and by name, on the container and on the Service.
CONSOLE_PORT = 3001

#: Decision F's per-pod row, DERIVED from the reserve the apps budget held.
CONSOLE_RESOURCES: dict[str, str] = {
    key: str(CONSOLE_RESERVE[key]) for key in ("requests.memory", "requests.cpu", "limits.memory")
}
CONSOLE_REPLICAS = int(str(CONSOLE_RESERVE["pods"]))

#: Decision E: the in-cluster gateway address, and the key that must NOT be set.
CONSOLE_API_URL_KEY = "CONSOLE_API_URL"
CONSOLE_API_URL = "http://gateway.apps.svc.cluster.local:8080"
TRADEOFF_PATH_KEY = "CONSOLE_TRADEOFF_PATH"

#: The Secret contract, by key. Also the names that may never sit in the ConfigMap.
CONSOLE_SECRET_KEYS = frozenset(
    {"CONSOLE_PASSWORD_SHA256", "CONSOLE_SESSION_SECRET", "CONSOLE_ADMIN_KEY"}
)
GATEWAY_ADMIN_DIGEST_KEY = "OPSCONSOLE_ADMIN_KEY_SHA256"
PLACEHOLDER = "REPLACE_ME"

#: The apps budget's total row, which the console moving from reserve to built does not change.
APPS_BUDGET_TOTAL = ["**total**", "**2880Mi**", "**1250m**", "**6784Mi**"]
APPS_BUDGET_HEADING = "## The apps budget"

#: Criterion: each router's path, `pathType`, TLS Secret and middleware set, by value.
CONSOLE_ROUTERS: dict[str, dict[str, Any]] = {
    CONSOLE_INGRESS: {
        "path": "/",
        "path_type": PATH_TYPE_PREFIX,
        "tls": "console-tls",
        "middlewares": frozenset({"basic-auth", "rate-limit", "security-headers"}),
    },
    CONSOLE_HEALTHZ_INGRESS: {
        "path": LIVENESS_PATH,
        "path_type": PATH_TYPE_EXACT,
        "tls": "console-healthz-tls",
        "middlewares": frozenset({"rate-limit", "security-headers"}),
    },
    CONSOLE_LOGIN_INGRESS: {
        "path": "/api/login",
        "path_type": PATH_TYPE_EXACT,
        "tls": "console-login-tls",
        "middlewares": frozenset({"basic-auth", CONSOLE_LOGIN_RATE_LIMIT, "security-headers"}),
    },
}

#: Decision D, exactly.
LOGIN_RATE_LIMIT_SPEC = {"rateLimit": {"average": 5, "period": "1m", "burst": 5}}

#: `(request path, matched by console-login)`: both sides of `Exact`.
LOGIN_MATCH_ROWS: tuple[tuple[str, bool], ...] = (
    ("/api/login", True),
    ("/api/login/x", False),
    ("/api/loginx", False),
    ("/api/logout", False),
    ("/", False),
)

#: The base image, pinned by digest (decision A).
BASE_IMAGE = re.compile(r"node:22-bookworm-slim@sha256:[0-9a-f]{64}")

#: The only paths the runtime stage may copy.
RUNTIME_COPY_SOURCES = (".next/standalone", ".next/static", "public")
REQUIRED_RUNTIME_COPY_SOURCES = frozenset({".next/standalone", ".next/static"})
RUNTIME_ENV = {"PORT": "3001", "HOSTNAME": "0.0.0.0", "NODE_ENV": "production"}

#: The ignore entries criterion 4 names, compared after normalising `/x`, `**/x` and `x/`.
DOCKERIGNORE_EXCLUDED = frozenset({"node_modules", ".next", "tests", ".env", ".env.*"})
DOCKERIGNORE_KEPT = "!.env.example"

NEXT_CONFIG_REQUIRED = (
    re.compile(r"""\boutput\s*:\s*["']standalone["']"""),
    re.compile(r"\breactStrictMode\s*:\s*true\b"),
    re.compile(r"\bpoweredByHeader\s*:\s*false\b"),
)


def console_owner(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """The console Deployment, asserted to own a pod template."""
    owner = sole_object(documents, "Deployment", CONSOLE)
    assert owner in pod_owners(documents), f"{CONSOLE} carries no pod template"
    return owner


def labels_of(document: dict[str, Any]) -> dict[str, str]:
    """`metadata.labels`, empty when the object declares none."""
    metadata = document.get("metadata", {})
    labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
    return {str(key): str(value) for key, value in labels.items()} if labels else {}


def raw_documents(path: Path) -> list[dict[str, Any]]:
    """Every mapping document in a YAML file, as WRITTEN rather than as built."""
    return [
        document
        for document in yaml.safe_load_all(read(path))
        if isinstance(document, dict) and document
    ]


def router_paths(ingress: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    """Every `(host, path, pathType, backend service, backend port name)` of an Ingress."""
    found: list[tuple[str, str, str, str, str]] = []
    for rule in ingress["spec"]["rules"]:
        for path in rule["http"]["paths"]:
            service = path["backend"]["service"]
            port = service.get("port", {})
            found.append(
                (
                    str(rule.get("host")),
                    str(path["path"]),
                    str(path.get("pathType")),
                    str(service["name"]),
                    str(port.get("name")) if isinstance(port, dict) else str(port),
                )
            )
    return found


def instructions(text: str) -> list[tuple[str, str]]:
    """A Dockerfile as `(KEYWORD, arguments)`, continuation lines joined, comments dropped."""
    logical: list[str] = []
    pending = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not pending and (stripped == "" or stripped.startswith("#")):
            continue
        if pending and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        logical.append(pending + stripped)
        pending = ""
    if pending.strip():
        logical.append(pending)
    parsed: list[tuple[str, str]] = []
    for entry in logical:
        keyword, _, arguments = entry.strip().partition(" ")
        parsed.append((keyword.upper(), arguments.strip()))
    return parsed


def stages(parsed: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """The instructions grouped by stage, each group starting with its `FROM`."""
    grouped: list[list[tuple[str, str]]] = []
    for keyword, arguments in parsed:
        if keyword == "FROM":
            grouped.append([])
        if grouped:
            grouped[-1].append((keyword, arguments))
    return grouped


def without_flags(arguments: str) -> list[str]:
    """An instruction's tokens with its `--flag` options removed."""
    return [token for token in shlex.split(arguments) if not token.startswith("--")]


def env_assignments(arguments: str) -> dict[str, str]:
    """One `ENV` instruction's assignments, in either the `K=V ...` or the `K V` form."""
    tokens = shlex.split(arguments)
    if tokens and "=" in tokens[0]:
        return dict(token.split("=", 1) for token in tokens if "=" in token)
    return {tokens[0]: " ".join(tokens[1:])} if tokens else {}


def dockerignore_entries(text: str) -> list[str]:
    """The ignore file's patterns in order, normalised so `/x`, `**/x` and `x/` read as `x`."""
    entries: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        negated = stripped.startswith("!")
        body = stripped[1:] if negated else stripped
        while body.startswith("/") or body.startswith("**/"):
            body = body[1:] if body.startswith("/") else body[3:]
        body = body.rstrip("/")
        entries.append(f"!{body}" if negated else body)
    return entries


def test_the_build_emits_exactly_the_console_objects_the_plan_names(
    built: list[dict[str, Any]],
) -> None:
    """Catches `console.yaml` written and left out of `apps/kustomization.yaml`'s `resources`,
    a console object filed outside `apps`, an extra console object nobody agreed to, and any of
    the six `console.yaml` objects missing `app.kubernetes.io/name: console`. It also catches
    that label left off the Deployment's selector, its pod template or the Service's selector,
    which is a rollout of zero matching pods or a Service with no endpoints and no error."""
    named = {
        (str(document.get("kind")), name_of(document), str(namespace_of(document)))
        for document in built
        if name_of(document) == CONSOLE or name_of(document).startswith(f"{CONSOLE}-")
    }
    assert named == CONSOLE_NAMED, (
        f"unexpected: {sorted(named - CONSOLE_NAMED)}; missing: {sorted(CONSOLE_NAMED - named)}"
    )

    labelled = {
        (str(document.get("kind")), name_of(document))
        for document in built
        if labels_of(document).get(SELECTOR_LABEL) == CONSOLE
    }
    missing = sorted(CONSOLE_LABELLED - labelled)
    assert missing == [], f"console objects without {SELECTOR_LABEL}: {CONSOLE}: {missing}"
    stray = sorted(labelled - {(kind, name) for kind, name, _ in CONSOLE_NAMED})
    assert stray == [], f"objects labelled as the console that the plan does not name: {stray}"

    owner = console_owner(built)
    match_labels = owner["spec"]["selector"]["matchLabels"]
    assert isinstance(match_labels, dict) and match_labels.get(SELECTOR_LABEL) == CONSOLE, (
        f"{CONSOLE} selects {match_labels}"
    )
    labels = template_labels(owner)
    assert labels.get(SELECTOR_LABEL) == CONSOLE, f"the pod template is labelled {labels}"
    unmatched = {key: value for key, value in match_labels.items() if labels.get(key) != value}
    assert unmatched == {}, f"{CONSOLE}'s selector {unmatched} matches no pod: {labels}"

    selector = sole_object(built, "Service", CONSOLE)["spec"].get("selector", {})
    assert isinstance(selector, dict) and selector.get(SELECTOR_LABEL) == CONSOLE, (
        f"Service/{CONSOLE} selects {selector}"
    )
    stray_selector = {key: value for key, value in selector.items() if labels.get(key) != value}
    assert stray_selector == {}, f"Service/{CONSOLE} selector {stray_selector} matches no pod"


def test_the_console_image_is_the_ghcr_console_repository_at_the_unbuilt_tag(
    built: list[dict[str, Any]],
) -> None:
    """Catches an `images:` entry whose `name` matches no container (kustomize accepts it and
    leaves the bare `console` in the stream), an entry pointed at the gateway's repository, a
    floating or real tag committed to git, and a Deployment that writes the full GHCR
    reference itself: that bypasses the transformer, so `deploy.yml`'s `set image console=`
    changes nothing and every deploy runs the placeholder."""
    image = str(sole_container(console_owner(built)).get("image", ""))
    assert image == f"{CONSOLE_IMAGE_REPOSITORY}:{CONSOLE_IMAGE_TAG}", f"{CONSOLE} runs {image!r}"

    kustomization = yaml.safe_load(read(APPS_KUSTOMIZATION))
    assert isinstance(kustomization, dict)
    assert "console.yaml" in kustomization.get("resources", []), (
        f"apps/kustomization.yaml lists {kustomization.get('resources')}"
    )
    entries = [entry for entry in kustomization["images"] if entry.get("name") == CONSOLE]
    assert entries == [
        {
            "name": CONSOLE,
            "newName": CONSOLE_IMAGE_REPOSITORY,
            "newTag": CONSOLE_IMAGE_TAG,
        }
    ], f"the overlay's console image entries are {entries}"

    written = [
        document
        for document in raw_documents(CONSOLE_MANIFEST)
        if document.get("kind") == "Deployment"
    ]
    assert len(written) == 1, f"{CONSOLE_MANIFEST.name} writes {len(written)} Deployments"
    containers = written[0]["spec"]["template"]["spec"]["containers"]
    assert [str(container.get("image")) for container in containers] == [CONSOLE], (
        f"{CONSOLE_MANIFEST.name} writes images {[c.get('image') for c in containers]}"
    )


def test_the_console_declares_the_quantities_the_reserve_held(
    built: list[dict[str, Any]],
) -> None:
    """Catches the console given the gateway's `150m` or llama's quantities by a copied file, a
    limit written `512M` where the reserve says `512Mi`, and a `limits.cpu` added - equality on
    the whole flattened mapping sees the extra key. The README half catches the apps budget
    still showing a `console reserve` row beside the new built row (the console priced twice
    in prose), no built `console` row at all, a built row with other numbers, or the total row
    moved when the manifest was written to the reserve and nothing should change."""
    quantities = resource_quantities(sole_container(console_owner(built)))
    assert quantities == CONSOLE_RESOURCES, f"{CONSOLE} declares {quantities}"

    lines = read(K8S_README).splitlines()
    assert lines.count(APPS_BUDGET_HEADING) == 1, f"the README has no single {APPS_BUDGET_HEADING}"
    start = lines.index(APPS_BUDGET_HEADING) + 1
    section: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        section.append(line)
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section
        if line.strip().startswith("|")
    ]
    assert rows, f"the README's {APPS_BUDGET_HEADING} section carries no table"

    reserves = [row for row in rows if "reserve" in row[0].lower()]
    assert reserves == [], f"the apps budget still holds reserve rows: {reserves}"
    console_rows = [row for row in rows if row[0].startswith(f"`{CONSOLE}`")]
    assert len(console_rows) == 1, f"the apps budget has {len(console_rows)} built console rows"
    assert console_rows[0][1:4] == [
        CONSOLE_RESOURCES["requests.memory"],
        CONSOLE_RESOURCES["requests.cpu"],
        CONSOLE_RESOURCES["limits.memory"],
    ], f"the console row reads {console_rows[0]}"
    totals = [row for row in rows if row[0] == APPS_BUDGET_TOTAL[0]]
    assert totals == [APPS_BUDGET_TOTAL], f"the apps budget totals read {totals}"


def test_the_console_runs_one_replica_with_no_autoscaler_and_no_disruption_budget(
    built: list[dict[str, Any]],
) -> None:
    """Decision F. Catches `replicas` omitted or raised (a second pod is a second 256Mi the
    quota does not hold), an HPA scaling the console under any name, and a PDB copied from the
    gateway's: `minAvailable: 1` over one replica blocks every node drain on a one-node
    cluster. The PDB check is by SELECTOR, not by name, so a budget named otherwise that still
    selects the console pod is red too."""
    owner = console_owner(built)
    assert owner["spec"].get("replicas") == CONSOLE_REPLICAS, (
        f"{CONSOLE} declares replicas {owner['spec'].get('replicas')!r}"
    )
    assert replicas_of(owner) == CONSOLE_REPLICAS

    scalers = [
        name_of(hpa)
        for hpa in objects_of_kind(built, "HorizontalPodAutoscaler")
        if str(hpa["spec"]["scaleTargetRef"].get("name")) == CONSOLE
    ]
    assert scalers == [], f"HPAs scale the console: {scalers}"

    labels = template_labels(owner)
    selecting: list[str] = []
    for name, budget in sorted(in_apps(built, "PodDisruptionBudget").items()):
        declared = budget["spec"]["selector"].get("matchLabels", {})
        if all(labels.get(str(key)) == str(value) for key, value in declared.items()):
            selecting.append(name)
    assert selecting == [], f"disruption budgets select the console pod: {selecting}"


def test_the_console_probes_readyz_for_readiness_and_healthz_for_liveness(
    built: list[dict[str, Any]],
) -> None:
    """Catches the two probe paths swapped (a config-only `/readyz` as liveness restarts a pod
    that is merely unconfigured), either probe missing, a probe on a numeric port that drifts
    from the one the container declares, and the container or the Service on another port or
    another port name - `3000` from `next start`'s default, or an unnamed port the Ingress
    backends' `http` resolves to nothing."""
    container = sole_container(console_owner(built))
    for key, path in (
        ("readinessProbe", READINESS_PATH),
        ("livenessProbe", LIVENESS_PATH),
    ):
        probe = container.get(key, {})
        http_get = probe.get("httpGet", {}) if isinstance(probe, dict) else {}
        assert http_get.get("path") == path, f"{CONSOLE}'s {key} is {probe}"
        assert http_get.get("port") == CONTAINER_PORT_NAME, f"{CONSOLE}'s {key} is {probe}"

    ports = [
        (str(port.get("name")), int(port["containerPort"])) for port in container.get("ports", [])
    ]
    assert ports == [(CONTAINER_PORT_NAME, CONSOLE_PORT)], f"{CONSOLE} declares ports {ports}"

    service_ports = sole_object(built, "Service", CONSOLE)["spec"].get("ports", [])
    assert len(service_ports) == 1, f"Service/{CONSOLE} publishes {service_ports}"
    published = service_ports[0]
    assert published.get("name") == CONTAINER_PORT_NAME, f"Service/{CONSOLE} is {published}"
    assert published.get("port") == CONSOLE_PORT, f"Service/{CONSOLE} is {published}"
    assert published.get("targetPort", CONSOLE_PORT) in (
        CONTAINER_PORT_NAME,
        CONSOLE_PORT,
    ), f"Service/{CONSOLE} targets {published.get('targetPort')!r}"


def test_the_console_config_names_the_gateway_by_its_fully_qualified_address_and_no_tradeoff_path(
    built: list[dict[str, Any]],
) -> None:
    """Decision E. Catches `CONSOLE_API_URL` written as a bare `gateway:8080` or a public
    `api.` host (the page reads would leave the cluster through basic auth), a wrong port, and
    `CONSOLE_TRADEOFF_PATH` set to a file the image does not carry, in the ConfigMap or in a
    container `env` entry. Catches any of the three credentials put in the ConfigMap, which is
    a secret in git."""
    data = sole_object(built, "ConfigMap", CONSOLE_CONFIG_MAP).get("data", {})
    assert isinstance(data, dict), f"{CONSOLE_CONFIG_MAP} carries no data"
    assert data.get(CONSOLE_API_URL_KEY) == CONSOLE_API_URL, (
        f"{CONSOLE_CONFIG_MAP} sets {CONSOLE_API_URL_KEY}={data.get(CONSOLE_API_URL_KEY)!r}"
    )
    assert TRADEOFF_PATH_KEY not in data, f"{CONSOLE_CONFIG_MAP} sets {TRADEOFF_PATH_KEY}"
    leaked = sorted(CONSOLE_SECRET_KEYS & set(data))
    assert leaked == [], f"{CONSOLE_CONFIG_MAP} carries credentials {leaked}"

    env_names = [
        str(entry.get("name")) for entry in sole_container(console_owner(built)).get("env", [])
    ]
    assert TRADEOFF_PATH_KEY not in env_names, f"{CONSOLE} sets {TRADEOFF_PATH_KEY} by env"


def test_the_console_reads_its_config_map_and_console_secrets_by_env_from(
    built: list[dict[str, Any]],
) -> None:
    """Catches `envFrom` losing either source (a console that never reaches `/readyz` 200), a
    `secretRef` naming `gateway-secrets` by copy, which hands the console every provider key,
    and a third source. The contract half catches `console-secrets.example.yaml` in the wrong
    namespace, with a key missing or extra, or with a real-looking value, and a README row that
    omits the pairing of `CONSOLE_ADMIN_KEY` with the gateway's `OPSCONSOLE_ADMIN_KEY_SHA256`,
    whose drift turns every page `unauthorized`."""
    container = sole_container(console_owner(built))
    sources = [
        (key, str(entry[key].get("name", "")))
        for entry in container.get("envFrom", [])
        for key in entry
        if isinstance(entry.get(key), dict)
    ]
    assert sorted(sources) == [
        ("configMapRef", CONSOLE_CONFIG_MAP),
        ("secretRef", CONSOLE_SECRET),
    ], f"{CONSOLE} takes its environment from {sources}"

    documents = raw_documents(CONSOLE_SECRETS_EXAMPLE)
    assert len(documents) == 1, f"{CONSOLE_SECRETS_EXAMPLE.name} holds {len(documents)} documents"
    secret = documents[0]
    assert secret.get("kind") == "Secret" and name_of(secret) == CONSOLE_SECRET
    assert namespace_of(secret) == APPS, f"{CONSOLE_SECRET} is in {namespace_of(secret)}"
    values = {**secret.get("data", {}), **secret.get("stringData", {})}
    assert set(values) == CONSOLE_SECRET_KEYS, f"{CONSOLE_SECRET} carries {sorted(values)}"
    assert all(str(value) == PLACEHOLDER for value in values.values()), values

    rows = [
        line
        for line in read(K8S_README).splitlines()
        if line.strip().startswith(f"| `{CONSOLE_SECRET}` |")
    ]
    assert len(rows) == 1, f"the README carries {len(rows)} `{CONSOLE_SECRET}` rows"
    assert GATEWAY_ADMIN_DIGEST_KEY in rows[0], (
        f"the `{CONSOLE_SECRET}` row does not pair CONSOLE_ADMIN_KEY with "
        f"{GATEWAY_ADMIN_DIGEST_KEY}"
    )
    assert "envFrom" in rows[0], f"the `{CONSOLE_SECRET}` row does not say it is read by envFrom"


def test_the_three_console_routers_publish_one_host_with_their_paths_middlewares_and_tls(
    built: list[dict[str, Any]],
) -> None:
    """Catches basic auth missing from `console` or `console-login` (the console login form
    public to the internet), basic auth ON `console-healthz` (an uptime monitor reading a 401),
    the login limit attached to the `/` router (the whole console at five requests a minute)
    or missing from the login router, a staging issuer, a router on another host, a backend
    that is not `Service/console` port `http`, a TLS Secret shared or swapped between routers,
    and a middleware reference that resolves to no `apps` Middleware, which Traefik treats as
    no middleware. Any fourth router on the console host is red too."""
    emitted = {(APPS, name) for name in in_apps(built, "Middleware")}
    on_host = sorted(
        name_of(ingress)
        for ingress in objects_of_kind(built, "Ingress")
        if any(host == CONSOLE_HOST for host, *_ in router_paths(ingress))
    )
    assert on_host == sorted(CONSOLE_ROUTERS), f"routers on {CONSOLE_HOST}: {on_host}"

    for ingress_name, expected in CONSOLE_ROUTERS.items():
        ingress = sole_object(built, "Ingress", ingress_name)
        issuer = ingress["metadata"].get("annotations", {}).get(CLUSTER_ISSUER_ANNOTATION)
        assert issuer == CLUSTER_ISSUER, f"{ingress_name} is issued by {issuer!r}"

        assert router_paths(ingress) == [
            (
                CONSOLE_HOST,
                expected["path"],
                expected["path_type"],
                CONSOLE,
                CONTAINER_PORT_NAME,
            )
        ], f"{ingress_name} routes {router_paths(ingress)}"

        tls = ingress["spec"].get("tls", [])
        assert tls == [{"hosts": [CONSOLE_HOST], "secretName": expected["tls"]}], (
            f"{ingress_name} declares TLS {tls}"
        )

        parsed = middleware_names(ingress)
        assert parsed == {(APPS, name) for name in expected["middlewares"]}, (
            f"{ingress_name} names {sorted(parsed)}"
        )
        assert parsed <= emitted, f"{ingress_name} names unresolved {sorted(parsed - emitted)}"


def test_the_login_router_matches_exactly_api_login(
    built: list[dict[str, Any]],
) -> None:
    """Catches `console-login` written `Prefix`, which moves `/api/login/...` under the strict
    limit, or on `/api` or `/`, which puts every console request at five a minute; and catches
    a router whose path is not `/api/login` at all, where the login POST is served by the `/`
    router at 50 a second and decision D's limit protects nothing. Both sides of `Exact` are
    rows of `LOGIN_MATCH_ROWS`, run through the same matcher the gateway cases use."""
    routes = router_paths(sole_object(built, "Ingress", CONSOLE_LOGIN_INGRESS))
    assert len(routes) == 1, f"{CONSOLE_LOGIN_INGRESS} routes {routes}"
    _, path, path_type, _, _ = routes[0]
    for request_path, expected in LOGIN_MATCH_ROWS:
        assert router_matches(path, path_type, request_path) is expected, (
            f"{CONSOLE_LOGIN_INGRESS} ({path} {path_type}) against {request_path} is not {expected}"
        )


def test_the_login_rate_limit_is_five_per_minute_with_a_burst_of_five(
    built: list[dict[str, Any]],
) -> None:
    """Decision D by whole-`spec` equality. Catches `period: 1s` copied from `rate-limit` (five
    a SECOND is 300 a minute), a burst of 100, an `average` of 50, an extra field such as a
    `sourceCriterion` nobody agreed to, and the Middleware filed in `platform`, where an `apps`
    Ingress cannot resolve it."""
    middleware = sole_object(built, "Middleware", CONSOLE_LOGIN_RATE_LIMIT)
    assert middleware.get("spec") == LOGIN_RATE_LIMIT_SPEC, (
        f"{CONSOLE_LOGIN_RATE_LIMIT} is {middleware.get('spec')}"
    )


def test_no_public_router_matches_readyz(built: list[dict[str, Any]]) -> None:
    """Catches a `console-readyz` router added beside `console-healthz`, `/readyz` added to any
    healthz router, and `console-healthz` widened to `Prefix /`, each of which publishes the
    console's probe with no basic auth in front of it. A host's `Prefix /` catch-all cannot help
    matching `/readyz` (Grafana's and the Helpdesk's do, and serve no such probe), so the rule
    read here is the satisfiable one: no router anywhere names a path at or under `/readyz`,
    and on the console host the ONLY router matching it is `console`, which carries basic auth.
    A matcher answering False for everything is ruled out first."""
    assert router_matches("/", PATH_TYPE_PREFIX, READINESS_PATH) is True
    assert router_matches(READINESS_PATH, PATH_TYPE_EXACT, READINESS_PATH) is True

    named: list[str] = []
    on_console_host: list[str] = []
    for ingress in objects_of_kind(built, "Ingress"):
        for host, path, path_type, _, _ in router_paths(ingress):
            if not router_matches(path, path_type, READINESS_PATH):
                continue
            if path != "/":
                named.append(f"{name_of(ingress)} {host}{path} ({path_type})")
            if host == CONSOLE_HOST:
                on_console_host.append(name_of(ingress))
    assert named == [], f"routers publishing {READINESS_PATH} by path: {named}"
    assert on_console_host == [CONSOLE_INGRESS], (
        f"routers matching {READINESS_PATH} on {CONSOLE_HOST}: {on_console_host}"
    )
    console = sole_object(built, "Ingress", CONSOLE_INGRESS)
    assert (APPS, "basic-auth") in middleware_names(console), (
        f"{CONSOLE_INGRESS} serves {READINESS_PATH} without basic auth"
    )


def test_the_console_admits_only_traefik_and_reaches_only_the_gateway(
    built: list[dict[str, Any]],
) -> None:
    """Catches `console-ingress` admitting every pod in `apps` or `monitoring` (the console's
    session surface reachable around basic auth), a numeric port that drifts from the
    container's, `console-egress` reaching a provider by `ipBlock`, Postgres or llama directly,
    or the gateway on a port other than 8080, and either policy selecting pods other than the
    console. Equality both ways: the set of non-catch-all `apps` policies selecting the console
    pod is exactly these two, so a widened neighbour selecting it is red as well."""
    ingress = policy(built, APPS, "console-ingress")
    assert pod_selector(ingress) == {"matchLabels": {SELECTOR_LABEL: CONSOLE}}
    rules = rules_of(ingress, "ingress")
    assert rules, "console-ingress declares no ingress rule"
    peers: set[tuple[str, str]] = set()
    for rule in rules:
        listed = peers_of(rule, "from")
        assert listed, "console-ingress has a rule with no `from`, which admits everyone"
        for peer in listed:
            peers |= peer_targets(built, peer, APPS)
        assert declared_ports(rule, "apps/console-ingress") == [("TCP", CONTAINER_PORT_NAME)]
        assert resolved_ports(built, ingress, rule) == {CONSOLE: {("TCP", CONSOLE_PORT)}}
    assert peers == {(KUBE_SYSTEM, ANY_POD)}, f"console-ingress admits {sorted(peers)}"

    egress = policy(built, APPS, "console-egress")
    assert pod_selector(egress) == {"matchLabels": {SELECTOR_LABEL: CONSOLE}}
    assert egress_destinations(built, egress) == {(GATEWAY, 8080)}, (
        f"console-egress reaches {sorted(egress_destinations(built, egress))}"
    )
    assert ip_blocks(egress) == [], f"console-egress carries {ip_blocks(egress)}"

    selecting = sorted(
        name_of(document)
        for document in policies(built)
        if namespace_of(document) == APPS
        and pod_selector(document) != {}
        and CONSOLE in matched_workloads(built, pod_selector(document), APPS)
    )
    assert selecting == ["console-egress", "console-ingress"], (
        f"apps policies selecting the console pod: {selecting}"
    )


def test_the_gateway_ingress_still_admits_every_pod_in_apps(
    built: list[dict[str, Any]],
) -> None:
    """a project rule, the partner of `console-egress`. Catches `gateway-ingress` narrowed
    from its in-namespace `podSelector: {}` peer to the Helpdesk pods by name while the
    console's egress was being written - `default-deny` then refuses every console page read
    while `console-egress` looks correct. Its three peers and its one port are pinned by
    equality, and the console pod is asserted to exist, so the peer is not vacuously every pod
    of an `apps` holding no console."""
    document = policy(built, APPS, "gateway-ingress")
    peers: set[tuple[str, str]] = set()
    for rule in rules_of(document, "ingress"):
        for peer in peers_of(rule, "from"):
            peers |= peer_targets(built, peer, APPS)
        assert resolved_ports(built, document, rule) == {GATEWAY: {("TCP", 8080)}}
    assert peers == {(KUBE_SYSTEM, ANY_POD), (APPS, ANY_POD), (MONITORING, ANY_POD)}, (
        f"gateway-ingress admits {sorted(peers)}"
    )
    assert CONSOLE in matched_workloads(built, {}, APPS), f"{APPS} holds no console pod"


def test_the_console_dockerfile_runs_the_standalone_server_as_uid_1000_on_3001() -> None:
    """Decision A. Catches a floating `node:22` or `node:lts` base, a tag with no digest, a
    single-stage image carrying the whole build context, a runtime stage copying `node_modules`
    or the source tree, running as root, `HOSTNAME` left to the container hostname (the server
    binds one interface and the kubelet's probes are refused), `PORT` at Next's 3000, a
    development `NODE_ENV`, and `next start` in place of the standalone `node server.js`."""
    parsed = stages(instructions(read(CONSOLE_DOCKERFILE)))
    assert len(parsed) == 2, f"the console Dockerfile has {len(parsed)} stages"
    for stage in parsed:
        tokens = without_flags(stage[0][1])
        assert tokens and BASE_IMAGE.fullmatch(tokens[0]), f"a stage starts FROM {stage[0][1]!r}"

    runtime = parsed[-1]
    copies = [arguments for keyword, arguments in runtime if keyword in ("COPY", "ADD")]
    assert all(keyword != "ADD" for keyword, _ in runtime), "the runtime stage uses ADD"
    copied: set[str] = set()
    for arguments in copies:
        assert "--from=" in arguments, f"the runtime stage copies from the context: {arguments}"
        for source in without_flags(arguments)[:-1]:
            normalised = source.rstrip("/")
            matched = [
                allowed
                for allowed in RUNTIME_COPY_SOURCES
                if normalised == allowed or normalised.endswith(f"/{allowed}")
            ]
            assert matched, f"the runtime stage copies {source!r}"
            copied.update(matched)
    assert copied >= REQUIRED_RUNTIME_COPY_SOURCES, f"the runtime stage copies only {copied}"

    users = [arguments for keyword, arguments in runtime if keyword == "USER"]
    assert users and users[-1].split(":")[0] == "1000", f"the runtime runs as {users}"
    exposed = [
        token
        for keyword, arguments in runtime
        if keyword == "EXPOSE"
        for token in arguments.split()
    ]
    assert exposed and all(token.split("/")[0] == str(CONSOLE_PORT) for token in exposed), exposed

    environment: dict[str, str] = {}
    for keyword, arguments in runtime:
        if keyword == "ENV":
            environment.update(env_assignments(arguments))
    assert {key: environment.get(key) for key in RUNTIME_ENV} == RUNTIME_ENV, environment

    commands = [arguments for keyword, arguments in runtime if keyword == "CMD"]
    assert commands, "the runtime stage declares no CMD"
    last = commands[-1]
    tokens = json.loads(last) if last.startswith("[") else shlex.split(last)
    assert len(tokens) == 2 and tokens[0] == "node", f"the runtime runs {tokens}"
    assert Path(str(tokens[1])).name == "server.js", f"the runtime runs {tokens}"


def test_the_console_dockerignore_keeps_env_files_node_modules_and_tests_out() -> None:
    """Catches a real `.env` or `.env.local` with a password hash sent into the build context
    and baked into a layer, a host `node_modules` of the wrong architecture overwriting the
    image's, a stale `.next`, and `tests` shipped. The `!.env.example` re-include must come
    AFTER `.env.*`, because the last matching line wins and an earlier one is overridden."""
    entries = dockerignore_entries(read(CONSOLE_DOCKERIGNORE))
    missing = sorted(DOCKERIGNORE_EXCLUDED - set(entries))
    assert missing == [], f"the console .dockerignore does not exclude {missing}: {entries}"
    assert DOCKERIGNORE_KEPT in entries, f"the console .dockerignore drops .env.example: {entries}"
    last_exclude = max(index for index, entry in enumerate(entries) if entry == ".env.*")
    last_keep = max(index for index, entry in enumerate(entries) if entry == DOCKERIGNORE_KEPT)
    assert last_keep > last_exclude, f"`.env.*` overrides `{DOCKERIGNORE_KEPT}`: {entries}"


def test_next_config_builds_standalone() -> None:
    """Catches `output: "standalone"` missing, which leaves no `.next/standalone/server.js` for
    the image's CMD, and the edit dropping `reactStrictMode: true` or `poweredByHeader: false`,
    the second of which advertises the framework in every response header."""
    text = read(NEXT_CONFIG)
    missing = [pattern.pattern for pattern in NEXT_CONFIG_REQUIRED if not pattern.search(text)]
    assert missing == [], f"next.config.ts does not carry {missing}"
