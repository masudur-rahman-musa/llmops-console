r"""`.sops.yaml`, `infra/secrets/` and the three Makefile targets an earlier change agrees: the POLICY
and the GUARD, never the ciphertext. `sops` and `age` are not installed on this machine (hard
rule 12 makes installing them a question, not a step), so every case here asserts on file
structure, on `git ls-files`, and on the stream the session-scoped `built` fixture already
produces - and the encrypted-file case is written to be GREEN over an empty glob, because the
four `*.enc.yaml` land in a later `/quick` and the guard has to hold from today.

The build stream comes from the `built` fixture in `infra/tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, shared with
`test_k8s_base.py`, `test_k8s_apps.py`, `test_k8s_platform.py` and `test_k8s_netpol.py`.
Nothing here builds a second time. The two cases that need the stream are SPLIT from the two
that read files only, so the file-against-README comparison - the half that catches a drifting
template - still runs on a machine with no `kubectl`, where `built` skips.

Criterion 3 is the reason this file has a README parser at all. The contract
`{name: (namespaces, keys)}` is compared file-against-README, never file-against-a-literal:
a constant restated here would let the templates and the table a human reads before an apply
drift together in silence, which is exactly the failure the criterion names. Only the five
NAMES and their NAMESPACES are written down (the plans pin them, and the build's own
references are the third side of the comparison); the KEY SETS live in the two places the
criterion says they must agree, and nowhere else - `grafana-admin`'s included, which is why
an earlier change's own module does not restate them here either.

The five `tls[*].secretName` values (`gateway-tls`, `gateway-healthz-tls`, `helpdesk-tls`,
`helpdesk-healthz-tls`, `grafana-tls`) are EXCLUDED from the reference comparison and pinned
separately: cert-manager CREATES those, so "every Secret the build references is one this repo
supplies"
and "every Ingress names a tls secret" cannot both be read literally. The
exclusion is written as a subtraction of a pinned set rather than as a walk that never looks
at `tls`, so an exclusion gone stale is a readable red rather than an invisible pass.

`test_k8s_apps.py` already asserts build-wide that the stream emits no `Secret` and that every
referenced name is one of the `apps` LITERALS; the case here is the other direction - the names come
from the example FILES, so it catches a template renamed or never written, which a literal set
cannot see.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

#: This file's own location, never the working directory - `pytest` runs from anywhere. Both
#: sibling modules derive it this way rather than importing it.
REPO_ROOT = Path(__file__).resolve().parents[3]

K8S_ROOT = REPO_ROOT / "infra" / "k8s"
K8S_README = K8S_ROOT / "README.md"
SECRETS_DIR = REPO_ROOT / "infra" / "secrets"
SOPS_CONFIG = REPO_ROOT / ".sops.yaml"
GITIGNORE = REPO_ROOT / ".gitignore"

#: The eight Secrets, by name. The plans pin these; the key SETS are deliberately absent (see
#: the module docstring) because criterion 3 compares them file-against-README. `grafana-admin`
#: is an earlier change's criterion 8, the fifth contract this repo publishes and does not supply.
#: `alertmanager-secrets` is an earlier change's, the sixth. `backup-secrets` is an earlier change's, the seventh.
#: `evals-secrets` is an earlier change's, the eighth: the nightly-eval CronJob's `OPSCONSOLE_DATABASE_URL`.
POSTGRES_CREDENTIALS = "postgres-credentials"
TRAEFIK_BASIC_AUTH = "traefik-basic-auth"
GATEWAY_SECRETS = "gateway-secrets"
HELPDESK_SECRETS = "helpdesk-secrets"
GRAFANA_ADMIN = "grafana-admin"
ALERTMANAGER_SECRETS = "alertmanager-secrets"
BACKUP_SECRETS = "backup-secrets"
EVALS_SECRETS = "evals-secrets"
CONSOLE_SECRETS = "console-secrets"

#: An earlier change criterion 16: the pull secret the four Helpdesk pod specs name, the one contract here
#: a workload reads through `imagePullSecrets` rather than `envFrom`.
GHCR_PULL = "ghcr-pull"

PLATFORM = "platform"
APPS = "apps"
MONITORING = "monitoring"
EVALS = "evals"

#: Criterion 3's namespaces, which the plans pin. `traefik-basic-auth` exists TWICE: a
#: `Middleware` reads its Secret from its own namespace, so one copy per routing namespace.
EXPECTED_NAMESPACES: dict[str, frozenset[str]] = {
    POSTGRES_CREDENTIALS: frozenset({PLATFORM}),
    TRAEFIK_BASIC_AUTH: frozenset({PLATFORM, APPS}),
    GATEWAY_SECRETS: frozenset({APPS}),
    HELPDESK_SECRETS: frozenset({APPS}),
    GRAFANA_ADMIN: frozenset({MONITORING}),
    ALERTMANAGER_SECRETS: frozenset({MONITORING}),
    BACKUP_SECRETS: frozenset({PLATFORM}),
    EVALS_SECRETS: frozenset({EVALS}),
    CONSOLE_SECRETS: frozenset({APPS}),
    # An earlier change: the Helpdesk's registry credential.
    GHCR_PULL: frozenset({APPS}),
}

#: One template per Secret (decision B), `traefik-basic-auth`'s being one file of two
#: documents so the two namespace copies cannot drift.
EXAMPLE_FILENAMES: frozenset[str] = frozenset(
    {
        f"{POSTGRES_CREDENTIALS}.example.yaml",
        f"{TRAEFIK_BASIC_AUTH}.example.yaml",
        f"{GATEWAY_SECRETS}.example.yaml",
        f"{HELPDESK_SECRETS}.example.yaml",
        f"{GRAFANA_ADMIN}.example.yaml",
        f"{ALERTMANAGER_SECRETS}.example.yaml",
        f"{BACKUP_SECRETS}.example.yaml",
        f"{EVALS_SECRETS}.example.yaml",
        f"{CONSOLE_SECRETS}.example.yaml",
        f"{GHCR_PULL}.example.yaml",
    }
)

#: The two keys a Secret's values live under, and the only mapping either criterion 1 or
#: criterion 4 reads. `metadata.name`, `metadata.namespace`, `apiVersion`, `kind` and every
#: label are NOT values in this sense and are never scanned for the placeholder.
DATA_KEYS = ("data", "stringData")

SECRET_KIND = "Secret"

#: Traefik's own group. A `Middleware` is the second shape a Secret reference takes, and the
#: only one that is not an `envFrom`.
MIDDLEWARE_KIND = "Middleware"
TRAEFIK_API_VERSION = "traefik.io/v1alpha1"

#: Criterion 1: the exact prefix `sops` writes in front of every encrypted value. Pinned as a
#: literal prefix rather than as a loose "contains ENC" band: a value
#: encrypted with a different algorithm, or a half-written placeholder, is not this string.
ENCRYPTED_VALUE_PREFIX = "ENC[AES256_GCM,data:"

#: Criterion 2, exactly: the keys `.sops.yaml` encrypts, as the literal string the policy file
#: has to carry. Asserted by EQUALITY - a regex that merely mentions `data` could also match
#: `metadata`, which would encrypt a Secret's own name.
ENCRYPTED_REGEX = "^(data|stringData)$"

#: Criterion 2's two probe paths: the file the rule MUST cover and the one it must NOT. The
#: example templates live in the same directory and are plaintext by design, so a rule written
#: as `infra/secrets/.*\.yaml` would tell `sops` to encrypt them too.
ENCRYPTED_PROBE = "infra/secrets/gateway-secrets.enc.yaml"
EXAMPLE_PROBE = "infra/secrets/gateway-secrets.example.yaml"

#: Criterion 4, decision C: the literal placeholder, greppable and never read as a credential
#: by gitleaks. `<value>` would be neither.
PLACEHOLDER = "REPLACE_ME"

#: An earlier change criterion 10: the eight keys `helpdesk-secrets` carries, under the bare names the
#: companion repo reads. Pinned here as well as compared against the README, because a rename
#: made on both sides at once is the one edit the file-against-README equality cannot see.
HELPDESK_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "DATABASE_URL",
        "GATEWAY_API_KEYS",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "JWT_SECRET",
        "APP_DATABASE_URL",
        "APP_DB_PASSWORD",
    }
)

#: An earlier change criteria 16 and 17: the type the kubelet reads a registry credential from, and the
#: one key that type requires.
DOCKERCONFIGJSON_TYPE = "kubernetes.io/dockerconfigjson"
DOCKERCONFIGJSON_KEY = ".dockerconfigjson"

#: An earlier change criterion 17: the GHCR token scope the template's comment must name, since a token
#: without it cannot pull a private package.
READ_PACKAGES_SCOPE = "read:packages"

#: An earlier change criteria 11 to 13: what the `helpdesk-secrets` template's COMMENTS must state, as
#: `(what, pattern)`. Loose on wording on purpose: each pattern is the fact
#: the criterion names and nothing more.
HELPDESK_COMMENT_FACTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("the helpdesk_app role", re.compile(r"helpdesk_app")),
    ("the asyncpg DSN scheme", re.compile(re.escape("postgresql+asyncpg://"))),
    ("the tenant:key list shape", re.compile(re.escape("tenant:key,tenant:key"))),
    ("the gateway digest list", re.compile(r"OPSCONSOLE_API_KEYS")),
    ("the sha256 digest", re.compile(r"sha256")),
    ("the 32-byte floor", re.compile(r"\b32[ -]bytes?\b")),
)

#: The htpasswd shape `traefik-basic-auth`'s one key carries: `<user>:<password>`, one per
#: line. `fullmatch` and not `^...$`: `$` also matches before a trailing
#: newline, so a value ending in one would otherwise pass as a clean single line.
HTPASSWD_LINE = re.compile(r"(?P<user>[^:\s]+):(?P<password>\S+)")

#: Criterion 2: the three patterns that must be untracked AND ignored. `*.agekey` is already
#: in `.gitignore` today; `keys.txt` (age's own default output file) and `*.dec.yaml` (a
#: decrypted secret left in the tree) are what this plan adds.
IGNORED_PATTERNS = ("*.agekey", "keys.txt", "*.dec.yaml")

#: Criterion 5's targets, plus `secrets-check`. Only the two that SHELL OUT to `sops` are
#: parametrised for the unhappy path - `secrets-check` is the guard itself and is expected to
#: run with neither binary installed.
SOPS_TARGETS = ("secrets-encrypt", "secrets-decrypt")

#: A `PATH` with no `sops` anywhere on it, which on this machine is every `PATH`: `sops` is
#: not installed at all. Stripped down to the two directories `make`, `sh` and `command -v`
#: live in, so the case means the same thing on a machine where `sops` IS installed.
PATH_WITHOUT_SOPS = "/usr/bin:/bin"

#: cert-manager writes these five from the Ingresses' `spec.tls[*].secretName`; this repo
#: supplies none of them. They are subtracted from the referenced set in criterion 3's build
#: half, because "every referenced Secret is one of the five" and "every Ingress names a tls
#: secret" are both true only with this exclusion named. Pinned
#: rather than filtered by shape so a stale exclusion is a readable red. `grafana-tls` is plan
#: 23b's, and its disjointness from `grafana-admin` is what keeps an issued certificate from
#: overwriting the admin credential in one namespace.
INGRESS_TLS_SECRETS: frozenset[str] = frozenset(
    {
        "gateway-tls",
        "gateway-healthz-tls",
        "helpdesk-tls",
        "helpdesk-healthz-tls",
        "grafana-tls",
        # An earlier change's three console routers.
        "console-tls",
        "console-healthz-tls",
        "console-login-tls",
    }
)

#: The header the secrets table in `infra/k8s/README.md` carries, cell for cell. The table is
#: found by this row and not by a line number or by the first line mentioning "Secret".
README_HEADER_CELLS = ["Secret", "namespace", "keys", "read by"]

#: A backticked token in a README cell. The implementation rewrites the `keys` column from
#: prose to literal key names, so this is what turns a cell into a set.
BACKTICKED = re.compile(r"`([^`]+)`")

#: Every key a kustomization can name a PATH under, not `resources` alone: a `secretGenerator`
#: pointed at `infra/secrets/` puts a Secret in the stream exactly as a `resources` entry
#: would, and criterion 5's claim is about the directory, not about one field.
KUSTOMIZATION_PATH_KEYS = (
    "resources",
    "components",
    "bases",
    "patches",
    "files",
    "envs",
)

#: The two spellings a path into `infra/secrets/` takes, from the repo root and from a
#: kustomization one or more levels down.
FORBIDDEN_PATH_FRAGMENTS = ("infra/secrets", "../secrets")


def read(path: Path) -> str:
    """Read a file the plan owes, failing the one case that needs it when it is absent."""
    assert path.is_file(), f"{path} does not exist"
    return path.read_text(encoding="utf-8")


def documents_in(path: Path) -> list[dict[str, Any]]:
    """Every non-empty YAML document in a file, in file order. Multi-document by design:
    `traefik-basic-auth.example.yaml` is one file carrying both namespace copies."""
    return [
        document
        for document in yaml.safe_load_all(read(path))
        if isinstance(document, dict) and document
    ]


def encrypted_files() -> list[Path]:
    """Every `infra/secrets/*.enc.yaml`, sorted. EMPTY today and empty after this plan: the
    ciphertext lands in a later `/quick`, and the guard has to be green over zero files."""
    return sorted(SECRETS_DIR.glob("*.enc.yaml")) if SECRETS_DIR.is_dir() else []


def data_mappings(document: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """A Secret's `data` and `stringData` mappings as `(key-of-the-mapping, mapping)` pairs."""
    found: list[tuple[str, dict[str, Any]]] = []
    for key in DATA_KEYS:
        mapping = document.get(key)
        if mapping is None:
            continue
        assert isinstance(mapping, dict), f"`{key}` is not a mapping"
        found.append((key, mapping))
    return found


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


def example_contracts() -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """`{name: (namespaces, keys)}` read off the templates in `infra/secrets/`.

    Asserts the directory holds exactly the files `EXAMPLE_FILENAMES` names, so a missing
    template is a loud red rather than a contract map that is quietly one entry short.
    """
    assert SECRETS_DIR.is_dir(), f"{SECRETS_DIR} does not exist"
    present = {path.name for path in SECRETS_DIR.glob("*.example.yaml")}
    assert present == EXAMPLE_FILENAMES, (
        f"unexpected: {sorted(present - EXAMPLE_FILENAMES)}; "
        f"missing: {sorted(EXAMPLE_FILENAMES - present)}"
    )
    namespaces: dict[str, set[str]] = {}
    keys: dict[str, set[str]] = {}
    for filename in sorted(present):
        for document in documents_in(SECRETS_DIR / filename):
            assert document.get("kind") == SECRET_KIND, f"{filename} declares a non-Secret"
            name = name_of(document)
            assert name, f"{filename} has a document with no metadata.name"
            namespace = namespace_of(document)
            assert namespace is not None, f"{filename}'s {name} declares no metadata.namespace"
            namespaces.setdefault(name, set()).add(namespace)
            for _, mapping in data_mappings(document):
                keys.setdefault(name, set()).update(str(key) for key in mapping)
    return {
        name: (frozenset(found), frozenset(keys.get(name, set())))
        for name, found in namespaces.items()
    }


def table_cells(line: str) -> list[str]:
    """One markdown table row as stripped cells, or `[]` when the line is not a table row."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.split("|")[1:-1]]


def readme_contracts() -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """`{name: (namespaces, keys)}` read off the secrets table in `infra/k8s/README.md`.

    The table is located by its HEADER ROW, and every line of the page is scanned for that
    header rather than the first line mentioning "Secret" being taken: the
    page carries several tables and a lot of prose about Secrets, and a first-hit reader would
    happily parse the resource-envelope table instead. Exactly one header must match.
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
    assert len(rows) == len(EXPECTED_NAMESPACES), f"the secrets table has {len(rows)} data rows"
    contracts: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for cells in rows:
        assert len(cells) == len(README_HEADER_CELLS), f"a secrets row has {len(cells)} cells"
        names = BACKTICKED.findall(cells[0])
        assert len(names) == 1, f"the Secret cell {cells[0]!r} names {len(names)} backticked ids"
        contracts[names[0]] = (
            frozenset(BACKTICKED.findall(cells[1])),
            frozenset(BACKTICKED.findall(cells[2])),
        )
    return contracts


def envfrom_secret_names(document: dict[str, Any]) -> set[str]:
    """Every `spec.template.spec.containers[*].envFrom[*].secretRef.name` of one object.

    Covers `initContainers` too, and reads any object carrying a pod template rather than a
    list of kinds, so a Job added later is scanned without editing this file. A CronJob is
    NOT one of those: its template sits one level deeper, at `spec.jobTemplate.spec.template`,
    so this unwraps that one extra key when present - found by running this scan against
    an earlier change's two CronJobs and getting back an empty set, disproving the claim this
    docstring made before that plan existed.
    """
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return set()
    job_template = spec.get("jobTemplate")
    if isinstance(job_template, dict):
        spec = job_template.get("spec")
        if not isinstance(spec, dict):
            return set()
    template = spec.get("template")
    if not isinstance(template, dict):
        return set()
    pod = template.get("spec")
    if not isinstance(pod, dict):
        return set()
    found: set[str] = set()
    for key in ("initContainers", "containers"):
        containers = pod.get(key, [])
        if not isinstance(containers, list):
            continue
        for container in containers:
            if not isinstance(container, dict):
                continue
            sources = container.get("envFrom", [])
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                reference = source.get("secretRef")
                if isinstance(reference, dict) and reference.get("name") is not None:
                    found.add(str(reference["name"]))
    return found


def middleware_secret_names(document: dict[str, Any]) -> set[str]:
    """A Traefik `Middleware`'s `spec.basicAuth.secret`, the second shape a reference takes."""
    if document.get("kind") != MIDDLEWARE_KIND or document.get("apiVersion") != TRAEFIK_API_VERSION:
        return set()
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return set()
    basic_auth = spec.get("basicAuth")
    if not isinstance(basic_auth, dict) or basic_auth.get("secret") is None:
        return set()
    return {str(basic_auth["secret"])}


def tls_secret_names(document: dict[str, Any]) -> set[str]:
    """An Ingress's `spec.tls[*].secretName` - cert-manager's to create, never this plan's to
    supply. Collected so the exclusion can be a named subtraction rather than a silent gap."""
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return set()
    entries = spec.get("tls", [])
    if not isinstance(entries, list):
        return set()
    return {
        str(entry["secretName"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("secretName") is not None
    }


def image_pull_secret_names(document: dict[str, Any]) -> set[str]:
    """Every `imagePullSecrets[*].name` of one object's pod spec, the third shape a reference
    takes. A CronJob's pod spec is unwrapped the same way
    `envfrom_secret_names` unwraps it."""
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return set()
    job_template = spec.get("jobTemplate")
    if isinstance(job_template, dict):
        spec = job_template.get("spec")
        if not isinstance(spec, dict):
            return set()
    template = spec.get("template")
    if not isinstance(template, dict):
        return set()
    pod = template.get("spec")
    if not isinstance(pod, dict):
        return set()
    entries = pod.get("imagePullSecrets", [])
    if not isinstance(entries, list):
        return set()
    return {
        str(entry["name"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("name") is not None
    }


def referenced_secret_names(
    documents: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """`(supplied, issued)`: the Secret names the build expects someone to SUPPLY, and the
    ones cert-manager ISSUES. Walked over the parsed stream, never grepped out of text. The
    supplied set reads `envFrom`, a `Middleware`'s `basicAuth.secret` and, since an earlier change,
    `imagePullSecrets`."""
    supplied: set[str] = set()
    issued: set[str] = set()
    for document in documents:
        supplied |= envfrom_secret_names(document)
        supplied |= middleware_secret_names(document)
        supplied |= image_pull_secret_names(document)
        issued |= tls_secret_names(document)
    return supplied, issued


def kustomization_paths(path: Path) -> list[str]:
    """Every PATH one kustomization lists, from `resources` and from a generator's `files`.

    Parsed out of the YAML rather than grepped: a comment explaining why
    `infra/secrets/` is not a resource contains the same fragment as a resource entry would,
    and a text scan would redden against a correct, well-commented file.
    """
    listed: list[str] = []

    def walk(value: object, under_path_key: bool) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, under_path_key or str(key) in KUSTOMIZATION_PATH_KEYS)
        elif isinstance(value, list):
            for child in value:
                walk(child, under_path_key)
        elif isinstance(value, str) and under_path_key:
            listed.append(value)

    for document in documents_in(path):
        walk(document, False)
    return listed


def sops_creation_rules() -> list[dict[str, Any]]:
    """`.sops.yaml`'s `creation_rules`, as a list of mappings."""
    documents = documents_in(SOPS_CONFIG)
    assert len(documents) == 1, f"{SOPS_CONFIG} carries {len(documents)} documents, not 1"
    rules = documents[0].get("creation_rules")
    assert isinstance(rules, list), f"{SOPS_CONFIG} declares no creation_rules list"
    return [rule for rule in rules if isinstance(rule, dict)]


def tracked_files() -> list[str]:
    """Every path `git ls-files` reports, from the repo root rather than the cwd."""
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"`git ls-files` exited {completed.returncode}"
    return [line for line in completed.stdout.splitlines() if line]


#: Criterion 1's parametrisation. The fallback row keeps an EMPTY glob a passing case rather
#: than a skipped or an erroring one: zero encrypted files is the agreed state at the end of
#: this plan, and a skip there would read as "unknown" in every report from now on.
ENCRYPTED_FILE_PARAMS: list[Path | None] = list(encrypted_files()) or [None]
ENCRYPTED_FILE_IDS = [path.name for path in encrypted_files()] or ["no-enc-file-yet"]


@pytest.mark.parametrize("path", ENCRYPTED_FILE_PARAMS, ids=ENCRYPTED_FILE_IDS)
def test_every_encrypted_secret_is_ciphertext_under_a_named_age_recipient(
    path: Path | None,
) -> None:
    """Criterion 1. Catches an implementation that commits a `*.enc.yaml` written by hand or
    copied from a template - a file with the right name, the right keys, and the credentials
    in the clear, which every other case in this file would pass. Catches a file `sops` has
    only PARTIALLY encrypted: the prefix is asserted on every leaf under `data`/`stringData`,
    so one key added after the encrypt and never re-encrypted is red. Catches a `sops:` block
    with an empty `age` list, which is a file nobody holding an age key can decrypt, and which
    would fail at CD time rather than here."""
    if path is None:
        # Zero `*.enc.yaml` is the agreed state: `sops` and `age` are not installed, so the
        # ciphertext is a later `/quick`. The guard must be green over the empty glob.
        return
    document = yaml.safe_load(read(path))
    assert isinstance(document, dict), f"{path.name} is not a YAML mapping"
    metadata = document.get("sops")
    assert isinstance(metadata, dict), f"{path.name} carries no top-level `sops:` mapping"
    recipients = metadata.get("age")
    assert isinstance(recipients, list) and recipients, (
        f"{path.name}'s sops block lists no age recipient, so nothing can decrypt it"
    )
    mappings = data_mappings(document)
    assert mappings, f"{path.name} declares neither data nor stringData"
    for key, mapping in mappings:
        assert mapping, f"{path.name}'s `{key}` is empty"
        for name, value in mapping.items():
            assert str(value).startswith(ENCRYPTED_VALUE_PREFIX), (
                f"{path.name}: `{key}.{name}` is not ciphertext: {str(value)[:32]!r}"
            )


@pytest.mark.parametrize(
    ("candidate", "should_match"),
    [
        pytest.param(ENCRYPTED_PROBE, True, id="enc-file-is-encrypted"),
        pytest.param(EXAMPLE_PROBE, False, id="example-file-is-not"),
    ],
)
def test_the_sops_rule_covers_the_encrypted_file_and_leaves_the_example_alone(
    candidate: str, should_match: bool
) -> None:
    r"""Criterion 1's policy half, on the boundary that decides it: the two probe paths differ
    only in the segment before `.yaml`, and they live in the same directory. Catches a
    `path_regex` written as `infra/secrets/.*` or `infra/secrets/.*\.yaml$`, which tells `sops`
    to encrypt the plaintext templates too - after which criterion 4's placeholder case and
    every reader of the examples get ciphertext where the contract should be. Catches a
    `path_regex` anchored so tightly (a leading `^/`, a missing directory segment) that it
    matches NOTHING, which is the silent failure: `sops` then writes a file it has not
    encrypted and reports success."""
    rules = sops_creation_rules()
    assert len(rules) == 1, f"{SOPS_CONFIG} has {len(rules)} creation rules, not 1"
    pattern = rules[0].get("path_regex")
    assert isinstance(pattern, str) and pattern, "the creation rule declares no path_regex"
    matched = re.search(pattern, candidate) is not None
    assert matched is should_match, (
        f"path_regex {pattern!r} {'misses' if should_match else 'matches'} {candidate}"
    )


def test_the_sops_policy_encrypts_exactly_the_two_data_keys() -> None:
    r"""Criterion 1's second policy half, by EXACT string equality. Catches
    an `encrypted_regex` of `data` or `.*data.*`, which also matches `metadata` and encrypts
    the Secret's own name and namespace - a file `kubectl apply` cannot route anywhere and
    `kustomize` cannot read. Catches `^(data)$` alone, which leaves every `stringData` value
    in the clear in a file whose `sops:` block claims it is encrypted, and which criterion 1's
    leaf scan cannot see until such a file exists."""
    rules = sops_creation_rules()
    assert len(rules) == 1, f"{SOPS_CONFIG} has {len(rules)} creation rules, not 1"
    assert rules[0].get("encrypted_regex") == ENCRYPTED_REGEX, (
        f"encrypted_regex is {rules[0].get('encrypted_regex')!r}, not {ENCRYPTED_REGEX!r}"
    )


@pytest.mark.parametrize("pattern", IGNORED_PATTERNS, ids=list(IGNORED_PATTERNS))
def test_gitignore_carries_every_private_key_and_decrypted_secret_pattern(
    pattern: str,
) -> None:
    """Criterion 2's `.gitignore` half, which never skips. Catches an implementation that
    relies on nobody happening to run `age-keygen -o keys.txt` or `sops -d > x.dec.yaml` in
    the repo: the tracked-files case is red only AFTER the accident, and this one is red
    before it. Each pattern is required as a WHOLE line - `*.dec.yaml` appearing inside a
    comment explaining the workflow is not an ignore rule, and a substring check would count
    it."""
    lines = [line.strip() for line in read(GITIGNORE).splitlines()]
    assert pattern in lines, f"{GITIGNORE} has no line reading exactly {pattern!r}"


def test_no_private_key_and_no_decrypted_secret_is_tracked_by_git() -> None:
    """Criterion 2's tracked half. Catches the accident the whole plan exists to prevent: an
    age private key or a decrypted Secret committed - `git ls-files` and not the working tree,
    because a file deleted from the tree after the commit is still in history and still in
    everybody's clone. Runs over the whole index rather than over `infra/secrets/`, since the
    key a human generates lands wherever they were standing."""
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH; CI's checkout is the copy that cannot skip")
    tracked = tracked_files()
    assert tracked, "`git ls-files` reported nothing, so this scan read no files"
    offending = sorted(
        path
        for path in tracked
        for pattern in IGNORED_PATTERNS
        if fnmatch.fnmatch(Path(path).name, pattern)
    )
    assert offending == [], f"tracked files matching {list(IGNORED_PATTERNS)}: {offending}"


def test_the_example_templates_state_the_same_contracts_as_the_readme_table() -> None:
    """Criterion 3's file half, by EQUALITY of `{name: (namespaces, keys)}` - and neither side
    is a literal in this file, which is the whole point: the templates an operator fills in
    and the table a reader consults before an apply are the two places the contract lives, and
    a constant restated here would let them drift together while this case stayed green.
    Catches a key added to `gateway-secrets.example.yaml` for a new provider and never
    documented, and the reverse - a key documented and missing from the template, which is an
    operator supplying a Secret the gateway starts without and fails on at runtime. Catches
    `traefik-basic-auth` shipped for one namespace, since the namespace sets are compared too
    and a `Middleware` reads its Secret only from its own namespace. The NAME set and the
    namespaces are additionally pinned against the plan's own constants, so a wholesale rename
    on BOTH sides at once - the one edit equality alone cannot see - is still red."""
    from_files = example_contracts()
    from_readme = readme_contracts()
    differing = sorted(
        name for name in set(from_files) & set(from_readme) if from_files[name] != from_readme[name]
    )
    assert from_files == from_readme, (
        f"only in the templates: {sorted(set(from_files) - set(from_readme))}; "
        f"only in the README: {sorted(set(from_readme) - set(from_files))}; "
        f"differing: {differing}"
    )
    assert {name: namespaces for name, (namespaces, _) in from_files.items()} == (
        EXPECTED_NAMESPACES
    ), f"the templates state namespaces {[(n, sorted(ns)) for n, (ns, _) in from_files.items()]}"


#: `infra/secrets/README.md`, whose contracts table and its two counts are compared against
#: the templates by `test_the_secrets_readme_lists_every_template_and_counts_them`.
SECRETS_README = SECRETS_DIR / "README.md"

#: The header of that README's contracts table, cell for cell.
SECRETS_README_HEADER_CELLS = ["Secret", "namespace", "template"]

#: Spelled-out counts the two READMEs use for the number of Secret contracts.
NUMBER_WORDS = {"eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def test_the_secrets_readme_lists_every_template_and_counts_them() -> None:
    """Criterion 18's `infra/secrets/README.md` half. The contracts table names exactly the
    templates in the directory, each with its own filename, and the counts in its opening line,
    its heading and `infra/k8s/README.md`'s "Secrets nothing here writes" line equal the number
    of templates. Catches the next contract landing without a row. That table had already
    fallen three rows behind before an earlier change, and nothing went red."""
    lines = read(SECRETS_README).splitlines()
    headers = [
        index
        for index, line in enumerate(lines)
        if table_cells(line) == SECRETS_README_HEADER_CELLS
    ]
    assert len(headers) == 1, f"{len(headers)} contracts tables in {SECRETS_README}"
    rows: dict[str, str] = {}
    for line in lines[headers[0] + 1 :]:
        cells = table_cells(line)
        if not cells:
            break
        if all(cell and set(cell) <= {"-", ":"} for cell in cells):
            continue
        names = BACKTICKED.findall(cells[0])
        templates = BACKTICKED.findall(cells[2])
        assert len(names) == 1 and len(templates) == 1, f"a contracts row reads {cells!r}"
        assert names[0] not in rows, f"{names[0]} has two rows in the contracts table"
        rows[names[0]] = templates[0]
    listed = frozenset(rows.values())
    assert listed == EXAMPLE_FILENAMES, (
        f"only in the table: {sorted(listed - EXAMPLE_FILENAMES)}; "
        f"missing from it: {sorted(EXAMPLE_FILENAMES - listed)}"
    )
    assert frozenset(rows) == frozenset(EXPECTED_NAMESPACES), f"table names {sorted(rows)}"
    assert rows == {name: f"{name}.example.yaml" for name in EXPECTED_NAMESPACES}, (
        f"a row points at another Secret's template: {rows}"
    )

    expected = len(EXAMPLE_FILENAMES)
    counts = {
        "secrets README opening": re.search(
            r"^(\w+) Secrets the cluster needs", read(SECRETS_README), re.MULTILINE
        ),
        "secrets README heading": re.search(
            r"^## The (\w+) contracts$", read(SECRETS_README), re.MULTILINE
        ),
        "k8s README": re.search(
            r"^The (\w+) Secrets nothing here writes", read(K8S_README), re.MULTILINE
        ),
    }
    for where, match in counts.items():
        assert match, f"the {where} count sentence is gone"
        assert NUMBER_WORDS.get(match.group(1).lower()) == expected, (
            f"the {where} says {match.group(1)!r}; there are {expected} templates"
        )


def test_every_secret_the_build_references_is_one_the_templates_supply(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 3's build half, by EQUALITY between the templates' NAME set and the names the
    built stream actually references (`envFrom[*].secretRef.name`, each `Middleware`'s
    `spec.basicAuth.secret`, and since an earlier change criterion 26 each pod's
    `imagePullSecrets[*].name`). Catches `ghcr-pull` contracted and named by no pod spec, and a
    pod spec naming a pull secret nothing contracts. Catches a template named for a Secret
    nothing reads - a file an operator fills in for nothing - and catches a manifest naming a
    fifth Secret this plan supplies no template for, which is an apply that fails on a Secret
    nobody wrote down.
    `test_k8s_apps.py` pins the same names against four LITERALS build-wide; that case cannot
    see a template renamed or never written, because no literal changes when a FILE does, and
    this one is read off the files. The four `tls[*].secretName` values are subtracted by name: cert-manager creates those, so including them would make a correct
    build red, and the subtraction is asserted to have removed exactly the four pinned ones so
    a stale exclusion list cannot hide a real reference behind it."""
    supplied, issued = referenced_secret_names(built)
    assert issued == INGRESS_TLS_SECRETS, (
        f"the build's Ingress tls secrets are {sorted(issued)}, not {sorted(INGRESS_TLS_SECRETS)}"
    )
    assert supplied.isdisjoint(INGRESS_TLS_SECRETS), (
        f"a workload reads a cert-manager Secret: {sorted(supplied & INGRESS_TLS_SECRETS)}"
    )
    named = set(example_contracts())
    assert supplied == named, (
        f"referenced and unsupplied: {sorted(supplied - named)}; "
        f"supplied and unreferenced: {sorted(named - supplied)}"
    )


def test_no_kustomization_lists_a_path_under_infra_secrets() -> None:
    """Criterion 5's file half. Catches the convenient repair of an apply failing on a missing
    Secret: adding `../../secrets/gateway-secrets.enc.yaml` to a kustomization's `resources`,
    which makes `kustomize build` emit the ciphertext as a resource the API server rejects,
    and makes `make verify` the place credentials start flowing through the build. Scans every
    path-bearing key, generators included, because a `secretGenerator` pointed at the same
    directory puts a Secret in the stream with no `resources` entry at all. Parsed as YAML and
    not grepped, so a COMMENT saying `infra/secrets` is decrypted in CD -
    which is exactly the comment the implementation is likely to write - cannot redden it."""
    kustomizations = sorted(K8S_ROOT.rglob("kustomization.yaml"))
    assert kustomizations, f"{K8S_ROOT} holds no kustomization.yaml, so this scan read nothing"
    offending = sorted(
        f"{path.relative_to(REPO_ROOT)}: {listed}"
        for path in kustomizations
        for listed in kustomization_paths(path)
        if any(fragment in listed for fragment in FORBIDDEN_PATH_FRAGMENTS)
    )
    assert offending == [], f"kustomizations pointing into the secrets directory: {offending}"


def test_the_build_emits_no_secret_object(built: list[dict[str, Any]]) -> None:
    """Criterion 5's build half, over the WHOLE stream: whatever a kustomization lists, what
    reaches a cluster is what kustomize emits. Catches the same repair as the case above
    arriving by a route the path scan cannot see - a `secretGenerator` with inline `literals`,
    which names no path at all and produces a Secret with the credentials in the build output.
    `test_k8s_apps.py` asserts the same emptiness; that case is scoped by its own module to
    the `apps` story, and criterion 5 is the claim for the whole build, so it is stated here
    against the stream rather than inherited."""
    emitted = [document for document in built if document.get("kind") == SECRET_KIND]
    assert emitted == [], f"the build emits Secrets: {[name_of(document) for document in emitted]}"


@pytest.mark.parametrize("filename", sorted(EXAMPLE_FILENAMES), ids=sorted(EXAMPLE_FILENAMES))
def test_every_example_value_is_the_literal_placeholder(filename: str) -> None:
    """Criterion 4, across EVERY document in a file - `traefik-basic-auth.example.yaml` is two
    documents, and a reader of the first would never see a real password left in the second.
    Only values under `data`/`stringData` are in scope: `metadata.name`, `metadata.namespace`,
    `apiVersion`, `kind` and every label are the contract itself, not credentials, and a scan
    that required `REPLACE_ME` of those would be red against a correct file. Catches a
    template filled in with a working credential and committed - the exact accident a `.enc`
    workflow exists to remove, and one gitleaks does not necessarily flag for a Postgres
    password or an htpasswd hash. Catches an empty value (`POSTGRES_PASSWORD: ""`), which
    applies cleanly and fails at runtime as an authentication error nobody traces back here.
    `users` is the one key allowed a longer shape, and its PASSWORD field is still required to
    be exactly the placeholder, so a real htpasswd hash committed under it is red."""
    path = SECRETS_DIR / filename
    documents = documents_in(path)
    assert documents, f"{filename} carries no YAML document"
    checked = 0
    for document in documents:
        mappings = data_mappings(document)
        assert mappings, f"{filename}'s {name_of(document)} declares neither data nor stringData"
        for key, mapping in mappings:
            assert mapping, f"{filename}'s `{key}` is empty"
            for name, value in mapping.items():
                checked += 1
                text = str(value)
                if str(name) != "users":
                    assert text == PLACEHOLDER, f"{filename}: `{key}.{name}` is {text!r}"
                    continue
                lines = [line for line in text.splitlines() if line.strip()]
                assert lines, f"{filename}: `{key}.users` carries no htpasswd line"
                for line in lines:
                    matched = HTPASSWD_LINE.fullmatch(line.strip())
                    assert matched is not None, f"{filename}: {line!r} is not `<user>:<password>`"
                    assert matched.group("password") == PLACEHOLDER, (
                        f"{filename}: `users` carries a real-looking password"
                    )
    assert checked, f"{filename} declares no values at all, so this scan asserted nothing"


def comment_text(path: Path) -> str:
    """Every comment line of a template, joined, `#` markers included. A key NAME or a value is
    never a comment, so a fact stated only as a key does not count as stated."""
    return "\n".join(line for line in read(path).splitlines() if line.lstrip().startswith("#"))


def test_the_ghcr_pull_contract_is_a_dockerconfigjson_secret_in_apps() -> None:
    """An earlier change criteria 16 and 17, read off the template itself rather than through
    `example_contracts`, so a red here names this file and not whichever other template is
    missing. Catches the template written as `type: Opaque`, which applies cleanly and which
    the kubelet ignores for a pull, so every Helpdesk pod sits in `ImagePullBackOff`. Catches it
    in `platform` or with no namespace, where a pod in `apps` cannot read it. Catches a key other
    than `.dockerconfigjson`, or a second key beside it, which the API server rejects for that
    type or which the operator fills in for nothing. Catches the comment that never says which
    token scope to mint, which is the one step an operator gets wrong."""
    path = SECRETS_DIR / f"{GHCR_PULL}.example.yaml"
    documents = documents_in(path)
    assert len(documents) == 1, f"{path.name} carries {len(documents)} documents, not 1"
    document = documents[0]
    assert document.get("kind") == SECRET_KIND, f"{path.name} declares a {document.get('kind')}"
    assert name_of(document) == GHCR_PULL, f"{path.name} names {name_of(document)!r}"
    assert namespace_of(document) == APPS, f"{path.name} is in {namespace_of(document)!r}"
    assert document.get("type") == DOCKERCONFIGJSON_TYPE, (
        f"{path.name} is of type {document.get('type')!r}, not {DOCKERCONFIGJSON_TYPE}"
    )
    assert "data" not in document, f"{path.name} carries a `data` mapping beside `stringData`"
    assert document.get("stringData") == {DOCKERCONFIGJSON_KEY: PLACEHOLDER}, (
        f"{path.name}'s stringData is {document.get('stringData')!r}"
    )
    assert READ_PACKAGES_SCOPE in comment_text(path), (
        f"no comment in {path.name} names the `{READ_PACKAGES_SCOPE}` token the value is built from"
    )


def test_the_helpdesk_secret_template_carries_the_published_keys() -> None:
    """An earlier change criterion 10, the exact key set, read off the template itself rather than through
    `example_contracts`, so a missing neighbouring template cannot be what reddens it. Catches a
    prefixed name left where the companion repo reads a bare one, which is a pod that starts on
    its localhost default and never says so. Catches the singular gateway key name kept where
    the companion repo reads the plural list, and `JWT_SECRET` left out, which is an agent API
    signing sessions with whatever default it ships. Catches both spellings of one value kept
    side by side, since equality refuses the extra key."""
    path = SECRETS_DIR / f"{HELPDESK_SECRETS}.example.yaml"
    documents = documents_in(path)
    assert len(documents) == 1, f"{path.name} carries {len(documents)} documents, not 1"
    keys: set[str] = set()
    values: dict[str, str] = {}
    for _, mapping in data_mappings(documents[0]):
        keys.update(str(key) for key in mapping)
        values.update({str(key): str(value) for key, value in mapping.items()})
    assert keys == HELPDESK_SECRET_KEYS, (
        f"unexpected: {sorted(keys - HELPDESK_SECRET_KEYS)}; "
        f"missing: {sorted(HELPDESK_SECRET_KEYS - keys)}"
    )
    filled = sorted(key for key, value in values.items() if value != PLACEHOLDER)
    assert filled == [], f"{path.name} carries values other than {PLACEHOLDER} for {filled}"


@pytest.mark.parametrize(
    ("fact", "pattern"),
    HELPDESK_COMMENT_FACTS,
    ids=[fact for fact, _ in HELPDESK_COMMENT_FACTS],
)
def test_the_helpdesk_secret_template_comments_state_the_published_formats(
    fact: str, pattern: re.Pattern[str]
) -> None:
    """An earlier change criteria 11 to 13, one stated fact per row, over the template's COMMENT lines only.
    Every value is `REPLACE_ME`, so the comment is the only place these formats can live.
    Catches a template whose comments leave out one of the three things an operator cannot
    guess from a key name: the role and driver scheme of the DSN (guessed wrong, the Helpdesk
    runs as the owner role or fails to load its driver), the `tenant:key` list shape and where
    the digests go (guessed wrong, every model call is a 401), and the 32-byte floor on the
    signing secret. Each row is only the fact the criterion names, never wording (standing rule
    64)."""
    path = SECRETS_DIR / f"{HELPDESK_SECRETS}.example.yaml"
    assert pattern.search(comment_text(path)), (
        f"no comment in {path.name} states {fact} (`{pattern.pattern}`)"
    )


@pytest.mark.parametrize("target", SOPS_TARGETS, ids=list(SOPS_TARGETS))
def test_the_sops_targets_refuse_to_run_when_sops_is_missing(target: str) -> None:
    """Criterion 6, the unhappy path. Catches a target written as a bare `sops --encrypt ...`
    over a glob: with no `sops` on `PATH` the shell exits 127 with `sops: command not found`,
    which `make` reports as an error nobody reads - and with `-` or `|| true` in front of it,
    or a `for` loop over an empty glob, it reports SUCCESS over a no-op, which is an operator
    believing four files were encrypted when nothing was written. The exit code and the
    message are asserted together: a non-zero exit that names something else (a missing
    directory, a `sops` config error) is a different failure, and only the name in the message
    tells the operator what to install. `PATH` is stripped to `/usr/bin:/bin` so the case
    means the same thing on a machine where `sops` IS installed."""
    if shutil.which("make") is None:
        pytest.skip("make is not on PATH; the Makefile targets cannot be exercised")
    completed = subprocess.run(
        ["make", target],
        cwd=REPO_ROOT,
        env={"PATH": PATH_WITHOUT_SOPS, "HOME": os.environ.get("HOME", "")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0, (
        f"`make {target}` exited 0 with no sops on PATH:\n{completed.stdout}"
    )
    assert "sops" in completed.stderr, (
        f"`make {target}` exited {completed.returncode} without naming sops:\n{completed.stderr}"
    )


@pytest.mark.parametrize("target", SOPS_TARGETS, ids=list(SOPS_TARGETS))
def test_the_sops_targets_refuse_to_report_success_over_an_empty_glob(
    target: str, tmp_path: Path
) -> None:
    """Criterion 6's second half, and the branch the case above can never reach: with `sops`
    absent the guard fires first, so the loop underneath it is unexercised. A STUB `sops` on
    `PATH` - an executable that exits 0 and does nothing - gets past the guard without needing
    the real binary installed, and leaves the empty glob as the only thing
    that can decide the exit code.

    Catches `for f in infra/secrets/*.dec.yaml` with no match, which in `sh` iterates ONCE
    over the literal pattern and, guarded by the `[ -e ]` this implementation carries, simply
    does nothing and exits 0. An operator who ran the target in the wrong directory, or who
    edited `postgres-credentials.yaml` rather than `postgres-credentials.dec.yaml`, would be
    told four Secrets were encrypted when none were - the no-op reported as success that this
    criterion exists to forbid. The message is asserted with the exit code: `nothing to` is
    what separates it from the missing-binary refusal above, which would otherwise satisfy a
    bare non-zero check."""
    if shutil.which("make") is None:
        pytest.skip("make is not on PATH; the Makefile targets cannot be exercised")
    stub = tmp_path / "sops"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    # An empty folder of its own, not `infra/secrets`: since 2026-10-12 the real one
    # holds the six committed `*.enc.yaml`, so the glob there is no longer empty.
    empty = tmp_path / "secrets"
    empty.mkdir()
    completed = subprocess.run(
        ["make", target, f"SECRETS_DIR={empty}"],
        cwd=REPO_ROOT,
        env={
            "PATH": f"{tmp_path}{os.pathsep}{PATH_WITHOUT_SOPS}",
            "HOME": os.environ.get("HOME", ""),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0, (
        f"`make {target}` exited 0 with nothing to do:\n{completed.stdout}"
    )
    assert "nothing to" in completed.stderr, (
        f"`make {target}` exited {completed.returncode} without saying it did nothing:"
        f"\n{completed.stderr}"
    )
