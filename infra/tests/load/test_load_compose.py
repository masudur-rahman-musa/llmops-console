r"""`infra/load/compose.yml`, the Makefile's `load` target and the results
`.gitignore` entry.

Parsing only: the override is read as YAML and the Makefile recipe as text, with
`$(VAR)` make variables resolved. Nothing here runs Docker or k6.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_LOAD = REPO_ROOT / "infra" / "load" / "compose.yml"
MAKEFILE = REPO_ROOT / "Makefile"
GITIGNORE = REPO_ROOT / ".gitignore"

LOAD_COMPOSE = (
    "docker compose -f docker-compose.yml -f infra/load/compose.yml --profile load"
)
POLICY_SOURCE = "./infra/load/routing.load.yaml"
POLICY_TARGET = "/etc/opsconsole/routing.load.yaml"
CREDENTIAL_WORDS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
INTERPOLATION = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")

# ---- compose helpers ------------------------------------------------------


def load(path: Path) -> dict[str, Any]:
    """Parse a YAML document that must be a mapping."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def environment_of(service: dict[str, Any]) -> dict[str, str]:
    """A service's `environment`, in either compose form (mapping or `K=V` list)."""
    raw = service.get("environment", {})
    if isinstance(raw, list):
        pairs = [str(item).split("=", 1) for item in raw]
        return {pair[0]: pair[1] if len(pair) == 2 else "" for pair in pairs}
    assert isinstance(raw, dict)
    return {str(key): str(value) for key, value in raw.items()}


def volumes_of(service: dict[str, Any]) -> list[str]:
    raw = service.get("volumes", [])
    assert isinstance(raw, list)
    return [str(item) for item in raw]


@pytest.fixture(scope="module")
def services() -> dict[str, Any]:
    listed = load(COMPOSE_LOAD)["services"]
    assert isinstance(listed, dict)
    return listed


def test_mock_provider_is_built_from_its_directory_under_the_load_profile(
    services: dict[str, Any],
) -> None:
    """Criterion 8: catches a mock pulled from a registry instead of built from
    this repo, or one with no healthcheck for the gateway to wait on."""
    mock = services["mock-provider"]
    assert mock["build"] == "infra/load/mock-provider"
    assert isinstance(mock["healthcheck"], dict)
    assert "test" in mock["healthcheck"]


@pytest.mark.parametrize("name", ["mock-provider", "k6"])
def test_the_load_services_run_only_under_the_load_profile(
    services: dict[str, Any], name: str
) -> None:
    """Criterion 8: catches a load service with no profile, which a plain
    `docker compose up` with the override would start."""
    assert services[name]["profiles"] == ["load"]


def test_k6_is_pinned_to_one_exact_grafana_tag(services: dict[str, Any]) -> None:
    r"""Criterion 9: catches `grafana/k6:latest`, an untagged `grafana/k6`, or a
    partial tag like `grafana/k6:1`, any of which changes k6 under a rerun."""
    assert re.fullmatch(r"grafana/k6:\d+\.\d+\.\d+", str(services["k6"]["image"]))


def test_k6_runs_the_mounted_script_and_writes_results_to_the_host(
    services: dict[str, Any],
) -> None:
    """Criteria 11 and 17: catches k6 running a script path nothing mounts, or a
    results directory that stays inside the container and is lost at `--rm`."""
    k6 = services["k6"]
    assert k6["command"] == ["run", "/scripts/gateway.js"]
    volumes = volumes_of(k6)
    assert "./infra/load/k6:/scripts:ro" in volumes
    assert "./infra/load/results:/results" in volumes


def test_k6_reads_the_key_and_gateway_address_from_the_environment(
    services: dict[str, Any],
) -> None:
    """Criterion 18: catches a key written into the compose file for k6, or a
    gateway address that is not the compose service name."""
    environment = environment_of(services["k6"])
    assert environment["LOAD_API_KEY"] == "${LOAD_API_KEY}"
    assert environment["GATEWAY_URL"] == "http://gateway:8080"


def test_the_gateway_policy_path_is_the_mounted_load_routing_file(
    services: dict[str, Any],
) -> None:
    """Criterion 10: catches a policy path nothing is mounted at, or the shipped
    `routing.yaml` mounted there instead of the load file."""
    gateway = services["gateway"]
    policy_path = environment_of(gateway)["OPSCONSOLE_ROUTING_POLICY_PATH"]
    assert policy_path == POLICY_TARGET
    mounts = [volume.split(":") for volume in volumes_of(gateway)]
    at_target = [
        mount for mount in mounts if len(mount) >= 2 and mount[1] == policy_path
    ]
    assert at_target, f"nothing is mounted at {policy_path}"
    assert all(mount[0] == POLICY_SOURCE for mount in at_target)
    assert f"{POLICY_SOURCE}:{POLICY_TARGET}:ro" in volumes_of(gateway)


def test_the_gateway_takes_its_keys_from_the_environment_and_waits_for_the_mock(
    services: dict[str, Any],
) -> None:
    """Criterion 18: catches a tenant key digest written into the override, or a
    gateway that can start before the mock it routes to exists."""
    gateway = services["gateway"]
    assert environment_of(gateway)["OPSCONSOLE_API_KEYS"] == "${OPSCONSOLE_API_KEYS}"
    depends_on = gateway["depends_on"]
    assert isinstance(depends_on, list | dict)
    assert "mock-provider" in depends_on


def test_every_credential_environment_value_is_an_interpolation(
    services: dict[str, Any],
) -> None:
    """Criterion 10: catches any credential-named variable given a literal value
    (for example `load:changeme`) in any service of the override."""
    checked: set[str] = set()
    for service in services.values():
        for key, value in environment_of(service).items():
            if any(word in key.upper() for word in CREDENTIAL_WORDS):
                checked.add(key)
                assert INTERPOLATION.fullmatch(value), f"{key} carries a literal"
    assert {"LOAD_API_KEY", "OPSCONSOLE_API_KEYS"} <= checked


def test_no_sha256_digest_literal_appears_in_the_override() -> None:
    r"""Criterion 10: catches a precomputed key digest pasted anywhere in the file,
    including outside `environment:` (a comment, a command, an `x-` block)."""
    text = COMPOSE_LOAD.read_text(encoding="utf-8")
    assert re.search(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])", text) is None


# ---- Makefile helpers -----------------------------------------------------


def make_variables() -> dict[str, str]:
    """Every simple `NAME = value` / `NAME := value` definition in the Makefile.
    A name defined twice is refused rather than resolved to either one."""
    found: dict[str, list[str]] = {}
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)\s*[:?]?=\s*(.*)$", line)
        if match:
            found.setdefault(match.group(1), []).append(match.group(2).strip())
    for name, values in found.items():
        assert len(values) == 1, f"{name} is defined {len(values)} times"
    return {name: values[0] for name, values in found.items()}


def load_recipe_lines() -> list[str]:
    """The raw recipe lines of `load:`. Every target line is scanned: the
    Makefile must define `load:` exactly once, and the recipe runs until the first
    line that is not tab-indented."""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    starts = [
        index for index, line in enumerate(lines) if re.match(r"^load\s*:(?!=)", line)
    ]
    assert len(starts) == 1, f"`load:` is defined {len(starts)} times"
    recipe: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if not line.startswith("\t"):
            break
        recipe.append(line[1:])
    assert recipe, "the load target has no recipe"
    return recipe


def logical_lines() -> list[str]:
    """Recipe lines with `\\` continuations joined and make variables resolved.
    Each is one shell invocation, exactly as make hands it over."""
    variables = make_variables()

    def resolve(text: str) -> str:
        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            return variables.get(name, match.group(0))

        return re.sub(r"(?<!\$)\$[({]([A-Z][A-Z0-9_]*)[)}]", substitute, text)

    joined: list[str] = []
    pending = ""
    for line in load_recipe_lines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        joined.append(resolve(pending + stripped))
        pending = ""
    if pending:
        joined.append(resolve(pending))
    return joined


def statements() -> list[str]:
    """The recipe's shell statements in order: every logical line with its make
    prefixes (`@`, `-`, `+`) stripped, split on `;` and `&&`."""
    result: list[str] = []
    for line in logical_lines():
        body = line.lstrip().lstrip("@-+").strip()
        for part in re.split(r";|&&", body):
            if part.strip():
                result.append(part.strip())
    return result


def is_printer(statement: str) -> bool:
    """Rule 45: an `echo` or `printf` statement only PRINTS a command fragment, so it
    is excluded from every "this command runs" match. A `printf` inside a pipeline
    (`digest=$$(printf ...)`) does not start the statement and is not excluded."""
    return statement.startswith(("echo ", "echo\t", "printf ", "@echo "))


def running(fragment: str) -> list[int]:
    """Indexes of every non-printing statement containing `fragment`."""
    return [
        index
        for index, statement in enumerate(statements())
        if fragment in statement and not is_printer(statement)
    ]


def test_make_load_generates_a_fresh_key_at_runtime() -> None:
    """Criterion 18: catches a hard-coded or pre-shared load key; the key must be
    32 random bytes generated inside the recipe itself."""
    assert running("openssl rand -hex 32")


def test_make_load_derives_the_tenant_digest_with_sha256() -> None:
    """Criterion 18: catches `OPSCONSOLE_API_KEYS` set to the plaintext key or to a
    non-sha256 digest, which the gateway's key store would never match."""
    assignments = [
        statement
        for statement in statements()
        if statement.startswith("digest=") and not is_printer(statement)
    ]
    assert assignments, "no `digest=` assignment in the load recipe"
    assert all("sha256" in statement for statement in assignments)


def test_make_load_exports_both_keys_as_environment_only() -> None:
    """Criterion 18: catches a recipe that computes the key but never exports it to
    compose, or exports a tenant other than `load`."""
    exported = [
        statement for statement in statements() if statement.startswith("export ")
    ]
    assert any(
        re.match(r"^export\s+LOAD_API_KEY\b", statement) for statement in exported
    )
    assert any(
        re.match(r'^export\s+OPSCONSOLE_API_KEYS="load:\$\$digest"$', statement)
        for statement in exported
    )


def test_every_compose_invocation_uses_both_files_and_the_load_profile() -> None:
    """Criterion 20: catches any `docker compose` call in the recipe missing the
    override file or `--profile load`, for which `k6` and `mock-provider` do not
    exist."""
    invocations = running("docker compose")
    assert invocations, "the load recipe never calls docker compose"
    for index in invocations:
        assert LOAD_COMPOSE in statements()[index], statements()[index]


def test_make_load_brings_the_stack_up_and_migrates_before_k6_runs() -> None:
    """Criterion 19 and risk 2: catches k6 started before `ops-migrate` (every
    ledger write fails and the run reads as errors) or before the stack is up."""
    all_statements = statements()
    ups = [
        index
        for index in running(LOAD_COMPOSE)
        if all_statements[index].split(LOAD_COMPOSE, 1)[1].split()[:1] == ["up"]
    ]
    migrates = running("exec -T gateway ops-migrate")
    k6_runs = running("run --rm k6")
    assert ups and migrates and k6_runs
    assert max(ups) < min(migrates)
    assert max(migrates) < min(k6_runs)


def test_make_load_exits_with_k6s_own_exit_code() -> None:
    """Criterion 19 and the unhappy path: catches a teardown's status replacing
    k6's, a `|| true`, or a `-` recipe prefix, any of which turns a threshold
    breach into a green `make load`."""
    all_statements = statements()
    k6_runs = running("run --rm k6")
    assert len(k6_runs) == 1
    assert all_statements[k6_runs[0] + 1] == "status=$$?"
    assert all_statements[-1] == "exit $$status"
    recipe_text = "\n".join(logical_lines())
    assert re.search(r"\|\|\s*(?:true\b|:)", recipe_text) is None
    for line in logical_lines():
        assert not line.lstrip().lstrip("@+").startswith("-"), line


def test_make_load_writes_nothing_to_disk_through_a_redirect() -> None:
    """Criterion 19: catches the key or digest redirected into a file (`> .key`,
    `>> .env`). Printers are NOT excluded here: an `echo` into a file is exactly
    the write this forbids. Only the harmless fd redirects are allowed."""
    for statement in statements():
        cleaned = re.sub(r"\d?>&\d|\d?>\s*/dev/null", "", statement)
        assert ">" not in cleaned, statement
        assert not re.search(r"\btee\b", cleaned), statement


def test_the_results_directory_is_gitignored() -> None:
    """Criterion 17: catches `summary.json` from a local run landing in a commit."""
    lines = [
        line.strip() for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
    ]
    assert "infra/load/results/" in lines
