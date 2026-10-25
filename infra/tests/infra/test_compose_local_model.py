r"""The local embedding server is off by default and comes back with one profile.

An earlier change put the llama.cpp servers in `docker-compose.yml` behind the `local-model`
profile, so `docker compose up -d` starts the same model-free stack the node runs.
An earlier change removed the local chat model; only `llama-embed` is left behind the
profile. Its config stays whole, so turning it on is `--profile local-model` (or
`COMPOSE_PROFILES=local-model` in `.env.demo`), never a rewrite.

Most cases read files as TEXT or as parsed YAML (`yaml.safe_load`), through `read` and
`load_yaml` from `test_hetzner_module.py`, spelled as the package `__init__` files make it. A file the plan has not changed yet fails the case that reads it, never
the collection.

One case asks Docker Compose itself which services start, with and without the profile.
That pair of `docker compose config --services` calls is made ONCE per module, and skips
only when `docker` is not on `PATH`. A non-zero exit fails with Compose's stderr, because a
config that cannot be read is the failure this file exists to report.

No case needs a backing service, and none writes anywhere.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from infra.test_hetzner_module import load_yaml, read

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
TRADEOFF_SCRIPT = REPO_ROOT / "infra" / "scripts" / "tradeoff_run.sh"
README = REPO_ROOT / "README.md"

#: The profile the embedding server sits behind.
PROFILE = "local-model"

#: The services that start with a plain `docker compose up -d`.
DEFAULT_SERVICES = frozenset({"gateway", "redis", "postgres", "pushgateway", "qdrant"})

#: The local embedding server, with the host port mapping it must keep.
LLAMA_PORTS = {"llama-embed": ["8083:8083"]}

LLAMA_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-b10902"
MODELS_MOUNT = "${MODELS_DIR:-./.cache/models}:/models:ro"


def services() -> dict[str, dict[str, Any]]:
    """The `services:` mapping of the base compose file."""
    found = load_yaml(COMPOSE_FILE).get("services")
    assert isinstance(found, dict), "docker-compose.yml has no services mapping"
    return found


def compose_services(*extra: str) -> set[str]:
    """The service names `docker compose config --services` prints, with `extra` flags first."""
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            *extra,
            "config",
            "--services",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"docker compose config failed ({completed.returncode}): {completed.stderr}")
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


@pytest.fixture(scope="module")
def listed() -> dict[str, set[str]]:
    """Both `config --services` runs, made ONCE for this module; skips without `docker`."""
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH; the YAML cases in this file still pin the profiles")
    return {
        "default": compose_services(),
        "profile": compose_services("--profile", PROFILE),
    }


def test_the_llama_services_sit_behind_the_local_model_profile() -> None:
    """The embedding server carries exactly the `local-model` profile, and the chat
    model is gone.

    Catches a compose file that leaves it unprofiled, or brings back `llama`, or names the profile
    something the script and README do not use.
    """
    found = services()
    assert "llama" not in found
    for name in LLAMA_PORTS:
        assert name in found, f"{name} is gone from docker-compose.yml; it must be kept"
        assert found[name].get("profiles") == [PROFILE], name


def test_every_other_base_service_starts_by_default() -> None:
    """The services without `profiles` are exactly the five the default stack needs.

    Catches a profile added to the gateway or a backing store, and a llama service whose
    profile was dropped, since either changes this set.
    """
    unprofiled = {name for name, body in services().items() if "profiles" not in body}
    assert unprofiled == DEFAULT_SERVICES


def test_the_llama_services_keep_their_config() -> None:
    """Image, ports, models mount, healthcheck and command survive the change.

    Catches a change that turns the models off by gutting their config instead of
    profiling it, which the request forbids.
    """
    found = services()
    for name, ports in LLAMA_PORTS.items():
        body = found[name]
        assert body.get("image") == LLAMA_IMAGE, name
        assert body.get("ports") == ports, name
        assert MODELS_MOUNT in body.get("volumes", []), name
        healthcheck = body.get("healthcheck")
        assert isinstance(healthcheck, dict) and healthcheck.get("test"), name
        command = body.get("command")
        assert isinstance(command, list) and command, name


def test_the_gateway_does_not_wait_on_a_local_model() -> None:
    """The gateway waits on Redis, Postgres and Qdrant, and nothing else.

    Catches a gateway `depends_on` naming a llama service, which would make the default
    `up` fail on a service the profile left out.
    """
    depends_on = services()["gateway"].get("depends_on")
    assert isinstance(depends_on, dict), "gateway depends_on is not a mapping"
    assert set(depends_on) == {"redis", "postgres", "qdrant"}


def test_the_default_stack_starts_no_local_model(listed: dict[str, set[str]]) -> None:
    """Compose itself leaves the embedding server out by default and lists it with the profile.

    Catches a profile spelled in a way the YAML cases accept but Compose ignores, and a
    profile name `--profile local-model` does not bring back.
    """
    assert listed["default"] == set(DEFAULT_SERVICES)
    assert listed["profile"] == set(DEFAULT_SERVICES) | set(LLAMA_PORTS)


def test_the_tradeoff_run_starts_no_local_model() -> None:
    """The one `COMPOSE=` line carries `--profile tradeoff` and not `--profile local-model`.

    Catches a tradeoff script still starting a local model: no option routes to one
    since the `local-cpu` option left with the local chat model.
    """
    lines = [line for line in read(TRADEOFF_SCRIPT).splitlines() if line.startswith("COMPOSE=")]
    assert len(lines) == 1, f"expected exactly one COMPOSE= line, found {lines}"
    flags = re.findall(r"--profile\s+(\S+?)\)?(?=\s|$)", lines[0])
    assert "tradeoff" in flags, lines[0]
    assert PROFILE not in flags, lines[0]


def test_the_readme_quickstart_names_the_profile() -> None:
    """The README shows how to start the stack with the local embedding server.

    Catches a README that turns it off without saying how to turn it back on.
    """
    assert "docker compose --profile local-model up -d" in read(README)


def test_the_readme_default_up_drops_the_local_model() -> None:
    """The commented `docker compose up -d` line no longer promises a local model.

    Catches a quickstart comment still listing "local model" among what the default starts.
    """
    commented = [
        line for line in read(README).splitlines() if "docker compose up -d" in line and "#" in line
    ]
    assert commented, "README has no commented `docker compose up -d` line"
    for line in commented:
        assert "local model" not in line.lower(), line
