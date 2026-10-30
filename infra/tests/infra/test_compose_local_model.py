r"""The local embedding server starts by default, from a model file one `make` step fetches.

The llama.cpp servers used to sit behind a `local-model` profile. The local chat model is gone, and
the profile is off the one that is left, `llama-embed`:
the semantic cache needs it, so a plain `docker compose up -d` starts it. Its model file
(`bge-small-en-v1.5-q8_0.gguf`) comes from `make embed-model`, which `make up` runs first.

Most cases read files as TEXT or as parsed YAML (`yaml.safe_load`), through `read` and
`load_yaml` from `test_hetzner_module.py`, spelled as the package `__init__` files make it. A file
that has not been changed yet fails the case that reads it, never
the collection.

One case asks Docker Compose itself which services start by default. That
`docker compose config --services` call is made ONCE per module, and skips only when `docker`
is not on `PATH`. A non-zero exit fails with Compose's stderr, because a config that cannot
be read is the failure this file exists to report.

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
MAKEFILE = REPO_ROOT / "Makefile"

#: The profile the embedding server used to sit behind. It is gone now.
PROFILE = "local-model"

#: The services that start with a plain `docker compose up -d`.
DEFAULT_SERVICES = frozenset(
    {
        "gateway",
        "redis",
        "postgres",
        "pushgateway",
        "qdrant",
        "prometheus",
        "grafana",
        "llama-embed",
    }
)

#: The model file `make embed-model` fetches and compose mounts.
EMBED_MODEL_FILE = "bge-small-en-v1.5-q8_0.gguf"

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
    """The `config --services` run, made ONCE for this module; skips without `docker`."""
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH; the YAML cases in this file still pin the profiles")
    return {"default": compose_services()}


def test_llama_embed_has_no_profile_and_starts_by_default() -> None:
    """`llama-embed` carries no `profiles` key at all, and the chat model is still gone.

    Catches a compose file that leaves the embedding server behind a profile (the semantic
    cache would find no embedder in the default stack), one that swaps the profile for an
    empty list, and one that brings back `llama`.
    """
    found = services()
    assert "llama" not in found
    for name in LLAMA_PORTS:
        assert name in found, f"{name} is gone from docker-compose.yml; it must be kept"
        assert "profiles" not in found[name], name


def test_every_other_base_service_starts_by_default() -> None:
    """The services without `profiles` are exactly the eight the default stack needs.

    Catches a profile added to the gateway or a backing store, and a profile left on
    `llama-embed`, since either changes this set.
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


def test_the_default_stack_starts_the_embedding_server(listed: dict[str, set[str]]) -> None:
    """Compose itself lists `llama-embed` among the services a plain `up` starts.

    Catches a profile spelled in a way the YAML cases accept but Compose still honours, so
    the default list would lack the embedding server.
    """
    assert listed["default"] == set(DEFAULT_SERVICES)
    assert "llama-embed" in listed["default"]


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


def test_the_readme_quickstart_names_the_model_step() -> None:
    """The README tells the reader to run `make embed-model` to get the model file.

    Catches a README that starts the embedding server by default without saying where its
    model file comes from, which leaves a fresh clone with a server that cannot load.
    """
    assert "make embed-model" in read(README)


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


def target_line(makefile: str, target: str) -> str:
    """The `target: prerequisites` line of a Makefile, without its `##` help text."""
    match = re.search(rf"^{re.escape(target)}:[ \t]*([^#\n]*)", makefile, flags=re.MULTILINE)
    assert match is not None, f"Makefile has no `{target}:` target"
    return match.group(1)


def target_recipe(makefile: str, target: str) -> str:
    """The tab-indented recipe lines under a Makefile target, joined."""
    match = re.search(
        rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n?)+)", makefile, flags=re.MULTILINE
    )
    assert match is not None, f"Makefile has no recipe under `{target}:`"
    return match.group(1)


def test_make_embed_model_fetches_the_file_compose_mounts() -> None:
    """The target exists, names the model file, the `.cache/models` folder compose mounts,
    and checks a sha256.

    Catches a target that downloads a different file than `LLAMA_EMBED_MODEL_FILE` expects,
    puts it where compose does not mount it, or skips the checksum (a truncated download
    would then start a server that fails to load).
    """
    makefile = read(MAKEFILE)
    recipe = target_recipe(makefile, "embed-model")
    assert EMBED_MODEL_FILE in makefile
    assert EMBED_MODEL_FILE in recipe or "EMBED_MODEL" in recipe
    assert ".cache/models" in makefile
    assert "sha256" in makefile.lower()
    assert re.search(r"[0-9a-f]{64}", makefile), "the sha256 is not pinned in the Makefile"


def test_make_up_fetches_the_model_first() -> None:
    """`up` lists `embed-model` as a prerequisite.

    Catches an `up` that starts the stack before the model file exists, which makes Docker
    create an empty folder where the file should be.
    """
    assert "embed-model" in target_line(read(MAKEFILE), "up").split()


@pytest.mark.parametrize("path", [README], ids=["README.md"])
def test_the_docs_no_longer_name_the_local_model_profile(path: Path) -> None:
    """The README does not name `--profile local-model`.

    Catches a doc that still sends the reader to a profile that no longer exists, where
    the command would start nothing extra and say nothing.
    """
    assert "--profile local-model" not in read(path)


def test_make_run_starts_the_backing_services_without_the_compose_gateway() -> None:
    """`run` brings compose up with the gateway scaled to zero, then serves the gateway on
    this machine with reload.

    Catches a `run` that starts the compose gateway as well, which takes port 8080 before
    the host gateway can, and one that stops serving on the host or drops reload.
    """
    makefile = read(MAKEFILE)
    recipe = target_recipe(makefile, "run")
    compose_lines = [line for line in recipe.splitlines() if "docker compose" in line]
    assert len(compose_lines) == 1, compose_lines
    assert "--scale gateway=0" in compose_lines[0]
    assert "--profile" not in compose_lines[0]
    assert "serve(reload=True)" in recipe
    assert "embed-model" in target_line(makefile, "run").split()


def test_make_run_reads_the_demo_env_before_the_real_one() -> None:
    """The host gateway loads `backend/.env.demo` and then `backend/.env`, as compose does.

    Catches a `run` whose gateway starts with none of the local values, and one that loads
    the two in the wrong order so the stand-in values hide the real keys.
    """
    recipe = target_recipe(read(MAKEFILE), "run")
    demo = recipe.find("backend/.env.demo")
    real = recipe.find("backend/.env;")
    assert demo != -1 and real != -1, recipe
    assert demo < real
