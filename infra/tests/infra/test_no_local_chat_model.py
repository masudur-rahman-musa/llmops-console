"""The local chat model is gone from everything that ships; the embedding server stays.

Chat runs on Anthropic and OpenAI only. These cases read files
as text or as parsed YAML. No backing service, no network, nothing written.

The one-off GPU run keeps its own Qwen manifests (`infra/k8s/gpu/`, the `gpu-vllm`
tradeoff option and its provider file). Those are out of the `qwen` sweep, and only
there: the other words are swept everywhere.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_PROVIDERS = REPO_ROOT / "backend" / "config" / "providers"
LOAD_PROVIDERS = REPO_ROOT / "infra" / "load" / "providers"

#: Where the local chat model must not appear (files and folders, from the repo root).
SWEPT = (
    "docker-compose.yml",
    "infra/k8s",
    "infra/tradeoff",
    "infra/load",
    "README.md",
    "docs",
    "web/README.md",
    "backend/config",
    "backend/.env.example",
)

#: The kept GPU run. Allowed to name Qwen, nothing else.
QWEN_ALLOWED = (
    "infra/k8s/gpu/",
    "infra/tradeoff/gpu-vllm.yaml",
    "infra/tradeoff/providers/",
    "infra/tradeoff/tradeoff.json",
    "infra/tradeoff/tradeoff.example.json",
)

TEXT_SUFFIXES = {".yaml", ".yml", ".md", ".json", ".sh", ".example", ".txt", ".tf", ""}

#: word -> pattern. `provider: local` is the old routing id; `\b` stops it matching `local_embed`.
FORBIDDEN = {
    "qwen": re.compile(r"qwen", re.IGNORECASE),
    "llama:8082": re.compile(r"llama:8082"),
    "provider local": re.compile(r"provider:\s*['\"]?local['\"]?\s*(?=[,}\n]|$)", re.MULTILINE),
    "local-cpu": re.compile(r"local-cpu"),
}


def swept_files() -> list[Path]:
    found: list[Path] = []
    for entry in SWEPT:
        path = REPO_ROOT / entry
        candidates = [path] if path.is_file() else sorted(path.rglob("*")) if path.exists() else []
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix in TEXT_SUFFIXES:
                found.append(candidate)
    return found


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_nothing_ships_the_local_chat_model() -> None:
    """Catches a leftover mention of the local chat model in compose, k8s, the
    tradeoff table, the load test, the docs or the config: a stale `qwen` line, the
    `llama:8082` address, a `provider: local` row or the `local-cpu` option."""
    files = swept_files()
    assert len(files) > 20, "the sweep must find the shipped files to be a test"
    hits: list[str] = []
    for path in files:
        name = relative(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in FORBIDDEN.items():
            if label == "qwen" and name.startswith(QWEN_ALLOWED):
                continue
            if pattern.search(text):
                hits.append(f"{name}: {label}")
    assert hits == []


def test_the_embedding_server_still_ships() -> None:
    """Catches a clean-up that removes `llama-embed` along with the chat model: the
    embedding server stays in compose (port 8083), in the k8s apps and in the
    network policy."""
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    embed = compose["services"]["llama-embed"]
    assert "8083:8083" in embed["ports"]
    assert "llama" not in compose["services"]

    apps = (REPO_ROOT / "infra/k8s/apps/llama-embed.yaml").read_text(encoding="utf-8")
    assert "llama-embed" in apps and "8083" in apps
    assert not (REPO_ROOT / "infra/k8s/apps/llama.yaml").exists()

    netpol = (REPO_ROOT / "infra/k8s/base/networkpolicies.yaml").read_text(encoding="utf-8")
    assert "llama-embed" in netpol and "8083" in netpol
    # 8082 is the help desk scheduler's port now; the chat model's own pod is what must be gone.
    assert "app.kubernetes.io/name: llama\n" not in netpol


def load_provider_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), path
    return document


def without_stated(document: dict[str, Any]) -> dict[str, Any]:
    """A provider file minus the two fields a copy is allowed to change."""
    return {k: v for k, v in document.items() if k not in {"enabled", "base_url"}}


def test_the_load_provider_files_match_the_shipped_ones() -> None:
    """Catches a copy that drifts: a price or model that differs from the shipped file,
    or a missing file. A copy may differ in `enabled` and `base_url` only. The load
    test's `local_embed` is held to that rule, and its `mock_chat` must be a provider
    file of the same shape (no shipped twin)."""
    shipped = {p.stem: load_provider_yaml(p) for p in sorted(SHIPPED_PROVIDERS.glob("*.yaml"))}
    assert set(shipped) == {"anthropic", "openai", "local_embed"}

    load = {p.stem: load_provider_yaml(p) for p in sorted(LOAD_PROVIDERS.glob("*.yaml"))}
    assert set(load) == {"mock_chat", "local_embed"}
    assert without_stated(load["local_embed"]) == without_stated(shipped["local_embed"])
    assert load["mock_chat"]["provider"] == "mock_chat"
    assert set(load["mock_chat"]) <= set(shipped["openai"])


#: The changed files that name paths, from the repo root.
PATH_FILES = (
    "backend/Dockerfile",
    "docker-compose.yml",
    "Makefile",
    "infra/k8s/kustomization.yaml",
    "infra/k8s/apps/kustomization.yaml",
    "infra/k8s/aws/kustomization.yaml",
    "infra/k8s/kind/kustomization.yaml",
    "infra/k8s/README.md",
    "README.md",
    "ARCHITECTURE.md",
    "docs/customizing.md",
    "docs/azure.md",
    "docs/aws-evidence/README.md",
    "docs/tradeoff-run/README.md",
    "web/README.md",
)

#: Paths that are named on purpose and do not exist in this repo: the companion
#: repo's files, private folders that never ship, and build outputs.
PATH_EXEMPT_PREFIXES = ("backend/evals/conversations/", "backend/app/evals/prompts/")

#: Named on purpose, written by a run or filled in by the owner, so absent from a clean checkout.
PATH_EXEMPT_EXACT = frozenset(
    {
        "backend/evals/labels/calibration.json",
        "docs/aws-evidence/cost.json",
        "docs/tradeoff-run/cost.json",
        "docs/tradeoff.json",
        "docs/tradeoff.md",
        "infra/load/results",
        "infra/terraform/aws/terraform.tfvars",
        "infra/terraform/aws-gpu/terraform.tfvars",
    }
)

PATH_TOKEN = re.compile(r"(?<![\w/.-])((?:backend|infra|docs|web)/[\w./-]*[\w/])")


def named_paths(text: str) -> set[str]:
    found = set()
    for token in PATH_TOKEN.findall(text):
        if any(mark in token for mark in ("*", "<", "$", "{")):
            continue
        if token.startswith(PATH_EXEMPT_PREFIXES) or token.rstrip("/.") in PATH_EXEMPT_EXACT:
            continue
        found.add(token.rstrip("/."))
    return found


def git_ignored(paths: list[str]) -> set[str]:
    """The given repo-relative paths that git ignores, whether or not they exist. Each
    is asked about as a file and as a folder, since a folder rule (`scratch/`) only
    matches a path git knows is a folder."""
    if not paths:
        return set()
    done = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "--stdin"],
        input="\n".join(form for path in paths for form in (path, path.rstrip("/") + "/")),
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.rstrip("/") for line in done.stdout.split()}


def test_every_path_the_changed_files_name_exists() -> None:
    """Catches a doc or build file that still names `backend/config/prices.yaml`,
    `infra/k8s/apps/llama.yaml` or any other file that was deleted or moved, and a
    new path (the provider folders) that was written with a typo."""
    absent: list[tuple[str, str]] = []
    for file_name in PATH_FILES:
        path = REPO_ROOT / file_name
        assert path.exists(), file_name
        for named in sorted(named_paths(path.read_text(encoding="utf-8"))):
            if not (REPO_ROOT / named).exists():
                absent.append((file_name, named))
    # A path git ignores (an env file, a scratch folder) is never committed, so it
    # is absent from every fresh clone without being a broken reference.
    ignored = git_ignored([named for _, named in absent])
    missing = [f"{file_name}: {named}" for file_name, named in absent if named not in ignored]
    assert missing == []

    readme = (REPO_ROOT / "docs/customizing.md").read_text(encoding="utf-8")
    assert "backend/config/providers" in readme
