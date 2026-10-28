r"""`infra/scripts/tradeoff_policy.py` and `infra/tradeoff/`: the per-option effective policy.

The measurement run resolves one alias three different ways. Rather than three full
copies of `routing.yaml` that drift silently, each option ships a PARTIAL overlay that is
deep-merged over the shipped file. Two things can go wrong quietly and are pinned here.

The merge. A `dict.update` at the top level would drop every tenant the overlay does not
mention; a merge that EXTENDS lists would leave the shipped three candidates in place and the
option would measure whichever provider answers first. So the merge is exercised twice: once as
a pure function over synthetic mappings whose siblings are all observable, and once end to end
over the real `routing.yaml` and the three real overlays, each result loaded through
`app.model_routing.policy.load_policy`.

The pre-check. The option id names the overlay file AND the evidence directory, so it is a
store-shaping input and is checked with `re.fullmatch` against
`app.evals.tradeoff.OPTION_ID_PATTERN` before anything is read or written (standing rule
4: `^...$` with `.match` accepts a trailing newline, which is exactly the `cheap-api\n` row
below). A refusal and a missing overlay BOTH exit 2, so the two are told apart by the message:
a pattern refusal prints the pattern, a missing overlay prints the path it looked for. That is
what makes the boundary rows (1 and 32 characters accepted, 33 refused) observable at all.

The script lives in `infra/scripts/`, which is not a package, so its pure functions are loaded by
file
path with `importlib.util` - the same technique `test_eval_target_stub.py` uses for the stub.
Exit codes go through `subprocess`; merge semantics go through the import. No backing service,
no network: the three end-to-end runs are made once for the module.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml

from app.evals.tradeoff import OPTION_ID_PATTERN
from app.model_routing.policy import load_policy

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "infra" / "scripts" / "tradeoff_policy.py"
OVERLAY_DIR = REPO_ROOT / "infra" / "tradeoff"
ROUTING = REPO_ROOT / "backend" / "config" / "routing.yaml"
GPU_VLLM_MANIFEST = REPO_ROOT / "infra" / "k8s" / "gpu" / "vllm.yaml"
COMPOSE_OVERLAY = "compose.yml"

# The secret scan's three patterns and its reader, kept here rather than imported:
# they live in `infra/tests/infra/test_hetzner_module.py`, a tree a backend test
# cannot import (pinned by `tests/test_test_tree.py`). Keep the two copies equal.
SECRETISH_NAME = re.compile(
    r"[A-Za-z0-9_.-]*(token|secret|password|api_key)[A-Za-z0-9_.-]*", re.IGNORECASE
)
TOKEN_LITERAL = re.compile(r"[A-Za-z0-9+/=_-]{20,}")
ASSIGNMENT = re.compile(r"""([A-Za-z0-9_.-]+)\s*[:=]\s*["']([^"'\n]*)["']""")


def read(path: Path) -> str:
    """Read a file the case needs, failing this case alone when it is absent."""
    assert path.is_file(), f"{path} does not exist"
    return path.read_text(encoding="utf-8")


#: The three overlays, in the order the spec lists them, with the ONE candidate each pins.
#: Provider and model are required TOGETHER: a file naming the right
#: provider with the wrong model is the drift this table exists to see.
OPTIONS: tuple[tuple[str, dict[str, str]], ...] = (
    ("cheap-api", {"provider": "anthropic", "model": "claude-haiku-5-5"}),
    ("strong-api", {"provider": "anthropic", "model": "claude-sonnet-5-5"}),
    ("gpu-vllm", {"provider": "vllm", "model": "qwen2.5-7b-instruct"}),
)

OPTION_IDS = tuple(option_id for option_id, _candidate in OPTIONS)

#: The two options whose cost comes from the ledger.
API_OPTIONS = ("cheap-api", "strong-api")

#: The requirement's URL, written out literally; its port is compared to the manifest's nodePort.
VLLM_BASE_URL = "http://host.docker.internal:30800/v1"

#: The alias every overlay narrows, and the tenant the run is made as.
ALIAS = "agent-default"
TENANT = "evals"

#: Ids that are well-formed but have no overlay: the two inclusive boundary rows.
SHORTEST_ID = "a"
LONGEST_ID = "a" + "b" * 31

#: A well-formed id whose overlay will never exist.
ABSENT_ID = "no-such-option"

#: Literals no overlay may carry. Each is checked against the overlays' own
#: required content first: no candidate or URL above contains them.
FORBIDDEN_LITERALS = ("key_env", "Bearer", "sk-")


def _load_script() -> ModuleType:
    """Load `tradeoff_policy.py` by path: `infra/scripts/` carries no `__init__.py`."""
    assert SCRIPT.is_file(), f"{SCRIPT} does not exist"
    spec = importlib.util.spec_from_file_location("tradeoff_policy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load_script()


def providers_of(output: Path) -> Path:
    """The provider folder `build` writes for the policy at `output`."""
    return output.with_name(f"{output.stem}-providers")


def build(option_id: str, output: Path) -> subprocess.CompletedProcess[str]:
    """Run the script for `option_id`, writing the policy to `output` and its provider
    folder to `providers_of(output)`. Never raises on a non-zero exit."""
    assert SCRIPT.is_file(), f"{SCRIPT} does not exist"
    return subprocess.run(
        [sys.executable, str(SCRIPT), option_id, str(output), str(providers_of(output))],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def said(completed: subprocess.CompletedProcess[str]) -> str:
    """Everything the run printed, so a message on either stream is found."""
    return completed.stdout + completed.stderr


def overlay_path(option_id: str) -> Path:
    return OVERLAY_DIR / f"{option_id}.yaml"


def overlay(option_id: str) -> dict[str, Any]:
    document = yaml.safe_load(read(overlay_path(option_id)))
    assert isinstance(document, dict), f"{option_id}.yaml is not a mapping"
    return document


def alias_of(document: dict[str, Any]) -> dict[str, Any]:
    """The overlay's or the effective policy's `aliases.agent-default` mapping."""
    aliases = document.get("aliases")
    assert isinstance(aliases, dict), f"no aliases mapping: {sorted(document)}"
    entry = aliases.get(ALIAS)
    assert isinstance(entry, dict), f"no aliases.{ALIAS} mapping: {sorted(aliases)}"
    return entry


def node_port() -> int:
    """The one `nodePort` the gpu build publishes, read from every Service in the manifest.

    Every document is scanned and the name is required alongside the port:
    "the first port entry with a nodePort" would read a second Service added later.
    """
    found: list[tuple[str, int]] = []
    for document in yaml.safe_load_all(read(GPU_VLLM_MANIFEST)):
        if not isinstance(document, dict) or document.get("kind") != "Service":
            continue
        name = str(document.get("metadata", {}).get("name"))
        for entry in document.get("spec", {}).get("ports", []) or []:
            if isinstance(entry, dict) and entry.get("nodePort") is not None:
                found.append((name, int(entry["nodePort"])))
    assert len(found) == 1, f"{len(found)} nodePorts in {GPU_VLLM_MANIFEST.name}: {found}"
    assert found[0][0] == "vllm-node", f"the nodePort is on Service {found[0][0]}"
    return found[0][1]


@pytest.fixture(scope="module")
def effective(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One successful build per option, made ONCE for the module."""
    directory = tmp_path_factory.mktemp("effective")
    written: dict[str, Path] = {}
    for option_id in OPTION_IDS:
        output = directory / f"{option_id}.yaml"
        completed = build(option_id, output)
        assert completed.returncode == 0, (
            f"{option_id}: exit {completed.returncode}\n{said(completed)}"
        )
        written[option_id] = output
    return written


# ---------------------------------------------------------------------------------------------
# The option id pre-check


@pytest.mark.parametrize(
    "option_id",
    [
        pytest.param("1cheap", id="leading-digit"),
        pytest.param("Cheap-api", id="uppercase"),
        pytest.param("cheap_api", id="underscore"),
        pytest.param("", id="empty"),
        pytest.param("cheap-api\n", id="trailing-newline"),
        pytest.param("cheap-api\r\n", id="crlf"),
        pytest.param("cheap-api ", id="trailing-space"),
        pytest.param("../a", id="parent-traversal"),
        pytest.param("a" + "b" * 32, id="33-characters"),
    ],
)
def test_a_malformed_option_id_is_refused_before_any_file_is_touched(
    tmp_path: Path, option_id: str
) -> None:
    """The refused rows. Catches a check written as `re.match(r"^...$")`,
    which accepts the trailing-newline and CRLF rows and then happily merges `cheap-api`'s real
    overlay and writes an output file; a `{0,32}` bound, which accepts the 33-character row; a
    class holding `A-Z` or `_`; and no check at all, which reaches the filesystem with `../a`
    and reports the missing PATH instead of the pattern."""
    output = tmp_path / "effective.yaml"

    completed = build(option_id, output)

    assert completed.returncode == 2, f"{option_id!r} exited {completed.returncode}"
    assert not output.exists(), f"{option_id!r} wrote {output}"
    assert OPTION_ID_PATTERN.pattern in said(completed), (
        f"{option_id!r} was not refused by the pattern check: {said(completed)}"
    )


@pytest.mark.parametrize(
    "option_id",
    [pytest.param(SHORTEST_ID, id="one-character"), pytest.param(LONGEST_ID, id="32-characters")],
)
def test_the_inclusive_length_boundaries_pass_the_pre_check(tmp_path: Path, option_id: str) -> None:
    """The requirement's accepted rows. Both ids are well-formed and have no overlay, so the only
    observable difference from a refusal is WHICH message the script prints: this passes when
    the run reached the filesystem and named the path it wanted. Catches a bound written as
    `{1,31}` (the 32-character row is then refused) or a pattern demanding two characters (the
    one-character row is), neither of which any other case can see."""
    output = tmp_path / "effective.yaml"

    completed = build(option_id, output)

    assert completed.returncode == 2, f"{option_id!r} exited {completed.returncode}"
    assert not output.exists(), f"{option_id!r} wrote {output}"
    assert OPTION_ID_PATTERN.pattern not in said(completed), (
        f"{option_id!r} was refused by the pattern: {said(completed)}"
    )
    assert f"{option_id}.yaml" in said(completed), (
        f"{option_id!r} did not name the overlay it looked for: {said(completed)}"
    )


def test_a_well_formed_id_with_no_overlay_names_the_path_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """Catches a script that treats a missing overlay as an empty one and writes
    `routing.yaml` back out unchanged - the run would then measure the shipped three-candidate
    alias under the option's name, which is a wrong MEASUREMENT rather than a crash. Catches an
    output file opened for writing before the overlay is read, too."""
    output = tmp_path / "effective.yaml"

    completed = build(ABSENT_ID, output)

    assert completed.returncode == 2, f"exit {completed.returncode}: {said(completed)}"
    assert not output.exists(), f"the run wrote {output}"
    assert str(overlay_path(ABSENT_ID)) in said(completed) or f"{ABSENT_ID}.yaml" in said(
        completed
    ), f"the message does not name the overlay path: {said(completed)}"


# ---------------------------------------------------------------------------------------------
# The merge, as a pure function


def test_the_merge_is_recursive_over_mappings(builder: ModuleType) -> None:
    """The first half of the merge rule. The overlay names ONE leaf three levels down;
    catches a top-level `dict.update` (which drops `nordlicht` and `budget_usd_month` whole),
    a merge that replaces the whole `data_class_rules` mapping (which drops `pii`, so the
    tenant's strictest rule silently disappears), and one that replaces `confidential` whole
    (which drops `redact: true` and sends unredacted text to a provider)."""
    base = {
        "tenants": {
            "nordlicht": {"budget_usd_month": "50.00"},
            "evals": {
                "budget_usd_month": "10.00",
                "data_class_rules": {
                    "confidential": {"redact": True, "note": "a"},
                    "pii": {"redact": True},
                },
            },
        }
    }
    patch = {"tenants": {"evals": {"data_class_rules": {"confidential": {"note": "b"}}}}}

    merged = builder.deep_merge(base, patch)
    evals = merged["tenants"]["evals"]

    assert evals["data_class_rules"]["confidential"]["note"] == "b"
    assert evals["data_class_rules"]["confidential"]["redact"] is True
    assert evals["data_class_rules"]["pii"] == {"redact": True}
    assert evals["budget_usd_month"] == "10.00"
    assert merged["tenants"]["nordlicht"] == {"budget_usd_month": "50.00"}


def test_the_merge_replaces_a_list_whole(builder: ModuleType) -> None:
    """The requirement's second half. Catches a merge that EXTENDS lists: the effective alias would
    then carry the overlay's candidate plus the shipped three, the resolver would answer from
    whichever is first, and every option would measure the same provider."""
    base = {"aliases": {ALIAS: {"candidates": [{"provider": "a"}, {"provider": "b"}]}}}
    patch = {"aliases": {ALIAS: {"candidates": [{"provider": "c"}]}}}

    merged = builder.deep_merge(base, patch)

    assert merged["aliases"][ALIAS]["candidates"] == [{"provider": "c"}]


def test_the_merge_does_not_mutate_the_shipped_document(builder: ModuleType) -> None:
    """The requirement's "the shipped file never edited", at the object level. Catches a merge
    written as an in-place walk over `base`: the write-back would then be correct by accident
    and any second merge in the same process would start from the first one's result."""
    base = {"aliases": {ALIAS: {"semantic_cache": True, "candidates": [{"provider": "a"}]}}}
    patch = {"aliases": {ALIAS: {"semantic_cache": False, "candidates": [{"provider": "c"}]}}}

    builder.deep_merge(base, patch)

    assert base == {"aliases": {ALIAS: {"semantic_cache": True, "candidates": [{"provider": "a"}]}}}


# ---------------------------------------------------------------------------------------------
# The effective policies


def test_routing_yaml_is_byte_identical_after_a_merge(tmp_path: Path) -> None:
    """The requirement's "the shipped file never edited", at the file level. Catches a script that
    writes the merged document back over its input when the output path is defaulted or
    mis-parsed - the shipped policy would then carry one option's narrowing for every tenant."""
    before = ROUTING.read_bytes()

    completed = build(OPTION_IDS[0], tmp_path / "effective.yaml")

    assert completed.returncode == 0, said(completed)
    assert ROUTING.read_bytes() == before, f"{ROUTING} changed"


@pytest.mark.parametrize(("option_id", "candidate"), OPTIONS, ids=OPTION_IDS)
def test_each_effective_policy_loads_and_carries_exactly_one_candidate(
    effective: dict[str, Path], option_id: str, candidate: dict[str, str]
) -> None:
    """`load_policy` is the gateway's own reader, so a merge that
    produces valid YAML the resolver refuses is caught here and not four hours into a GPU run.
    Catches the shipped three candidates surviving the merge, the wrong provider/model pair for
    an option, `semantic_cache` left at the shipped `true` (option two would then
    read option one's answers and report a cost no provider charged), and a merge that loses the
    other aliases or tenants entirely."""
    policy = load_policy(effective[option_id], providers_of(effective[option_id]))
    document = yaml.safe_load(read(effective[option_id]))
    entry = alias_of(document)

    assert entry["candidates"] == [candidate]
    assert entry["semantic_cache"] is False
    assert {"judge", "embed"} <= set(document["aliases"]), sorted(document["aliases"])
    assert {"nordlicht", TENANT} <= set(document["tenants"]), sorted(document["tenants"])
    assert policy.aliases[ALIAS].semantic_cache is False


@pytest.mark.parametrize("option_id", API_OPTIONS)
def test_each_api_option_keeps_the_shipped_redaction_rules(
    effective: dict[str, Path], option_id: str
) -> None:
    """Catches an overlay that replaces the data class rules whole, which would drop
    `redact: true` and send unredacted text to a provider. The overlay says nothing about
    data classes, so the shipped rules must come through the merge unchanged."""
    document = yaml.safe_load(read(effective[option_id]))
    rules = document["tenants"][TENANT]["data_class_rules"]

    assert rules["confidential"] == {"redact": True}
    assert rules["pii"] == {"redact": True}
    assert rules["public"] == {"redact": False}


def test_the_gpu_option_enables_vllm_at_the_published_node_port(
    effective: dict[str, Path],
) -> None:
    """The gateway runs in a container and the model server behind a NodePort, so
    the two numbers live in two files and are compared here. Catches the option's own
    `vllm` provider file left out of its folder or switched off (every GPU turn would
    resolve to nothing), the container reaching `vllm.gpu.svc.cluster.local` which does not
    resolve from compose, and the URL and the manifest drifting apart - the run would then
    measure a connection refused."""
    provider = yaml.safe_load(read(providers_of(effective["gpu-vllm"]) / "vllm.yaml"))
    parsed = urlparse(str(provider["base_url"]))

    assert provider["enabled"] is True
    assert str(provider["base_url"]) == VLLM_BASE_URL
    assert parsed.port == node_port()


# ---------------------------------------------------------------------------------------------
# The overlay files themselves


def test_the_overlay_directory_holds_exactly_the_option_files() -> None:
    """Catches a fifth overlay left behind from a trial run, which `tradeoff.json`
    does not name and no test reads, and an option file named for an id the manifest never uses
    - the run would build a policy nobody measures."""
    assert OVERLAY_DIR.is_dir(), f"{OVERLAY_DIR} does not exist"
    # `compose.yml` is the run's Docker Compose overlay, kept with its kit, not a policy overlay.
    names = sorted(
        path.name
        for path in OVERLAY_DIR.iterdir()
        if path.suffix in {".yaml", ".yml"} and path.name != COMPOSE_OVERLAY
    )

    assert names == sorted(f"{option_id}.yaml" for option_id in OPTION_IDS), names


@pytest.mark.parametrize(("option_id", "candidate"), OPTIONS, ids=OPTION_IDS)
def test_each_overlay_pins_its_one_candidate_and_turns_the_cache_off(
    option_id: str, candidate: dict[str, str]
) -> None:
    """Read from the overlay rather than from the merged result. The merge
    could produce the right answer from a WRONG overlay if it appended defaults, so this is the
    independent half. Provider and model are required together: an overlay
    naming `anthropic` with `claude-haiku-5-5` under `strong-api` measures the cheap tier twice
    and the table's conclusion is then inverted."""
    entry = alias_of(overlay(option_id))

    assert entry["candidates"] == [candidate]
    assert entry["semantic_cache"] is False


def test_no_overlay_names_a_credential() -> None:
    """Catches a `key_env` redefined so an option reads a
    different environment variable (the run then silently uses the wrong account), an
    `Authorization: Bearer ...` header pasted in, and any token-shaped value assigned to a
    secret-shaped name. The name-anchored scan is deliberate: a bare
    20-character-run ban would redden on the overlays' own model and host names."""
    assert OVERLAY_DIR.is_dir(), f"{OVERLAY_DIR} does not exist"
    files = sorted(path for path in OVERLAY_DIR.iterdir() if path.suffix in {".yaml", ".yml"})
    assert files, f"{OVERLAY_DIR} holds no overlay"
    findings: list[str] = []
    for path in files:
        text = read(path)
        findings += [f"{path.name}: {literal}" for literal in FORBIDDEN_LITERALS if literal in text]
        findings += [
            f"{path.name}: {name}"
            for name, value in ASSIGNMENT.findall(text)
            if SECRETISH_NAME.fullmatch(name) and TOKEN_LITERAL.fullmatch(value)
        ]

    assert findings == [], f"credential-shaped content in the overlays: {findings}"
