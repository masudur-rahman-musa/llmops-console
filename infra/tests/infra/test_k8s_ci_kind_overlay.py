r"""`infra/k8s/kind/`: the kind e2e's own kustomization.

Its own `built` fixture rather than `infra/tests/infra/conftest.py`'s: that one builds
`infra/k8s` (the production root), and this overlay is a second, INDEPENDENT root
kustomize does not let escape its own directory without
`--load-restrictor=LoadRestrictionsNone` (it selects individual files out of
`infra/k8s/apps` and `infra/k8s/platform` rather than each directory whole, which
is what trims Helpdesk, Ingress and the two CRD-backed platform files out at all -
a project rule says a shared helper is copied rather than imported across test
modules, and the two builds do not even share a root to make importing tempting).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_KIND_ROOT = REPO_ROOT / "infra" / "k8s" / "kind"
ROUTING_POLICY = REPO_ROOT / "backend" / "config" / "routing.yaml"

#: Criterion 2: the object set this overlay must contain, by `(kind, name, namespace)`,
#: read as a strict SUBSET check (not equality) - `base`'s four `Namespace`s, four
#: `ResourceQuota`s, four `LimitRange`s and every platform/apps `NetworkPolicy` also
#: ride along unavoidably (base is used whole, per the plan's own wording), and this
#: overlay only has to prove what it ADDS is exactly these, not that base emits
#: nothing else. `test_no_excluded_kind_or_name_appears` is the equality half, over
#: the excluded set instead.
REQUIRED_OBJECTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("Deployment", "gateway", "apps"),
        ("Service", "gateway", "apps"),
        ("Deployment", "eval-target-stub", "apps"),
        ("Service", "eval-target-stub", "apps"),
        ("Secret", "gateway-secrets", "apps"),
        ("Secret", "eval-target-stub-secrets", "apps"),
        ("Service", "postgres", "platform"),
        ("Service", "redis", "platform"),
        ("Service", "qdrant", "platform"),
    }
)

#: Criterion 2's other half: names no CI overlay may emit, whatever their kind -
#: nothing here can pull the four Helpdesk images or `web`, and kind
#: has no controller for either CRD kind.
EXCLUDED_NAMES: frozenset[str] = frozenset(
    {
        "helpdesk-agent-api",
        "helpdesk-mcp-server",
        "helpdesk-scheduler",
        "helpdesk-web",
        "helpdesk-config",
        "helpdesk-secrets",
        "helpdesk",
        "helpdesk-healthz",
        "gateway-healthz",
        "basic-auth",
        "rate-limit",
        "security-headers",
        "letsencrypt-staging",
        "letsencrypt-prod",
    }
)

EXCLUDED_KINDS: frozenset[str] = frozenset({"Ingress", "Middleware", "ClusterIssuer", "CronJob"})

GATEWAY_IMAGE_TAG = "e2e-kind"
STUB_IMAGE_TAG = "e2e-kind"
LLAMA_IMAGE_REPOSITORY = "ghcr.io/ggml-org/llama.cpp"
LLAMA_IMAGE_TAG = "server-b10902"

#: The exact upstream file and its published git-LFS `oid sha256`, resolved from
#: `Qwen/Qwen2.5-3B-Instruct-GGUF` at implementation time - copied here rather than
#: parsed out of the patch's inline Python (a project rule: both files say the
#: value, so the two cannot drift silently).
MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/"
    "resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
)
MODEL_SHA256 = "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"


@pytest.fixture(scope="module")
def built() -> list[dict[str, Any]]:
    """`kubectl kustomize` over `infra/k8s/kind`, once for the module.

    `LoadRestrictionsNone` is what lets this overlay select individual files out
    of `infra/k8s/apps` and `infra/k8s/platform` at all - the workflow builds with
    the identical flag, pinned by `infra/tests/ci/test_e2e_kind_workflow.py`, so the two
    cannot disagree about how this directory is meant to be built."""
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl is not on PATH; CI's `k8s` job is the copy that cannot skip")
    completed = subprocess.run(
        [
            "kubectl",
            "kustomize",
            "--load-restrictor=LoadRestrictionsNone",
            str(CI_KIND_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"`kubectl kustomize {CI_KIND_ROOT}` exited {completed.returncode}:\n{completed.stderr}"
    )
    documents = [
        document
        for document in yaml.safe_load_all(completed.stdout)
        if isinstance(document, dict) and document
    ]
    assert documents, f"`kubectl kustomize {CI_KIND_ROOT}` emitted no objects"
    return documents


def _objects(documents: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (
            document["kind"],
            document["metadata"]["name"],
            document["metadata"].get("namespace", ""),
        )
        for document in documents
    }


def _deployment(documents: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [
        document
        for document in documents
        if document["kind"] == "Deployment" and document["metadata"]["name"] == name
    ]
    assert len(matches) == 1, f"expected exactly one Deployment named {name!r}"
    return matches[0]


def test_the_overlay_contains_every_required_object(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 2's inclusion half: the gateway, both llama.cpp servers and the
    stub target, each with its Service, plus the two Secrets and `platform`'s
    three data services - a subset check, since `base` rides along whole."""
    assert _objects(built) >= REQUIRED_OBJECTS


def test_no_excluded_kind_or_name_appears(built: list[dict[str, Any]]) -> None:
    """Criterion 2's exclusion half: no Helpdesk object, no Ingress, no Traefik
    Middleware, no cert-manager ClusterIssuer, no CronJob - a build containing any
    of these is one that would fail `kubectl apply` on a bare kind cluster with
    neither controller installed, or reach for an image nothing publishes."""
    names = {document["metadata"]["name"] for document in built}
    kinds = {document["kind"] for document in built}
    assert names.isdisjoint(EXCLUDED_NAMES)
    assert kinds.isdisjoint(EXCLUDED_KINDS)


def test_no_secret_reference_names_helpdesk_secrets(
    built: list[dict[str, Any]],
) -> None:
    """The same claim from the other direction: nothing built references
    `helpdesk-secrets` in an `envFrom`, which would fail the rollout wait on a
    Secret this overlay never supplies."""
    text = yaml.safe_dump_all(built)
    assert "helpdesk-secrets" not in text


def test_the_gateway_image_is_the_locally_built_ci_tag(
    built: list[dict[str, Any]],
) -> None:
    """Decision D: the gateway's image is what THIS workflow builds and
    `kind load docker-image`s, never a GHCR reference a PR run cannot publish to."""
    container = _deployment(built, "gateway")["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == f"gateway:{GATEWAY_IMAGE_TAG}"


def test_the_stub_image_is_the_locally_built_ci_tag(
    built: list[dict[str, Any]],
) -> None:
    """The stub has no production counterpart to copy a tag from at all - this
    overlay is the only place its image is ever named."""
    container = _deployment(built, "eval-target-stub")["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == f"eval-target-stub:{STUB_IMAGE_TAG}"


def test_no_llama_server_is_deployed_while_the_local_model_is_out_of_use(
    built: list[dict[str, Any]],
) -> None:
    """2026-10-12: the local model is out of use, and on a 2-CPU runner its weight
    never finished downloading before the run needed it. Catches either llama.cpp
    server creeping back into this build without the golden set that needs it."""
    names = {document["metadata"]["name"] for document in built}
    assert names.isdisjoint({"llama", "llama-embed", "llama-models", "llama-embed-models"})


def test_the_kept_model_fetch_patch_still_verifies_a_checksum() -> None:
    """`llama-model-init-patch.yaml` stays in this directory for the day the local
    model returns. Read directly, since nothing builds it now: it must still name
    the exact upstream URL and its sha256 and actually compare them (`hashlib` and
    `!=`), honouring `infra/k8s/README.md`'s no-unverified-download rule."""
    patch = (CI_KIND_ROOT / "llama-model-init-patch.yaml").read_text(encoding="utf-8")
    assert MODEL_URL in patch.replace("\n", "")
    assert MODEL_SHA256 in patch
    assert "hashlib" in patch
    assert "!=" in patch


def test_the_ci_secret_carries_no_provider_key(built: list[dict[str, Any]]) -> None:
    """The CI `gateway-secrets` carries only the tenant key digest - no
    `ANTHROPIC_API_KEY` or any other provider key, which is never what keeps this
    tenant off a non-local provider (the `evals` tenant's own `confidential` rule
    already does that), but naming one here would still be a live-looking
    credential in a public repo for no reason at all."""
    matches = [
        document
        for document in built
        if document["kind"] == "Secret" and document["metadata"]["name"] == "gateway-secrets"
    ]
    assert len(matches) == 1
    keys = set(matches[0].get("stringData", {})) | set(matches[0].get("data", {}))
    assert keys == {"OPSCONSOLE_API_KEYS"}


def test_the_alias_and_tenant_resolve_to_the_local_provider_only() -> None:
    """The routing-policy half of `main.py`'s own claim: with no `X-Data-Class`
    header (default `confidential`), the shipped `evals` tenant's own rule already
    narrows `agent-default` to the `self` region alone, because `anthropic_eu`
    ships disabled and `anthropic`'s region (`us`) is not in `[eu, self]` -
    independent of whether any provider key is configured."""
    from app.model_routing.policy import DataClass, load_policy
    from app.model_routing.resolver import resolve

    policy = load_policy(ROUTING_POLICY)
    resolved = resolve(
        policy,
        tenant="evals",
        alias="agent-default",
        data_class=DataClass.CONFIDENTIAL.value,
    )
    assert {candidate.provider for candidate in resolved} == {"local"}
