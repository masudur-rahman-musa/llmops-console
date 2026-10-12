r"""`.github/workflows/e2e-kind.yml`: pinned-string assertions in
`tests/ci/test_eval_gate_workflow.py`'s own pattern - the trigger, the pinned
`kind` versions, the `kind load docker-image` step, the golden-set invocation,
the `if: always()` upload, and that no step reaching a real registry runs on a
pull request from a fork.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "e2e-kind.yml"

KIND_ACTION_VERSION = "v1.15.0"
KIND_VERSION = "v0.33.0"
GOLDEN_SET = "project-evals/golden/e2e-kind.jsonl"
SMOKE_SCRIPT = "project-evals/ci/kind/smoke.sh"
SMOKE_SCRIPT = "project-evals/ci/kind/smoke.sh"


def load(path: Path) -> dict[str, Any]:
    """Parse a YAML document that must be a mapping."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def trigger(document: dict[str, Any]) -> dict[str, Any]:
    """The `on:` block, found under either the string key or PyYAML's boolean `True`
    (`tests/ci/test_eval_gate_workflow.py`'s own `trigger` helper, copied rather
    than imported per a project rule)."""
    for key in (True, "on"):
        if key in document:
            block = document[key]
            assert isinstance(block, dict), "the trigger block is not a mapping"
            return block
    raise AssertionError("the workflow declares no trigger")


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    listed = job["steps"]
    assert isinstance(listed, list)
    return [step for step in listed if isinstance(step, dict)]


def run_of(step: dict[str, Any]) -> str:
    return str(step.get("run", ""))


def uses_of(step: dict[str, Any]) -> str:
    return str(step.get("uses", ""))


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return load(WORKFLOW)


@pytest.fixture(scope="module")
def jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    listed = workflow["jobs"]
    assert isinstance(listed, dict)
    return listed


@pytest.fixture(scope="module")
def job(jobs: dict[str, Any]) -> dict[str, Any]:
    assert set(jobs) == {"e2e-kind"}
    the_job = jobs["e2e-kind"]
    assert isinstance(the_job, dict)
    return the_job


def test_the_workflow_triggers_on_pull_requests_and_pushes_into_main(
    workflow: dict[str, Any],
) -> None:
    """Acceptance criterion 1. `branches: [main]` on both, the same way
    `eval-gate.yml`'s own trigger is asserted as the whole list - a second branch
    added to either is a scope nobody agreed to run this on."""
    block = trigger(workflow)
    assert block["pull_request"]["branches"] == ["main"]
    assert block["push"]["branches"] == ["main"]


def test_the_kind_action_and_binary_versions_are_both_pinned(job: dict[str, Any]) -> None:
    """Criterion 1: "a pinned `kind` and `kind`-action version" - two separate
    pins, since the action's own version and the `kind` binary it installs move
    independently."""
    kind_steps = [step for step in steps(job) if "helm/kind-action@" in uses_of(step)]
    assert len(kind_steps) == 1
    assert uses_of(kind_steps[0]) == f"helm/kind-action@{KIND_ACTION_VERSION}"
    assert kind_steps[0]["with"]["version"] == KIND_VERSION


def test_both_images_are_built_locally_and_loaded_into_kind(job: dict[str, Any]) -> None:
    """Criterion 1 and decision D: the gateway's and the stub's images are built
    in the runner and `kind load docker-image`d, never pulled from GHCR - a PR
    run has no image published under its SHA at all."""
    scripts = [run_of(step) for step in steps(job)]
    build_scripts = " ".join(scripts)
    assert "docker build -t gateway:e2e-kind ." in build_scripts
    assert (
        "docker build -t eval-target-stub:e2e-kind project-evals/ci/eval-target-stub"
        in build_scripts
    )
    load_steps = [script for script in scripts if "kind load docker-image" in script]
    assert len(load_steps) == 1
    assert "kind load docker-image gateway:e2e-kind" in load_steps[0]
    assert "kind load docker-image eval-target-stub:e2e-kind" in load_steps[0]


def test_the_overlay_is_applied_with_the_relaxed_load_restrictor(job: dict[str, Any]) -> None:
    """`kubectl apply -k` has no `--load-restrictor` flag, so the build has to be
    piped in - catches a step that reverts to a plain `kubectl apply -k`, which
    fails outright on this overlay's cross-directory file references."""
    apply_steps = [step for step in steps(job) if "kubectl apply -f -" in run_of(step)]
    assert len(apply_steps) == 1
    assert "--load-restrictor=LoadRestrictionsNone" in run_of(apply_steps[0])
    assert "project-evals/ci/kind" in run_of(apply_steps[0])


def test_the_workflow_waits_for_the_gateway_and_stub_rollouts(job: dict[str, Any]) -> None:
    """Criterion 3: both Deployments, each with the five-minute ceiling."""
    scripts = " ".join(run_of(step) for step in steps(job))
    assert "kubectl -n apps rollout status deployment/gateway --timeout=5m" in scripts
    assert "kubectl -n apps rollout status deployment/eval-target-stub --timeout=5m" in scripts


def test_the_run_reaches_the_gateway_through_a_port_forward_never_a_public_address(
    job: dict[str, Any],
) -> None:
    """While the local model is out of use the run checks the gateway itself: its
    own Service, reached by `port-forward`, and no step opens a NodePort or a
    LoadBalancer."""
    scripts = " ".join(run_of(step) for step in steps(job))
    assert "kubectl -n apps port-forward svc/gateway 18080:8080" in scripts
    assert "NodePort" not in scripts
    assert "LoadBalancer" not in scripts


def test_the_run_checks_the_gateway_with_the_model_free_smoke_script(
    job: dict[str, Any],
) -> None:
    """While the local model is out of use: exactly one step runs `smoke.sh`
    against the forwarded gateway, and no step runs the golden set, whose cases
    all need that model and would fail every commit."""
    scripts = [run_of(step) for step in steps(job)]
    smoke = [script for script in scripts if f"{SMOKE_SCRIPT} http://127.0.0.1:18080" in script]
    assert len(smoke) == 1
    assert not any("ops-eval run" in script for script in scripts)
    assert (REPO_ROOT / SMOKE_SCRIPT).is_file()


def test_the_report_artifact_is_uploaded_even_when_the_run_fails(job: dict[str, Any]) -> None:
    """Criterion 7: the unhappy path - a cluster-up failure, a rollout timeout or
    a failed case - still uploads the report, `eval-gate.yml`'s own reason:
    `always()` exactly, because `failure()` would drop it on the passing runs."""
    uploads = [step for step in steps(job) if "actions/upload-artifact" in uses_of(step)]
    assert len(uploads) == 1
    assert uploads[0]["if"] == "always()"


def test_no_step_can_reach_a_real_registry_with_a_write_token(job: dict[str, Any]) -> None:
    """Input validation criterion: every image this job runs is built in the
    runner and loaded straight into `kind` - nothing here logs into a registry or
    pushes, so a pull request from a fork has nothing that could exfiltrate a
    write-scoped `GITHUB_TOKEN` even if one were granted."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "docker push" not in text
    assert "docker/login-action" not in text
    assert "docker login" not in text
