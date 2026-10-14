r"""`.github/workflows/load.yml`: pinned
assertions in `infra/tests/ci/test_e2e_kind_workflow.py`'s own pattern - a manual-only
trigger, read-only permissions, `make load` as the one run, and the results
upload under `if: always()`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "load.yml"

RESULTS_PATH = "infra/load/results/"


def load(path: Path) -> dict[str, Any]:
    """Parse a YAML document that must be a mapping."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def trigger(document: dict[str, Any]) -> dict[str, Any]:
    """The `on:` block, found under either the string key or PyYAML's boolean `True`
    (`infra/tests/ci/test_e2e_kind_workflow.py`'s own `trigger` helper, copied rather
    than imported per a project rule). `workflow_dispatch:` with no body parses
    to `None`, so the block's values are not required to be mappings."""
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
def job(workflow: dict[str, Any]) -> dict[str, Any]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"load"}
    the_job = jobs["load"]
    assert isinstance(the_job, dict)
    return the_job


def test_the_workflow_triggers_on_manual_dispatch_only(
    workflow: dict[str, Any],
) -> None:
    """Criterion 20 and decision A: catches a `pull_request`, `push` or `schedule`
    trigger, each of which spends ten noisy runner minutes nobody asked for."""
    assert set(trigger(workflow)) == {"workflow_dispatch"}


def test_the_workflow_token_is_read_only(workflow: dict[str, Any]) -> None:
    """Input validation: catches a workflow granting more than `contents: read`,
    or a `packages: write` anywhere (a job-level block included)."""
    assert workflow["permissions"] == {"contents": "read"}
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"packages:\s*write", text) is None


def test_the_job_runs_exactly_make_load(job: dict[str, Any]) -> None:
    """Criterion 21: catches a workflow re-spelling the compose commands inline,
    which drifts from the recipe `infra/tests/load/test_load_compose.py` pins."""
    make_steps = [step for step in steps(job) if run_of(step).strip() == "make load"]
    assert len(make_steps) == 1


def test_the_results_are_uploaded_even_when_a_threshold_fails(
    job: dict[str, Any],
) -> None:
    """Criterion 21 and the unhappy path: catches an upload without `always()`
    (skipped exactly when k6 exits non-zero), the wrong path, or an upload placed
    before the run it is meant to collect."""
    listed = steps(job)
    uploads = [
        index
        for index, step in enumerate(listed)
        if "actions/upload-artifact" in uses_of(step)
    ]
    assert len(uploads) == 1
    upload = listed[uploads[0]]
    assert uses_of(upload) == "actions/upload-artifact@v4"
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == RESULTS_PATH
    runs = [
        index
        for index, step in enumerate(listed)
        if run_of(step).strip() == "make load"
    ]
    assert runs and max(runs) < uploads[0]
