r"""`.github/workflows/nightly-eval.yml`: SPEC line 344's documented backup path to the k3s
CronJob, an earlier change's #43.

This file does not exist yet. It is wired the SAME way `eval-gate.yml`'s `eval` job is
(`infra/tests/ci/test_eval_gate_workflow.py`'s own convention, which this module copies rather than
imports - a test module importing another test module makes the second one's collection a
dependency of the first's): `if: vars.EVAL_TARGET_URL != ''`, so it ships correct but inert
until a repository variable makes it real, exactly as `eval-gate.yml`'s own inert half does.

What is asserted is wiring and never behaviour - no target exists to run this workflow against,
so what a green run of it would print is not something this suite can observe. The document is
read with `yaml.safe_load` and the repository root is taken from this file's own `__file__`,
never from the working directory, for the same reason `test_eval_gate_workflow.py` gives.

One PyYAML quirk carried over from that file: the bare key `on` is YAML 1.1's boolean, so
`yaml.safe_load` returns the trigger under the key `True` rather than under `"on"`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly-eval.yml"


def load(path: Path) -> dict[str, Any]:
    """Parse a YAML document that must be a mapping."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def trigger(document: dict[str, Any]) -> dict[str, Any]:
    """The `on:` block, found under either the string key or PyYAML's boolean `True`."""
    for key in (True, "on"):
        if key in document:
            block = document[key]
            assert isinstance(block, dict), "the trigger block is not a mapping"
            return block
    raise AssertionError("the workflow declares no trigger")


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    """The job's steps, each a mapping."""
    listed = job["steps"]
    assert isinstance(listed, list)
    return [step for step in listed if isinstance(step, dict)]


def run_of(step: dict[str, Any]) -> str:
    """The step's `run` script, or the empty string for a `uses` step."""
    return str(step.get("run", ""))


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """The parsed workflow, read once for the module.

    A missing file is a failure of this fixture and therefore of every case, which is the
    honest report while the workflow does not exist yet.
    """
    return load(WORKFLOW)


@pytest.fixture(scope="module")
def jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    """The `jobs:` mapping."""
    listed = workflow["jobs"]
    assert isinstance(listed, dict)
    return listed


def test_the_workflow_triggers_on_a_schedule_and_manually(
    workflow: dict[str, Any],
) -> None:
    """Catches a workflow that only triggers on `workflow_dispatch:`, which is not a nightly
    job at all - a human has to remember to click it every night forever - and catches
    `schedule:` written with no `cron:` entry, which GitHub Actions accepts as valid YAML
    and then simply never runs. Both triggers are required: `workflow_dispatch` is what lets
    an operator run the backup path once by hand to check it still works between the nights
    nobody is watching."""
    block = trigger(workflow)
    assert "schedule" in block, "the workflow declares no schedule trigger"
    assert "workflow_dispatch" in block, "the workflow cannot be run by hand"
    crons = [str(entry.get("cron", "")) for entry in block["schedule"] if isinstance(entry, dict)]
    assert any(crons), "the schedule trigger names no cron expression"


def test_the_workflow_carries_exactly_one_job(jobs: dict[str, Any]) -> None:
    """Catches a second job nobody agreed to: this workflow's whole purpose is one inert-until-
    configured eval run, and a required-check-by-name mistake here is `eval-gate.yml`'s own
    decision E risk repeated on a workflow that runs unattended at night."""
    assert set(jobs) == {"nightly-eval"}


def test_the_job_runs_only_when_a_target_url_is_configured(
    jobs: dict[str, Any],
) -> None:
    """Mirrors `eval-gate.yml`'s decision A exactly, per this plan's own acceptance criteria
    ("gated the same way `eval-gate.yml` is"). Catches the condition dropped, which would
    fail every scheduled run on a compose stack with nothing to evaluate against, since no
    prod target exists yet."""
    assert "EVAL_TARGET_URL" in str(jobs["nightly-eval"]["if"])


def test_the_job_invokes_ops_eval_run_with_the_nightly_run_kind(
    jobs: dict[str, Any],
) -> None:
    """Catches a copy-pasted `eval-gate.yml` step left at `--run-kind ci`, which would tag
    every nightly row in `eval_runs` as a PR run and make the alert's own 7-day window read a
    series mixing two different populations. `--run-kind nightly` is what `NightlyEvalRegression`
    and its `quantile_over_time` window are read against."""
    scripts = [run_of(step) for step in steps(jobs["nightly-eval"])]
    matches = [
        script for script in scripts if "ops-eval run" in script and "--run-kind nightly" in script
    ]
    assert len(matches) == 1, f"{len(matches)} steps run `ops-eval run --run-kind nightly`"


def test_the_job_passes_a_pushgateway_url(jobs: dict[str, Any]) -> None:
    """Criterion 1 end to end at the CI wiring layer: a nightly run with no `--pushgateway-url`
    never feeds `NightlyEvalRegression` anything, and the alert stays permanently pending on
    a manifest that is otherwise correct. Catches the flag dropped from the one step that
    matters even though the run itself is wired correctly in every other respect."""
    scripts = [run_of(step) for step in steps(jobs["nightly-eval"])]
    matches = [script for script in scripts if "--pushgateway-url" in script]
    assert len(matches) == 1, f"{len(matches)} steps pass --pushgateway-url"
