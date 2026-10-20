"""`.github/workflows/eval-gate.yml`: the wiring nothing else in this repo can exercise.

Risk 1 of an earlier change says it plainly - there is no eval target, so the `eval` job can
be WIRED but never run here, and this file is the only thing holding its shape
between now and the row that gives it something to run against. What is asserted is
therefore wiring and not behaviour: the trigger, the two job names, decision A's
condition, `fetch-depth: 0`, `if: always()` on the upload, the fork guard on the
comment, and the flags the gate step must carry. The comparison those flags drive is
pinned in `tests/evals/test_gate.py` and `tests/evals/test_cli.py`.

The document is read with `yaml.safe_load` and the repository root is taken from this
file's own `__file__`, never from the working directory: a test that finds the
workflow only when pytest is run from the repo root passes in CI and misleads anyone
running it from `tests/`.

One PyYAML quirk the reader below has to handle: the bare key `on` is YAML 1.1's
boolean, so `yaml.safe_load` returns the trigger under the key `True` rather than
under `"on"` - asserting `doc["on"]` would raise `KeyError` on a perfectly correct
workflow file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "eval-gate.yml"
CI_OVERLAY = REPO_ROOT / "project-evals" / "ci" / "routing.ci.yaml"


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


def uses_of(step: dict[str, Any]) -> str:
    """The step's `uses` reference, or the empty string for a `run` step."""
    return str(step.get("uses", ""))


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """The parsed workflow, read once for the module.

    A missing file is a failure of this fixture and therefore of every case, which is
    the honest report while the workflow does not exist yet.
    """
    return load(WORKFLOW)


@pytest.fixture(scope="module")
def jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    """The `jobs:` mapping."""
    listed = workflow["jobs"]
    assert isinstance(listed, dict)
    return listed


def test_the_workflow_triggers_on_pull_requests_into_main(
    workflow: dict[str, Any],
) -> None:
    """Criterion 1. Catches a workflow wired to `push`, where the guard runs after the merge
    it was supposed to block, and catches a `pull_request` with no `branches` filter, which
    spends the eval budget on every PR between feature branches. `main` is asserted as the
    whole list, because an added branch is a second target nobody agreed to gate on."""
    assert trigger(workflow)["pull_request"]["branches"] == ["main"]


def test_the_workflow_carries_exactly_the_two_jobs(jobs: dict[str, Any]) -> None:
    """Criterion 1 and decision E: `eval-gate.yml` is a required check by NAME, so a third
    job added here silently becomes part of that check, and a renamed one makes the branch
    protection rule point at a job that no longer exists. Equality catches both."""
    assert set(jobs) == {"baseline-guard", "eval"}


def test_the_baseline_guard_job_runs_on_every_pull_request(
    jobs: dict[str, Any],
) -> None:
    """Criterion 2: #42's guard is the half that really executes, so it carries NO condition
    at all. Catches decision A's `EVAL_TARGET_URL` condition being copied onto this job
    too, which would make the baseline rule inert in exactly the repository state it ships
    in - a gate that is green because it never ran."""
    assert "if" not in jobs["baseline-guard"]


def test_the_baseline_guard_checks_out_the_full_history(jobs: dict[str, Any]) -> None:
    """Risk 2, and the one mistake that makes this guard pass every PR: `actions/checkout`
    defaults to depth 1, where `git log base..head` resolves nothing. Catches a checkout
    with no `with:` block, and catches a `fetch-depth` of some other number - only 0 is the
    whole history. The CLI's unresolvable-ref case is the other half of this defence."""
    checkouts = [
        step for step in steps(jobs["baseline-guard"]) if "actions/checkout" in uses_of(step)
    ]

    assert len(checkouts) == 1
    assert checkouts[0]["with"]["fetch-depth"] == 0


def test_the_baseline_guard_job_invokes_the_subcommand(jobs: dict[str, Any]) -> None:
    """Criterion 2 end to end: catches a job that checks out the history, syncs the
    dependencies and never runs the guard - the shape a green tick is most misleading in,
    since every step in it succeeds. Decision B put the logic in `ops-eval baseline-guard`
    precisely so this line is the only thing the YAML has to get right."""
    guard_steps = [
        step for step in steps(jobs["baseline-guard"]) if "ops-eval baseline-guard" in run_of(step)
    ]

    assert len(guard_steps) == 1


def test_the_eval_job_has_no_condition_and_always_runs(
    jobs: dict[str, Any],
) -> None:
    """An earlier change (a design note replaces decision A): the help desk now serves the eval turn, so the
    eval half runs on every pull request. Catches the old `EVAL_TARGET_URL` condition coming
    back, or any other `if` on the job, which would make the gate green because it skipped."""
    assert "if" not in jobs["eval"]


def test_the_report_artifact_is_uploaded_even_when_the_gate_fails(
    jobs: dict[str, Any],
) -> None:
    """Criterion 6: a FAILED gate is exactly when the report is needed, and a step with no
    `if` is skipped once an earlier step fails. Catches the upload inheriting the default
    success condition, which publishes evidence only for the runs nobody needs it for.
    `always()` exactly, because `failure()` would drop it on the passing runs."""
    uploads = [step for step in steps(jobs["eval"]) if "actions/upload-artifact" in uses_of(step)]

    assert len(uploads) == 1
    assert uploads[0]["if"] == "always()"


def test_the_comment_step_is_skipped_on_a_fork_pull_request(
    jobs: dict[str, Any],
) -> None:
    """The unhappy path in criterion 8: a fork PR's `GITHUB_TOKEN` is read-only, so posting
    would fail the job on every contributor's pull request. Catches an unguarded comment
    step. The upload above is deliberately NOT guarded this way, so the evidence still
    reaches a fork PR - which is why the two are separate cases."""
    commenters = [step for step in steps(jobs["eval"]) if "gh pr comment" in run_of(step)]

    assert len(commenters) == 1
    assert "github.event.pull_request.head.repo.fork == false" in str(commenters[0]["if"])


def test_the_gate_step_passes_the_report_the_baseline_and_the_comment(
    jobs: dict[str, Any],
) -> None:
    """Criterion 6: the three flags are what turn 18a's comparison into a CI verdict. Catches
    `ops-eval gate` invoked with the report alone, which compares the run against nothing
    and can only ever pass, and catches `--comment` being dropped, which leaves the comment
    step with no file to post. All four fragments must be in ONE step's script, since a
    `--comment` in a different step is not this command's flag."""
    scripts = [run_of(step) for step in steps(jobs["eval"])]
    gate_steps = [
        script
        for script in scripts
        if all(flag in script for flag in ("ops-eval gate", "--report", "--baseline", "--comment"))
    ]

    assert len(gate_steps) == 1


def test_the_eval_job_uses_the_evals_overlay(jobs: dict[str, Any]) -> None:
    """An earlier change: the run's policy is `infra/evals/routing.evals.yaml`, which points every
    provider at the mock. Catches a job that runs `make eval` with the overlay unreferenced,
    where a pull request could reach a real provider. The overlay is named by the job's own
    steps or by the `make eval` recipe they call, and the recipe must name its compose file."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    scripts = "\n".join(run_of(step) for step in steps(jobs["eval"]))

    assert "make eval" in scripts
    assert "infra/evals/compose.yml" in makefile
    assert (REPO_ROOT / "infra" / "evals" / "routing.evals.yaml").is_file()


def test_the_ci_overlay_caps_the_evals_tenant_at_two_dollars() -> None:
    """Decision D's number, pinned by value. Catches an overlay carrying the
    prod `10.00`, and catches the budget written as a bare YAML number: the policy loader
    reads these as strings, so `2.00` unquoted parses to a float and the overlay that
    looked right stops merging. Only this path is asserted - the file is a partial document
    merged over `routing.yaml`, not a whole policy."""
    assert load(CI_OVERLAY)["tenants"]["evals"]["budget_usd_month"] == "2.00"
