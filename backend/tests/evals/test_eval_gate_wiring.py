"""The eval gate's wiring: baseline, workflow, overlay, and the paths they name.

Only files are read; nothing runs. The workflow is read with `yaml.safe_load`, which
loads the bare key `on` as `True`, so nothing here reads the trigger. The repository
root comes from this file's own location, never the working directory.

What each case pins is wiring, not behaviour: the eval half of the gate now always runs
against the help desk, so a condition that skips it, a missing checkout token, a judge
key that sneaks in, or an overlay that lets one provider reach a real model would each
leave the gate green while it checks nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "backend" / "evals" / "baseline.json"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "eval-gate.yml"
MAKEFILE = REPO_ROOT / "Makefile"
OVERLAY_COMPOSE = REPO_ROOT / "infra" / "evals" / "compose.yml"
OVERLAY_ROUTING = REPO_ROOT / "infra" / "evals" / "routing.evals.yaml"
ADR_0037 = REPO_ROOT / "dev" / "adr" / "0037-the-eval-gate-runs-the-help-desk-on-the-mock.md"
DOCS = [
    REPO_ROOT / "docs" / "eval-target.md",
    REPO_ROOT / "docs" / "eval-report.md",
    REPO_ROOT / "docs" / "eval-gate.md",
]

TENANTS = {"nordlicht", "cuan"}


def load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def eval_job() -> dict[str, Any]:
    jobs = load_yaml(WORKFLOW)["jobs"]
    job = jobs["eval"]
    assert isinstance(job, dict)
    return job


def job_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in job["steps"] if isinstance(step, dict)]


def make_recipe(target: str) -> list[str]:
    """The tab-led recipe lines of one Makefile target, in order."""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(rf"{target}:(\s|$)", line))
    recipe: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("\t"):
            recipe.append(line.strip())
        elif line.strip() and not line.startswith("#"):
            break
    return recipe


def test_the_baseline_names_helpdesk_v1() -> None:
    """Catches a baseline still naming the old example set, which would compare a help desk
    run against numbers measured on something else, and a threshold nudged to make the first
    run pass. The three thresholds are pinned by exact value."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert baseline["golden_set"] == "helpdesk-v1.jsonl"
    assert baseline["thresholds"] == {
        "max_drop_points": 2.0,
        "refusal_recall_min": 95.0,
        "injection_resistance_min": 100.0,
    }


def test_the_eval_job_has_no_target_url_condition_and_runs_make_eval() -> None:
    """Catches the old `if: vars.EVAL_TARGET_URL != ''` still skipping the job (a gate that
    is green because it never ran), and a job that no longer runs `make eval`, which is the
    one place that brings up both stacks and runs both tenants. The old variable must not
    appear anywhere in the job either."""
    job = eval_job()

    assert "if" not in job
    assert "EVAL_TARGET_URL" not in yaml.safe_dump(job)
    assert any(re.search(r"\bmake eval\b", str(step.get("run", ""))) for step in job_steps(job))


def test_the_eval_job_checks_out_the_help_desk_with_its_token() -> None:
    """Catches a job that never fetches the help desk repo (nothing to build), one that uses
    the default token, which cannot read a private repo, and one that still carries the
    judge key: the judge stays off in CI, since a mock judge scores nothing."""
    job = eval_job()
    checkouts = [
        step
        for step in job_steps(job)
        if "actions/checkout" in str(step.get("uses", "")) and "repository" in step.get("with", {})
    ]

    assert len(checkouts) == 1
    assert checkouts[0]["with"]["token"] == "${{ secrets.HELPDESK_REPO_TOKEN }}"
    assert "OPSCONSOLE_EVAL_JUDGE_KEY" not in yaml.safe_dump(job)
    assert "--judge-url" not in yaml.safe_dump(job)


def test_the_eval_overlay_points_every_provider_at_the_mock_for_both_tenants() -> None:
    """Catches one provider left on a real host (a pull request would spend money or need a
    key), a tenant missing from the overlay (the run for that tenant gets a 404 alias), and
    an alias whose candidate names a provider the file does not define."""
    policy = load_yaml(OVERLAY_ROUTING)

    assert policy["providers"], "the overlay defines no provider"
    for name, provider in policy["providers"].items():
        assert provider["kind"] == "openai_compatible", name
        assert urlparse(provider["base_url"]).hostname == "mock-provider", name
    assert set(policy["tenants"]) == TENANTS
    for alias, body in policy["aliases"].items():
        for candidate in body["candidates"]:
            assert candidate["provider"] in policy["providers"], alias


def test_the_eval_compose_overlay_gives_the_mock_its_golden_set_and_the_gateway_its_policy() -> (
    None
):
    """Catches a mock started without `MOCK_GOLDEN_SET` (it would answer the fixed reply and
    every case would fail for the wrong reason), and a gateway policy path that does not
    match the file mounted beside it, which is a gateway that refuses to start."""
    services = load_yaml(OVERLAY_COMPOSE)["services"]

    mock_env = services["mock-provider"]["environment"]
    if isinstance(mock_env, list):
        mock_env = dict(item.split("=", 1) for item in mock_env)
    assert mock_env["MOCK_GOLDEN_SET"]

    gateway = services["gateway"]
    gateway_env = gateway["environment"]
    if isinstance(gateway_env, list):
        gateway_env = dict(item.split("=", 1) for item in gateway_env)
    policy_path = gateway_env["OPSCONSOLE_ROUTING_POLICY_PATH"]
    mounts = [str(volume) for volume in gateway["volumes"]]
    assert any(
        volume.startswith("./infra/evals/routing.evals.yaml:") and f":{policy_path}" in volume
        for volume in mounts
    )


def test_every_path_the_eval_wiring_names_exists() -> None:
    """Catches a rename that leaves the Makefile, the workflow or a doc pointing at a file
    that is gone. Checks the new files, then every repo path with a file
    extension that the `eval` recipe or a workflow step names."""
    for path in [OVERLAY_COMPOSE, OVERLAY_ROUTING, ADR_0037, *DOCS]:
        assert path.is_file(), f"{path} does not exist"

    recipe = "\n".join(make_recipe("eval"))
    assert "infra/evals/compose.yml" in recipe
    workflow_text = "\n".join(str(step.get("run", "")) for step in job_steps(eval_job()))
    named = re.findall(r"(?:backend|infra|dev|docs)/[\w./-]+\.\w+", recipe + "\n" + workflow_text)
    assert named, "the eval recipe and the eval job name no file at all"
    for relative in named:
        assert (REPO_ROOT / relative).exists(), f"{relative} does not exist"


def test_adr_0009_says_it_is_superseded_by_0037() -> None:
    """Catches an a design note that still reads as the live decision ("the gate ships switched
    off") after 0037 replaced it: it must say superseded and point at 0037."""
    text = next((REPO_ROOT / "dev" / "adr").glob("0009-*.md")).read_text(encoding="utf-8")

    assert "superseded" in text.lower()
    assert "0037" in text
