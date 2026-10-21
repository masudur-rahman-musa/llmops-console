"""The eval gate's wiring for the conversation run: baseline, workflow, overlay, ADRs.

Only files are read; nothing runs. The workflow is read with `yaml.safe_load`, which loads
the bare key `on` as `True`, so nothing here reads the trigger. The repository root comes
from this file's own location, never the working directory.

What each case pins is wiring, not behaviour: the gate now plays the help desk's 33 cases on
the real cheap model, so a condition that skips it, a missing checkout token, a missing model
key, the old baseline still sitting there, or a mock still answering from a golden set would
each leave the gate green while it checks nothing real.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALS = REPO_ROOT / "backend" / "evals"
BASELINE = EVALS / "conversation-baseline.json"
OLD_BASELINE = EVALS / "baseline.json"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "eval-gate.yml"
MAKEFILE = REPO_ROOT / "Makefile"
OVERLAY_COMPOSE = REPO_ROOT / "infra" / "evals" / "compose.yml"
ADR_DIR = REPO_ROOT / "dev" / "adr"


def load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def eval_job() -> dict[str, Any]:
    job = load_yaml(WORKFLOW)["jobs"]["eval"]
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


def test_the_conversation_baseline_replaces_the_old_one() -> None:
    """Catches the old `baseline.json` still on disk (a gate could read it by mistake and the
    guard would watch a file nobody uses) and a conversation baseline that is not two plain
    percent numbers: `check_score` is required and between 0 and 100, `judge_score` is
    optional, and nothing else is in the file (the old thresholds block would be dead
    weight)."""
    assert not OLD_BASELINE.exists()
    document = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert set(document) <= {"check_score", "judge_score"}
    assert 0.0 <= document["check_score"] <= 100.0
    assert 0.0 <= document.get("judge_score", 0.0) <= 100.0


def test_the_old_golden_set_and_its_label_readme_are_gone() -> None:
    """Catches the single-reply golden example, the 1-to-5 label readme, or the whole golden
    folder surviving: they would keep the old format alive and the old docs honest-looking."""
    assert not (EVALS / "golden").exists()
    assert not (EVALS / "labels" / "README.md").exists()


def test_the_eval_job_has_no_condition_and_runs_make_eval() -> None:
    """Catches a skip condition on the eval job (a gate that is green because it never ran)
    and a job that no longer runs `make eval`. The old `EVAL_TARGET_URL` variable must not
    appear in the job either."""
    job = eval_job()

    assert "if" not in job
    assert "EVAL_TARGET_URL" not in yaml.safe_dump(job)
    assert any(re.search(r"\bmake eval\b", str(step.get("run", ""))) for step in job_steps(job))


def test_the_eval_job_checks_out_the_help_desk_and_carries_the_model_key() -> None:
    """Catches a job that never fetches the help desk (no cases, nothing to build), one using
    the default token (cannot read a private repo), and one with no `ANTHROPIC_API_KEY`
    secret: the gate plays the real cheap model, so without the key it fails, like the
    checkout without its token."""
    job = eval_job()
    checkouts = [
        step
        for step in job_steps(job)
        if "actions/checkout" in str(step.get("uses", "")) and "repository" in step.get("with", {})
    ]

    assert len(checkouts) == 1
    assert checkouts[0]["with"]["token"] == "${{ secrets.HELPDESK_REPO_TOKEN }}"
    assert "secrets.ANTHROPIC_API_KEY" in yaml.safe_dump(job)


def test_the_eval_recipe_gates_against_the_conversation_baseline() -> None:
    """Catches a recipe that plays conversations and gates against nothing, or against the
    removed baseline: `ops-eval gate` is given `conversation-baseline.json`."""
    recipe = "\n".join(make_recipe("eval"))

    assert "ops-eval gate" in recipe
    assert "backend/evals/conversation-baseline.json" in recipe


def test_the_eval_compose_overlay_has_no_golden_set_mode() -> None:
    """Catches the mock still started in golden-set mode (a design note removes it): the real
    cheap model answers, and a mock reading the help desk's old set would answer from the
    wrong file. Checked on the whole file text so no spelling hides it."""
    text = OVERLAY_COMPOSE.read_text(encoding="utf-8")

    assert "MOCK_GOLDEN_SET" not in text
    assert "/golden" not in text


def test_adr_0037_says_it_is_superseded_by_0039() -> None:
    """Catches an a design note that still reads as the live decision ("the gate runs on the mock")
    after 0039 replaced it: it must say superseded and point at 0039."""
    text = next(ADR_DIR.glob("0037-*.md")).read_text(encoding="utf-8")

    assert "superseded" in text.lower()
    assert "0039" in text


def test_adr_0009_still_says_it_is_superseded() -> None:
    """Catches the older chain being broken: 0009 stays superseded by 0037."""
    text = next(ADR_DIR.glob("0009-*.md")).read_text(encoding="utf-8")

    assert "superseded" in text.lower()
    assert "0037" in text
