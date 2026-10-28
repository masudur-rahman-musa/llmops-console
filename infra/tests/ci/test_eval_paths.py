"""Every eval path plays the help desk's conversations.

Files are read; nothing runs. The paths are the PR gate (`eval-gate.yml` through
`make eval`), the nightly (`nightly-eval.yml`), the kind end-to-end run (`e2e-kind.yml`)
and the self-hosted versus API comparison (`infra/scripts/tradeoff_run.sh`).

The repository root comes from this file's own location, never the working directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
MAKEFILE = REPO_ROOT / "Makefile"
TRADEOFF_SCRIPT = REPO_ROOT / "infra" / "scripts" / "tradeoff_run.sh"

#: The files that run, describe or ship an eval path. A reference to the removed single-reply
#: format in any of them is a path someone will follow into nothing.
SCANNED = [
    WORKFLOWS / "eval-gate.yml",
    WORKFLOWS / "nightly-eval.yml",
    WORKFLOWS / "e2e-kind.yml",
    MAKEFILE,
    REPO_ROOT / "backend" / "Dockerfile",
    TRADEOFF_SCRIPT,
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "tradeoff-run" / "README.md",
    *sorted((REPO_ROOT / "docs").glob("eval-*.md")),
    *sorted((REPO_ROOT / "infra" / "evals").rglob("*")),
    *sorted((REPO_ROOT / "infra" / "tradeoff").rglob("*")),
    *sorted((REPO_ROOT / "infra" / "k8s" / "evals").rglob("*")),
]

#: What may no longer be named. `baseline.json` alone: the conversation baseline is
#: `conversation-baseline.json`, which has a hyphen in front and so does not match.
FORBIDDEN = {
    "the golden folder": re.compile(r"golden/"),
    "the old baseline": re.compile(r"(?<![-\w])baseline\.json"),
    "the single-reply route": re.compile(r"/v1/eval/turn"),
    "the retired command": re.compile(r"ops-eval run\b"),
}


def load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def run_scripts(workflow: str, job: str) -> list[str]:
    steps = load(WORKFLOWS / workflow)["jobs"][job]["steps"]
    return [str(step.get("run", "")) for step in steps if isinstance(step, dict)]


def make_recipe(target: str) -> str:
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(rf"{target}:(\s|$)", line))
    recipe: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("\t"):
            recipe.append(line.strip())
        elif line.strip() and not line.startswith("#"):
            break
    return "\n".join(recipe)


def test_the_pr_gate_plays_conversations_through_make_eval() -> None:
    """Catches a PR gate that still plays the old set: the `eval` job runs `make eval`, whose
    recipe runs `ops-eval conversations` on the help desk's `cases.jsonl`, with no `--set`,
    `--tenant` or `--policy` left over from the old command."""
    scripts = " ".join(run_scripts("eval-gate.yml", "eval"))
    recipe = make_recipe("eval")

    assert re.search(r"\bmake eval\b", scripts)
    assert "ops-eval conversations" in recipe
    assert "backend/evals/conversations/cases.jsonl" in MAKEFILE.read_text(encoding="utf-8")
    assert not re.search(r"--(set|tenant|policy)\b", recipe)


def test_the_nightly_plays_conversations() -> None:
    """Catches a nightly still on the removed command, or one that reads a copy of the cases
    inside this repo instead of the help desk's own file from its checkout."""
    scripts = run_scripts("nightly-eval.yml", "nightly-eval")
    plays = [s for s in scripts if "ops-eval conversations" in s]

    assert len(plays) == 1
    assert "helpdesk/backend/evals/conversations/cases.jsonl" in plays[0]


def test_the_kind_run_plays_conversations() -> None:
    """Catches a kind run that plays nothing eval-shaped, or the removed command."""
    jobs = load(WORKFLOWS / "e2e-kind.yml")["jobs"]
    scripts = [
        str(step.get("run", ""))
        for job in jobs.values()
        for step in job["steps"]
        if isinstance(step, dict)
    ]

    assert any("ops-eval conversations" in script for script in scripts)


def test_the_tradeoff_run_plays_conversations() -> None:
    """Catches the comparison script still calling the removed command: one `ops-eval
    conversations` call with `--cases` and none of the old flags."""
    text = TRADEOFF_SCRIPT.read_text(encoding="utf-8")

    assert "ops-eval conversations" in text
    assert "--cases" in text
    assert not re.search(r"ops-eval run\b|--set\b|--tenant\b", text)


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_eval_file_names_the_removed_single_reply_format(path: Path) -> None:
    """Catches any workflow, job, doc, manifest or script that still names `golden/`, the old
    `baseline.json`, the `/v1/eval/turn` route or `ops-eval run`. One case per file, so the
    failing file is the test id. A file that does not exist yet is skipped by the scan (a
    deleted file names nothing), which is why the paths test below checks the ones that must
    exist."""
    if not path.is_file():
        pytest.skip(f"{path} is not a file")

    text = path.read_text(encoding="utf-8")

    found = [label for label, pattern in FORBIDDEN.items() if pattern.search(text)]
    assert found == [], f"{path.relative_to(REPO_ROOT)} still names {found}"


def test_no_job_copies_the_golden_set_into_the_image() -> None:
    """Catches the gateway image still carrying `COPY evals/golden/`: the cases
    live in the help desk repo and would be a copy."""
    assert "evals/golden" not in (REPO_ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")


def test_the_in_cluster_nightly_cronjob_is_gone() -> None:
    """Catches the CronJob manifest or its mention in the kustomization: `nightly-eval.yml` is
    the one nightly path."""
    evals = REPO_ROOT / "infra" / "k8s" / "evals"

    assert not (evals / "nightly-eval-cronjob.yaml").exists()
    assert not any("kind: CronJob" in p.read_text() for p in evals.glob("*.yaml"))


#: A repo path as a workflow, recipe or script writes it. Not preceded by a path character or
#: a `)`, so `helpdesk/backend/...` and `$(EVAL_HELPDESK)/backend/...` (the other repo) are
#: skipped, and not under `reports/`, which a run creates.
NAMED_PATH = re.compile(r"(?<![\w./)-])((?:backend|infra|docs)/[\w./-]*\w\.\w+)")


def test_every_path_the_eval_files_name_exists() -> None:
    """Catches a rename that leaves the Makefile, a workflow or the script pointing at a file
    that is gone. Reads the `eval` recipe, the three workflows and the
    tradeoff script. Paths that only a run creates (under `reports/`) are left out, and at
    least one path must be found so the scan cannot pass over nothing."""
    texts = [
        make_recipe("eval"),
        *(
            (WORKFLOWS / name).read_text(encoding="utf-8")
            for name in ("eval-gate.yml", "nightly-eval.yml", "e2e-kind.yml")
        ),
        TRADEOFF_SCRIPT.read_text(encoding="utf-8"),
    ]
    named = {m for text in texts for m in NAMED_PATH.findall(text) if "/reports/" not in m}

    assert named, "no repo path found in the eval files"
    missing = sorted(p for p in named if not (REPO_ROOT / p).exists())
    assert missing == [], f"named but missing: {missing}"
