r"""An earlier change's nightly-eval gap: the k8s CronJob reads its golden set from an in-image path, so
the gateway image must actually carry that set.

The CronJob used `ops-eval run --set backend/evals/golden/example.jsonl`, a repo-relative path
the gateway image never shipped (the Dockerfile copies `app/`, `config/` and `migrations/`, and
`/app` is the workdir), so the file was simply absent at run time. This plan ships `evals/golden/`
into the image and points `--set` at the absolute in-image `/app/evals/golden/example.jsonl`.

These two cases read the Dockerfile and the CronJob manifest as FILES, not through the `built`
fixture: the Dockerfile is not a kustomize object at all, and reading the CronJob directly keeps
the in-image-path claim independent of kubectl. The parsing helpers are module-local copies, the
same way the other `infra/tests/infra/` modules copy theirs: nothing here imports
another test module, so this one's collection depends on nothing else's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
GOLDEN_FILE = REPO_ROOT / "backend" / "evals" / "golden" / "example.jsonl"
NIGHTLY_EVAL_CRONJOB = REPO_ROOT / "infra" / "k8s" / "evals" / "nightly-eval-cronjob.yaml"

#: The absolute in-image path the CronJob's `--set` must name after this plan: `/app` is the
#: Dockerfile's WORKDIR, and the golden set is copied to `/app/evals/golden/`.
IN_IMAGE_GOLDEN_PATH = "/app/evals/golden/example.jsonl"

#: The two forms a Dockerfile `COPY` destination for the golden set may take, both resolving
#: under `/app`: workdir-relative with the leading `./`, or the bare relative form.
VALID_COPY_DESTINATIONS = ("./evals/golden/", "evals/golden/")


def value_after(args: list[str], flag: str) -> str:
    """The item directly after `flag`; the flag must appear exactly once, so
    a second `--set` added later cannot hide a wrong first one."""
    positions = [index for index, item in enumerate(args) if item == flag]
    assert len(positions) == 1, f"{flag} appears {len(positions)} times in {args}"
    assert positions[0] + 1 < len(args), f"{flag} is the last arg in {args}"
    return args[positions[0] + 1]


def sole_container(document: dict[str, Any]) -> dict[str, Any]:
    """The one container of a CronJob's pod template, asserting exactly one. The pod template of
    a CronJob sits under `spec.jobTemplate.spec.template.spec`."""
    pod_spec = (
        document.get("spec", {})
        .get("jobTemplate", {})
        .get("spec", {})
        .get("template", {})
        .get("spec", {})
    )
    containers = pod_spec.get("containers", [])
    assert isinstance(containers, list) and len(containers) == 1, (
        f"nightly-eval declares {len(containers) if isinstance(containers, list) else 0} "
        "containers, and this plan's CronJob runs one"
    )
    container = containers[0]
    assert isinstance(container, dict)
    return container


def test_the_image_ships_the_golden_set() -> None:
    """Catches an image whose Dockerfile never copies the golden set, so the nightly CronJob's
    in-image `--set` path points at a file that is not in the image. One `COPY` line, stripped,
    must start with `COPY evals/golden/` and name a destination that resolves under `/app`; and
    the source file the COPY reads must really exist in the repo today, or the COPY copies
    nothing. Without this, the CronJob path case could pass while the image stays empty."""
    assert BACKEND_DOCKERFILE.is_file(), f"{BACKEND_DOCKERFILE} does not exist"
    lines = [line.strip() for line in BACKEND_DOCKERFILE.read_text(encoding="utf-8").splitlines()]

    copy_lines = [line for line in lines if line.startswith("COPY evals/golden/")]
    assert copy_lines, "the Dockerfile has no `COPY evals/golden/` line, so the image ships no set"
    resolving = [
        line
        for line in copy_lines
        if any(dest in line.split()[1:] for dest in VALID_COPY_DESTINATIONS)
    ]
    assert resolving, (
        f"no `COPY evals/golden/` line names a destination under /app "
        f"({VALID_COPY_DESTINATIONS}): {copy_lines}"
    )
    assert GOLDEN_FILE.is_file(), f"{GOLDEN_FILE} does not exist, so the COPY would copy nothing"


def test_the_cronjob_reads_the_golden_set_from_the_image() -> None:
    """Catches the CronJob still running `--set backend/evals/golden/example.jsonl`, the
    repo-relative path the gateway image never carries: the value right after `--set` must be the
    absolute in-image `/app/evals/golden/example.jsonl`, under `/app` and never `backend/`-
    relative. Read from the file directly, not the `built` fixture, so this case does not need
    kubectl to redden on a wrong path."""
    documents = list(yaml.safe_load_all(NIGHTLY_EVAL_CRONJOB.read_text(encoding="utf-8")))
    documents = [document for document in documents if isinstance(document, dict)]
    assert len(documents) == 1, f"{NIGHTLY_EVAL_CRONJOB} holds {len(documents)} documents, not one"

    command = [str(part) for part in sole_container(documents[0]).get("command", [])]
    set_value = value_after(command, "--set")

    assert set_value == IN_IMAGE_GOLDEN_PATH, f"--set reads {set_value!r}"
    assert set_value.startswith("/app/"), f"--set is not an absolute in-image path: {set_value!r}"
    assert not set_value.startswith("backend/"), f"--set is still repo-relative: {set_value!r}"
