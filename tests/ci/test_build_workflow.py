"""`.github/workflows/build.yml`: the image pipeline nothing else in this repo can run.

No image has ever been pushed to `ghcr.io/developermusa/ai-operations-console/gateway`, and
nothing in this repository can push one - `docker buildx`, GHCR and Trivy all live inside
the runner. So this file holds WIRING and not behaviour: the trigger set, the platform
pair, the single SHA tag, the GHCR login, the SBOM, the Trivy thresholds and the
`permissions` block. The only thing here that is checked against another file in the repo
is the image reference, which is read out of `infra/k8s/apps/kustomization.yaml` rather
than written down twice: the workflow that pushes and the overlay that deploys cannot
drift if the test compares them.

The document is read with `yaml.safe_load` and the repository root is taken from this
file's own `__file__`, never from the working directory, so the suite behaves the same run
from `tests/` as from the repo root.

One PyYAML quirk `trigger()` below handles: the bare key `on` is YAML 1.1's boolean, so
`yaml.safe_load` returns the trigger block under the key `True` rather than under `"on"`,
and `document["on"]` would raise `KeyError` on a perfectly correct workflow.

Two deliberate choices about selection:

* The plan pins no job names (this workflow is not a required check by name), so steps are
  gathered across EVERY job and each step is located by a substring of its `uses`. Actions
  in this repo are pinned by major tag (`actions/checkout@v4`), so an exact ref would go
  red on a legitimate bump; the substring is the action's `owner/name`.
* Selection is by `uses` only, never by a fragment of a `run` script, because a step that
  merely ECHOES `--severity CRITICAL` or the image name would match a fragment search. For the same reason the "no floating tag" case reads the `tags` value
  of the build step and nothing else: the workflow's own trigger says `main`, and so will
  its comments, so a whole-file search for that word could never distinguish a correct
  file from a `:main` tag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build.yml"
KUSTOMIZATION = REPO_ROOT / "infra" / "k8s" / "apps" / "kustomization.yaml"

BUILD_ACTION = "docker/build-push-action"
LOGIN_ACTION = "docker/login-action"
UPLOAD_ACTION = "actions/upload-artifact"
TRIVY_ACTION = "aquasecurity/trivy-action"

EXPECTED_PLATFORMS = "linux/amd64,linux/arm64"
EXPECTED_PERMISSIONS = {"contents": "read", "packages": "write", "id-token": "write"}


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


def uses_of(step: dict[str, Any]) -> str:
    """The step's `uses` reference, or the empty string for a `run` step."""
    return str(step.get("uses", ""))


def all_steps(jobs: dict[str, Any]) -> list[dict[str, Any]]:
    """Every step of every job, in declaration order."""
    collected: list[dict[str, Any]] = []
    for job in jobs.values():
        assert isinstance(job, dict), "a job is not a mapping"
        collected.extend(steps(job))
    return collected


def step_using(jobs: dict[str, Any], action: str) -> dict[str, Any]:
    """The one step whose `uses` names `action`; a second one is a failure, not a pick."""
    matches = [step for step in all_steps(jobs) if action in uses_of(step)]
    assert len(matches) == 1, f"expected exactly one step using {action}, found {len(matches)}"
    return matches[0]


def with_of(step: dict[str, Any]) -> dict[str, Any]:
    """The step's `with:` inputs."""
    block = step["with"]
    assert isinstance(block, dict), "the step's `with` block is not a mapping"
    return block


def gateway_image_name() -> str:
    """The `newName` of the `gateway` entry in the apps overlay's `images:` list."""
    images = load(KUSTOMIZATION)["images"]
    assert isinstance(images, list)
    entries = [
        entry for entry in images if isinstance(entry, dict) and entry.get("name") == "gateway"
    ]
    assert len(entries) == 1, "the apps overlay does not carry exactly one `gateway` image"
    return str(entries[0]["newName"])


def tag_lines(build: dict[str, Any]) -> list[str]:
    """The build step's `tags`, one full image reference per entry, blanks dropped."""
    declared = with_of(build)["tags"]
    lines = (
        [str(entry) for entry in declared]
        if isinstance(declared, list)
        else str(declared).splitlines()
    )
    stripped = [line.strip() for line in lines if line.strip()]
    assert stripped != [], "the build step publishes no tag at all"
    return stripped


def repository_of(reference: str) -> str:
    """The registry/repository half of `ghcr.io/owner/repo/image:tag`."""
    return reference.rsplit(":", 1)[0]


def tag_of(reference: str) -> str:
    """The tag half of `ghcr.io/owner/repo/image:tag`, empty when the reference carries none."""
    return reference.rsplit(":", 1)[1] if ":" in reference else ""


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """The parsed workflow, read once for the module.

    A missing file is a failure of this fixture and therefore of every case, which is the
    honest report while `build.yml` does not exist yet.
    """
    return load(WORKFLOW)


@pytest.fixture(scope="module")
def jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    """The `jobs:` mapping."""
    listed = workflow["jobs"]
    assert isinstance(listed, dict)
    return listed


@pytest.fixture(scope="module")
def build(jobs: dict[str, Any]) -> dict[str, Any]:
    """The one `docker/build-push-action` step, wherever it sits."""
    return step_using(jobs, BUILD_ACTION)


def test_the_workflow_builds_on_a_push_to_main_and_on_tags(workflow: dict[str, Any]) -> None:
    """Criterion 1. Catches a `push` trigger with no `branches` filter, which builds and
    pushes an image for every branch anyone happens to push, and catches a workflow wired to
    `main` alone: with no `tags` entry a release tag produces no image at all, and the tag
    that names a release points at nothing in the registry. `main` is asserted as the whole
    list, because a second branch here is another producer of pushed images nobody agreed
    to."""
    push = trigger(workflow)["push"]

    assert push["branches"] == ["main"]
    assert "tags" in push


def test_the_workflow_can_be_dispatched_by_hand(workflow: dict[str, Any]) -> None:
    """Criterion 1. Catches a workflow that can only be reached by pushing a commit, which
    makes the only way to rebuild an image - after a base-image CVE, or a runner change -
    an empty commit on `main`."""
    assert "workflow_dispatch" in trigger(workflow)


def test_the_workflow_has_no_pull_request_trigger(workflow: dict[str, Any]) -> None:
    """Criteria 1 and 7, the unhappy path: a `pull_request` run from a FORK would carry a
    `GITHUB_TOKEN` and reach the login step with `packages: write` requested. Catches the
    trigger a reviewer would add for the best of reasons ("scan the image before merge"),
    which is the one shape that lets an untrusted contributor's Dockerfile run against this
    repository's registry credentials. Absence is the control, so absence is the
    assertion."""
    assert "pull_request" not in trigger(workflow)


def test_the_permissions_are_exactly_the_three_the_push_needs(
    workflow: dict[str, Any], jobs: dict[str, Any]
) -> None:
    """Criterion 4, by dict EQUALITY. Catches a missing `packages: write`,
    where the login succeeds and the push 403s; catches a missing `id-token: write`, which
    the attestation half needs; and catches a FOURTH permission added later - `contents:
    write`, say, which would let a compromised action in this pipeline rewrite the
    repository. Read from the job when it declares its own block, since a job-level
    `permissions` replaces the workflow's rather than adding to it."""
    declared = [job["permissions"] for job in jobs.values() if "permissions" in job]
    if declared == []:
        declared = [workflow["permissions"]]

    assert declared != [], "neither the workflow nor any job declares `permissions`"
    for block in declared:
        assert block == EXPECTED_PERMISSIONS


def test_the_login_step_authenticates_to_ghcr_with_the_workflow_token(
    jobs: dict[str, Any],
) -> None:
    """Criterion 4. Catches a login pointed at Docker Hub (the action's default registry when
    `registry` is omitted), which would push this image somewhere nobody deploys from, and
    catches a password taken from a long-lived personal access token in `secrets`: the
    `GITHUB_TOKEN` is scoped to this run and expires with it, which is the whole reason the
    plan names it."""
    login = with_of(step_using(jobs, LOGIN_ACTION))

    assert login["registry"] == "ghcr.io"
    assert "secrets.GITHUB_TOKEN" in str(login["password"])


def test_the_build_pushes_the_repositorys_one_dockerfile(build: dict[str, Any]) -> None:
    """Criterion 2. Catches `push: false`, which leaves a green tick, an SBOM and a Trivy
    scan over an image that exists only inside the runner - the most convincing shape of a
    pipeline that ships nothing. Also catches a `file:` naming a path that is not in the
    tree, which fails only once the workflow runs; the path is resolved against the repo
    root here, so a typo reddens now."""
    assert build["with"]["push"] is True
    assert (REPO_ROOT / str(with_of(build)["file"])).is_file()


def test_the_build_targets_both_architectures(build: dict[str, Any]) -> None:
    """Criterion 2, pinned exactly. Catches a single-platform build: the
    cluster's CAX21 nodes are arm64, so an amd64-only image fails at the pull with
    `exec format error`, and an arm64-only one cannot run on a developer's amd64 machine.
    The string is compared whole because the action takes `platforms` as one
    comma-separated value - a list with the same two members is a different input."""
    assert with_of(build)["platforms"] == EXPECTED_PLATFORMS


def test_the_pushed_image_is_the_name_the_apps_overlay_deploys(build: dict[str, Any]) -> None:
    """Criterion 2, checked against `infra/k8s/apps/kustomization.yaml` rather than against a
    hardcoded string. Catches a push to a repository the overlay does not deploy from - a
    typo in the owner or the image segment, or `.../ai-operations-console` without
    `/gateway` - which produces a green build and a cluster that still cannot pull. Every
    tag line is checked, so adding a second reference to another repository reddens too."""
    expected = gateway_image_name()

    assert {repository_of(line) for line in tag_lines(build)} == {expected}


def test_every_published_tag_is_the_commit_sha(build: dict[str, Any]) -> None:
    """Criterion 3. Catches a tag built from anything but the commit - `github.ref_name`,
    `github.run_number`, a `metadata-action` output - which gives the cluster an image
    reference that no commit in this repository identifies. `github.sha` is the 40-character
    SHA; `github.sha` truncated by an expression would still contain the fragment, which is
    why the neighbouring case pins the absence of floating names separately."""
    for line in tag_lines(build):
        assert "github.sha" in tag_of(line), f"tag {tag_of(line)!r} does not come from the SHA"


@pytest.mark.parametrize("floating", ["latest", "main"])
def test_no_floating_tag_is_published(build: dict[str, Any], floating: str) -> None:
    """Criterion 3's prohibition, one row per name a pipeline drifts to. Catches a build that
    publishes the SHA tag AND a moving alias, which is the shape that passes the case above
    while making a rollback meaningless: `:latest` and `:main` mean different bytes on
    different days, so the cluster's running code is no longer named by a commit. Scoped to
    the TAG half of the build step's own `tags` - the trigger block and
    the file's comments both contain the word `main`, so a whole-file search would redden a
    correct workflow and could never fail for this reason."""
    for line in tag_lines(build):
        assert floating not in tag_of(line), f"tag {tag_of(line)!r} is a floating tag"


def test_the_build_produces_an_sbom(build: dict[str, Any]) -> None:
    """Criterion 5. Catches the input being dropped or written as the string `"true"`, which
    buildx does not read as the flag: with no SBOM there is nothing to answer "which
    version of that library was in the image we shipped in March" with, and the upload step
    below would publish an empty artefact. `is True`, because a truthy string is a
    different input."""
    assert with_of(build)["sbom"] is True


def test_the_sbom_is_uploaded_as_an_artifact(jobs: dict[str, Any]) -> None:
    """Criterion 5's second half. Catches an SBOM generated inside the runner and thrown away
    with it, which is criterion 5 held only in appearance, and catches an upload step
    retargeted at the Trivy report or the build log - the inputs must NAME the SBOM (`sbom`
    or `spdx`), or what is published is some other file. Located by `uses`, so a step that
    merely echoes the word cannot satisfy it."""
    upload = with_of(step_using(jobs, UPLOAD_ACTION))
    described = " ".join(str(value) for value in upload.values()).lower()

    assert "sbom" in described or "spdx" in described


@pytest.mark.parametrize(
    ("input_name", "expected"),
    [
        ("severity", "CRITICAL"),
        ("exit-code", "1"),
        ("ignore-unfixed", True),
    ],
)
def test_the_trivy_step_fails_the_job_on_a_critical_finding(
    jobs: dict[str, Any], input_name: str, expected: object
) -> None:
    """Criterion 6, one row per input, each pinned exactly.

    `severity` catches a widened band (`CRITICAL,HIGH` fails the build on findings nobody
    agreed to block, `HIGH` blocks on the wrong class and lets a CRITICAL through as a
    warning). `exit-code` catches the action's DEFAULT of `0`, which is criterion 7's exact
    failure: Trivy prints a CRITICAL table and the job is a green tick. It is compared to
    the STRING `"1"` because that is what the action reads and what `yaml.safe_load` returns
    only when the YAML says `'1'` - a bare `1` parses to the int and reddens here, which is
    intended. `ignore-unfixed` catches the scan being left to fail on CVEs with no fixed
    version in the base image, where the pipeline is red for something no rebuild can
    clear and someone soon deletes the whole step."""
    assert with_of(step_using(jobs, TRIVY_ACTION))[input_name] == expected


def test_trivy_scans_the_image_the_build_just_pushed(
    jobs: dict[str, Any], build: dict[str, Any]
) -> None:
    """Criterion 6, the target rather than the thresholds. Catches an `image-ref` pointed at
    a floating tag or at the base image - a scan that passes while the artefact actually
    published was never looked at. The reference must name the overlay's repository and
    resolve from `github.sha`, the same two properties the tag cases pin, so the scanned
    image and the pushed image are one image."""
    reference = str(with_of(step_using(jobs, TRIVY_ACTION))["image-ref"])

    assert repository_of(reference) == gateway_image_name()
    assert "github.sha" in tag_of(reference)
    assert repository_of(reference) in {repository_of(line) for line in tag_lines(build)}
