r"""`.github/workflows/ci.yml` and the `Makefile`: the security checks run as `make security`.

An earlier change moves the three security checks - the gitleaks scan over the whole history, the
Python dependency audit (`uv export` then pip-audit 2.9.0) and the console's `pnpm audit` -
out of `ci.yml` and into ONE Make target, `security`, which `make verify` also runs. CI keeps
one `security` job that does its own setup (a full-history checkout, uv, pnpm and Node, the
console install) and then runs `make security` and nothing else. So the commands live in the
`security` RECIPE, and this file pins two things:

* the WIRING in `ci.yml`: the `security` job exists, checks out every commit, runs
  `make security`, cannot fail without failing the build, and no job anywhere runs gitleaks,
  pip-audit or `pnpm audit` directly. The old `secrets` and `audit` jobs are gone.
* the RECIPE in the `Makefile`: the scan runs the gitleaks image pinned by tag and digest,
  over history (`git`), redacted, failing on a finding, with no staging-area flag; the export
  carries the locked flags and writes into the ignored `backend/scratch/`; pip-audit is pinned
  and strict and reads the file the export wrote; the console audit runs through
  `$(call console,audit)`.

None of these checks can be OBSERVED from a pytest run: the scan needs Docker and a
full-depth clone, the audits need the advisory databases over the network. The plan's own
last step runs `make security` once and pastes the result; that evidence is not a case here.

Three things are checked against another file rather than written down twice:

* the gitleaks version, read out of `.pre-commit-config.yaml`'s `rev: v8.24.3` and compared
  to the image tag the recipe must carry LITERALLY. a project rule is the point and the
  cost: a tag factored into a Make variable is a reference this static reader sees as equal
  to nothing, so the drift this comparison catches - `make security` scanning with one
  version while the commit hook uses another - becomes invisible. The repetition is
  deliberate. The digest after the tag takes over the job the old checksum check of the
  downloaded archive did: it pins the exact bytes that run.
* the requirements file the audit reads, extracted from BOTH the export's output flag and
  the `-r` argument of pip-audit and compared to each other. Extracting both ends means an
  export writing one name and an audit reading another is red, which is the shape that
  audits an empty or stale file and reports green.
* `.gitleaksignore`'s fingerprints, whose PATHS are read out of the file itself rather than
  listed here. The file is the record of which literals were allowed; this suite keeps that
  list to the exact shape decision A chose.

The workflow is read with `yaml.safe_load` and the repository root comes from this file's
own `__file__`, never from the working directory, so the suite behaves the same run from
anywhere. A Make recipe is read as the tab-indented lines under a start-of-line `name:`,
the same way `backend/tests/test_tooling_fast_suite.py` reads one; the helper is copied
here rather than imported across trees.

`ci.yml`'s `on:` block is not read here. Worth knowing for the case that is added the day it
matters: the bare key `on` is YAML 1.1's boolean, so `yaml.safe_load` returns that block
under the key `True` rather than under `"on"`.

Selection, and a project rule. A line that merely ECHOES a fragment matches a fragment
search exactly as the real command does - a banner, a visible-skip message, a comment
explaining what is deliberately NOT passed. Every case that selects or scans by command
text therefore goes through `command_lines()` (for `ci.yml`) or `recipe_commands()` (for the
`Makefile`), both of which drop comment lines and lines carrying `echo` or `printf`. A comment
saying "no --staged flag here" must not redden the case that forbids `--staged`, and a
banner naming `pnpm audit` must neither satisfy nor redden a case.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"
GITIGNORE = REPO_ROOT / ".gitignore"
BACKEND = REPO_ROOT / "backend"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
GITLEAKS_IGNORE = REPO_ROOT / ".gitleaksignore"
GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"

#: The one CI job and the one Make target the plan adds, and the exact step that joins them.
SECURITY_JOB = "security"
SECURITY_TARGET = "security"
MAKE_SECURITY = "make security"

#: The two jobs an earlier change merges into `security`. Each one surviving is a second copy of a
#: command the Makefile now owns, which is the drift the whole plan exists to end.
RETIRED_JOBS = ("secrets", "audit")

#: The tools no `ci.yml` step may run itself, each as a pattern over a COMMAND line. `pnpm`
#: and `audit` may have flags between them (`pnpm -C web audit`), so that row is not a
#: plain substring.
DIRECT_TOOLS = {
    "gitleaks": re.compile(r"gitleaks"),
    "pip-audit": re.compile(r"pip-audit"),
    "pnpm audit": re.compile(r"\bpnpm\b.*\baudit\b"),
}

#: The console audit, in the skip-when-absent form every other console check uses.
CONSOLE_AUDIT = "$(call console,audit)"

#: The image the scan runs, and its full pin: the tag the commit hook pins plus a 64-hex
#: digest, run with `re.fullmatch` over the one image token in the recipe.
GITLEAKS_IMAGE_NAME = "zricethezav/gitleaks"
PINNED_IMAGE = re.compile(r"zricethezav/gitleaks:v8\.24\.3@sha256:[0-9a-f]{64}")

#: The scan, its redaction and its failing exit - the old three fragments. The scan
#: itself is now "the image, then the `git` subcommand" rather than the literal
#: `gitleaks git`, because the image's entrypoint is the binary. `--exit-code 1` is what
#: makes a finding fail the build rather than log it.
REQUIRED_SCAN_FRAGMENTS = (
    ("git", re.compile(r"zricethezav/gitleaks:\S+\s+git(?=\s|$)")),
    ("--redact", re.compile(r"(?:^|\s)--redact(?=\s|$)")),
    ("--exit-code 1", re.compile(r"(?:^|\s)--exit-code\s+1(?=\s|$)")),
)

#: These are the local hook's flags, and both narrow the scan to a staging area that a
#: full-history scan does not have. Either one turns the scan into a scan of nothing.
FORBIDDEN_SCAN_FLAGS = ("--staged", "--pre-commit")

#: Pinned by value and cross-read from `.pre-commit-config.yaml` (standing
#: rule 53). The hook pins it as `v8.24.3`; the recipe's image tag must say it too.
GITLEAKS_VERSION = "8.24.3"
GITLEAKS_REPO = "github.com/gitleaks/gitleaks"

#: The table rows for the image pin, built from a repeated hex character rather than typed
#: as a realistic digest: this repository's own scan reads this file's history.
SAMPLE_DIGEST = "a" * 64

#: Where the export writes, and the `.gitignore` line that keeps a local run's file out
#: of git.
AUDIT_FILE = "backend/scratch/requirements-audit.txt"
IGNORED_SCRATCH = "/backend/scratch/"

#: Every path this plan names. The pytest fixture paths are here too, because a rename of
#: any of them would leave the suite reading a file the build no longer uses.
SECURITY_PATHS = (MAKEFILE, WORKFLOW, PRE_COMMIT, GITLEAKS_IGNORE, BACKEND)

#: The count the full-history run produced at drafting time, exact, plus the two
#: `web/tests/screens.test.tsx` fingerprints reviewed after. A count is an absolute
#:, so it is asserted by equality and not as a ceiling.
EXPECTED_FINGERPRINTS = 16

#: The fingerprint shape, run with `re.fullmatch` and therefore written WITHOUT the
#: `^`/`$` the plan quotes it with: a project rule, since in Python `$` also matches before a
#: trailing newline and `.match` with anchors would accept a fingerprint carrying one.
FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{40}:[^:]+:generic-api-key:\d+")

#: Every allowed literal is a fake one written by a test, so a fingerprint naming anything
#: else is an exception nobody reviewed as one.
ALLOWED_FINGERPRINT_PREFIXES = ("tests/", "services/console/tests/", "web/tests/")

#: The fingerprint unhappy rows, built from a repeated hex character rather than typed as a
#: realistic digest: this repository's own scan reads this file's history with the
#: `generic-api-key` rule, and a long random-looking literal here would be its first finding.
SAMPLE_COMMIT = "a" * 40
SHORT_COMMIT = "a" * 39
REJECTED_FINGERPRINT_PATH = "packages/opsconsole/x.py"

#: The two `[allowlist]` keys decision A rejects. A path or a regex allowlist also hides a
#: real key pasted into the same file later; a fingerprint pins one line of one blob.
FORBIDDEN_ALLOWLIST_KEYS = ("paths", "regexes")

#: The export, all four flags. `--locked` is the one that matters most: without it
#: `uv export` may RESOLVE afresh and audit a dependency set the lockfile never had.
EXPORT_COMMAND = "uv export"
EXPORT_FLAGS = (
    "--format requirements-txt",
    "--locked",
    "--no-emit-project",
    "--all-groups",
)

#: The audit, with the version pinned in the `--from` reference (decision C: the tool is
#: the check's, not the package's, so it never enters the lockfile).
AUDIT_COMMAND = "pip-audit"
AUDIT_FLAGS = ("pip-audit==2.9.0", "--strict", "--require-hashes", "--disable-pip")

#: Where the export writes its file, accepting the three ways a shell command can say it -
#: `-o path`, `--output-file path` and a `>` redirect - and where the audit reads it. `-r`
#: is required to stand alone, so `--require-hashes` cannot be read as an `-r`.
EXPORT_OUTPUT = re.compile(
    r"(?:-o|--output-file)[= ]\s*(?P<flagged>[^\s'\"|>]+)|>\s*(?P<redirected>[^\s'\"|>]+)"
)
AUDIT_INPUT = re.compile(r"(?:^|\s)-r\s+(?P<path>[^\s'\"|>]+)")


def load(path: Path) -> dict[str, Any]:
    """Parse a YAML document that must be a mapping."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    """The job's steps, each a mapping."""
    listed = job["steps"]
    assert isinstance(listed, list)
    return [step for step in listed if isinstance(step, dict)]


def uses_of(step: dict[str, Any]) -> str:
    """The step's `uses` reference, or the empty string for a `run` step."""
    return str(step.get("uses", ""))


def with_of(step: dict[str, Any]) -> dict[str, Any]:
    """The step's `with:` inputs, or an empty mapping when it declares none."""
    block = step.get("with", {})
    assert isinstance(block, dict), "the step's `with` block is not a mapping"
    return block


def is_command(stripped: str) -> bool:
    """a project rule: a blank line, a comment or a printer is not a command."""
    if stripped == "" or stripped.startswith("#"):
        return False
    return "echo" not in stripped and "printf" not in stripped


def command_lines(jobs: dict[str, Any]) -> list[str]:
    """Every EXECUTED line of every `run:` script: comment lines and printers dropped.

    a project rule. A step that echoes `pnpm audit` into the log matches a fragment search
    exactly as the real command does, and a comment explaining that a tool deliberately left
    this file carries the tool's name. Lines carrying `echo` or `printf` and lines whose first
    character is `#` are therefore not commands here.
    """
    collected: list[str] = []
    for job in jobs.values():
        assert isinstance(job, dict), "a job is not a mapping"
        for step in steps(job):
            for line in str(step.get("run", "")).splitlines():
                stripped = line.strip()
                if is_command(stripped):
                    collected.append(stripped)
    return collected


def makefile_recipe(name: str) -> list[str]:
    """The raw recipe lines of Make target `name`.

    Anchored on a start-of-line `name:` so a target name that only shows up inside another
    recipe, or a longer name such as `security-images:`, is not matched. Every head is
    counted: two `security:` rules would be two recipes, and reading the
    first would hide the second. Collects the tab-indented lines that follow until the next
    unindented line, which a column-0 `#` comment between targets also is.
    """
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    target = re.compile(rf"{re.escape(name)}:(?!=)")
    heads = [index for index, line in enumerate(lines) if target.match(line)]
    assert len(heads) == 1, f"the Makefile has {len(heads)} `{name}:` targets"
    recipe: list[str] = []
    for line in lines[heads[0] + 1 :]:
        if line.startswith(("\t", " ")) or line.strip() == "":
            recipe.append(line)
        else:
            break
    return recipe


def recipe_commands(name: str) -> list[str]:
    """The EXECUTED lines of a Make recipe: comments and printers dropped.

    Make's `@` (do not echo) and `+` (run under `-n`) prefixes are stripped so a command
    reads the same with or without them. Make's `-` (ignore failure) is NOT stripped.
    """
    collected: list[str] = []
    for line in makefile_recipe(name):
        stripped = line.strip().lstrip("@+").strip()
        if is_command(stripped):
            collected.append(stripped)
    return collected


def target_prerequisites(name: str) -> list[str]:
    """The prerequisites on the one `name:` rule line, its `##` help text dropped."""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    target = re.compile(rf"{re.escape(name)}:(?!=)")
    heads = [line for line in lines if target.match(line)]
    assert len(heads) == 1, f"the Makefile has {len(heads)} `{name}:` targets"
    return heads[0].split(":", 1)[1].split("##", 1)[0].split()


def joined_commands(lines: list[str]) -> list[str]:
    """Command lines with every trailing-`\\` continuation joined to the next, so an export or
    an audit split across lines reads as the one command it is (rule 45's filter runs first)."""
    joined: list[str] = []
    pending = ""
    for line in lines:
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        joined.append(pending + line)
        pending = ""
    if pending:
        joined.append(pending.rstrip())
    return joined


def one_command(lines: list[str], fragment: str) -> str:
    """The one joined command containing `fragment`; every match is counted, never the first
    one taken, because a second export or a second audit is a second
    contract and picking either silently would hide it."""
    matches = [line for line in joined_commands(lines) if fragment in line]
    assert len(matches) == 1, f"{len(matches)} command lines carry {fragment!r}: {matches}"
    return matches[0]


def gitleaks_images() -> list[str]:
    """Every token of the `security` recipe's commands naming the gitleaks image, quotes
    stripped. Every match is returned, so a second scan is visible."""
    tokens = [
        token.strip("'\"")
        for line in joined_commands(recipe_commands(SECURITY_TARGET))
        for token in line.split()
    ]
    return [token for token in tokens if GITLEAKS_IMAGE_NAME in token]


def gitleaks_rev() -> str:
    """The `rev:` of the gitleaks repo in `.pre-commit-config.yaml`, e.g. `v8.24.3`."""
    repos = load(PRE_COMMIT)["repos"]
    assert isinstance(repos, list)
    entries = [
        entry
        for entry in repos
        if isinstance(entry, dict) and GITLEAKS_REPO in str(entry.get("repo", ""))
    ]
    assert len(entries) == 1, f"{len(entries)} pre-commit repos are {GITLEAKS_REPO}"
    return str(entries[0]["rev"])


def fingerprints() -> list[str]:
    """Every non-comment, non-blank line of `.gitleaksignore`, in file order."""
    assert GITLEAKS_IGNORE.is_file(), f"{GITLEAKS_IGNORE} does not exist"
    lines = GITLEAKS_IGNORE.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for line in lines
        if (stripped := line.strip()) != "" and not stripped.startswith("#")
    ]


def fingerprint_path(fingerprint: str) -> str:
    """The path field of `<commit>:<path>:<rule>:<line>`."""
    fields = fingerprint.split(":")
    assert len(fields) == 4, f"a fingerprint with {len(fields)} fields: {fingerprint}"
    return fields[1]


def allowlists() -> list[dict[str, Any]]:
    """Every `allowlist` table `.gitleaks.toml` declares, whether written as one table or as an
    array of them. An absent file has none, which criterion 6 accepts."""
    if not GITLEAKS_CONFIG.is_file():
        return []
    document = tomllib.loads(GITLEAKS_CONFIG.read_text(encoding="utf-8"))
    declared = document.get("allowlist", [])
    tables = declared if isinstance(declared, list) else [declared]
    return [table for table in tables if isinstance(table, dict)]


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """The parsed workflow, read once for the module."""
    assert WORKFLOW.is_file(), f"{WORKFLOW} does not exist"
    return load(WORKFLOW)


@pytest.fixture(scope="module")
def jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    """The `jobs:` mapping."""
    listed = workflow["jobs"]
    assert isinstance(listed, dict)
    return listed


def test_the_security_job_runs_make_security(jobs: dict[str, Any]) -> None:
    """Catches a `security` job that runs the tools itself, or a neighbouring target
    (`make verify`, `make security-images`), instead of exactly `make security`: the laptop
    and CI would then run different commands again, which is the drift the plan ends. Exactly
    one such step, so a second copy cannot hide beside the first."""
    assert SECURITY_JOB in jobs, f"`ci.yml` declares no `{SECURITY_JOB}` job"
    runs = [str(step.get("run", "")).strip() for step in steps(jobs[SECURITY_JOB])]

    assert runs.count(MAKE_SECURITY) == 1, f"the `{SECURITY_JOB}` job runs {runs}"


@pytest.mark.parametrize("tool", list(DIRECT_TOOLS))
def test_no_job_runs_a_security_tool_directly(jobs: dict[str, Any], tool: str) -> None:
    """Unhappy path, one row per tool, over EVERY job. Catches a `console` job still running
    `pnpm audit`, or any job running gitleaks or pip-audit itself or through a `uses:` action:
    that command then lives in two files, and the copy CI runs drifts from the one
    `make verify` runs. Command lines only, so a comment saying the tool moved to
    the Makefile cannot redden the row."""
    pattern = DIRECT_TOOLS[tool]
    offenders = [line for line in command_lines(jobs) if pattern.search(line)]
    offenders += [
        uses_of(step)
        for job in jobs.values()
        for step in steps(job)
        if pattern.search(uses_of(step))
    ]

    assert offenders == [], f"`ci.yml` runs {tool} directly: {offenders}"


@pytest.mark.parametrize("name", RETIRED_JOBS)
def test_the_old_secrets_and_audit_jobs_are_gone(jobs: dict[str, Any], name: str) -> None:
    """Catches the new `security` job added while the old `secrets` or `audit` job is left
    in place: the old job keeps its own copy of the commands, so CI runs every check twice
    and the two copies drift apart."""
    assert name not in jobs, f"`ci.yml` still declares the `{name}` job"


def test_verify_runs_security() -> None:
    """Catches a `security` target that `verify` does not depend on: `make verify` is then
    green on a laptop while CI's `security` job is red on the same commit, which is exactly
    the broken promise this plan closes."""
    prerequisites = target_prerequisites("verify")

    assert SECURITY_TARGET in prerequisites, f"`verify` runs {prerequisites}"


def test_the_security_recipe_runs_the_console_audit() -> None:
    """Catches `pnpm audit` dropped on the move: it leaves the `console` job, so a recipe
    without `$(call console,audit)` means nothing audits the console anywhere. Exact, so a
    bare `pnpm audit` that skips the web-absent check, or runs from the wrong directory,
    is red too."""
    lines = recipe_commands(SECURITY_TARGET)

    assert lines.count(CONSOLE_AUDIT) == 1, f"the `security` recipe runs {lines}"


def test_the_gitleaks_image_is_pinned_by_digest() -> None:
    """Catches the image pulled by tag alone, `latest`, or through a Make variable: a tag
    can be moved to other bytes on the registry, so without the digest the scan runs
    whatever the tag points at that day. The digest is what replaced the old archive
    checksum. Exactly one image token, so a second unpinned scan cannot sit beside the
    pinned one."""
    images = gitleaks_images()

    assert len(images) == 1, f"the `security` recipe names {len(images)} gitleaks images: {images}"
    assert PINNED_IMAGE.fullmatch(images[0]) is not None, f"not pinned by digest: {images[0]}"


@pytest.mark.parametrize(
    ("candidate", "accepted"),
    [
        (f"zricethezav/gitleaks:v8.24.3@sha256:{SAMPLE_DIGEST}", True),
        ("zricethezav/gitleaks:v8.24.3", False),
        (f"zricethezav/gitleaks:v8.24.3@sha256:{SAMPLE_DIGEST[:-1]}", False),
        (f"zricethezav/gitleaks:v8.24.3@sha256:{SAMPLE_DIGEST}a", False),
        (f"zricethezav/gitleaks:v8.24.3@sha256:{SAMPLE_DIGEST.upper()}", False),
        (f"zricethezav/gitleaks:latest@sha256:{SAMPLE_DIGEST}", False),
        (f"zricethezav/gitleaks:v8.24.3@sha256:{SAMPLE_DIGEST}\n", False),
        (f"zricethezav/gitleaks:v8.24.3@sha256:{SAMPLE_DIGEST}\r\n", False),
        (f"zricethezav/gitleaks:v8.24.3@sha256:{SAMPLE_DIGEST} ", False),
    ],
)
def test_the_image_pin_refuses_a_tag_without_a_digest(candidate: str, accepted: bool) -> None:
    r"""The pin's unhappy rows, run against the same pattern the case above runs over the
    recipe. Catches a pattern loosened to make the digest optional, a quantifier written as a
    minimum or a range (63 or 65 hex accepted), a tag other than the hook's, and (standing rule
    4) an anchored `.match` accepting a trailing newline, CRLF or space. The digest is a
    repeated hex character on purpose: this repository's own scan reads this file."""
    assert (PINNED_IMAGE.fullmatch(candidate) is not None) is accepted


def test_the_audit_file_sits_in_an_ignored_folder() -> None:
    """Catches the export left writing `requirements-audit.txt` at the repo root, as the old
    CI job did: on a laptop that leaves an untracked file at the root after every
    `make verify`. Catches the `/backend/scratch/` line removed from `.gitignore`, where the
    file would show up in `git status` and could be committed."""
    export = one_command(recipe_commands(SECURITY_TARGET), EXPORT_COMMAND)
    written = [
        match.group("flagged") or match.group("redirected")
        for match in EXPORT_OUTPUT.finditer(export)
    ]
    ignored = [line.strip() for line in GITIGNORE.read_text(encoding="utf-8").splitlines()]

    assert written == [AUDIT_FILE], f"the export writes {written}: {export}"
    assert IGNORED_SCRATCH in ignored, f"`.gitignore` does not list {IGNORED_SCRATCH}"


@pytest.mark.parametrize("path", SECURITY_PATHS, ids=lambda path: path.name)
def test_every_named_security_path_exists(path: Path) -> None:
    """a project rule. Catches one of the paths this plan names moved or renamed without the
    plan following; the `backend/` row is the one no other case reads, and it catches the
    project moved while the recipe still exports `--project backend`."""
    assert path.exists(), f"{path} does not exist"


def test_the_secrets_job_checks_out_the_whole_history(jobs: dict[str, Any]) -> None:
    """Now reads the `security` job. Catches its checkout left at the default
    `fetch-depth: 1`, which gives the runner ONE commit: the history scan then reports clean
    over a repository whose secret is three commits back. Catches the checkout dropped
    entirely. Every checkout step is counted, so a second shallow one cannot be the one the
    scan reads."""
    assert SECURITY_JOB in jobs, f"`ci.yml` declares no `{SECURITY_JOB}` job"
    checkouts = [step for step in steps(jobs[SECURITY_JOB]) if "actions/checkout" in uses_of(step)]

    assert len(checkouts) == 1, f"{len(checkouts)} checkout steps in the `{SECURITY_JOB}` job"
    depth = with_of(checkouts[0]).get("fetch-depth")
    assert str(depth) == "0", f"the checkout fetches depth {depth!r}, not the whole history"


@pytest.mark.parametrize(
    ("label", "pattern"), REQUIRED_SCAN_FRAGMENTS, ids=[row[0] for row in REQUIRED_SCAN_FRAGMENTS]
)
def test_the_secrets_job_scans_history_redacted_and_fails_on_a_finding(
    label: str, pattern: re.Pattern[str]
) -> None:
    """Now reads the `security` recipe's one scan command, one row per fragment. `git` catches
    the scan pointed at the working directory (`dir`), which never reads the history;
    `--redact` catches a finding printed in full into a public Actions log; `--exit-code 1`
    catches a scan that reports and exits 0. Each row reads the SCAN command itself, so the
    flag on some other recipe line cannot satisfy it."""
    scan = one_command(recipe_commands(SECURITY_TARGET), GITLEAKS_IMAGE_NAME)

    assert pattern.search(scan) is not None, f"the scan does not carry {label}: {scan}"


@pytest.mark.parametrize("flag", FORBIDDEN_SCAN_FLAGS)
def test_the_secrets_job_carries_neither_staging_area_flag(flag: str) -> None:
    """Now reads the `security` recipe, one row per flag. Both are the LOCAL hook's flags,
    and the likeliest way the recipe gets written is by copying the hook's entry from
    `.pre-commit-config.yaml` - at which point the scan narrows to a staging area, finds
    nothing by construction, and exits 0 forever. Command lines only, so a comment
    explaining why the flags are absent cannot fail the row."""
    lines = recipe_commands(SECURITY_TARGET)

    assert lines != [], "the `security` recipe runs no command at all"
    offenders = [line for line in lines if flag in line]
    assert offenders == [], f"the scan is narrowed by {flag}: {offenders}"


def test_the_installed_version_is_the_one_the_commit_hook_pins() -> None:
    """Now reads the recipe's image TAG, and a project rule. Catches `make security`
    scanning with a different gitleaks than the commit hook uses - say the hook bumped to a
    new `rev` while the recipe stayed: the two then disagree about what a secret is. The digest
    pin above cannot see this row, because it pins its own copy of the tag."""
    images = gitleaks_images()

    assert len(images) == 1, f"the `security` recipe names {len(images)} gitleaks images: {images}"
    tag = images[0].split(":", 1)[1].split("@", 1)[0]
    assert tag == gitleaks_rev(), f"the commit hook pins {gitleaks_rev()}, the recipe runs {tag}"
    assert tag == f"v{GITLEAKS_VERSION}", f"the recipe runs {tag}, not v{GITLEAKS_VERSION}"


@pytest.mark.parametrize("name", [SECURITY_JOB])
def test_neither_new_job_may_fail_without_failing_the_build(
    jobs: dict[str, Any], name: str
) -> None:
    """Now over the one `security` job, checked at BOTH levels. `continue-on-error` is the
    repair somebody reaches for the first time a scan lands red on an unrelated pull request,
    and it turns a gate into a notification: the job still runs and the build is green
    regardless. The step level is checked separately because one flagged step - the
    `make security` step - is enough to swallow the only command that fails."""
    assert name in jobs, f"`ci.yml` declares no `{name}` job"
    job = jobs[name]

    assert job.get("continue-on-error", False) is False, f"the `{name}` job cannot fail the build"
    for step in steps(job):
        assert step.get("continue-on-error", False) is False, (
            f"a `{name}` step cannot fail the job: {step.get('name', step)}"
        )


def test_the_ignore_file_holds_exactly_the_sixteen_reviewed_fingerprints() -> None:
    """Case 7, by COUNT equality. The count is the review: 16 findings
    were read and each judged a fake literal in a test. A 17th line is a
    finding nobody read, added by whoever needed the job green that afternoon - which is exactly
    how a fingerprint file stops being an audited exception list. A count below 16 catches the
    other direction, a fingerprint deleted so the job goes red on committed test data. Comment
    lines are dropped, so the file may explain itself as freely as it likes."""
    lines = fingerprints()

    assert len(lines) == EXPECTED_FINGERPRINTS, f"{len(lines)} fingerprints: {lines}"


def test_every_ignored_line_is_a_single_generic_api_key_fingerprint() -> None:
    """Case 8, with `re.fullmatch`. Catches a line that is not a
    fingerprint at all: gitleaks also accepts a bare commit SHA there, which allowlists EVERY
    finding in that commit, and a rule id on its own, which disables the rule repository-wide.
    Both are one short line away from the correct form and neither is visible as a widening in
    review. The rule field is pinned to `generic-api-key` so an exception for some other rule -
    a real private key, an AWS credential - cannot be slipped in beside these."""
    for fingerprint in fingerprints():
        assert FINGERPRINT_PATTERN.fullmatch(fingerprint) is not None, (
            f"not a `<commit>:<path>:generic-api-key:<line>` fingerprint: {fingerprint}"
        )


@pytest.mark.parametrize(
    ("candidate", "accepted"),
    [
        (f"{SAMPLE_COMMIT}:tests/x.py:generic-api-key:1", True),
        (f"{SHORT_COMMIT}:tests/x.py:generic-api-key:1", False),
        (f"{SAMPLE_COMMIT}:tests/x.py:generic-api-key:1\n", False),
    ],
)
def test_the_fingerprint_pattern_refuses_a_short_commit(candidate: str, accepted: bool) -> None:
    r"""Case 9's boundary, 40 against 39, asserted directly against the module's pattern so
    the rows exercise the same constant the case above runs over the real file. The 39 row
    catches a quantifier written as a minimum or a range (`{7,40}`, `{40,}`), which would accept
    an ABBREVIATED commit - and an abbreviated prefix in a fingerprint matches nothing gitleaks
    emits, so the exception silently stops applying and the job goes red with no line to point
    at. The trailing-newline row is a project rule: `$` in Python matches before a final
    newline, so an anchored `.match` would accept a fingerprint with one and `re.fullmatch` is
    what refuses it. The commit fields are a repeated hex character on purpose - a realistic
    digest written here is the first thing this plan's own scan would flag."""
    assert (FINGERPRINT_PATTERN.fullmatch(candidate) is not None) is accepted


def test_every_ignored_path_sits_under_a_test_directory() -> None:
    """Case 10. Catches a fingerprint allowing a literal in shipped code - a key pasted into
    a provider adapter, a settings default, a manifest - which is precisely the find this whole
    job exists to make impossible, granted an exemption in the one file nobody re-reads. The
    directories are the plan's two, so a path under `packages/` or `infra/` is red however
    plausible its line looks. The failing row below shares this predicate, so a widened
    predicate cannot pass both."""
    offenders = [
        fingerprint
        for fingerprint in fingerprints()
        if not fingerprint_path(fingerprint).startswith(ALLOWED_FINGERPRINT_PREFIXES)
    ]

    assert offenders == [], f"fingerprints outside the test directories: {offenders}"


def test_a_source_path_is_not_an_allowed_fingerprint_location() -> None:
    """Case 10's unhappy row, run against the same predicate the file-wide case uses.
    Without it that case passes vacuously the day somebody widens the prefix tuple to `("",)`
    or to `"packages/"` while adding the exception they wanted: every real fingerprint still
    passes, and nothing says the control stopped controlling. This row is the one that flips."""
    source_fingerprint = f"{SAMPLE_COMMIT}:{REJECTED_FINGERPRINT_PATH}:generic-api-key:1"

    assert FINGERPRINT_PATTERN.fullmatch(source_fingerprint) is not None, "the row is malformed"
    assert not fingerprint_path(source_fingerprint).startswith(ALLOWED_FINGERPRINT_PREFIXES)


@pytest.mark.parametrize("key", FORBIDDEN_ALLOWLIST_KEYS)
def test_no_allowlist_widens_the_scan_by_path_or_regex(key: str) -> None:
    """Case 11 and decision A, parsed with `tomllib`. Catches the cheap repair for a future
    finding: one `paths = ['tests/']` line allows every current literal AND every key pasted
    into any test file from then on, with no diff to review when it happens, whereas a
    fingerprint pins one line of one blob in one commit. `regexes` is the same widening by
    another spelling. An absent `.gitleaks.toml` is the state the plan expects and passes
    without asserting anything about a file nobody wrote; both forms of the table - one table
    or an array of them - are walked, so a second allowlist cannot hide behind the first."""
    for table in allowlists():
        assert key not in table, f"`.gitleaks.toml` widens the scan with `{key}`: {table[key]}"


@pytest.mark.parametrize("flag", EXPORT_FLAGS)
def test_the_audit_exports_the_locked_dependency_set(flag: str) -> None:
    """Now reads the `security` recipe, one row per flag. `--locked` is the row that matters:
    without it `uv export` may re-resolve, so the audit reads a dependency set the lockfile
    never had. `--format requirements-txt` catches an export pip-audit cannot read.
    `--no-emit-project` catches the local package emitted as an editable entry, which has no
    hash and fails `--require-hashes`. `--all-groups` catches an export of the runtime group
    alone, leaving every dev and test dependency unaudited. Continuations are joined, so a
    flag on its own continuation line still reads."""
    command = one_command(recipe_commands(SECURITY_TARGET), EXPORT_COMMAND)

    assert flag in command, f"the export does not carry {flag}: {command}"


@pytest.mark.parametrize("flag", AUDIT_FLAGS)
def test_the_audit_runs_the_pinned_pip_audit_strictly(flag: str) -> None:
    """Now reads the `security` recipe, one row per flag. `pip-audit==2.9.0` catches an
    unpinned `uvx pip-audit`, whose behaviour changes without a commit. `--strict` catches the
    default, where a dependency pip-audit cannot resolve is only a WARNING. `--require-hashes`
    catches an audit that accepts whatever the index serves rather than what the lockfile
    pinned. `--disable-pip` catches the resolver being allowed to reach out and install, which
    lets the audited set drift from the exported one."""
    command = one_command(recipe_commands(SECURITY_TARGET), AUDIT_COMMAND)

    assert flag in command, f"the audit does not carry {flag}: {command}"


def test_the_audit_reads_exactly_the_file_the_export_wrote() -> None:
    """Now reads the `security` recipe, both ends EXTRACTED and compared rather than either
    one retyped here. Catches the export and the audit drifting apart - an export renamed
    while the audit still reads the old name - where pip-audit either fails unreadably or, if
    a stale file of that name exists from an earlier local run, audits the wrong dependency
    set and reports green. Exactly one export and one audit, so the comparison cannot be a
    pick."""
    lines = recipe_commands(SECURITY_TARGET)
    export = one_command(lines, EXPORT_COMMAND)
    audit = one_command(lines, AUDIT_COMMAND)

    written = [
        match.group("flagged") or match.group("redirected")
        for match in EXPORT_OUTPUT.finditer(export)
    ]
    read = [match.group("path") for match in AUDIT_INPUT.finditer(audit)]

    assert len(written) == 1, f"the export writes {written} files: {export}"
    assert len(read) == 1, f"the audit reads {read} files: {audit}"
    assert written[0] == read[0], f"the export writes {written[0]}, the audit reads {read[0]}"
