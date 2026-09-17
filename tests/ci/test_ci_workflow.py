r"""`.github/workflows/ci.yml`: the two scans #59 names that nothing in this repository runs.

An earlier change adds a `secrets` job and an `audit` job. Neither can be OBSERVED from a pytest
run: the first needs a full-depth clone and a downloaded binary, the second needs a network
fetch of an advisory database, and both are gates whose whole value is that they run on a
GitHub runner over every commit. So this file pins WIRING, exactly as
`tests/ci/test_deploy_workflow.py` does for the deploy: the jobs exist, they carry the flags
that make them scan the thing they claim to scan, they carry none of the flags that would
quietly narrow that scan, and nothing lets either one fail without failing the build. The
plan's own step 5 runs the pinned binary over full history and pastes the result; that
evidence is not a case here.

Three things are checked against another file rather than written down twice:

* the gitleaks version, read out of `.pre-commit-config.yaml`'s `rev: v8.24.3` and compared
  to the `8.24.3` the workflow must carry LITERALLY. a project rule is the whole point and
  also the cost: a version factored into an `env:` var or a YAML anchor is correct YAML that
  a static reader sees as an expression equal to nothing, so the drift this comparison exists
  to catch - CI scanning with one version while the commit hook uses another - becomes
  invisible. The repetition in the workflow is deliberate.
* the requirements file the audit reads, extracted from BOTH the export step's output flag
  and the `-r` argument of the pip-audit invocation and compared to each other. A path
  retyped into this module would prove only that this module can retype it; extracting both
  ends means an export writing one name and an audit reading another is red, which is the
  shape that audits an empty or stale file and reports green.
* `.gitleaksignore`'s fingerprints, whose PATHS are read out of the file itself rather than
  listed here. The file is the record of which literals were allowed; this suite's job is to
  keep that list to the exact shape decision A chose.

The document is read with `yaml.safe_load` and the repository root comes from this file's
own `__file__`, never from the working directory, so the suite behaves the same run from
`tests/` as from the repo root. `ci.yml` exists today, so the `workflow` fixture is green
before the plan starts; the `secrets` and `audit` jobs and `.gitleaksignore` do not, which is
where every case below is currently red.

`ci.yml`'s `on:` block is not read here: the plan's §6 lists fourteen cases and the trigger
is not one of them, since both new jobs simply inherit the block the file already has. Worth
knowing for the case that is added the day it matters: the bare key `on` is YAML 1.1's
boolean, so `yaml.safe_load` returns that block under the key `True` rather than under
`"on"`, and `document["on"]` raises `KeyError` on a perfectly correct workflow.

Selection, and a project rule. A step that merely ECHOES a fragment matches a fragment
search exactly as the real command does - a banner, a visible-skip message, a comment
explaining what is deliberately NOT passed here. Every case that selects or scans by `run:`
text therefore goes through `command_lines()`, which drops comment lines and lines carrying
`echo` or `printf`. That matters in both directions for this plan: a download step printing
"verifying with sha256sum --check" must not satisfy the case that requires the verification,
and a comment saying "no --staged flag here, CI has no staging area" must not redden the case
that forbids `--staged`. The version case is the one deliberate whole-JOB scan and says so:
there the claim is about the job's text, `env:` and comments included, because a version
named anywhere in that job is a version somebody will read as the one being installed.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
GITLEAKS_IGNORE = REPO_ROOT / ".gitleaksignore"
GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"

#: Cases 1 and 12's two new jobs, by name. Read individually rather than by set equality
#: over `jobs:`, because the six jobs already in `ci.yml` are not this plan's business.
SECRETS_JOB = "secrets"
AUDIT_JOB = "audit"

#: Case 2: the scan, its redaction and its failing exit, each written as the plan writes
#: them. `--exit-code 1` is what makes a finding fail the build rather than log it.
REQUIRED_SCAN_FRAGMENTS = ("gitleaks git", "--redact", "--exit-code 1")

#: Case 3: these are the local hook's flags, and both narrow the scan to a staging area
#: a runner does not have. Either one turns a full-history scan into a scan of nothing.
FORBIDDEN_SCAN_FLAGS = ("--staged", "--pre-commit")

#: Case 4's verification, and the fragment that identifies the line the scan runs on.
CHECKSUM_CHECK = "sha256sum --check"
SCAN_COMMAND = "gitleaks git"

#: Case 5, pinned by value and cross-read from `.pre-commit-config.yaml`
#:. The hook pins it as `v8.24.3`; the workflow must say the number itself.
GITLEAKS_VERSION = "8.24.3"
GITLEAKS_REPO = "github.com/gitleaks/gitleaks"

#: Case 7: the count the full-history run produced at drafting time, exact. A count is an
#: absolute, so it is asserted by equality and not as a ceiling.
EXPECTED_FINGERPRINTS = 14

#: Case 8's fingerprint shape, run with `re.fullmatch` and therefore written WITHOUT the
#: `^`/`$` the plan quotes it with: a project rule, since in Python `$` also matches before a
#: trailing newline and `.match` with anchors would accept a fingerprint carrying one.
FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{40}:[^:]+:generic-api-key:\d+")

#: Case 10's directories. Every allowed literal is a fake one written by a test, so a
#: fingerprint naming anything else is an exception nobody reviewed as one.
ALLOWED_FINGERPRINT_PREFIXES = ("tests/", "services/console/tests/")

#: Cases 9 and 10's unhappy rows, built from a repeated hex character rather than typed as a
#: realistic digest: this repository's own `secrets` job scans this file's history with the
#: `generic-api-key` rule, and a long random-looking literal here would be its first finding.
SAMPLE_COMMIT = "a" * 40
SHORT_COMMIT = "a" * 39
REJECTED_FINGERPRINT_PATH = "packages/opsconsole/x.py"

#: Case 11: the two `[allowlist]` keys decision A rejects. A path or a regex allowlist also
#: hides a real key pasted into the same file later; a fingerprint pins one line of one blob.
FORBIDDEN_ALLOWLIST_KEYS = ("paths", "regexes")

#: Case 12's export, all four flags. `--locked` is the one that matters most: without it
#: `uv export` may RESOLVE afresh and audit a dependency set the lockfile never had.
EXPORT_COMMAND = "uv export"
EXPORT_FLAGS = ("--format requirements-txt", "--locked", "--no-emit-project", "--all-groups")

#: Case 13's audit, with the version pinned in the `--from` reference (decision C: the
#: tool is CI's, not the package's, so it never enters the lockfile).
AUDIT_COMMAND = "pip-audit"
AUDIT_FLAGS = ("pip-audit==2.9.0", "--strict", "--require-hashes", "--disable-pip")

#: Case 14: where the export writes its file, accepting the three ways a shell command can
#: say it - `-o path`, `--output-file path` and a `>` redirect - and where the audit reads it.
#: `-r` is required to stand alone, so `--require-hashes` cannot be read as an `-r`.
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


def render(value: object) -> str:
    """A whole YAML subtree as one searchable string, keys and values alike."""
    return yaml.safe_dump(value, default_flow_style=False, sort_keys=True)


def command_lines(jobs: dict[str, Any]) -> list[str]:
    """Every EXECUTED line of every `run:` script: comment lines and printers dropped.

    a project rule. A step that echoes `gitleaks git --redact` or prints `sha256sum --check`
    into the log matches a fragment search exactly as the real command does, and a comment
    explaining that `--staged` is deliberately absent carries `--staged`. So a workflow with a
    helpful banner would satisfy the cases that require a command, and a workflow documenting
    its own control would redden the case that forbids one. Lines carrying `echo` or `printf`
    and lines whose first character is `#` are therefore not commands here, and every case
    below says which side of that filter its claim is on.
    """
    collected: list[str] = []
    for job in jobs.values():
        assert isinstance(job, dict), "a job is not a mapping"
        for step in steps(job):
            for line in str(step.get("run", "")).splitlines():
                stripped = line.strip()
                if stripped == "" or stripped.startswith("#"):
                    continue
                if "echo" in stripped or "printf" in stripped:
                    continue
                collected.append(stripped)
    return collected


def job_commands(jobs: dict[str, Any], name: str) -> list[str]:
    """One named job's command lines, in file order; a missing job is the failure it is."""
    assert name in jobs, f"`ci.yml` declares no `{name}` job"
    return command_lines({name: jobs[name]})


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


def test_the_secrets_job_checks_out_the_whole_history(jobs: dict[str, Any]) -> None:
    """Case 1's first half. Catches the job added with a plain `actions/checkout`, whose
    default `fetch-depth: 1` gives the runner ONE commit: `gitleaks git` then reports clean
    over a repository whose secret is three commits back, and the job is green for the same
    reason it is useless. Catches the checkout dropped entirely, where the scan runs over an
    empty directory. Every checkout step in the job is required to carry the depth, so a second
    one added later cannot be the shallow one the scan actually reads."""
    assert SECRETS_JOB in jobs, f"`ci.yml` declares no `{SECRETS_JOB}` job"
    checkouts = [step for step in steps(jobs[SECRETS_JOB]) if "actions/checkout" in uses_of(step)]

    assert len(checkouts) == 1, f"{len(checkouts)} checkout steps in the `{SECRETS_JOB}` job"
    depth = with_of(checkouts[0]).get("fetch-depth")
    assert str(depth) == "0", f"the checkout fetches depth {depth!r}, not the whole history"


@pytest.mark.parametrize("fragment", REQUIRED_SCAN_FRAGMENTS)
def test_the_secrets_job_scans_history_redacted_and_fails_on_a_finding(
    jobs: dict[str, Any], fragment: str
) -> None:
    """Case 2, one row per fragment. `gitleaks git` catches the scan pointed
    at the working directory (`gitleaks dir`), which reads the checked-out files and never the
    history the plan exists to cover; `--redact` catches a finding printed in full into a public
    Actions log, which turns the alarm into a disclosure; `--exit-code 1` catches a scan that
    reports and exits 0, where the finding is a line in a green job nobody opens. Read over
    COMMAND lines only, so a banner naming the command cannot satisfy the row."""
    lines = job_commands(jobs, SECRETS_JOB)

    assert lines != [], f"the `{SECRETS_JOB}` job runs no command at all"
    assert any(fragment in line for line in lines), f"no command carries {fragment!r}: {lines}"


@pytest.mark.parametrize("flag", FORBIDDEN_SCAN_FLAGS)
def test_the_secrets_job_carries_neither_staging_area_flag(jobs: dict[str, Any], flag: str) -> None:
    """Case 3, one row per flag. Both are the LOCAL hook's flags, and the likeliest way
    this job gets written is by copying the hook's entry from `.pre-commit-config.yaml` - at
    which point the scan narrows to a staging area a runner never has, finds nothing by
    construction, and exits 0 on every run forever. That is the failure this plan was drafted
    to close, arriving through the back door. Absence is the control, so absence is the
    assertion, and it is scanned over COMMAND lines so a comment explaining why the flags are
    absent cannot be what fails the row."""
    lines = job_commands(jobs, SECRETS_JOB)

    assert lines != [], f"the `{SECRETS_JOB}` job runs no command at all"
    offenders = [line for line in lines if flag in line]
    assert offenders == [], f"the scan is narrowed by {flag}: {offenders}"


def test_the_archive_is_verified_before_the_binary_runs(jobs: dict[str, Any]) -> None:
    """Case 4, and the ORDER is the whole claim. Catches the verification omitted, where
    a job that exists to find secrets downloads and executes an unverified binary over the
    entire history of the repository - the checksum is what makes the pin mean anything.
    Catches the check placed AFTER the scan, which is the version that looks right in a diff
    and verifies an artefact that has already run. Indices come from COMMAND lines, so a step
    echoing either command is neither of the two lines being ordered."""
    lines = job_commands(jobs, SECRETS_JOB)
    checks = [index for index, line in enumerate(lines) if CHECKSUM_CHECK in line]
    scans = [index for index, line in enumerate(lines) if SCAN_COMMAND in line]

    assert len(checks) == 1, f"{len(checks)} command lines run {CHECKSUM_CHECK}: {lines}"
    assert len(scans) == 1, f"{len(scans)} command lines run {SCAN_COMMAND}: {lines}"
    assert checks[0] < scans[0], f"the binary runs before it is verified: {lines}"


def test_the_installed_version_is_the_one_the_commit_hook_pins(jobs: dict[str, Any]) -> None:
    """Case 5 and a project rule. Catches CI installing a different gitleaks than the
    commit hook uses: the two then disagree about what a secret is, and the divergence surfaces
    as a job that fails on a commit which passed locally - or, far worse, passes on one the hook
    would have caught. Catches the version factored into an `env:` var, a `matrix` entry or a
    `latest` tag, all of which leave the job with no literal to compare and turn the pin into
    whatever ships that week. A whole-JOB scan on purpose (`env:` and comments included): a
    version named anywhere in this job is the version the next reader believes is installed."""
    assert SECRETS_JOB in jobs, f"`ci.yml` declares no `{SECRETS_JOB}` job"
    text = render(jobs[SECRETS_JOB])

    assert GITLEAKS_VERSION in text, f"the `{SECRETS_JOB}` job never says {GITLEAKS_VERSION}"
    assert gitleaks_rev().removeprefix("v") == GITLEAKS_VERSION, (
        f"the commit hook pins {gitleaks_rev()}, the workflow installs {GITLEAKS_VERSION}"
    )


@pytest.mark.parametrize("name", [SECRETS_JOB, AUDIT_JOB])
def test_neither_new_job_may_fail_without_failing_the_build(
    jobs: dict[str, Any], name: str
) -> None:
    """Case 6, one row per job, checked at BOTH levels. `continue-on-error` is the repair
    somebody reaches for the first time either scan lands red on an unrelated pull request, and
    it converts a gate into a notification: the job still runs, still reports its finding, and
    the build is green regardless - which is indistinguishable from having no scan at all, while
    looking in the Actions tab exactly like having one. The step level is checked separately
    because a single flagged step is enough to swallow the only command that fails."""
    assert name in jobs, f"`ci.yml` declares no `{name}` job"
    job = jobs[name]

    assert job.get("continue-on-error", False) is False, f"the `{name}` job cannot fail the build"
    for step in steps(job):
        assert step.get("continue-on-error", False) is False, (
            f"a `{name}` step cannot fail the job: {step.get('name', step)}"
        )


def test_the_ignore_file_holds_exactly_the_fourteen_reviewed_fingerprints() -> None:
    """Case 7, by COUNT equality. The count is the review: 14 findings
    were read at drafting time and each judged a fake literal in a test. A 15th line is a
    finding nobody read, added by whoever needed the job green that afternoon - which is exactly
    how a fingerprint file stops being an audited exception list. A count below 14 catches the
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
def test_the_audit_exports_the_locked_dependency_set(jobs: dict[str, Any], flag: str) -> None:
    """Case 12, one row per flag. `--locked` is the row that matters: without it `uv
    export` may re-resolve, so the audit reads a dependency set the lockfile never had and
    reports on packages CI will not install. `--format requirements-txt` catches an export in
    uv's own format, which pip-audit cannot read. `--no-emit-project` catches the local package
    emitted as an editable entry, which has no hash and fails `--require-hashes` outright.
    `--all-groups` catches an export of the runtime group alone, leaving every dev and test
    dependency - the ones that pull the largest transitive trees - unaudited. COMMAND lines with
    continuations joined, so a flag on its own continuation line still reads."""
    command = one_command(job_commands(jobs, AUDIT_JOB), EXPORT_COMMAND)

    assert flag in command, f"the export does not carry {flag}: {command}"


@pytest.mark.parametrize("flag", AUDIT_FLAGS)
def test_the_audit_runs_the_pinned_pip_audit_strictly(jobs: dict[str, Any], flag: str) -> None:
    """Case 13, one row per flag. `pip-audit==2.9.0` catches an unpinned `uvx pip-audit`,
    whose behaviour and vulnerability set change under CI without a commit - the class of gate
    that fails on a Tuesday for no reason in the diff. `--strict` catches the default, where a
    dependency pip-audit cannot resolve is a WARNING and the job is green having audited less
    than it says. `--require-hashes` catches an audit of an unhashed requirements file, which
    accepts whatever the index serves rather than what the lockfile pinned. `--disable-pip`
    catches the resolver being allowed to reach out and install, which both slows the job and
    lets the audited set drift from the exported one."""
    command = one_command(job_commands(jobs, AUDIT_JOB), AUDIT_COMMAND)

    assert flag in command, f"the audit does not carry {flag}: {command}"


def test_the_audit_reads_exactly_the_file_the_export_wrote(jobs: dict[str, Any]) -> None:
    """Case 14, both ends EXTRACTED and compared rather than either one retyped here. This
    is the case that catches the two steps drifting apart - an export renamed to
    `requirements-ci.txt` while the audit still reads `requirements.txt` - which on a runner is
    not an error at the point it matters: pip-audit is handed a path, and the failure is either
    a red job nobody can read or, if a file of that name exists from some earlier step, a green
    audit of the wrong dependency set. Neither step's own flag case above can see it, because
    each is individually correct. Requires exactly one export and one audit, so a second
    invocation of either cannot make the comparison a pick."""
    lines = job_commands(jobs, AUDIT_JOB)
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
