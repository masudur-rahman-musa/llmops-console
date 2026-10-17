"""Argument checking and parser for `ops-eval`.

Everything that happens before a command runs: the class that signals a bad
argument, every pre-check function, and the `_parser` that wires them up.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import DEFAULT_ROUTING_POLICY_PATH, Settings
from app.core.validation import checked_identifier
from app.evals.baseline_guard import (
    MAX_REF_LENGTH,
    REF_PATTERN,
    Commit,
)
from app.evals.calibration import (
    read_calibration,
)
from app.evals.judge import judge_prompt_version
from app.evals.report import (
    SET_SHA_PATTERN,
    RunCalibration,
)
from app.model_routing.errors import PolicyInvalid
from app.model_routing.policy import Policy, load_policy

ALLOWED_TARGET_SCHEMES: frozenset[str] = frozenset({"http", "https"})

RUN_KINDS: tuple[str, ...] = ("ci", "nightly", "manual")

DEFAULT_RUN_KIND: str = "manual"


class UsageError(Exception):
    """One argument was refused. Reported on stderr, and answered with `EXIT_USAGE`."""


def _checked_tenant(value: str) -> str:
    """Validate `--tenant` against the gateway's own identifier vocabulary.

    The cap and the pattern are imported rather than restated, so a tenant the
    running gateway serves cannot be one the harness refuses to evaluate
    (`test_the_tenant_length_cap_is_the_resolvers_own`). `re.fullmatch` and never
    `.match`: in Python `$` also matches before a trailing newline (standing
    rule 4), and this value names a directory under `--out`.
    """
    return checked_identifier(value, "tenant")


def _checked_target(value: str) -> str:
    """Refuse anything but an absolute `http` or `https` URL.

    An unchecked value produces an `httpx` error mid-run, after the seed has been
    reset and the first cases have been paid for; `file://` is a scheme no eval
    target speaks.
    """
    parsed = urlparse(value)
    if parsed.scheme not in ALLOWED_TARGET_SCHEMES or not parsed.netloc:
        raise UsageError(f"--target: {value!r} is not an absolute http(s) URL")
    return value


def _checked_judge_url_value(value: str) -> str:
    """Validate a `--judge-url` that was given, by `_checked_target`'s own rule.

    The same function and not a second copy of the rule: the two URLs this command
    takes would otherwise drift, and a judge URL of `gateway.example` is the same
    defect as a target of one
    (`test_a_judge_url_that_is_not_an_absolute_http_url_is_refused` and
    `test_a_calibrate_judge_url_that_is_not_an_absolute_http_url_is_refused` both
    parametrise the target's own rejection table). The message still names the
    flag the operator typed.

    Split from `_checked_judge_url` because the flag is OPTIONAL on `run` and
    REQUIRED on `calibrate` (17e): one function answering `str | None` would
    leave the required caller narrowing a value argparse already guaranteed.
    """
    try:
        return _checked_target(value)
    except UsageError:
        raise UsageError(f"--judge-url: {value!r} is not an absolute http(s) URL") from None


def _checked_judge_url(value: str | None) -> str | None:
    """`--judge-url` on `run`, where absent means the judge is simply off."""
    if value is None:
        return None
    return _checked_judge_url_value(value)


def _checked_judge_key() -> str:
    """The judge key `calibrate` cannot run without.

    Where `run` turns the judge OFF with no key and keeps its own exit code
    (17c decision B), a calibration with no judge has nothing left to do, so it
    refuses rather than writing an artifact with no kappa in it - a
    `calibration.json` that exists is read by 17f as trust
    (`test_calibrate_with_no_judge_key_is_a_usage_refusal`). An empty string is
    the same as absent: a `JudgeClient` built on one would spend a 401 per sample.
    """
    api_key = Settings().eval_judge_key
    if not api_key:
        raise UsageError(
            "OPSCONSOLE_EVAL_JUDGE_KEY: not set, and a calibration cannot be made without the judge"
        )
    return api_key


def _checked_calibration(value: Path | None, judge_url: str | None) -> RunCalibration | None:
    """Read `--calibration`, or answer `None` when the caller passed none.

    Three refusals, all of them here in the pre-check block and none of them after
    a request has been made: the file cannot be read or is not a calibration
    (`read_calibration`'s own `CalibrationSetInvalid`, printed as one line by the
    caller's guard); the artifact was computed under a DIFFERENT judge prompt than
    this run will score with, which is a stale number worn as trust and whose fix
    is to recalibrate (decision D,
    `test_a_calibration_computed_against_another_prompt_is_refused_before_the_run`);
    and `--calibration` with no `--judge-url`, where the run judges nothing and a
    kappa beside a null `judge_mean` is agreement about scores this run never
    produced (decision E,
    `test_a_calibration_with_no_judge_url_is_refused_before_the_run`).

    What comes back is the narrowed `RunCalibration` a report publishes, not 17e's
    whole document: the rows, the drop count and the clock stay in
    `calibration.json`, which is the file a reader is sent to by `labels_name` and
    `labels_sha`.
    """
    if value is None:
        return None
    if judge_url is None:
        raise UsageError(
            "--calibration: needs --judge-url, or the run judges nothing and the kappa "
            "would stand beside a null judge_mean"
        )
    report = read_calibration(value)
    if report.judge_prompt_version != judge_prompt_version():
        raise UsageError(
            f"--calibration: {value} was computed under judge prompt "
            f"{report.judge_prompt_version}, and this run scores with "
            f"{judge_prompt_version()}"
        )
    return RunCalibration(
        judge_prompt_version=report.judge_prompt_version,
        labels_name=report.labels_name,
        labels_sha=report.labels_sha,
        pairs=report.pairs,
        kappa=report.kappa,
        trusted=report.trusted,
    )


def _checked_ref(value: str, flag: str) -> str:
    """Validate one `--base` or `--head` before git is asked anything (risk 5).

    The cap and the pattern come from `baseline_guard`, and the order is the point:
    both values reach a subprocess argument list, so a value starting `-` is an
    option to `git log` and a value carrying a newline is a second argument. Neither
    is refusable once the command line is built, which is why this runs first
    (`test_a_ref_failing_the_pre_check_is_refused_before_any_git_call`).

    The length is measured before the pattern is applied, and the boundary is
    inclusive - 256 in, 257 out
    (`test_a_ref_of_exactly_the_cap_gets_past_the_pre_check`).
    """
    if len(value) > MAX_REF_LENGTH:
        raise UsageError(f"{flag}: longer than {MAX_REF_LENGTH} characters")
    if REF_PATTERN.fullmatch(value) is None:
        raise UsageError(
            f"{flag}: not a ref this command will pass to git "
            "(letters, digits, `.`, `_`, `/` and `-`, no leading `-`, no `..`)"
        )
    return value


def _commits(base: str, head: str) -> tuple[Commit, ...]:
    r"""Read `base..head` once and answer what each commit touched.

    `git log --format=%H%x00%s --name-status -M100%` prints a header line
    `<sha>\x00<subject>`, then a blank line, then one `<status>\t<path>` line per file
    until the next header. The NUL is what separates the sha from a subject that may
    contain anything, including the colon every well-formed baseline commit's subject
    carries.

    `-M100%` makes git report only a byte-identical move as `R100\told\tnew`; any edit
    turns it into a delete and an add, so it is judged like any change
    (`test_baseline_guard_fails_a_move_that_also_edits_the_baseline`). Both sides of
    a move go into `files`, so moving the baseline away is still seen
    (`test_baseline_guard_fails_a_commit_that_moves_the_baseline_away`).

    A range git cannot resolve exits non-zero, and that is reported as a refusal
    rather than as an empty commit list: a shallow checkout would otherwise let this
    guard report clean having read nothing, which is risk 2
    (`test_a_ref_git_cannot_resolve_is_one_usage_line_and_never_a_clean_verdict`).
    git's own message runs to three lines, so what is printed is this command's own
    one line naming the range.
    """
    revisions = f"{base}..{head}"
    try:
        finished = subprocess.run(
            ["git", "log", "--format=%H%x00%s", "--name-status", "-M100%", revisions],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UsageError(f"the range {revisions} could not be read by git: {exc}") from exc
    if finished.returncode != 0:
        raise UsageError(f"the range {revisions} could not be read by git")

    commits: list[Commit] = []
    shas: list[str] = []
    subjects: list[str] = []
    touched: list[list[str]] = []
    moves: list[list[tuple[str, str]]] = []
    for line in finished.stdout.splitlines():
        if "\x00" in line:
            sha, _, subject = line.partition("\x00")
            shas.append(sha)
            subjects.append(subject)
            touched.append([])
            moves.append([])
        elif line and touched:
            status, *paths = line.split("\t")
            touched[-1].extend(paths)
            if status == "R100" and len(paths) == 2:
                moves[-1].append((paths[0], paths[1]))
    for sha, subject, files, exact in zip(shas, subjects, touched, moves, strict=True):
        commits.append(
            Commit(sha=sha, subject=subject, files=tuple(files), exact_moves=tuple(exact))
        )
    return tuple(commits)


def _checked_out(value: Path) -> Path:
    """Create the report directory, or refuse the path.

    A path that exists as a FILE is refused rather than replaced: the two report
    writes would fail at the end of a paid-for run, and an implementation that
    "fixed" it would delete whatever an operator named
    (`test_an_out_path_that_is_an_existing_file_is_refused`).
    """
    if value.exists() and not value.is_dir():
        raise UsageError(f"--out: {value} exists and is not a directory")
    value.mkdir(parents=True, exist_ok=True)
    return value


def _checked_policy(path: Path, tenant: str) -> Policy:
    """Load the routing policy and confirm it defines this tenant.

    A well-formed tenant no policy defines would otherwise reach the run with no
    allow list at all, and deny-by-default would report a residency violation per
    case instead of an operator's typo
    (`test_a_tenant_the_policy_does_not_define_is_refused`).
    """
    try:
        policy = load_policy(path)
    except PolicyInvalid as exc:
        raise UsageError(str(exc)) from exc
    if tenant not in policy.tenants:
        raise UsageError(f"--tenant: {tenant!r} is not a tenant of {path}")
    return policy


def _other_tenant_markers(cases: tuple, tenant: str) -> tuple[str, ...]:
    """The other tenants the set names, which a reply may not mention.

    The seed's markers, derived from the set itself rather than configured: a
    `cross_tenant` case exists to prove the agent cannot see another tenant's
    data, and the other tenants in the file are exactly the ones the seed carries.
    With none, 16a's leak assertion reports `skipped` rather than claiming a pass
    nothing checked.
    """
    return tuple(sorted({case.tenant for case in cases} - {tenant}))


def _checked_set_sha(value: str | None) -> str | None:
    """Validate `--set-sha`, or answer `None` when the caller gave none.

    7 to 64 lowercase hex, inclusive at both ends, with `re.fullmatch` and never
    `.match`: `$` also matches before a trailing newline in Python (standing
    rule 4), and this value is stored beside a run as the version of the set that
    produced it (`test_a_malformed_set_sha_is_refused_before_the_target_is_touched`).
    """
    if value is None:
        return None
    if SET_SHA_PATTERN.fullmatch(value) is None:
        raise UsageError(f"--set-sha: {value!r} is not 7 to 64 lowercase hex characters")
    return value


def _set_sha(supplied: str | None, case_set: Path) -> tuple[str, str]:
    """Which version of the set this was, and how that answer was arrived at.

    The golden set lives in the Helpdesk repo, whose commit this process cannot
    see, so a caller who knows it passes it and everyone else gets the sha256 of
    the file's BYTES (`16c` decision A). The source travels with the value so no
    reader is ever told a content hash is a commit
    (`test_without_a_set_sha_the_field_is_the_sets_own_sha256`).
    """
    if supplied is not None:
        return supplied, "commit"
    return hashlib.sha256(case_set.read_bytes()).hexdigest(), "content"


def _git_sha() -> str | None:
    """This repo's `HEAD`, or `None` when git cannot answer.

    Never a guess and never a default string: `unknown` in a stored column is a
    value a query has to know about, while `null` is the absence it already
    handles. All three shapes of failure answer `None` - no git binary, a
    non-zero exit outside a repository, and a raising call
    (`test_the_git_sha_is_null_when_git_rev_parse_does_not_answer`) - and the
    trailing newline `rev-parse` prints is stripped, or every equality query
    against a commit would miss
    (`test_the_git_sha_is_this_repos_head_with_no_trailing_newline`).
    """
    try:
        finished = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip() or None


def _parser() -> argparse.ArgumentParser:
    """`ops-eval` and its five subcommands: `run`, `calibrate`, `gate`, `baseline-guard`,
    `tradeoff`."""
    parser = argparse.ArgumentParser(prog="ops-eval", description="Run a golden set.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run a golden set against a target")
    run.add_argument("--target", required=True, help="base URL of the target under evaluation")
    run.add_argument("--set", required=True, dest="case_set", help="the golden set, JSONL")
    run.add_argument("--tenant", required=True, help="the tenant the run is made as")
    run.add_argument("--out", required=True, type=Path, help="where the two reports are written")
    run.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_ROUTING_POLICY_PATH,
        help=f"the routing policy to read residency from (default: {DEFAULT_ROUTING_POLICY_PATH})",
    )
    run.add_argument(
        "--set-sha",
        dest="set_sha",
        help="the commit the golden set came from; without it, the file's own sha256",
    )
    run.add_argument(
        "--run-kind",
        dest="run_kind",
        choices=RUN_KINDS,
        default=DEFAULT_RUN_KIND,
        help=f"why this run happened (default: {DEFAULT_RUN_KIND})",
    )
    run.add_argument(
        "--judge-url",
        dest="judge_url",
        help="base URL of the gateway that serves the judge alias; without it, no judging",
    )
    run.add_argument(
        "--calibration",
        type=Path,
        help="a calibration.json from `ops-eval calibrate`; publishes its kappa and trust",
    )
    run.add_argument(
        "--pushgateway-url",
        dest="pushgateway_url",
        help="base URL of a Pushgateway; without it, gateway_eval_* metrics are not pushed",
    )
    calibrate_command = subcommands.add_parser(
        "calibrate", help="score human labels through the judge and report the agreement"
    )
    calibrate_command.add_argument(
        "--labels", required=True, type=Path, help="the human-labelled replies, JSONL"
    )
    calibrate_command.add_argument(
        "--judge-url",
        dest="judge_url",
        required=True,
        help="base URL of the gateway that serves the judge alias",
    )
    calibrate_command.add_argument(
        "--out", required=True, type=Path, help="where calibration.json is written"
    )
    gate_command = subcommands.add_parser(
        "gate", help="compare a published run against the baseline (SPEC 12.6)"
    )
    gate_command.add_argument("--report", required=True, type=Path, help="the run's report.json")
    gate_command.add_argument(
        "--baseline", required=True, type=Path, help="the checked-in backend/evals/baseline.json"
    )
    gate_command.add_argument(
        "--comment", type=Path, help="where the PR comment markdown is written; without it, none"
    )
    guard_command = subcommands.add_parser(
        "baseline-guard",
        help="refuse a baseline change arriving without its prefix and its ADR",
    )
    guard_command.add_argument(
        "--base", required=True, help="the ref the range starts after (exclusive)"
    )
    guard_command.add_argument("--head", required=True, help="the ref the range ends at")
    tradeoff_command = subcommands.add_parser(
        "tradeoff", help="build the tradeoff table from N published runs (SPEC 11)"
    )
    tradeoff_command.add_argument(
        "--manifest", required=True, type=Path, help="the manifest naming the options and runs"
    )
    tradeoff_command.add_argument(
        "--out", required=True, type=Path, help="where tradeoff.md and tradeoff.json are written"
    )
    return parser
