r"""`ops-eval`: `run`, the command `make eval` names, and `calibrate` (#33, #38, #36).

A thin shell over `runner.run_set`. Everything it owns happens BEFORE the target
is touched - every argument pre-check, the policy, the golden set - because
after the reset the seed has been spent and a refusal costs a whole run.

    ops-eval run --target URL --set PATH --tenant T --out DIR
                 [--policy PATH] [--set-sha SHA] [--run-kind KIND] [--judge-url URL]
                 [--calibration PATH]
    ops-eval calibrate --labels PATH --judge-url URL --out DIR

`calibrate` (17e) is the same shell over `calibration.calibrate`: it scores a
file of human-labelled replies through the SAME `JudgeClient` a run uses and
writes one `calibration.json` into `--out`. Its two flags in common with `run`
are checked by the same two functions - `_checked_judge_url_value` and
`_checked_out` - and both refusals it can end in, an unloadable labels file and a
judge that scored too few labels, are one line on stderr and `EXIT_USAGE`. The
one place it deliberately differs from `run` is the key: `run` turns the judge
OFF without `OPSCONSOLE_EVAL_JUDGE_KEY` and keeps its own exit code, while a
calibration with no judge has nothing left to do and refuses
(`test_calibrate_with_no_judge_key_is_a_usage_refusal`).

`run --calibration` (17f) is the other end of that file: it reads one artifact
back through `read_calibration`, refuses it when the judge prompt it was computed
under is not the one this run scores with (decision D) and when there is no
`--judge-url` to judge with at all (decision E), and otherwise publishes what it
says - the kappa as the eighteenth metric, the `RunCalibration` record on
`report.json`, and `eval_runs.judge_kappa`. An artifact whose kappa is under
17e's `JUDGE_TRUST_MIN` does not refuse the run: it prints `UNTRUSTED_JUDGE_LINE`
once and the two reports mark the judge metrics untrusted (decision I,
`test_an_untrusted_calibration_still_runs_and_says_so_once`). The eval gate excludes
those metrics from every rule by construction (18a decision F); this command
only publishes the number it reads.

The exit code is the interface, not the output (decision C). The PR gate (18a) has
to tell an accuracy regression from a residency breach without parsing stdout,
so the four codes are module constants pinned exactly by
`test_the_exit_codes_are_the_published_ones`, and a residency violation or a
cross-tenant leak outranks a failed assertion
(`test_a_residency_violation_exits_three_and_outranks_a_failed_assertion`).

`--judge-url` is pre-checked by the same `_checked_target` the target URL is, in
the same block and for the same reason: an unchecked judge URL surfaces as a
transport error per rubric case after the seed has been reset
(`test_the_judge_url_is_refused_before_the_target_is_touched`). The judge itself
is OFF unless that URL and `OPSCONSOLE_EVAL_JUDGE_KEY` are BOTH present (17c
decision B, the shape decision 16d took for the store): the run then judges
nothing, prints `UNJUDGED_LINE` once on stderr, reports `judge_mean` as `null`
and keeps its own exit code
(`test_with_no_judge_key_the_run_judges_nothing_and_says_so_once`).

`--tenant` and `--set-sha` are pre-checked with `re.fullmatch` (a project rule
and 4) before any policy is consulted - the tenant against the gateway's OWN
`IDENTIFIER_PATTERN` and `MAX_IDENTIFIER_LENGTH`, the SHA against
`report.SET_SHA_PATTERN`. The tenant selects a tenant's residency rules and both
are written into `report.json` and into `16d`'s `eval_runs` row, so `acme\n` and
`../a1b2c3d` are refused before anything runs. Neither shapes a path: the two
report file names are constants (`16c` decision B). This module, `evals/store.py`
and `evals/residency.py` are three of the TWENTY-ONE `IdentifierInvalid` raisers
that r`grep -rlE "raise Identifier[I]nvalid" packages/` returns as of an earlier change,
and the enumeration in `routing/errors.py` was re-derived whole from that command.

`--out` is validated and created before the run and written after it: exactly
`report.json` and `report.md`, both named by `report.py`'s constants
(`test_a_run_writes_exactly_the_two_report_files_into_the_out_directory`). The
write happens BEFORE the exit code is chosen and on a failed run too (`16c`
decision D), because a failed run is the one somebody opens the report for, and
a write that fails is `EXIT_USAGE` naming the path rather than a traceback over
a run already paid for
(`test_an_out_directory_that_cannot_be_written_exits_two_naming_the_path`). What
this command prints is still ONE line naming the four counts.

The `eval_runs` and `eval_cases` rows are stored AFTER those two files and never
before: the artefact is what survives a failed insert, and
the eval gate reads it off disk. With no `OPSCONSOLE_DATABASE_URL` configured the
run stores nothing, says so in one line on stderr and keeps its own exit code
(decision B) - `make eval` is usually run on a machine with no PostgreSQL up, and
refusing there would turn the whole harness off rather than turn storage off.
When the store DOES fail, the exit code is `EXIT_USAGE` unless the run already
answered `EXIT_RESIDENCY`, which wins: a boundary breach may never be downgraded
to a storage complaint
(`test_a_store_that_fails_reports_two_unless_the_run_already_answered_three`).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import httpx

from app.core.config import DEFAULT_ROUTING_POLICY_PATH, Settings
from app.core.validation import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH
from app.evals.baseline_guard import (
    GUARD_PASSED_LINE,
    MAX_REF_LENGTH,
    REF_PATTERN,
    Commit,
    check,
    offence_line,
)
from app.evals.calibration import (
    CalibrationRefused,
    CalibrationReport,
    CalibrationSetInvalid,
    calibrate,
    load_labels,
    read_calibration,
    write_calibration,
)
from app.evals.cases import EvalCase, EvalSetInvalid, load_golden_set
from app.evals.gate import (
    GateInputInvalid,
    evaluate,
    gate_comment,
    load_baseline,
    load_report,
    reason_line,
    verdict_line,
)
from app.evals.judge import JudgeClient, judge_prompt_version
from app.evals.metrics_push import push_metrics
from app.evals.report import (
    REPORT_JSON_NAME,
    SET_SHA_PATTERN,
    RunCalibration,
    RunReport,
    build_report,
    write_reports,
)
from app.evals.runner import RunResult, run_set
from app.evals.store import PostgresEvalStore, rows_from_report
from app.evals.target import TargetClient, TargetUnreachable
from app.evals.tradeoff import (
    TradeoffInputInvalid,
    build_table,
    load_manifest,
    write_tradeoff,
)
from app.ledger.db import build_pool
from app.model_routing.errors import IdentifierInvalid, PolicyInvalid
from app.model_routing.policy import Policy, load_policy

EXIT_OK: Final[int] = 0
"""Every case passed and every turn stayed in region."""

EXIT_FAILED: Final[int] = 1
"""At least one case failed its assertions, or its turn never answered."""

EXIT_USAGE: Final[int] = 2
"""A usage error, an unloadable set, or a target that could not be reached or reset.

Distinct from `EXIT_FAILED` so a gate can tell "the harness never ran" from "the
agent got worse" - the same reason the ledger's CLI keeps its own `2`."""

EXIT_RESIDENCY: Final[int] = 3
"""A residency violation or a cross-tenant leak. Not a score: a human looks at it."""

ALLOWED_TARGET_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
"""What `--target` may be. Everything else is refused before the seed is reset
(`test_a_target_that_is_not_an_absolute_http_url_is_refused`)."""


RUN_KINDS: Final[tuple[str, ...]] = ("ci", "nightly", "manual")
"""What `--run-kind` may be, and the same three words `report.RunKind` accepts.

An argparse `choices`, so a fourth word is a usage error rather than a new kind
in a stored row (`test_a_run_kind_outside_the_three_is_a_usage_error`)."""

DEFAULT_RUN_KIND: Final[str] = "manual"
"""What a run with no `--run-kind` is. An operator at a terminal is the common
case, and defaulting to `ci` would label a laptop run as a gate run
(`test_the_run_kind_defaults_to_manual`)."""

UNSTORED_LINE: Final[str] = (
    "no database URL configured: this run's rows were not stored, and its two report files were"
)
"""The ONE line an unstored run prints on stderr (decision B).

Silence would leave an operator believing rows were written, a line per case is
noise in a CI log, and refusing to run at all would break `make eval` on every
machine without PostgreSQL up - which is where a golden set is usually run
(`test_with_no_database_url_nothing_is_stored_and_the_run_keeps_its_own_exit_code`)."""


UNJUDGED_LINE: Final[str] = (
    "no judge URL or no OPSCONSOLE_EVAL_JUDGE_KEY: this run judged nothing, and judge_mean is null"
)
"""The ONE line an unjudged run prints on stderr (17c decision B).

`UNSTORED_LINE`'s shape and its reasons: silence leaves an operator who passed
`--judge-url` believing the rubric cases were scored, a line per case is noise in
a CI log, and refusing to run would turn the harness off on every machine with no
gateway up rather than turn the judge off
(`test_with_no_judge_url_the_run_judges_nothing_and_keeps_its_exit_code`)."""


UNTRUSTED_JUDGE_LINE: Final[str] = (
    "the calibration supplied is not trusted: this run's judge metrics are marked "
    "untrusted in both reports"
)
"""The ONE line a run carrying an untrusted calibration prints on stderr (17f decision I).

`UNSTORED_LINE`'s and `UNJUDGED_LINE`'s shape and their reason: a CI log is where
this is noticed, and the two existing notices set the precedent that a run
reporting a weaker fact than the operator asked for says so exactly once. It is
not a refusal - the eval gate excludes the judge metrics from every rule whatever
the kappa says (18a decision F), so the run still writes both reports, still stores
its rows and keeps its own exit code
(`test_an_untrusted_calibration_still_runs_and_says_so_once`)."""


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
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(f"tenant: longer than {MAX_IDENTIFIER_LENGTH} characters")
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("tenant: not a lowercase slug")
    return value


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

    `git log --format=%H%x00%s --name-only` prints a header line `<sha>\x00<subject>`, then a
    blank line, then one path per line until the next header. The NUL is what
    separates the sha from a subject that may contain anything, including the colon
    every well-formed baseline commit's subject carries.

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
            ["git", "log", "--format=%H%x00%s", "--name-only", revisions],
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
    for line in finished.stdout.splitlines():
        if "\x00" in line:
            sha, _, subject = line.partition("\x00")
            shas.append(sha)
            subjects.append(subject)
            touched.append([])
        elif line and touched:
            touched[-1].append(line)
    for sha, subject, files in zip(shas, subjects, touched, strict=True):
        commits.append(Commit(sha=sha, subject=subject, files=tuple(files)))
    return tuple(commits)


def _baseline_guard(options: argparse.Namespace) -> int:
    """Answer #42's verdict over one range of commits and the process exit code (18b).

    `_gate`'s shape over git rather than over two files: every refusal is one line on
    stderr answered with `EXIT_USAGE`, stdout carries exactly the one summary line,
    and the offences go to stderr one per sha so an author sees every commit to fix
    in one CI round
    (`test_baseline_guard_prints_one_stderr_line_per_offence_and_exits_one`).

    `EXIT_USAGE` and `EXIT_FAILED` are kept apart here for the reason the rest of
    this module keeps them apart: "the guard could not read the range" and "the
    range breaks the rule" have different fixes, and only the second is the
    author's.
    """
    try:
        base = _checked_ref(options.base, "--base")
        head = _checked_ref(options.head, "--head")
        commits = _commits(base, head)
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    verdict = check(commits)
    if verdict.passed:
        print(GUARD_PASSED_LINE.format(checked=len(commits)))
        return EXIT_OK

    for offence in verdict.offences:
        print(offence_line(offence), file=sys.stderr)
    return EXIT_FAILED


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


def _other_tenant_markers(cases: tuple[EvalCase, ...], tenant: str) -> tuple[str, ...]:
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


def _summary(result: RunResult) -> str:
    """The one line this command prints, in the order `report.md`'s counts repeat.

    One line because `make eval`'s output is read in a CI log, and the counts are
    the run's real ones (`test_the_summary_is_one_line_naming_the_four_counts`).
    """
    return (
        f"cases {result.cases} · passed {result.passed_count} · "
        f"failed {result.failed_count} · residency violations {result.residency_violations}"
    )


def _exit_code(result: RunResult) -> int:
    """Turn one run into the integer the eval gate reads.

    Residency and a leak outrank the assertion counts: a boundary breach read as
    an accuracy dip is a breach somebody merges over.
    """
    if result.residency_violations or result.leaked:
        return EXIT_RESIDENCY
    if result.failed_count:
        return EXIT_FAILED
    return EXIT_OK


def _store_rows(report: RunReport, out: Path) -> str | None:
    """Store the run and its cases, and answer what went wrong, or `None`.

    Called AFTER both report files are written (decision E's ordering): the
    artefact is what survives a failed insert, and the eval gate reads it off
    disk. The failure is answered rather than raised so the caller can apply
    decision E's exit rule in one place, and a storage complaint never becomes a
    traceback over a run already paid for.

    With no `OPSCONSOLE_DATABASE_URL` configured nothing is stored, `UNSTORED_LINE`
    is printed and the run keeps its own exit code (decision B).
    """
    database_url = Settings().database_url
    if database_url is None:
        print(UNSTORED_LINE, file=sys.stderr)
        return None
    try:
        run_row, case_rows = rows_from_report(report, report_path=out / REPORT_JSON_NAME)
        PostgresEvalStore(build_pool(database_url)).write(run_row, case_rows)
    except Exception as exc:
        return f"the eval store did not take this run: {exc}"
    return None


def _judge(judge_url: str | None, client: httpx.Client) -> JudgeClient | None:
    """The judge this run scores with, or `None` with one line on stderr.

    Both halves are required (decision B): a URL with no key would spend three
    401s per rubric case, and a key with no URL has nowhere to send them. The
    client is the run's own `httpx.Client`, so the judge opens no second pool and
    the whole path is drivable through one `MockTransport`.
    """
    api_key = Settings().eval_judge_key
    if judge_url is None or not api_key:
        print(UNJUDGED_LINE, file=sys.stderr)
        return None
    return JudgeClient(judge_url, client=client, api_key=api_key)


def _judged_prompt_version(result: RunResult) -> str | None:
    """The prompt's sha256 when a case was actually judged, and `None` otherwise.

    Read off the RECORDS rather than off whether a judge was built (criterion 7):
    a run whose every judge sample failed, and a set with no rubric in it, both
    judged nothing, and a hash recorded there would tell a later comparison the
    two runs used a prompt neither of them scored anything with
    (`test_with_no_judge_url_the_run_judges_nothing_and_keeps_its_exit_code`).
    """
    if any(run.judge is not None for run in result.runs):
        return judge_prompt_version()
    return None


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


def _calibration_summary(report: CalibrationReport) -> str:
    """The one line `calibrate` prints, in the order a reader asks the questions.

    One line for `_summary`'s reason - this output is read in a CI log - and the
    trust flag is on it because that is the whole point of the command: a kappa
    with no verdict beside it makes every reader apply the threshold themselves
    (`test_calibrate_writes_the_artifact_into_out_and_exits_zero`).
    """
    kappa = "null" if report.kappa is None else report.kappa
    return (
        f"pairs {report.pairs} · dropped {report.dropped} · "
        f"kappa {kappa} · trusted {str(report.trusted).lower()}"
    )


def _calibrate(options: argparse.Namespace) -> int:
    """Calibrate the judge against a labels file and answer the process exit code.

    Every pre-check happens before a request is made, for `run`'s own reason one
    level along: the spend here is `MIN_CALIBRATION_LABELS` times `JUDGE_SAMPLES`
    calls, and a URL refused afterwards has already cost all of it
    (`test_a_calibrate_judge_url_that_is_not_an_absolute_http_url_is_refused`
    asserts the request log is empty).

    Both refusals a calibration can end in - an unloadable file and a judge that
    scored too few labels - are reported as their own single line and answered
    with `EXIT_USAGE`, never as a traceback and never as `EXIT_FAILED`: a gate
    reads `2` as "the harness never ran", which is exactly what a judge outage is
    (`test_a_judge_that_answers_five_hundred_throughout_writes_nothing_and_exits_two`).
    The artifact is written LAST, so nothing on disk can be read as trust unless
    the whole calibration succeeded.
    """
    try:
        judge_url = _checked_judge_url_value(options.judge_url)
        out = _checked_out(options.out)
        api_key = _checked_judge_key()
        labels_path = Path(options.labels)
        labels = load_labels(labels_path)
    except (UsageError, CalibrationSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    with httpx.Client() as http:
        try:
            report = calibrate(
                labels,
                judge=JudgeClient(judge_url, client=http, api_key=api_key),
                labels_name=labels_path.name,
                labels_sha=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            )
        except CalibrationRefused as exc:
            print(exc, file=sys.stderr)
            return EXIT_USAGE

    try:
        write_calibration(out, report)
    except OSError as exc:
        print(f"--out: {out} could not be written: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(_calibration_summary(report))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Answer the process exit code for whichever subcommand was named.

    Argparse's own `required=True` on the subparsers is what refuses a bare
    `ops-eval`, so this dispatch never needs a branch for "no subcommand at all".
    """
    options = _parser().parse_args(argv)
    if options.command == "calibrate":
        return _calibrate(options)
    if options.command == "gate":
        return _gate(options)
    if options.command == "baseline-guard":
        return _baseline_guard(options)
    if options.command == "tradeoff":
        return _tradeoff(options)
    return _run(options)


def _tradeoff(options: argparse.Namespace) -> int:
    """Build the tradeoff table from a manifest of published runs (27b).

    `_gate`'s shape with no verdict in it: every input is read and checked before
    anything is written, so a refusal on the last option leaves `--out` empty
    rather than a half-written table for the next reader to quote
    (`test_every_refusal_exits_two_with_one_line_on_stderr_and_an_empty_out_directory`).
    A relative `report` in the manifest resolves against the manifest's own
    directory, which is what makes the committed evidence directory portable.
    """
    try:
        manifest = load_manifest(options.manifest)
        table = build_table(manifest, base_dir=options.manifest.parent)
    except TradeoffInputInvalid as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    try:
        write_tradeoff(options.out, table)
    except OSError as exc:
        print(f"--out: {options.out} could not be written: {exc.strerror}", file=sys.stderr)
        return EXIT_USAGE

    winner = "none" if table.winner_id is None else table.winner_id
    print(f"options {len(table.rows)} · winner {winner}")
    return EXIT_OK


def _run(options: argparse.Namespace) -> int:
    """Run one golden set and answer the process exit code.

    Every refusal goes to stderr and never to stdout, which is the stream the
    summary's counts are parsed from
    (`test_a_pre_check_failure_reports_on_stderr_rather_than_stdout`), and
    `EvalSetInvalid` and `TargetUnreachable` are reported as their own message
    rather than as a traceback: an operator who mistyped a path needs the path.
    """
    try:
        tenant = _checked_tenant(options.tenant)
        target = _checked_target(options.target)
        out = _checked_out(options.out)
        policy = _checked_policy(options.policy, tenant)
        supplied_sha = _checked_set_sha(options.set_sha)
        judge_url = _checked_judge_url(options.judge_url)
        calibration = _checked_calibration(options.calibration, judge_url)
        case_set = Path(options.case_set)
        cases = load_golden_set(case_set)
        set_sha, set_sha_source = _set_sha(supplied_sha, case_set)
    except (IdentifierInvalid, UsageError, EvalSetInvalid, CalibrationSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    if calibration is not None and not calibration.trusted:
        print(UNTRUSTED_JUDGE_LINE, file=sys.stderr)

    started_at = datetime.now(UTC)
    with httpx.Client() as http:
        client = TargetClient(target, client=http)
        try:
            result = run_set(
                client=client,
                cases=cases,
                policy=policy,
                tenant=tenant,
                other_tenant_markers=_other_tenant_markers(cases, tenant),
                judge=_judge(judge_url, http),
            )
        except TargetUnreachable as exc:
            print(exc, file=sys.stderr)
            return EXIT_USAGE
    finished_at = datetime.now(UTC)

    report = build_report(
        result,
        started_at=started_at,
        finished_at=finished_at,
        set_name=case_set.name,
        set_sha=set_sha,
        set_sha_source=set_sha_source,
        run_kind=options.run_kind,
        git_sha=_git_sha(),
        tenant=tenant,
        target=target,
        judge_prompt_version=_judged_prompt_version(result),
        calibration=calibration,
    )
    try:
        write_reports(out, report)
    except OSError as exc:
        print(f"--out: {out} could not be written: {exc}", file=sys.stderr)
        return EXIT_USAGE

    stored = _store_rows(report, out)

    if options.pushgateway_url is not None:
        # `model_dump(mode="json")`, not `report.metrics`: a Decimal cost serialises to a
        # STRING on the written report (`report.py`'s `_serialise_cost`), and this is the
        # SAME document `eval_runs.metrics` and the eval gate both read - pushing the raw
        # in-memory `Decimal` would push a value no other reader of this run's metrics sees.
        written_metrics = report.model_dump(mode="json")["metrics"]
        with httpx.Client() as push_client:
            pushed = push_metrics(written_metrics, options.pushgateway_url, client=push_client)
        if pushed is not None:
            print(pushed, file=sys.stderr)

    print(_summary(result))
    code = _exit_code(result)
    if stored is not None:
        print(stored, file=sys.stderr)
        return EXIT_RESIDENCY if code == EXIT_RESIDENCY else EXIT_USAGE
    return code


def _gate(options: argparse.Namespace) -> int:
    """Compare one published run against the baseline and answer the exit code (18a).

    `_run`'s shape with no run in it: every refusal is one line on stderr answered
    with `EXIT_USAGE`, stdout carries exactly the one verdict line the workflow reads
    (decision E), and the breached rules go to stderr one line each so a reviewer
    sees all of them at once
    (`test_gate_reports_every_breached_rule_on_its_own_line_and_exits_one`).

    The order here is the contract. Both inputs are validated before anything is
    written, so a malformed report never leaves a comment file behind claiming a
    run that never happened; the comment is written before the verdict is printed,
    so a run whose comment could not be written does not look, from stdout and the
    code, exactly like one whose comment reached the PR
    (`test_a_comment_path_inside_a_directory_that_does_not_exist_is_a_usage_refusal`).

    `EXIT_RESIDENCY` outranks `EXIT_FAILED` exactly as `_exit_code` ranks them for
    `run` (decision D): a boundary breach read as an accuracy dip is a breach
    somebody merges over
    (`test_gate_exits_three_when_a_residency_violation_sits_beside_a_threshold_breach`).
    """
    try:
        report = load_report(options.report)
        baseline = load_baseline(options.baseline)
    except GateInputInvalid as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    verdict = evaluate(report, baseline)

    if options.comment is not None:
        try:
            options.comment.write_text(gate_comment(verdict, report), encoding="utf-8")
        except OSError as exc:
            print(
                f"--comment: {options.comment} could not be written: {exc.strerror}",
                file=sys.stderr,
            )
            return EXIT_USAGE

    print(verdict_line(verdict))
    for breach in verdict.breaches:
        print(reason_line(breach), file=sys.stderr)

    if verdict.residency_breached:
        return EXIT_RESIDENCY
    return EXIT_OK if verdict.passed else EXIT_FAILED
