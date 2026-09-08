r"""`ops-eval run`: the command `make eval` names, and the four exit codes (#33, #38).

A thin shell over `runner.run_set`. Everything it owns happens BEFORE the target
is touched - the three argument pre-checks, the policy, the golden set - because
after the reset the seed has been spent and a refusal costs a whole run.

    ops-eval run --target URL --set PATH --tenant T --out DIR
                 [--policy PATH] [--set-sha SHA] [--run-kind KIND]

The exit code is the interface, not the output (decision C). Row 18's PR gate has
to tell an accuracy regression from a residency breach without parsing stdout,
so the four codes are module constants pinned exactly by
`test_the_exit_codes_are_the_published_ones`, and a residency violation or a
cross-tenant leak outranks a failed assertion
(`test_a_residency_violation_exits_three_and_outranks_a_failed_assertion`).

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
row 18's gate reads it off disk. With no `OPSCONSOLE_DATABASE_URL` configured the
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

from opsconsole.core.settings import DEFAULT_ROUTING_POLICY_PATH, Settings
from opsconsole.evals.cases import EvalCase, EvalSetInvalid, load_golden_set
from opsconsole.evals.report import (
    REPORT_JSON_NAME,
    SET_SHA_PATTERN,
    RunReport,
    build_report,
    write_reports,
)
from opsconsole.evals.runner import RunResult, run_set
from opsconsole.evals.store import PostgresEvalStore, rows_from_report
from opsconsole.evals.target import TargetClient, TargetUnreachable
from opsconsole.ledger.db import build_pool
from opsconsole.routing.errors import IdentifierInvalid, PolicyInvalid
from opsconsole.routing.policy import Policy, load_policy
from opsconsole.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH

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
    """Turn one run into the integer row 18's gate reads.

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
    artefact is what survives a failed insert, and row 18's gate reads it off
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


def _parser() -> argparse.ArgumentParser:
    """`ops-eval` and its one subcommand."""
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one golden set and answer the process exit code.

    Every refusal goes to stderr and never to stdout, which is the stream the
    summary's counts are parsed from
    (`test_a_pre_check_failure_reports_on_stderr_rather_than_stdout`), and
    `EvalSetInvalid` and `TargetUnreachable` are reported as their own message
    rather than as a traceback: an operator who mistyped a path needs the path.
    """
    options = _parser().parse_args(argv)
    try:
        tenant = _checked_tenant(options.tenant)
        target = _checked_target(options.target)
        out = _checked_out(options.out)
        policy = _checked_policy(options.policy, tenant)
        supplied_sha = _checked_set_sha(options.set_sha)
        case_set = Path(options.case_set)
        cases = load_golden_set(case_set)
        set_sha, set_sha_source = _set_sha(supplied_sha, case_set)
    except (IdentifierInvalid, UsageError, EvalSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

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
    )
    try:
        write_reports(out, report)
    except OSError as exc:
        print(f"--out: {out} could not be written: {exc}", file=sys.stderr)
        return EXIT_USAGE

    stored = _store_rows(report, out)

    print(_summary(result))
    code = _exit_code(result)
    if stored is not None:
        print(stored, file=sys.stderr)
        return EXIT_RESIDENCY if code == EXIT_RESIDENCY else EXIT_USAGE
    return code
