r"""`ops-eval run`: the command `make eval` names, and the four exit codes (#33, #38).

A thin shell over `runner.run_set`. Everything it owns happens BEFORE the target
is touched - the three argument pre-checks, the policy, the golden set - because
after the reset the seed has been spent and a refusal costs a whole run.

    ops-eval run --target URL --set PATH --tenant T --out DIR [--policy PATH]

The exit code is the interface, not the output (decision C). Row 18's PR gate has
to tell an accuracy regression from a residency breach without parsing stdout,
so the four codes are module constants pinned exactly by
`test_the_exit_codes_are_the_published_ones`, and a residency violation or a
cross-tenant leak outranks a failed assertion
(`test_a_residency_violation_exits_three_and_outranks_a_failed_assertion`).

`--tenant` is pre-checked against the gateway's OWN `IDENTIFIER_PATTERN` and
`MAX_IDENTIFIER_LENGTH` with `re.fullmatch` before any
policy is consulted: it selects a tenant's residency rules, becomes a report path
under `--out` and reaches `16c`'s `eval_cases` rows, so `acme\n` and `../acme`
are refused before anything runs. This module and `evals/residency.py` are two of
the TWENTY `IdentifierInvalid` raisers that
r`grep -rlE "raise Identifier[I]nvalid" packages/` returns as of an earlier change, and the
enumeration in `routing/errors.py` was re-derived whole from that command.

`--out` is validated and created here and written into by nobody (decision D):
`16c` owns `report.json` and `report.md`, and a half-written report shipped here
would be a file format nobody agreed
(`test_nothing_is_written_into_the_out_directory`). What this command prints is
ONE line naming the four counts.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import httpx

from opsconsole.core.settings import DEFAULT_ROUTING_POLICY_PATH
from opsconsole.evals.cases import EvalCase, EvalSetInvalid, load_golden_set
from opsconsole.evals.runner import RunResult, run_set
from opsconsole.evals.target import TargetClient, TargetUnreachable
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

    A path that exists as a FILE is refused rather than replaced: `16c`'s writes
    would fail at the end of a paid-for run, and an implementation that "fixed"
    it would delete whatever an operator named
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


def _summary(result: RunResult) -> str:
    """The one line this command prints, in the order `16c`'s report repeats.

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


def _parser() -> argparse.ArgumentParser:
    """`ops-eval` and its one subcommand."""
    parser = argparse.ArgumentParser(prog="ops-eval", description="Run a golden set.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run a golden set against a target")
    run.add_argument("--target", required=True, help="base URL of the target under evaluation")
    run.add_argument("--set", required=True, dest="case_set", help="the golden set, JSONL")
    run.add_argument("--tenant", required=True, help="the tenant the run is made as")
    run.add_argument("--out", required=True, type=Path, help="where 16c will write the report")
    run.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_ROUTING_POLICY_PATH,
        help=f"the routing policy to read residency from (default: {DEFAULT_ROUTING_POLICY_PATH})",
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
        _checked_out(options.out)
        policy = _checked_policy(options.policy, tenant)
        cases = load_golden_set(Path(options.case_set))
    except (IdentifierInvalid, UsageError, EvalSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

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

    print(_summary(result))
    return _exit_code(result)
