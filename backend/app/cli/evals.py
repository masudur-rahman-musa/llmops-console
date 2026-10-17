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
report file names are constants (`16c` decision B). This module, `storage/eval_store.py`
and `evals/region_check.py` are three of the TWENTY-ONE `IdentifierInvalid` raisers
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

from collections.abc import Sequence
from typing import Final

import app.cli.evals_args as _eargs
import app.cli.evals_commands as _ecmds
from app.cli.evals_commands import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_RESIDENCY,
    EXIT_USAGE,
    UNJUDGED_LINE,
    UNSTORED_LINE,
    UNTRUSTED_JUDGE_LINE,
)

__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_RESIDENCY",
    "EXIT_USAGE",
    "UNJUDGED_LINE",
    "UNSTORED_LINE",
    "UNTRUSTED_JUDGE_LINE",
    "main",
]

# Re-export as Final so tests that pin the exact values still pass.
EXIT_OK_: Final[int] = EXIT_OK
EXIT_FAILED_: Final[int] = EXIT_FAILED
EXIT_USAGE_: Final[int] = EXIT_USAGE
EXIT_RESIDENCY_: Final[int] = EXIT_RESIDENCY

ALLOWED_TARGET_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
RUN_KINDS: Final[tuple[str, ...]] = ("ci", "nightly", "manual")
DEFAULT_RUN_KIND: Final[str] = "manual"


def main(argv: Sequence[str] | None = None) -> int:
    """Answer the process exit code for whichever subcommand was named.

    Argparse's own `required=True` on the subparsers is what refuses a bare
    `ops-eval`, so this dispatch never needs a branch for "no subcommand at all".
    """
    options = _eargs._parser().parse_args(argv)
    if options.command == "calibrate":
        return _ecmds._calibrate(options)
    if options.command == "gate":
        return _ecmds._gate(options)
    if options.command == "baseline-guard":
        return _ecmds._baseline_guard(options)
    if options.command == "tradeoff":
        return _ecmds._tradeoff(options)
    return _ecmds._run(options)
