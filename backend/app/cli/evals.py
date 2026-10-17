r"""`ops-eval`: `run`, the command `make eval` names, and `calibrate`.

A thin shell over `runner.run_set`. Everything it owns happens BEFORE the target
is touched - every argument pre-check, the policy, the golden set - because
after the reset the seed has been spent and a refusal costs a whole run.

    ops-eval run --target URL --set PATH --tenant T --out DIR
                 [--policy PATH] [--set-sha SHA] [--run-kind KIND] [--judge-url URL]
                 [--calibration PATH]
    ops-eval calibrate --labels PATH --judge-url URL --out DIR

`calibrate` is the same shell over `calibration.calibrate`: it scores a
file of human-labelled replies through the SAME `JudgeClient` a run uses and
writes one `calibration.json` into `--out`. Its two flags in common with `run`
are checked by the same two functions - `_checked_judge_url_value` and
`_checked_out` - and both refusals it can end in, an unloadable labels file and a
judge that scored too few labels, are one line on stderr and `EXIT_USAGE`. The
one place it deliberately differs from `run` is the key: `run` turns the judge
OFF without `OPSCONSOLE_EVAL_JUDGE_KEY` and keeps its own exit code, while a
calibration with no judge has nothing left to do and refuses.

`run --calibration` is the other end of that file: it reads one artifact
back through `read_calibration`, refuses it when the judge prompt it was computed
under is not the one this run scores with and when there is no
`--judge-url` to judge with at all, and otherwise publishes what it
says - the kappa as the eighteenth metric, the `RunCalibration` record on
`report.json`, and `eval_runs.judge_kappa`. An artifact whose kappa is under
`JUDGE_TRUST_MIN` does not refuse the run: it prints `UNTRUSTED_JUDGE_LINE`
once and the two reports mark the judge metrics untrusted. The eval gate excludes
those metrics from every rule by construction; this command
only publishes the number it reads.

The exit code is the interface, not the output. The PR gate
has to tell an accuracy regression from a residency breach without parsing stdout,
so the four codes are module constants, and a residency violation or a
cross-tenant leak outranks a failed assertion.

`--judge-url` is pre-checked by the same `_checked_target` the target URL is, in
the same block and for the same reason: an unchecked judge URL surfaces as a
transport error per rubric case after the seed has been reset. The judge itself
is OFF unless that URL and `OPSCONSOLE_EVAL_JUDGE_KEY` are BOTH present, the same
shape the store uses: the run then judges
nothing, prints `UNJUDGED_LINE` once on stderr, reports `judge_mean` as `null`
and keeps its own exit code.

`--tenant` and `--set-sha` are pre-checked with `re.fullmatch`
before any policy is consulted - the tenant against the gateway's OWN
`IDENTIFIER_PATTERN` and `MAX_IDENTIFIER_LENGTH`, the SHA against
`report.SET_SHA_PATTERN`. The tenant selects a tenant's residency rules and both
are written into `report.json` and into the `eval_runs` row, so `acme\n` and
`../a1b2c3d` are refused before anything runs. Neither shapes a path: the two
report file names are constants. This module, `storage/eval_store.py`
and `evals/region_check.py` are three of the places that raise `IdentifierInvalid`,
and the list in `model_routing/errors.py` was re-derived whole from a search for them.

`--out` is validated and created before the run and written after it: exactly
`report.json` and `report.md`, both named by `report.py`'s constants. The
write happens BEFORE the exit code is chosen and on a failed run too,
because a failed run is the one somebody opens the report for, and
a write that fails is `EXIT_USAGE` naming the path rather than a traceback over
a run already paid for. What
this command prints is still ONE line naming the four counts.

The `eval_runs` and `eval_cases` rows are stored AFTER those two files and never
before: the artefact is what survives a failed insert, and
the eval gate reads it off disk. With no `OPSCONSOLE_DATABASE_URL` configured the
run stores nothing, says so in one line on stderr and keeps its own exit code
- `make eval` is usually run on a machine with no PostgreSQL up, and
refusing there would turn the whole harness off rather than turn storage off.
When the store DOES fail, the exit code is `EXIT_USAGE` unless the run already
answered `EXIT_RESIDENCY`, which wins: a boundary breach may never be downgraded
to a storage complaint.
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
