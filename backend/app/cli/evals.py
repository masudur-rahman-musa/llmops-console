r"""`ops-eval`: the commands `make eval` and the workflows name.

A thin shell over the eval modules. Everything a command owns happens BEFORE the help
desk is touched (every argument check, the cases file), because after the reset the seed
has been spent and a refusal costs a whole run.

    ops-eval conversations --target URL --cases PATH --out DIR [--run-id ID]
                           [--set-sha SHA] [--run-kind KIND] [--gateway-url URL]
                           [--plays N] [--reset-command CMD] [--pushgateway-url URL]
    ops-eval gate          --report PATH --baseline PATH [--comment PATH]
    ops-eval baseline-guard --base REF --head REF
    ops-eval tradeoff      --manifest PATH --out DIR

`conversations` plays each case `--plays` times, one turn per call, each play from freshly
reset data, and writes `report.json`, `report.md`, `needs-a-look.md` and `comment.md` into
`--out`. It exits 0 whenever the run completed: failed plays are the gate's business.

`gate` compares a run with `baseline.json`: safety must be 100, and outcome every play must
reach the baseline's number.

The exit code is the interface, not the output: 0 passed, 1 failed, 2 for anything that
stopped the command (a bad argument, an unreadable file, a failed reset).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import app.cli.evals_args as _eargs
import app.cli.evals_commands as _ecmds
from app.cli.evals_commands import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    UNJUDGED_LINE,
    UNSTORED_LINE,
)

__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_USAGE",
    "UNJUDGED_LINE",
    "UNSTORED_LINE",
]
# `main` is defined here, not re-exported, and is reached through the
# `ops-eval` script in `backend/pyproject.toml`, so it needs no entry above.

ALLOWED_TARGET_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
RUN_KINDS: Final[tuple[str, ...]] = ("ci", "nightly", "manual")
DEFAULT_RUN_KIND: Final[str] = "manual"


def main(argv: Sequence[str] | None = None) -> int:
    """Answer the process exit code for whichever subcommand was named.

    Argparse's own `required=True` on the subparsers refuses a bare `ops-eval`, and its
    choice check refuses the removed commands, so this dispatch needs no branch for them.
    """
    options = _eargs._parser().parse_args(argv)
    handlers = {
        "conversations": _ecmds._conversations,
        "gate": _ecmds._gate,
        "baseline-guard": _ecmds._baseline_guard,
        "tradeoff": _ecmds._tradeoff,
    }
    return handlers[options.command](options)
