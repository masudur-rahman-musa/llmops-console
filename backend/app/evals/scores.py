"""The two scores one conversation run reports, and the counts behind them.

Pure: `compute_metrics` reads the `CaseLine` records it is handed and nothing else.

* A case played several times passes when every run passed. One that both passed and
  failed is unstable: it is counted in `unstable` and left out of `check_score`, so chance
  alone cannot move the gate. With one run per case nothing is unstable.
* `check_score` is the percent of stable cases that passed all three checks. A case that
  never answered counts as not passed and stays in the denominator: leaving it out would
  shrink the number every gate is measured against. With no stable case it is `None`.
* `judge_score` is the percent of judged replies the judge passed. A reply the judge could
  not grade is out of the denominator. With nothing judged it is `None`, never `0.0`: zero
  would read as a judge that failed every reply.
* Both are rounded half up to two places through `Decimal`, never through `round()`, and the
  gate compares the written number.

The counts `cases`, `passed`, `failed` and `unstable` go beside them; `cases` is the sum
of the other three. Types are part of the contract:
the database stores this mapping as `jsonb`, so a score is a `float` or `None` and a count
is an `int`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from app.evals.runner import CaseLine

METRIC_NAMES: Final[tuple[str, ...]] = ("check_score", "judge_score")
"""The two published scores, in the order every consumer lists them."""

SCORE_SCALE: Final[int] = 2
"""Decimal places a score is published at."""

WILSON_Z: Final[float] = 1.959963984540054
"""The normal quantile for a two-sided 95% range."""


def _percentage(part: int, whole: int) -> float | None:
    """`part` of `whole` as a percent at `SCORE_SCALE` places, or `None` for no whole."""
    if whole == 0:
        return None
    value = Decimal(part * 100) / Decimal(whole)
    return float(value.quantize(Decimal(1).scaleb(-SCORE_SCALE), rounding=ROUND_HALF_UP))


def _rounded(value: float) -> float:
    return float(Decimal(repr(value)).quantize(Decimal(1).scaleb(-SCORE_SCALE), ROUND_HALF_UP))


def pass_rate(passes: int, runs: int) -> float:
    """`passes` of `runs` as a percent at `SCORE_SCALE` places. `runs` is at least 1."""
    return _percentage(passes, runs) or 0.0


def wilson_range(passes: int, runs: int) -> tuple[float, float]:
    """The 95% Wilson score range of a pass rate, in percent at `SCORE_SCALE` places.

    Chosen over the plain normal range because at 3 runs that one has no width at 3 of 3
    and runs past 0 or 100 elsewhere; `test_wilson_range_at_known_points` pins it.
    """
    share = passes / runs
    z2 = WILSON_Z * WILSON_Z
    centre = share + z2 / (2 * runs)
    spread = WILSON_Z * math.sqrt(share * (1 - share) / runs + z2 / (4 * runs * runs))
    scale = 1 + z2 / runs
    low = max(0.0, (centre - spread) / scale)
    high = min(1.0, (centre + spread) / scale)
    return _rounded(low * 100), _rounded(high * 100)


def runs_by_case(lines: Sequence[CaseLine]) -> dict[str, list[CaseLine]]:
    """Every line of each case, cases in first-seen order, runs in saved order."""
    grouped: dict[str, list[CaseLine]] = {}
    for line in lines:
        grouped.setdefault(line.case_id, []).append(line)
    return grouped


def is_unstable(runs: Sequence[CaseLine]) -> bool:
    """True when a case both passed and failed across its runs."""
    return any(line.passed for line in runs) and not all(line.passed for line in runs)


def check_score(lines: Sequence[CaseLine]) -> float | None:
    """The percent of stable cases passed, or `None` when no case is stable."""
    stable = [runs for runs in runs_by_case(lines).values() if not is_unstable(runs)]
    return _percentage(sum(1 for runs in stable if runs[0].passed), len(stable))


def judge_score(lines: Sequence[CaseLine]) -> float | None:
    """The percent of judged replies the judge passed, or `None` when none was judged."""
    verdicts = [reply.verdict for line in lines for reply in line.judged if reply.verdict]
    return _percentage(sum(1 for verdict in verdicts if verdict == "pass"), len(verdicts))


def compute_metrics(lines: Sequence[CaseLine]) -> dict[str, Any]:
    """The run's metrics: the two scores and the four counts."""
    grouped = list(runs_by_case(lines).values())
    unstable = sum(1 for runs in grouped if is_unstable(runs))
    passed = sum(1 for runs in grouped if all(line.passed for line in runs))
    return {
        "check_score": check_score(lines),
        "judge_score": judge_score(lines),
        "cases": len(grouped),
        "passed": passed,
        "failed": len(grouped) - passed - unstable,
        "unstable": unstable,
    }
