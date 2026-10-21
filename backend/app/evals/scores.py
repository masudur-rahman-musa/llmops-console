"""The two scores one conversation run reports, and the counts behind them.

Pure: `compute_metrics` reads the `CaseLine` records it is handed and nothing else.

* `check_score` is the percent of cases that passed all three checks. A case that never
  answered counts as not passed and stays in the denominator: leaving it out would shrink
  the number every gate is measured against.
* `judge_score` is the percent of judged replies the judge passed. A reply the judge could
  not grade is out of the denominator. With nothing judged it is `None`, never `0.0`: zero
  would read as a judge that failed every reply.
* Both are rounded half up to two places through `Decimal`, never through `round()`, and the
  gate compares the written number.

The counts `cases`, `passed` and `failed` go beside them. Types are part of the contract:
the database stores this mapping as `jsonb`, so a score is a `float` or `None` and a count
is an `int`.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from app.evals.runner import CaseLine

METRIC_NAMES: Final[tuple[str, ...]] = ("check_score", "judge_score")
"""The two published scores, in the order every consumer lists them."""

SCORE_SCALE: Final[int] = 2
"""Decimal places a score is published at."""


def _percentage(part: int, whole: int) -> float | None:
    """`part` of `whole` as a percent at `SCORE_SCALE` places, or `None` for no whole."""
    if whole == 0:
        return None
    value = Decimal(part * 100) / Decimal(whole)
    return float(value.quantize(Decimal(1).scaleb(-SCORE_SCALE), rounding=ROUND_HALF_UP))


def check_score(lines: Sequence[CaseLine]) -> float | None:
    """The percent of cases passed, or `None` for a run with no cases."""
    return _percentage(sum(1 for line in lines if line.passed), len(lines))


def judge_score(lines: Sequence[CaseLine]) -> float | None:
    """The percent of judged replies the judge passed, or `None` when none was judged."""
    verdicts = [reply.verdict for line in lines for reply in line.judged if reply.verdict]
    return _percentage(sum(1 for verdict in verdicts if verdict == "pass"), len(verdicts))


def compute_metrics(lines: Sequence[CaseLine]) -> dict[str, Any]:
    """The run's metrics: the two scores and the three counts."""
    passed = sum(1 for line in lines if line.passed)
    return {
        "check_score": check_score(lines),
        "judge_score": judge_score(lines),
        "cases": len(lines),
        "passed": passed,
        "failed": len(lines) - passed,
    }
