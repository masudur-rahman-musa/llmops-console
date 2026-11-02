"""The names of the scores a run publishes.

The scores themselves are computed in `results.py`, from the finished plays. This module
only fixes which of them leave the process: the gate reads `safety` and
`outcome_every_play`, and the nightly push sends the same two plus `outcome_at_least_once`.
"""

from __future__ import annotations

from typing import Final

METRIC_NAMES: Final[tuple[str, ...]] = (
    "safety",
    "outcome_every_play",
    "outcome_at_least_once",
)
"""The scores that become `gateway_eval_*` gauges."""
