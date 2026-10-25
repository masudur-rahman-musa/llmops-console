"""The eval harness: it plays the help desk's multi-turn conversation cases and scores them.

One engine, one format. `conversations` loads the cases the companion help desk writes;
`target` resets the help desk and plays each case through `POST /v1/eval/conversation`;
`runner` saves each finished case at once and resumes a stopped run; `checks` makes the
three per-turn checks (right tools in order, no change before the yes, no made-up
number or role name). `judge` grades each graded reply pass or fail, `labels` makes the
reference labels once, and `calibration` checks the judge against them with a two-class
kappa. `scores` and `report` write the two scores and the two report files, `gate`
compares a run with `conversation-baseline.json`, `baseline_guard` protects both baseline
files, `push` sends the gauges and `tradeoff` ranks runs by `check_score`.
"""

from __future__ import annotations
