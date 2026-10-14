"""The eval harness: the golden-case format, its verdict, and the run that fills it.

Two halves. The pure one came first: the format an adopter writes by hand
(`cases`, `docs/eval-format.md`, #74) and the deterministic assertions a run is
scored by (`assertions`, #34), both pinned before anything could reach a network.
The half that reaches one is `target`, `residency`, `runner` and `cli` (#33, #38,
`docs/eval-target.md`) - the `ops-eval run` command, the seed reset, the turn and
the per-turn residency check. `report` writes the two files a run leaves behind
and `store` writes the `eval_runs` and `eval_cases` rows built from the same
record (#39, `docs/eval-report.md`), the rows the console's eval page reads.
`taxonomy` tags every failed case, `metrics` scores every run, and `judge` is the
one module that asks a model anything (#35): the rubric cases, three samples at
temperature zero through the gateway's `judge` alias, the median taken.
"""

from __future__ import annotations
