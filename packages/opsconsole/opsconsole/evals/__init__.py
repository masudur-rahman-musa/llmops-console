"""The eval harness: the golden-case format, its verdict, and the run that fills it.

Two halves. The pure one came first: the format an adopter writes by hand
(`cases`, `docs/eval-format.md`, #74) and the deterministic assertions a run is
scored by (`assertions`, #34), both pinned before anything could reach a network.
The half that reaches one is `target`, `residency`, `runner` and `cli` (#33, #38,
`docs/eval-target.md`) - the `ops-eval run` command, the seed reset, the turn and
the per-turn residency check. `16c` adds the report files and the `eval_runs` and
`eval_cases` rows (#39); nothing here writes a file.
"""

from __future__ import annotations
