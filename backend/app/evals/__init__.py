"""The eval harness: it plays the help desk's cases and scores them.

One engine, one format. `cases` loads what the companion help desk writes; `target` resets
the help desk and plays each turn through `POST /v1/eval/conversation`; `person` is the
simulated person who answers each reply; `runner` plays every case several times, each from
reset data. `world` reads facts and end states, `rules` and `never_rules` are the code
checks, and `results` gives each play one of four results and keeps the scores apart.
`judge` answers each reply check on its own. `report` writes the four run files, `gate`
compares a run with `baseline.json`, `baseline_guard` protects both baseline files, `push`
sends the gauges and `tradeoff` ranks runs by `outcome_every_play`.
"""

from __future__ import annotations
