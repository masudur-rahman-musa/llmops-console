"""The eval harness's pure core: the golden-case format and its verdict.

Nothing in this package does I/O beyond reading a golden-set file. The runner
that reaches a target, resets the seed and writes a report is `16b`/`16c`; what
lives here is the format an adopter writes by hand (`docs/eval-format.md`, #74)
and the deterministic assertions a run is scored by (#34), so both are pinned
before anything can reach a network.
"""

from __future__ import annotations
