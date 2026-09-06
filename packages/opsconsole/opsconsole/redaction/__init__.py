"""Regex redaction: find personal data in free text, replace it with stable placeholders.

A pure library. It reads one string and returns another plus a map; it has no
HTTP surface, no store and no dependency on the routing policy. An earlier change adds
`POST /v1/redact`, the map store and the inline path for a data class marked
`redact` around this package.

Nine kinds are detected (`Kind`), each with its own pattern and, where the shape
alone is not enough, its own check: Luhn for cards, mod-97 for IBANs, ISO 7064
MOD 11,10 for German tax ids, mod-23 for Irish PPS numbers, and a cue window for
birth dates. Accuracy is measured rather than asserted: the 50-case fixture in
`tests/redaction/fixtures/cases.json` scores precision and recall, the floors
live in `tests/redaction/test_fixture_accuracy.py`, and the numbers reach the
README through `scripts/redaction_report.py`.
"""

from __future__ import annotations

from opsconsole.redaction.detectors import (
    DOB_CUE_DISTANCE,
    DOB_CUES,
    KIND_PRECEDENCE,
    Detection,
    Kind,
    compact_separators,
    detect_all,
)
from opsconsole.redaction.errors import RedactionInputTooLong
from opsconsole.redaction.redactor import (
    MAX_REDACTION_INPUT_CHARS,
    RedactionResult,
    redact,
    resolve_overlaps,
)

__all__ = [
    "DOB_CUES",
    "DOB_CUE_DISTANCE",
    "KIND_PRECEDENCE",
    "MAX_REDACTION_INPUT_CHARS",
    "Detection",
    "Kind",
    "RedactionInputTooLong",
    "RedactionResult",
    "compact_separators",
    "detect_all",
    "redact",
    "resolve_overlaps",
]
