"""Regex redaction: find personal data in free text, replace it with stable placeholders.

The engine is a pure library: it reads one string and returns another plus a
map, with no HTTP surface and no dependency on the routing policy. Alongside it,
`store.py` holds the placeholder maps a call produced, keyed by tenant and
expiring after an hour, so an earlier change can rehydrate from one. The gateway's
`POST /v1/redact` and its inline call site live in `opsconsole.gateway.routes`,
not here.

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
    taken_placeholders,
)
from opsconsole.redaction.store import (
    REDACTION_MAP_ID_BYTES,
    REDACTION_MAP_TTL_SECONDS,
    InProcessRedactionMapStore,
    RedactionMapStore,
)

__all__ = [
    "DOB_CUES",
    "DOB_CUE_DISTANCE",
    "KIND_PRECEDENCE",
    "MAX_REDACTION_INPUT_CHARS",
    "REDACTION_MAP_ID_BYTES",
    "REDACTION_MAP_TTL_SECONDS",
    "Detection",
    "InProcessRedactionMapStore",
    "Kind",
    "RedactionInputTooLong",
    "RedactionMapStore",
    "RedactionResult",
    "compact_separators",
    "detect_all",
    "redact",
    "resolve_overlaps",
    "taken_placeholders",
]
