"""Regex redaction: find personal data in free text, replace it with stable placeholders.

The engine is a pure library: it reads one string and returns another plus a
map, with no HTTP surface and no dependency on the routing policy. Alongside it,
`store.py` holds the placeholder maps a call produced, keyed by tenant and
expiring after an hour, `rehydrate.py` puts those values back into an answer -
whole, or fragment by fragment as a stream delivers it - and
`names_store.py` holds the display names a tenant's operator pushed. The gateway's
`POST /v1/redact`, `POST /admin/directory` and the inline call sites live in
`app.api.routes`, not here.

Ten kinds are detected (`Kind`). Nine have their own pattern and, where the shape
alone is not enough, their own check: Luhn for cards, mod-97 for IBANs, ISO 7064
MOD 11,10 for German tax ids, mod-23 for Irish PPS numbers, and a cue window for
birth dates. The tenth, `PERSON`, is the tenant's own directory compiled into one
alternation by `names.py` and passed to `detect_all` and `redact` as an optional
`directory`; no directory means no `PERSON` candidate, which is what keeps the
engine's older callers unchanged.

Accuracy is measured rather than asserted: the 57-case fixture in
`tests/redaction/fixtures/cases.json` scores precision and recall, and the numbers reach the
README through a report script.
"""

from __future__ import annotations

from app.redaction.detectors import (
    DOB_CUE_DISTANCE,
    DOB_CUES,
    KIND_PRECEDENCE,
    Detection,
    Kind,
    compact_separators,
    detect_all,
)
from app.redaction.errors import (
    DirectoryNameInvalid,
    DirectoryStoreUnavailable,
    RedactionInputTooLong,
    RedactionMapStoreUnavailable,
)
from app.redaction.names import NameDirectory
from app.redaction.names_store import (
    DIRECTORY_KEY_PREFIX,
    MAX_DIRECTORY_NAME_CHARS,
    MAX_DIRECTORY_NAMES,
    MIN_DIRECTORY_NAME_CHARS,
    DirectoryStore,
    InProcessDirectoryStore,
    RedisDirectoryStore,
    checked_entries,
    name_field,
    normalise_name,
)
from app.redaction.redactor import (
    MAX_REDACTION_INPUT_CHARS,
    RedactionResult,
    redact,
    resolve_overlaps,
    taken_placeholders,
)
from app.redaction.rehydrate import StreamRehydrator, rehydrate
from app.redaction.store import (
    MAP_ID_PATTERN,
    REDACTION_MAP_ID_BYTES,
    REDACTION_MAP_KEY_PREFIX,
    REDACTION_MAP_TTL_SECONDS,
    InProcessRedactionMapStore,
    RedactionMapStore,
    RedisRedactionMapStore,
)

__all__ = [
    "DIRECTORY_KEY_PREFIX",
    "DOB_CUES",
    "DOB_CUE_DISTANCE",
    "KIND_PRECEDENCE",
    "MAP_ID_PATTERN",
    "MAX_DIRECTORY_NAMES",
    "MAX_DIRECTORY_NAME_CHARS",
    "MAX_REDACTION_INPUT_CHARS",
    "MIN_DIRECTORY_NAME_CHARS",
    "REDACTION_MAP_ID_BYTES",
    "REDACTION_MAP_KEY_PREFIX",
    "REDACTION_MAP_TTL_SECONDS",
    "Detection",
    "DirectoryNameInvalid",
    "DirectoryStore",
    "DirectoryStoreUnavailable",
    "InProcessDirectoryStore",
    "InProcessRedactionMapStore",
    "Kind",
    "NameDirectory",
    "RedactionInputTooLong",
    "RedactionMapStore",
    "RedactionMapStoreUnavailable",
    "RedactionResult",
    "RedisDirectoryStore",
    "RedisRedactionMapStore",
    "StreamRehydrator",
    "checked_entries",
    "compact_separators",
    "detect_all",
    "name_field",
    "normalise_name",
    "redact",
    "rehydrate",
    "resolve_overlaps",
    "taken_placeholders",
]
