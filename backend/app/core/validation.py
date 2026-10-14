"""The one identifier check, shared by everything that keys on a slug.

A client-supplied tenant, alias, map id or cache key part becomes a dictionary
key, a metric label, a cache key part or a URL segment, so it is checked once
here and the same way everywhere. Before this module the check was copied into
every store; now the copies import `checked_identifier` from here, so a tenant
the running gateway serves cannot be one a store quietly refuses.

This lives in `core` rather than in the routing package on purpose: the resolver
was the first raiser, but the rule is a core concern that `core` must own with no
import back into `model_routing`, which is what lets `core/config.py` read
`ResidencyMode` from itself and the resolver import the check from here without a
cycle.

A check with a different message, `param` or width keeps its own wrapper and only
sources the constants from here (the `param=`-carrying stores, the Postgres
identifier limit in `storage/db.py`, the wider principal and set-name caps).
"""

from __future__ import annotations

import re
from typing import ClassVar

from app.core.errors import ErrorCode, GatewayError

MAX_IDENTIFIER_LENGTH = 64
"""Longest tenant or alias a request may name. Pinned by `test_max_identifier_length_is_pinned`."""

IDENTIFIER_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
"""Lowercase slug, matched with `re.fullmatch`.

`.match` with `^...$` would accept a trailing newline, and these strings become
dictionary keys, metric labels and cache key parts. The rejection table in
`test_identifier_rejection_table` covers a trailing newline, a trailing CRLF and
a trailing space.
"""


class IdentifierInvalid(GatewayError):
    """A client-supplied identifier failed its pre-check.

    One code covers every rejected input. It subclasses `GatewayError` directly
    rather than the routing package's `RoutingError`, because it now lives in
    `core` and `core` imports nothing from `model_routing`; the error contract
    test walks `GatewayError.__subclasses__()`, so VALIDATION is still carried by
    exactly this one class.

    r`grep -rlE "raise Identifier[I]nvalid" backend/app` names every module that
    raises it directly; the shared `checked_identifier` below raises it for the
    stores that dropped their own copy. The bracket keeps this quotation out of
    its own results so the command can be re-run and compared.
    """

    code: ClassVar[ErrorCode] = ErrorCode.VALIDATION
    http_status: ClassVar[int] = 422


def checked_identifier(value: str, field: str) -> str:
    """Validate one client-supplied identifier before it is used as a key.

    `re.fullmatch`, so a trailing newline is refused. The
    inclusive boundary at `MAX_IDENTIFIER_LENGTH` and the rejection table are
    pinned by `test_identifier_at_the_maximum_length_is_accepted` and
    `test_identifier_rejection_table`.
    """
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(f"{field}: longer than {MAX_IDENTIFIER_LENGTH} characters")
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{field}: not a lowercase slug")
    return value
