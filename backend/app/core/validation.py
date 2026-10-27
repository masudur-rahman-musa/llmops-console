"""The one identifier check, shared by everything that keys on a slug.

A client-supplied tenant, alias, map id or cache key part becomes a dictionary
key, a metric label, a cache key part or a URL segment, so it is checked once
here and the same way everywhere. Before this module the check was copied into
every store; now the copies import `checked_identifier` from here, so a tenant
the running gateway serves cannot be one a store quietly refuses.

This lives in `core` rather than in the routing package on purpose: the resolver
was the first raiser, but the rule is a core concern that `core` must own with no
import back into `model_routing`, so the resolver imports the check from here
without a cycle.

A check with a different message, `param` or width keeps its own wrapper and only
sources the constants from here (the `param=`-carrying stores, the Postgres
identifier limit in `storage/db.py`, the wider principal and set-name caps).
"""

from __future__ import annotations

import re
from typing import ClassVar

from app.core.errors import ErrorCode, GatewayError

MAX_IDENTIFIER_LENGTH = 64
"""Longest tenant or alias a request may name."""

IDENTIFIER_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
"""Lowercase slug, matched with `re.fullmatch`.

`.match` with `^...$` would accept a trailing newline, and these strings become
dictionary keys, metric labels and cache key parts. The tests cover a trailing newline, a trailing
CRLF and a trailing space.
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


def checked_identifier(value: str, field: str, *, param: str | None = None) -> str:
    """Validate one client-supplied identifier before it is used as a key.

    `re.fullmatch`, so a trailing newline is refused. The
    inclusive boundary at `MAX_IDENTIFIER_LENGTH` is allowed.

    `param` names the request field a rejection points at, for the stores and
    headers that answer a caller with a 422 naming the bad input; it defaults to
    `None` for the raisers that do not.
    """
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(
            f"{field}: longer than {MAX_IDENTIFIER_LENGTH} characters", param=param
        )
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{field}: not a lowercase slug", param=param)
    return value


USER_ID_PATTERN = re.compile(r"u_[0-9a-f]{32}")
"""A random user id: `u_` and exactly 32 lowercase hex characters, with `re.fullmatch`.

The shape is the whole check that the id says nothing about a person: a name or an
email cannot match it. a design note. Pinned by `test_user_id_rejection_table`.
"""


def checked_user_id(value: str, *, param: str | None = None) -> str:
    """Validate one random user id before it reaches a key, a cost row or a trace.

    The message never repeats the value: a caller that sent an email by mistake
    must not see it echoed back, or logged with the error.
    """
    if USER_ID_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("user id: not u_ and 32 lowercase hex characters", param=param)
    return value
