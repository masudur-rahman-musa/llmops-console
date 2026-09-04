"""What every `/v1/*` request is, once its headers have been checked.

One dependency builds the whole envelope, in the order an earlier change decision F fixes:

1. the bearer key, resolved to a tenant, or `UNAUTHORIZED`;
2. `X-Tenant`, present and a lowercase slug within the cap, or `VALIDATION`;
3. the two agreeing, or `TENANT_MISMATCH`;
4. `X-Data-Class`, defaulting to the most restrictive class rather than the most
   permissive one, and `X-Session`, optional and slug-checked;
5. `traceparent`, carried when it is a valid W3C header and replaced with a new
   trace id when it is absent or malformed.

Auth comes first so that an unauthenticated caller learns nothing about which of
its other headers were also wrong; `test_models_requires_auth` sends none of them
and pins the 401.

The identifier pre-check is the resolver's rule applied at the edge: the same
`IDENTIFIER_PATTERN` and `MAX_IDENTIFIER_LENGTH`, imported rather than restated,
so a header and a resolver argument can never disagree about what a name is.
`X-Session` gets it too because the session id becomes a cache key part and a
metric label.
"""

from __future__ import annotations

import re
import secrets
from typing import Annotated

from fastapi import Depends, Header, Request
from pydantic import BaseModel, ConfigDict

from opsconsole.core.errors import TenantMismatch, Unauthorized
from opsconsole.gateway.auth import KeyStore, bearer_key
from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.policy import DataClass
from opsconsole.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH

DEFAULT_DATA_CLASS = DataClass.CONFIDENTIAL
"""What a request with no `X-Data-Class` is treated as. Pinned by
`test_data_class_defaults_to_confidential`. Defaulting to `public` would route a
tenant's unlabelled traffic to the widest region set it has, which is the one
mistake residency cannot recover from."""

TRACEPARENT_PATTERN = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}")
"""W3C `traceparent`, version `00` only, matched with `re.fullmatch`. The
all-zero trace and span ids the spec forbids are rejected separately below.
The rejection table is `test_malformed_traceparent_generates_one`."""

_ZERO_TRACE_ID = "0" * 32
_ZERO_SPAN_ID = "0" * 16


class RequestContext(BaseModel):
    """The identity and labels every later stage of a request reads.

    No user identity, by construction: a tenant, an optional session and a trace
    id are the only things here, which is what keeps telemetry free of it.
    `test_context_carries_the_authenticated_tenant` compares the field set
    exactly, so a field added here goes red rather than quietly reaching a trace;
    that red is the invariant, not a stale expectation to update.
    """

    model_config = ConfigDict(frozen=True)

    tenant: str
    data_class: DataClass
    session: str | None
    trace_id: str


def _checked_identifier(value: str, field: str) -> str:
    """Validate one client-supplied header before it is used as a key or a label."""
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(
            f"{field}: longer than {MAX_IDENTIFIER_LENGTH} characters", param=field
        )
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{field}: not a lowercase slug", param=field)
    return value


def _checked_data_class(value: str | None) -> DataClass:
    """Coerce `X-Data-Class` into the closed enum, defaulting only when absent.

    An unknown value is rejected rather than replaced by the default: silently
    downgrading `secret` to `confidential` would route on a class the caller
    never asked for (`test_unknown_data_class_is_422`).
    """
    if value is None:
        return DEFAULT_DATA_CLASS
    try:
        return DataClass(value)
    except ValueError as exc:
        raise IdentifierInvalid(
            f"X-Data-Class: {value!r} is not a known data class", param="X-Data-Class"
        ) from exc


def trace_id_from(traceparent: str | None) -> str:
    """The upstream trace id, or a fresh one when there is not a usable header.

    Decision C: a malformed `traceparent` is ignored rather than rejected, which
    is what W3C asks of a receiver. A broken header upstream should not fail a
    request that is otherwise valid.
    """
    if traceparent is not None:
        match = TRACEPARENT_PATTERN.fullmatch(traceparent)
        if (
            match is not None
            and match.group(1) != _ZERO_TRACE_ID
            and match.group(2) != _ZERO_SPAN_ID
        ):
            return match.group(1)
    return secrets.token_hex(16)


def request_context(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_tenant: Annotated[str | None, Header()] = None,
    x_data_class: Annotated[str | None, Header()] = None,
    x_session: Annotated[str | None, Header()] = None,
    traceparent: Annotated[str | None, Header()] = None,
) -> RequestContext:
    """Authenticate the request and build its context, in the order above.

    `X-Tenant` is declared optional to FastAPI and required here, so its absence
    is answered after the key rather than before it: FastAPI raises its own
    validation error while resolving parameters, which would turn an
    unauthenticated call into a 422.
    """
    store: KeyStore = request.app.state.key_store

    key = bearer_key(authorization)
    tenant = store.tenant_for(key) if key is not None else None
    if tenant is None:
        raise Unauthorized("no usable API key was presented")

    if x_tenant is None:
        raise IdentifierInvalid("X-Tenant: required on every /v1 request", param="X-Tenant")
    claimed = _checked_identifier(x_tenant, "X-Tenant")
    if not secrets.compare_digest(claimed, tenant):
        raise TenantMismatch("X-Tenant: does not match the API key", param="X-Tenant")

    return RequestContext(
        tenant=tenant,
        data_class=_checked_data_class(x_data_class),
        session=_checked_identifier(x_session, "X-Session") if x_session is not None else None,
        trace_id=trace_id_from(traceparent),
    )


Context = Annotated[RequestContext, Depends(request_context)]
"""What a route asks for to get an authenticated, checked request."""
