"""What every `/v1/*` request is, once its headers have been checked.

One dependency builds the whole envelope, in the order an earlier change decision F fixes:

1. the bearer key, resolved to a tenant, or `UNAUTHORIZED`;
2. `X-Tenant`, present and a lowercase slug within the cap, or `VALIDATION`;
3. the two agreeing, or `TENANT_MISMATCH`;
4. `X-Data-Class`, defaulting to the most restrictive class rather than the most
   permissive one, and `X-Session`, optional and slug-checked;
5. the trace id, read off the root span `TracingMiddleware` has already started
   from the caller's `traceparent` - continued when that header is valid, fresh
   when it is absent or malformed.

Auth comes first so that an unauthenticated caller learns nothing about which of
its other headers were also wrong; `test_models_requires_auth` sends none of them
and pins the 401.

The identifier pre-check is the resolver's rule applied at the edge: the same
`IDENTIFIER_PATTERN` and `MAX_IDENTIFIER_LENGTH`, imported rather than restated,
so a header and a resolver argument can never disagree about what a name is.
`X-Session` gets it too because the session id becomes a cache key part and a
metric label.

This dependency is also the `auth` stage span, and it is where the request's
tenant, session and data class are put on the root span. Those three are the only
labels this file contributes, and they go through `SpanAttributes`, so a field
that is not on an earlier change's allowlist cannot be added here without failing to
validate (`test_the_attribute_key_set_is_exactly_the_allowlist`).
"""

from __future__ import annotations

import re
import secrets
from typing import Annotated

from fastapi import Depends, Header, Request
from pydantic import BaseModel, ConfigDict
from starlette.types import Scope

from opsconsole.core.errors import TenantMismatch, Unauthorized
from opsconsole.core.tracing import (
    SpanAttributes,
    annotate,
    request_trace,
    stage_span,
)
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

AUTH_SPAN = "auth"
"""The stage span this dependency is. the design names it; pinned by
`test_the_emitted_span_names_are_exactly_the_stages_that_exist`."""


def emitted_trace_id(scope: Scope) -> str:
    """The trace id of the span this request is actually being traced under.

    Reading it off the started span rather than off the header is the whole of
    an earlier change decision B: `X-Gateway-Trace-Id` is then the id an engineer can paste
    into a trace viewer, in the continued case and the fresh case alike
    (`test_the_trace_id_header_is_the_emitted_traces_id`).

    A request served by an app with no tracing middleware gets a fresh id, which
    keeps the `X-Gateway-Trace-Id` contract true rather than making tracing a
    requirement for answering at all
    (`test_an_untraced_scope_still_gets_a_trace_id`).
    """
    record = request_trace(scope)
    if record is None:
        return secrets.token_hex(16)
    return format(record.span.get_span_context().trace_id, "032x")


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


def parsed_traceparent(traceparent: str) -> tuple[str, str] | None:
    """The upstream trace id and span id, or `None` when the header is not usable.

    Parse and validate only. An earlier change decision B moved the MINTING out of here:
    the root span owns the trace id now, so this function answers what the caller
    sent and `TracingMiddleware` decides what the request's own trace is. Two
    places minting an id was the way the response header and the exported trace
    could describe two different traces, which would make "one trace shared with
    the Helpdesk" a claim rather than a fact.

    An earlier change decision C still holds for what a bad header does: a malformed
    `traceparent` is ignored rather than rejected, which is what W3C asks of a
    receiver. A broken header upstream should not fail a request that is otherwise
    valid, and the rejection table is `test_malformed_traceparent_generates_one`
    with `test_a_malformed_traceparent_starts_a_new_trace` as its tracing half.
    """
    match = TRACEPARENT_PATTERN.fullmatch(traceparent)
    if match is None or match.group(1) == _ZERO_TRACE_ID or match.group(2) == _ZERO_SPAN_ID:
        return None
    return match.group(1), match.group(2)


def request_context(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_tenant: Annotated[str | None, Header()] = None,
    x_data_class: Annotated[str | None, Header()] = None,
    x_session: Annotated[str | None, Header()] = None,
) -> RequestContext:
    """Authenticate the request and build its context, in the order above.

    `X-Tenant` is declared optional to FastAPI and required here, so its absence
    is answered after the key rather than before it: FastAPI raises its own
    validation error while resolving parameters, which would turn an
    unauthenticated call into a 422.
    """
    store: KeyStore = request.app.state.key_store

    with stage_span(request.scope, AUTH_SPAN):
        key = bearer_key(authorization)
        tenant = store.tenant_for(key) if key is not None else None
        if tenant is None:
            raise Unauthorized("no usable API key was presented")

        if x_tenant is None:
            raise IdentifierInvalid("X-Tenant: required on every /v1 request", param="X-Tenant")
        claimed = _checked_identifier(x_tenant, "X-Tenant")
        if not secrets.compare_digest(claimed, tenant):
            raise TenantMismatch("X-Tenant: does not match the API key", param="X-Tenant")

        context = RequestContext(
            tenant=tenant,
            data_class=_checked_data_class(x_data_class),
            session=_checked_identifier(x_session, "X-Session") if x_session is not None else None,
            trace_id=emitted_trace_id(request.scope),
        )

    annotate(
        request.scope,
        SpanAttributes(
            tenant=context.tenant,
            session=context.session,
            data_class=context.data_class.value,
        ),
    )
    return context


Context = Annotated[RequestContext, Depends(request_context)]
"""What a route asks for to get an authenticated, checked request."""
