"""The shared request dependencies: the key store, the request context, the admin key.

Three things every route reaches for and none of them owns:

* the per-tenant API key store and the bearer key a request presents;
* `request_context`, which authenticates a `/v1/*` request and builds the
  identity and labels every later stage reads;
* `admin_key`, the operator key space the `/admin/*` and `/console/*` routes sit
  behind.

Keys are held as sha256 digests and never as plaintext. A presented key is
hashed and compared with `hmac.compare_digest`, so the comparison does not leak
the position of the first differing byte, and nothing here logs the key: the
plaintext appears in no response body and no log record.

The store is a seam. `SettingsKeyStore` reads `tenant:sha256hex` pairs from one
setting, which is the implementation that costs nothing to delete once Postgres
owns the table. Its entries are parsed at construction, so
a malformed pair fails when the process starts and names the pair's 1-based
position, rather than surfacing as a puzzling 401 at request time.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import secrets
from collections.abc import Mapping
from typing import Annotated, Protocol

from fastapi import Depends, Header, Request
from starlette.types import Scope

from app.core.config import Settings
from app.core.context import RequestContext
from app.core.errors import TenantMismatch, Unauthorized
from app.core.metrics import label_tenant
from app.core.tracing import (
    SpanAttributes,
    annotate,
    mark_public,
    note_step,
    request_trace,
    stage_span,
)
from app.core.validation import (
    IDENTIFIER_PATTERN,
    MAX_IDENTIFIER_LENGTH,
    IdentifierInvalid,
    checked_identifier,
    checked_user_id,
)
from app.limits.prices import PriceStore
from app.model_routing.policy import DataClass, PolicyStore
from app.redaction import DirectoryStore, RedactionMapStore
from app.services.chat_accounting import Clients
from app.services.routing_reload import RUNTIME_SCOPE_KEY, RoutingRuntime, RoutingRuntimeStore
from app.storage import ConsoleEventWriter

MAX_KEY_LENGTH = 256
"""Longest bearer key the gateway will hash. The limit is inclusive: a key one
character over it is `UNAUTHORIZED`, like any other unusable key."""

BEARER_SCHEME = "bearer"
"""The `Authorization` scheme, matched case-insensitively: RFC 7235 makes the
scheme token case-insensitive, and a client that sends `bearer` would otherwise
get a 401 whose reason is deliberately never disclosed."""

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
"""A lowercase sha256 digest, matched with `re.fullmatch`: `.match` with `$`
would accept a trailing newline, and the digest is a dictionary key here."""


class KeyStore(Protocol):
    """Where the gateway looks a presented key up. Postgres fills this seam later."""

    def tenant_for(self, key: str) -> str | None:
        """The tenant the key belongs to, or `None` if no stored key matches."""
        ...


def _digest(key: str) -> str:
    """The stored form of a key. No plaintext is kept by the store or logged."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SettingsKeyStore:
    """`tenant:sha256hex` pairs from a single setting, parsed once.

    An empty setting is a valid store that authorises nobody, which is what the
    shipped default has to be.
    """

    def __init__(self, entries: str) -> None:
        self._by_digest: dict[str, str] = {}
        if not entries:
            return
        for position, entry in enumerate(entries.split(","), start=1):
            tenant, digest = self._parsed(entry, position)
            self._by_digest[digest] = tenant

    @staticmethod
    def _parsed(entry: str, position: int) -> tuple[str, str]:
        """Split and validate one pair, naming its position in every rejection."""
        tenant, separator, digest = entry.partition(":")
        if not separator:
            raise ValueError(
                f"api key entry {position}: expected 'tenant:sha256hex', got {entry!r}"
            )
        if len(tenant) > MAX_IDENTIFIER_LENGTH or IDENTIFIER_PATTERN.fullmatch(tenant) is None:
            raise ValueError(f"api key entry {position}: {tenant!r} is not a tenant name")
        if _DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"api key entry {position}: the digest is not a lowercase sha256")
        return tenant, digest

    def tenant_for(self, key: str) -> str | None:
        """Hash the presented key and return its tenant, or `None`.

        Every stored digest is compared with `hmac.compare_digest` rather than
        the presented one being looked up in the dictionary. A dictionary lookup
        is itself a comparison, on a value derived from what the caller sent, so
        the loop is the version with nothing timing-sensitive left in it. Do not
        "optimise" this back to `self._by_digest[presented]`: the cost is one
        hash comparison per configured tenant, and no test can catch the
        regression, because constant-time behaviour is not something an
        assertion observes.
        """
        presented = _digest(key)
        for digest, tenant in self._by_digest.items():
            if hmac.compare_digest(digest, presented):
                return tenant
        return None


def bearer_key(authorization: str | None) -> str | None:
    """The key inside an `Authorization` header, or `None` if there is not one.

    Absent, empty, a scheme other than `Bearer`, an empty value and a value over
    `MAX_KEY_LENGTH` all give `None`: the caller turns every one of them into the
    same `UNAUTHORIZED`. The key itself is
    taken verbatim and never stripped, so a trailing newline or space makes it a
    different key.
    """
    if authorization is None:
        return None
    scheme, separator, key = authorization.partition(" ")
    if not separator or scheme.lower() != BEARER_SCHEME:
        return None
    if not key or len(key) > MAX_KEY_LENGTH:
        return None
    return key


DEFAULT_DATA_CLASS = DataClass.CONFIDENTIAL
"""What a request with no `X-Data-Class` is treated as. Defaulting to `public` would skip
redaction for a tenant's unlabelled traffic, which is the mistake this default avoids."""

TRACEPARENT_PATTERN = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}")
"""W3C `traceparent`, version `00` only, matched with `re.fullmatch`. The
all-zero trace and span ids the spec forbids are rejected separately below."""

_ZERO_TRACE_ID = "0" * 32
_ZERO_SPAN_ID = "0" * 16

AUTH_SPAN = "auth"
"""The stage span this dependency is. The spec names it."""


def emitted_trace_id(scope: Scope) -> str:
    """The trace id of the span this request is actually being traced under.

    Reading it off the started span rather than off the header means
    `X-Gateway-Trace-Id` is the id an engineer can paste
    into a trace viewer, in the continued case and the fresh case alike.

    A request served by an app with no tracing middleware gets a fresh id, which
    keeps the `X-Gateway-Trace-Id` contract true rather than making tracing a
    requirement for answering at all.
    """
    record = request_trace(scope)
    if record is None:
        return secrets.token_hex(16)
    return format(record.span.get_span_context().trace_id, "032x")


def _checked_identifier(value: str, field: str) -> str:
    """Validate one client-supplied header before it is used as a key or a label."""
    return checked_identifier(value, field, param=field)


def _checked_data_class(value: str | None) -> DataClass:
    """Coerce `X-Data-Class` into the closed enum, defaulting only when absent.

    An unknown value is rejected rather than replaced by the default: silently
    downgrading `secret` to `confidential` would route on a class the caller
    never asked for.
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

    Parse and validate only. Minting an id is not done here:
    the root span owns the trace id, so this function answers what the caller
    sent and `TracingMiddleware` decides what the request's own trace is. Two
    places minting an id was the way the response header and the exported trace
    could describe two different traces, which would make "one trace shared with
    the Helpdesk" a claim rather than a fact.

    A malformed
    `traceparent` is ignored rather than rejected, which is what W3C asks of a
    receiver. A broken header upstream should not fail a request that is otherwise
    valid.
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
    x_prompt_version: Annotated[str | None, Header()] = None,
    x_user: Annotated[str | None, Header()] = None,
) -> RequestContext:
    """Authenticate the request and build its context, in a fixed order.

    1. the bearer key, resolved to a tenant, or `UNAUTHORIZED`;
    2. `X-Tenant`, present and a lowercase slug within the cap, or `VALIDATION`;
    3. the two agreeing, or `TENANT_MISMATCH`;
    4. `X-Data-Class`, defaulting to the most restrictive class, and `X-Session`
       and `X-Prompt-Version`, both optional and slug-checked (a SHA-256 hex digest
       is exactly the 64-character cap), and `X-User`, optional and held to the
       random id shape `u_` and 32 lowercase hex characters;
    5. the trace id, read off the root span `TracingMiddleware` started.

    Auth comes first so an unauthenticated caller learns nothing about which of
    its other headers were also wrong.

    `X-Tenant` is declared optional to FastAPI and required here, so its absence
    is answered after the key rather than before it: FastAPI raises its own
    validation error while resolving parameters, which would turn an
    unauthenticated call into a 422.
    """
    store: KeyStore = request.app.state.key_store

    with stage_span(request.scope, AUTH_SPAN) as span:
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
            prompt_version=_checked_identifier(x_prompt_version, "X-Prompt-Version")
            if x_prompt_version is not None
            else None,
            user_id=checked_user_id(x_user, param="X-User") if x_user is not None else None,
        )
        note_step(span, input="API key and X-Tenant", output=f"accepted: tenant {tenant}")
        if _trace_is_public(request, tenant):
            mark_public(request.scope, span)

    annotate(
        request.scope,
        SpanAttributes(
            tenant=context.tenant,
            session=context.session,
            user_id=context.user_id,
            data_class=context.data_class.value,
        ),
    )
    label_tenant(request.scope, context.tenant)
    return context


def _trace_is_public(request: Request, tenant: str) -> bool:
    """Whether this authenticated request's trace goes public.

    The setting is the master switch; a tenant can only opt out.
    """
    settings: Settings = request.app.state.settings
    if not settings.public_trace_langfuse:
        return False
    store: PolicyStore = request.app.state.policy
    regime = store.policy.tenants.get(tenant)
    return regime is None or regime.trace_public is not False


Context = Annotated[RequestContext, Depends(request_context)]
"""What a route asks for to get an authenticated, checked request."""


TENANT_PARAM = "tenant"
"""The query parameter the routing view filters on, and the `param` every
rejection of it names."""


def checked_tenant(value: str) -> str:
    """Validate the `?tenant=` filter before it is used to look anything up.

    a project rule applied to the one client-supplied string on this surface.
    The cap and the pattern are the resolver's own rather than numbers copied
    here, so the filter and the request path can never disagree about what a
    tenant name is; `IDENTIFIER_PATTERN` is a `fullmatch`, which is what keeps a
    trailing newline out. The length limit is inclusive.
    """
    return checked_identifier(value, TENANT_PARAM, param=TENANT_PARAM)


def admin_key(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    """Authorise an `/admin/*` call, or raise `Unauthorized`.

    The header shape is `bearer_key`'s, so the scheme handling and the length cap
    are the ones the tenant path already uses; only the store the key is checked
    against differs. The digest of the presented key is compared with the
    configured one under `hmac.compare_digest`, which does not leak the position
    of the first differing byte.

    An unset `admin_key_sha256` authorises nobody. It is compared rather than
    short-circuited, so there is no "not configured" branch a deployment could
    fall into open; the comparison simply never succeeds, because no key hashes
    to the empty string.

    Nothing here logs or echoes the presented key: the refusal message says only
    that no usable admin key was presented, exactly as the tenant path's does.
    """
    settings: Settings = request.app.state.settings
    key = bearer_key(authorization)
    presented = hashlib.sha256(key.encode("utf-8")).hexdigest() if key is not None else ""
    if not key or not hmac.compare_digest(settings.admin_key_sha256, presented):
        raise Unauthorized("no usable admin key was presented")


# --- typed state dependencies --------------------------------------
#
# Every `/v1`, `/admin` and `/console` route reaches the clients and stores the
# lifespan built through one of these getters instead of reading
# `request.app.state.<name>` by hand. A route takes the matching `*Dep` alias as a
# parameter; a test swaps what the route sees with
# `app.dependency_overrides[get_<name>]` rather than by mutating `app.state`. The getters live here,
# not in the route files, so a scan of those files finds no `app.state` read left.


def get_clients(request: Request) -> Clients:
    """All the stores and clients the chat and embeddings pipeline reads.

    The chat and embeddings routes take this as `ClientsDep` and pass it to
    `complete_chat` / `embed_texts`.  Building it here keeps those routes free
    of any `request.app.state` read and lets tests override it with
    `app.dependency_overrides[get_clients]`.

    `policy`, `adapters` and `breaker` come from the `RoutingRuntime` the
    middleware leased for this request, so they are always from the same reload
    generation.
    """
    state = request.app.state
    runtime: RoutingRuntime = request.scope[RUNTIME_SCOPE_KEY]
    return Clients(
        policy=runtime.policy,
        rate_limit_store=state.rate_limit_store,
        budget_store=state.budget_store,
        settings=state.settings,
        cost_ledger=state.cost_ledger,
        prices=state.prices,
        adapters=runtime.adapters,
        breaker=runtime.breaker,
        response_cache=state.response_cache,
        semantic_cache=state.semantic_cache,
        redaction_map_store=state.redaction_map_store,
        directory_store=state.directory_store,
        month_spend=state.month_spend,
    )


def get_policy(request: Request) -> PolicyStore:
    """The routing policy snapshot store, built in `create_app`."""
    policy: PolicyStore = request.app.state.policy
    return policy


def get_settings(request: Request) -> Settings:
    """The settings `create_app` was built with."""
    settings: Settings = request.app.state.settings
    return settings


def get_prices(request: Request) -> PriceStore:
    """The price table store, built in `create_app`."""
    prices: PriceStore = request.app.state.prices
    return prices


def get_redaction_map_store(request: Request) -> RedactionMapStore:
    """The redaction map store the lifespan built (Redis-backed or in-process)."""
    store: RedactionMapStore = request.app.state.redaction_map_store
    return store


def get_directory_store(request: Request) -> DirectoryStore:
    """The tenant directory store the lifespan built (Redis-backed or in-process)."""
    store: DirectoryStore = request.app.state.directory_store
    return store


def get_console_events(request: Request) -> ConsoleEventWriter:
    """The console event writer the lifespan built (Postgres-backed or in-memory)."""
    writer: ConsoleEventWriter = request.app.state.console_events
    return writer


def get_provider_limiter(request: Request) -> asyncio.Semaphore:
    """The provider concurrency semaphore `create_app` built.

    The chat and embeddings routes use this to bound how many provider calls run
    at once, so one slow provider fills only its own pool and cannot stall other
    tenants. Tests override this dep to inject a spy or a fake limiter.
    """

    limiter: asyncio.Semaphore = request.app.state.provider_limiter
    return limiter


def get_routing_store(request: Request) -> RoutingRuntimeStore:
    """The store that swaps the policy, adapters and breaker as one, built in `create_app`."""
    store: RoutingRuntimeStore = request.app.state.routing_store
    return store


def get_provider_environ(request: Request) -> Mapping[str, str]:
    """The environment the provider keys are read from, set in `create_app`."""
    environ: Mapping[str, str] = request.app.state.provider_environ
    return environ


ClientsDep = Annotated[Clients, Depends(get_clients)]
PolicyDep = Annotated[PolicyStore, Depends(get_policy)]
PricesDep = Annotated[PriceStore, Depends(get_prices)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
RedactionMapStoreDep = Annotated[RedactionMapStore, Depends(get_redaction_map_store)]
DirectoryStoreDep = Annotated[DirectoryStore, Depends(get_directory_store)]
ConsoleEventsDep = Annotated[ConsoleEventWriter, Depends(get_console_events)]
ProviderLimiterDep = Annotated[asyncio.Semaphore, Depends(get_provider_limiter)]
RoutingStoreDep = Annotated[RoutingRuntimeStore, Depends(get_routing_store)]
ProviderEnvironDep = Annotated[Mapping[str, str], Depends(get_provider_environ)]
