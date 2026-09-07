r"""One trace per request: the provider that makes the spans, and what may go on them.

Three things live here, and nothing else does. The process-wide `TracerProvider`
every span in this codebase comes from; `SpanAttributes`, the CLOSED set of
labels a span may carry; and `build_exporter`, the switch that decides whether
those spans leave the process at all. The middleware that opens a request's root
span is in `gateway/app.py`, because a root span is a gateway concern and `core/`
does not import from `gateway/`.

**Why the attribute set is a type.** "Telemetry never carries user identity" is a
rule someone has to keep, and a rule kept by review is kept until the review that
misses it. `SpanAttributes` forbids unknown fields, so a `user`, an `email` or a
`prompt` is a `ValidationError` at the call site rather than a value on a trace
backend that sits outside this system's residency and redaction guarantees. The
field set is pinned whole by `test_the_attribute_key_set_is_exactly_the_allowlist`
and the leak it prevents by `test_no_message_content_reaches_a_span`.

**Why there is no exporter by default.** An unconfigured gateway builds spans and
drops them. The alternative most SDKs reach for - a console
exporter - would print every request's tenant, session and region to stdout in
production, which is a disclosure nobody asked for and a log bill nobody costed.
`build_exporter` answers `None` unless an endpoint is configured, pinned by
`test_no_exporter_is_installed_without_an_endpoint`.

**Why the request's context is carried in the ASGI scope.** The routes are `def`,
so Starlette runs them in a worker thread, and a stage span that relied on the
ambient OpenTelemetry context surviving that hop would silently become the root
of a second trace on the day it does not. The root span's `Context` is put in the
scope under `TRACE_SCOPE_KEY` and passed to every stage span explicitly, so the
parent is an argument rather than an assumption. Told by
`test_the_stage_spans_share_the_roots_trace_id`, which asserts one trace id
across all five spans of one request.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    Tracer,
    set_span_in_context,
)
from pydantic import BaseModel, ConfigDict
from starlette.types import Scope

from opsconsole.core.settings import Settings

TRACER_NAME = "opsconsole.gateway"
"""The instrumentation scope every span here is created under."""

TRACE_SCOPE_KEY = "opsconsole.tracing"
"""The ASGI scope key holding this request's `RequestTrace`.

A vendor-prefixed key of our own rather than `scope["state"]`, which is shared
with whatever else a deployment mounts and would make a name collision someone
else's bug."""

ROOT_SPAN_NAME = "http.request"
"""What the root span is called. Deliberately not pinned by a test: an earlier change
fixes the four STAGE names and says nothing about this one, so the tests identify
the root structurally (the one span with no local parent) and stay true if it is
renamed."""

SAMPLED_PARENT_FLAGS = TraceFlags(TraceFlags.SAMPLED)
"""A continued trace is treated as sampled. The gateway samples everything in
this plan, so honouring an upstream "not sampled" would drop the very requests a
caller is trying to follow across two services."""

ERROR_STATUS_FLOOR = 400
"""The response status at and above which a root span is marked failed. Pinned by
`test_a_residency_refusal_records_the_code_on_the_span` (451) and
`test_an_unauthorized_request_still_emits_a_trace` (401)."""

_provider: TracerProvider | None = None


def tracer_provider() -> TracerProvider:
    """The one `TracerProvider` this process uses, built on first ask.

    Module-owned rather than installed as OpenTelemetry's global: the global is
    write-once per process, so a second `create_app` in the same test session
    would either warn and be ignored or quietly serve spans from the first app's
    provider. Everything here goes through this function instead, which is also
    what lets the suite attach one in-memory exporter to it once
    (`tests/gateway/conftest.py`, the `span_exporter` fixture).

    The service name comes from the environment rather than from the `Settings`
    an app was built with, because a resource is a property of the PROCESS and the
    provider outlives any one app. Two apps in one process would otherwise have to
    agree on it, and a trace viewer would believe whichever built first.
    """
    global _provider
    if _provider is None:
        _provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: Settings().otel_service_name})
        )
    return _provider


def tracer() -> Tracer:
    """The tracer every span in this codebase is started from."""
    return tracer_provider().get_tracer(TRACER_NAME)


class SpanAttributes(BaseModel):
    """Everything a span in this system is allowed to say, and nothing else.

    Ten optional fields, rendered onto a span as only the ones actually set, so an
    unresolved request does not claim a provider and an unpriced one does not
    claim a cost. `extra="forbid"` is the point of the class: it is what makes "no
    user identity on a trace" a property of the type rather than a promise in a
    review, and `test_the_attribute_key_set_is_exactly_the_allowlist` pins both
    halves - the exact field set, and that an unknown key raises.

    `session` is here and there is no user field to omit: a session is a
    conversation, not a person, and it is what an operator needs to follow one
    caller's traffic through the gateway.

    `cache` and `cost_usd` are named by the design and nothing sets them yet.
    They are declared here so the plans that need them add a value rather than a
    field. The `cache` STAGE SPAN is a different thing and does now exist, on
    every chat request (an earlier change,
    `test_the_cache_span_is_emitted_on_a_miss_and_on_a_hit`): the outcome it
    would put in this field is on the response header and on
    `gateway_requests_total`'s `cache` label instead. Only the `ledger` span is
    still absent, pinned by `test_no_ledger_span_is_emitted`.
    `fallback_count` was the third and is now set, by the routes' fallback loop
    and only on a request that actually fell back
    (`test_a_fallback_is_recorded_on_the_trace_as_a_count`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str | None = None
    session: str | None = None
    alias: str | None = None
    provider: str | None = None
    region: str | None = None
    data_class: str | None = None
    cache: str | None = None
    cost_usd: float | None = None
    redacted: bool | None = None
    fallback_count: int | None = None

    def as_attributes(self) -> dict[str, str | float | bool | int]:
        """The set fields only, as the primitive types OpenTelemetry accepts."""
        return {key: value for key, value in self.model_dump().items() if value is not None}


@dataclass
class RequestTrace:
    """One request's root span, the context to parent stage spans on, and its failure code.

    `error_code` is written by the error handlers in `gateway/errors.py`, the only
    place that knows which `ErrorCode` a failure carried; the middleware reads it
    when it closes the root span. The code travels as a string rather than as the
    exception itself, because an exception on a span brings its message with it
    and a provider's message is exactly what must not reach a trace backend
    (`test_a_provider_failure_records_the_code_on_the_span`).
    """

    span: Span
    context: Context
    error_code: str | None = None


def request_trace(scope: Scope) -> RequestTrace | None:
    """This request's trace record, or `None` when nothing started one."""
    record = scope.get(TRACE_SCOPE_KEY)
    return record if isinstance(record, RequestTrace) else None


def parent_context(trace_id: str, span_id: str) -> Context:
    """The remote parent an incoming `traceparent` describes.

    `is_remote=True` is what tells a backend the parent lives in another service,
    which is the whole of "one trace shared with the Helpdesk": without it the
    root would claim a local parent no exporter ever sent
    (`test_a_valid_traceparent_is_continued`).
    """
    return set_span_in_context(
        NonRecordingSpan(
            SpanContext(
                trace_id=int(trace_id, 16),
                span_id=int(span_id, 16),
                is_remote=True,
                trace_flags=SAMPLED_PARENT_FLAGS,
            )
        )
    )


def start_root_span(parent: Context | None) -> RequestTrace:
    """Open one request's root span, under a remote parent when there is one."""
    span = tracer().start_span(ROOT_SPAN_NAME, context=parent)
    return RequestTrace(span=span, context=set_span_in_context(span))


def end_root_span(record: RequestTrace, *, failed: bool) -> None:
    """Close the root span, recording the typed code when the request failed.

    The status DESCRIPTION carries the code and nothing else. An operator filters
    a trace store on "which requests failed and why" and gets the same closed
    `ErrorCode` set the wire contract uses, with no message, no provider payload
    and no exception text alongside it.
    """
    if failed:
        record.span.set_status(Status(StatusCode.ERROR, record.error_code))
    record.span.end()


def record_error_code(scope: Scope, code: str | None) -> None:
    """Note the typed code a failure carried, for the root span to report."""
    record = request_trace(scope)
    if record is not None:
        record.error_code = code


def annotate(scope: Scope, attributes: SpanAttributes) -> None:
    """Put allowlisted labels on this request's root span."""
    record = request_trace(scope)
    if record is not None:
        record.span.set_attributes(attributes.as_attributes())


@contextmanager
def stage_span(scope: Scope, name: str) -> Iterator[Span | None]:
    """One named stage of a request, parented explicitly on its root span.

    Labels are not taken here. Every attribute this gateway sets belongs to the
    REQUEST rather than to one stage of it - a tenant does not change between
    `auth` and `provider:` - so they go on the root span through `annotate` and an
    operator reads them in one place instead of hunting the child that happened to
    carry one.

    Yields `None` when no root span was started, which is what keeps every call
    site a plain `with` rather than a conditional: a router mounted on an app
    without the middleware still serves, it is just not traced
    (`test_an_untraced_scope_still_runs_every_stage`).

    The parent comes from the scope rather than from the ambient context on
    purpose - see this module's docstring on the worker-thread hop.

    Both of OpenTelemetry's exception defaults are turned OFF, and that is the
    security decision in this function rather than a tuning choice. Left on, a
    stage span whose body raises records an exception EVENT carrying `str(exc)`
    and sets a status description built from it - so a residency refusal would
    write the alias and the regions a tenant may not reach onto the trace, and a
    provider failure would write the provider's own message there, both of which
    the wire contract already refuses to disclose
    (`test_a_residency_refusal_records_the_code_on_the_span`,
    `test_a_provider_failure_records_the_code_on_the_span`). What a failure is
    allowed to say is the typed code, and the root span says it once.
    """
    record = request_trace(scope)
    if record is None:
        yield None
        return
    with tracer().start_as_current_span(
        name, context=record.context, record_exception=False, set_status_on_exception=False
    ) as span:
        yield span


def otlp_headers(raw: str | None) -> dict[str, str] | None:
    """Parse `key=value,key=value` into the header map the exporter takes.

    This is where a Langfuse EU project's basic-auth header comes from, which is
    why it is a setting and not a literal anywhere in this repository (hard rule
    5).
    """
    if raw is None or not raw.strip():
        return None
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        name, separator, value = pair.partition("=")
        if separator and name.strip():
            headers[name.strip()] = value.strip()
    return headers or None


def build_exporter(settings: Settings) -> SpanExporter | None:
    """The OTLP/HTTP exporter, or `None` when no endpoint is configured.

    Decision D: no endpoint means no exporter at all, not a fallback one. Pinned
    by `test_no_exporter_is_installed_without_an_endpoint`.
    """
    if settings.otlp_endpoint is None:
        return None
    return OTLPSpanExporter(
        endpoint=settings.otlp_endpoint, headers=otlp_headers(settings.otlp_headers)
    )


def install_exporter(settings: Settings) -> None:
    """Attach the configured exporter to the process provider, if there is one.

    `BatchSpanProcessor` rather than the simple one: it hands spans to a
    background thread, so a trace backend that is slow or unreachable costs a
    request nothing. That is the difference between an observability tool and an
    outage (`test_an_exporter_that_raises_changes_no_response`).
    """
    exporter = build_exporter(settings)
    if exporter is not None:
        tracer_provider().add_span_processor(BatchSpanProcessor(exporter))


def flush_tracing() -> None:
    """Push whatever is still buffered, on the way down.

    Without this a process that stops between two batch intervals loses the last
    spans it made, which are the ones a crash investigation wants most.
    """
    tracer_provider().force_flush()
