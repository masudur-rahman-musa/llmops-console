r"""One trace per request: the provider that makes the spans, and what may go on them.

Three things live here, and nothing else does. The process-wide `TracerProvider`
every span in this codebase comes from; `SpanAttributes`, the CLOSED set of
labels a span may carry; and `build_exporter`, the switch that decides whether
those spans leave the process at all. The middleware that opens a request's root
span is in the app module, because a root span is a gateway concern and `core/`
does not import from the app layer.

**Why the attribute set is a type.** "Telemetry never carries user identity" is a
rule someone has to keep, and a rule kept by review is kept until the review that
misses it. `SpanAttributes` forbids unknown fields, so a `user`, an `email` or a
`prompt` is a `ValidationError` at the call site rather than a value on a trace
backend that sits outside this system's redaction guarantees. The
field set is pinned whole by tests.

**Why there is no exporter by default.** An unconfigured gateway builds spans and
drops them. The alternative most SDKs reach for - a console
exporter - would print every request's tenant and session to stdout in
production, which is a disclosure nobody asked for and a log bill nobody costed.
`build_exporter` answers `None` unless an endpoint is configured.

**Why the request's context is carried in the ASGI scope.** The routes are `def`,
so Starlette runs them in a worker thread, and a stage span that relied on the
ambient OpenTelemetry context surviving that hop would silently become the root
of a second trace on the day it does not. The root span's `Context` is put in the
scope under `TRACE_SCOPE_KEY` and passed to every stage span explicitly, so the
parent is an argument rather than an assumption. A test asserts one trace id across all five spans
of one request.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

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
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.types import Scope

from app.core.config import Settings
from app.core.errors import GatewayError

TRACER_NAME = "app.gateway"
"""The instrumentation scope every span here is created under."""

TRACE_SCOPE_KEY = "app.tracing"
"""The ASGI scope key holding this request's `RequestTrace`.

A vendor-prefixed key of our own rather than `scope["state"]`, which is shared
with whatever else a deployment mounts and would make a name collision someone
else's bug."""

ROOT_SPAN_NAME = "http.request"
"""What the root span is called until the request ends and `root_span_name` renames
it after the matched route."""

SAMPLED_PARENT_FLAGS = TraceFlags(TraceFlags.SAMPLED)
"""A continued trace is treated as sampled. The gateway samples everything in
this system, so honouring an upstream "not sampled" would drop the very requests a
caller is trying to follow across two services."""

ERROR_STATUS_FLOOR = 400
"""The response status at and above which a root span is marked failed."""

_provider: TracerProvider | None = None


def tracer_provider() -> TracerProvider:
    """The one `TracerProvider` this process uses, built on first ask.

    Module-owned rather than installed as OpenTelemetry's global: the global is
    write-once per process, so a second `create_app` in the same test session
    would either warn and be ignored or quietly serve spans from the first app's
    provider. Everything here goes through this function instead, which is also
    what lets the suite attach one in-memory exporter to it once
    (the `span_exporter` fixture in the tests).

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

    Eleven optional fields, rendered onto a span as only the ones actually set, so an
    unresolved request does not claim a provider and an unpriced one does not
    claim a cost. `extra="forbid"` is the point of the class: it is what makes "no
    user identity on a trace" a property of the type rather than a promise in a
    review, and tests pin both
    halves - the exact field set, and that an unknown key raises.

    `session` is a conversation, not a person, and it is what an operator needs to
    follow one caller's traffic through the gateway. `user_id` is the one random id
    the help desk sends in `X-User`, never a name or an email; only the help desk
    can map it to a person.

    `cache` and `cost_usd` are named in the design and nothing sets them yet.
    They are declared here so the plans that need them add a value rather than a
    field. The `cache` STAGE SPAN is a different thing and does now exist, on
    every chat request: the outcome it
    would put in this field is on the response header and on
    `gateway_requests_total`'s `cache` label instead. Only the `ledger` span is
    still absent.
    `fallback_count` was the third and is now set, by the routes' fallback loop
    and only on a request that actually fell back.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str | None = None
    session: str | None = None
    user_id: str | None = None
    alias: str | None = None
    provider: str | None = None
    data_class: str | None = None
    cache: str | None = None
    cost_usd: float | None = None
    redacted: bool | None = None
    fallback_count: int | None = None

    def as_attributes(self) -> dict[str, str | float | bool | int]:
        """The set fields only, as the primitive types OpenTelemetry accepts.

        `session` goes out as `langfuse.session.id`, the key Langfuse groups its
        Sessions page by; the field keeps its name so no call site changes.
        Pinned by `test_the_session_goes_out_as_langfuse_session_id`. `user_id`
        goes out as `langfuse.user.id` for its Users page, pinned by
        `test_the_user_id_goes_out_as_langfuse_user_id`.
        """
        return {
            WIRE_KEYS.get(key, key): value
            for key, value in self.model_dump().items()
            if value is not None
        }


WIRE_KEYS: dict[str, str] = {
    "session": "langfuse.session.id",
    "user_id": "langfuse.user.id",
}
"""A `SpanAttributes` field whose key on the wire differs from its name."""


GENERATION_TYPE_KEY = "langfuse.observation.type"
"""The label that makes Langfuse show a span as a generation. Sent on purpose
rather than left to Langfuse to guess from the model label, which it does not do
reliably."""

GENERATION_KEYS: dict[str, str] = {
    "model": "gen_ai.request.model",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cost_usd": "gen_ai.usage.cost",
}
"""The standard `gen_ai.*` key each `GenerationAttributes` field goes out under."""

STEP_METADATA_PREFIX = "langfuse.observation.metadata."
"""Where a provider step shows the same model, tokens and cost: Langfuse shows
metadata but neither adds it to a total nor prices it again."""


class GenerationAttributes(BaseModel):
    """What the provider span may say beyond its name: model, tokens, cost.

    Closed for the same reason `SpanAttributes` is: a prompt or an answer on a
    generation would be text on a trace backend outside this system's redaction,
    so an unknown key is a `ValidationError`. The Input and Output are set by
    `Generation`, never through this class. Pinned by
    `test_the_generation_label_set_is_closed`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    def as_attributes(self) -> dict[str, str | float | bool | int]:
        """The type label, then the set fields under their `gen_ai.*` keys."""
        attributes: dict[str, str | float | bool | int] = {GENERATION_TYPE_KEY: "generation"}
        for field, value in self.model_dump().items():
            if value is not None:
                attributes[GENERATION_KEYS[field]] = value
        return attributes

    def as_metadata(self) -> dict[str, str | float | bool | int]:
        """The set fields as step metadata, with no type label and no `gen_ai.*` key."""
        return {
            STEP_METADATA_PREFIX + field: value
            for field, value in self.model_dump().items()
            if value is not None
        }


INPUT_NOTE_KEY = "langfuse.observation.input"
OUTPUT_NOTE_KEY = "langfuse.observation.output"
"""The labels Langfuse shows as a step's Input and Output. They carry a
`PreviewNote`, or, for a tenant with `trace_content` on, the redacted
conversation and the answer."""

FINISH_REASON_KEY = "langfuse.observation.metadata.finish_reason"
"""Where the finish reason goes when the Output is the answer itself."""

CONTENT_LIMIT = 8000
CUT_MARK = " [cut]"
"""Each message sent to Langfuse keeps at most `CONTENT_LIMIT` characters; a
longer one ends in `CUT_MARK` (`test_a_message_at_8001_characters_is_cut`)."""

TAGS_KEY = "langfuse.trace.tags"
"""The root span label Langfuse reads a trace's tags from."""

INPUT_NOT_SENT = "not sent: prompts stay inside the gateway"
OUTPUT_NOT_SENT = "not sent"

STEP_INPUT_DEFAULT = "not recorded"
STEP_OUTPUT_DEFAULT = "no decision recorded"
"""What a stage step (auth, route, cache, redact, semantic) says until it writes
its own decision with `note_step`. The root and the provider step keep the
`not sent` notes (`test_the_step_default_lines_are_the_published_ones`)."""

REFUSED_PREFIX = "refused: "
"""A stage whose body raises a typed error says this and the code, never the message
(`test_a_refused_auth_says_the_code`)."""

FINISH_REASON_PATTERN = re.compile(r"[a-z_]{1,32}")
"""What a finish reason must look like to be sent as given."""

OTHER_FINISH_REASON = "other"
"""What any other finish reason is sent as."""


class PreviewNote(BaseModel):
    """The short note Langfuse shows as a step's Input or Output, in place of the text.

    Left empty, Langfuse shows `undefined`, which reads as a bug. A step whose
    text is not sent says so, plus a few safe facts, as ONE plain line: the
    Formatted view shows a plain string whole, but only the first field of JSON
    (`test_the_input_note_is_one_plain_line`). Closed like
    `SpanAttributes`: a `prompt` or a `user` key is a `ValidationError`. Pinned by
    `test_the_preview_note_field_set_is_closed`.

    `finish_reason` is provider text, so anything that is not a short lower-case
    word becomes `other` (`test_a_bad_finish_reason_is_sent_as_other`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    alias: str | None = None
    messages: int | None = None
    finish_reason: str | None = None
    tokens_out: int | None = None

    @field_validator("finish_reason")
    @classmethod
    def _checked_finish_reason(cls, value: str | None) -> str | None:
        if value is None or FINISH_REASON_PATTERN.fullmatch(value):
            return value
        return OTHER_FINISH_REASON

    @classmethod
    def input(cls, *, alias: str | None = None, messages: int | None = None) -> PreviewNote:
        """An Input note: not sent, with what is known about the request."""
        return cls(text=INPUT_NOT_SENT, alias=alias, messages=messages)

    @classmethod
    def output(
        cls, *, finish_reason: str | None = None, tokens_out: int | None = None
    ) -> PreviewNote:
        """An Output note: not sent, with what is known about the answer."""
        return cls(text=OUTPUT_NOT_SENT, finish_reason=finish_reason, tokens_out=tokens_out)

    def as_text(self) -> str:
        """One line: the text, then the known facts in brackets; an unknown fact is left out."""
        facts: list[str] = []
        if self.alias is not None:
            facts.append(f"alias {self.alias}")
        if self.messages is not None:
            facts.append(f"{self.messages} message{'' if self.messages == 1 else 's'}")
        if self.finish_reason is not None:
            facts.append(f"finish reason {self.finish_reason}")
        if self.tokens_out is not None:
            facts.append(f"{self.tokens_out} tokens out")
        return f"{self.text} ({', '.join(facts)})" if facts else self.text


def cut(text: str) -> str:
    """The text as sent to Langfuse: whole up to `CONTENT_LIMIT`, else cut and marked."""
    return text if len(text) <= CONTENT_LIMIT else text[:CONTENT_LIMIT] + CUT_MARK


def conversation_json(messages: Sequence[tuple[str, str]]) -> str:
    """The Input of a step that sends content: a list of role and (cut) text."""
    return json.dumps([{"role": role, "content": cut(content)} for role, content in messages])


def answer_json(text: str) -> str:
    """The Output of a step that sends content: the answer as one assistant message."""
    return json.dumps({"role": "assistant", "content": cut(text)})


def plain_notes() -> dict[str, str]:
    """The Input and Output every step starts with."""
    return {
        INPUT_NOTE_KEY: PreviewNote.input().as_text(),
        OUTPUT_NOTE_KEY: PreviewNote.output().as_text(),
    }


def step_notes() -> dict[str, str]:
    """The Input and Output every stage step starts with."""
    return {INPUT_NOTE_KEY: STEP_INPUT_DEFAULT, OUTPUT_NOTE_KEY: STEP_OUTPUT_DEFAULT}


def note_step(span: Span | None, *, input: str | None = None, output: str | None = None) -> None:
    """Write a stage step's decision as its Input and Output, each one plain line.

    The callers pass only facts the gateway decided (a tenant auth accepted, an
    alias that resolved, counts), never text a caller sent.
    """
    if span is None or not span.is_recording():
        return
    if input is not None:
        span.set_attribute(INPUT_NOTE_KEY, input)
    if output is not None:
        span.set_attribute(OUTPUT_NOTE_KEY, output)


def score_text(score: float) -> str:
    """A similarity score with 3 decimals, rounded DOWN, so a miss never reads as reaching
    the threshold (`test_a_score_just_under_the_threshold_rounds_down`)."""
    return str(Decimal(repr(score)).quantize(Decimal("0.001"), rounding=ROUND_DOWN))


def counted(count: int, unit: str) -> str:
    """`1 message`, `3 messages`: a count and its unit, plural past one."""
    return f"{count} {unit}{'' if count == 1 else 's'}"


class Generation:
    """One provider call's span, opened by hand so it can outlive a `with` block.

    A plain call's cost is known only after `book_cost`, and a stream's usage only
    after its last frame, so the span cannot close where the call returns. The
    caller reports what it learns and ends it once; a second `end` is ignored, so
    an error path and a finished path can both end it safely. Holds `None` when
    the request is not traced, so call sites stay unconditional.

    `root` is the request's root span, which carries the Output note of the call
    that answered (`test_the_root_step_carries_the_provider_notes`).

    The root, not this step, is the request's one Langfuse generation:
    `report` puts the model, tokens and cost on the root under `gen_ai.*`, and on
    this step only as metadata, so Langfuse counts the cost once. Each attempt
    writes over the root, so the call that answered is the one it shows
    (`test_exactly_one_span_per_request_carries_a_cost`,
    `test_a_fallback_counts_the_cost_once`).

    With `content` on, the Output is the answer rather than a note, and the
    finish reason goes as metadata instead. A stream hands its text
    pieces to `add_text`, and `end` writes them joined
    (`test_a_streamed_answer_is_gathered_into_the_output`).
    """

    def __init__(
        self, span: Span | None, root: Span | None = None, *, content: bool = False
    ) -> None:
        self._span = span
        self._root = root
        self._content = content
        self._output = PreviewNote.output()
        self._pieces: list[str] | None = None

    def _set(self, key: str, value: str) -> None:
        """Set one label on this step and on the root."""
        if self._span is None or not self._span.is_recording():
            return
        self._span.set_attribute(key, value)
        if self._root is not None and self._root.is_recording():
            self._root.set_attribute(key, value)

    def report(self, attributes: GenerationAttributes) -> None:
        """Add what is now known: as metadata on this step, as the generation on the root."""
        if self._span is None or not self._span.is_recording():
            return
        self._span.set_attributes(attributes.as_metadata())
        if self._root is not None and self._root.is_recording():
            self._root.set_attributes(attributes.as_attributes())

    def note_output(
        self, *, finish_reason: str | None = None, tokens_out: int | None = None
    ) -> None:
        """Add a fact about the answer to the Output note, here and on the root.

        A stream learns the finish reason and the tokens from two events, so each
        call keeps what an earlier one set. With content on, only the finish
        reason is sent, as metadata.
        """
        self._output = PreviewNote.output(
            finish_reason=finish_reason
            if finish_reason is not None
            else self._output.finish_reason,
            tokens_out=tokens_out if tokens_out is not None else self._output.tokens_out,
        )
        if not self._content:
            self._set(OUTPUT_NOTE_KEY, self._output.as_text())
        elif self._output.finish_reason is not None:
            self._set(FINISH_REASON_KEY, self._output.finish_reason)

    def note_input(self, conversation: Sequence[tuple[str, str]]) -> None:
        """Send the redacted conversation as the Input, here and on the root."""
        self._set(INPUT_NOTE_KEY, conversation_json(conversation))

    def note_answer(self, text: str) -> None:
        """Send the answer as the Output, when content is on."""
        if self._content:
            self._set(OUTPUT_NOTE_KEY, answer_json(text))

    def add_text(self, piece: str) -> None:
        """Keep one streamed text piece, for `end` to send joined."""
        if self._pieces is None:
            self._pieces = []
        self._pieces.append(piece)

    def end(self) -> None:
        """Close the span; later calls do nothing."""
        if self._pieces is not None:
            pieces, self._pieces = self._pieces, None
            self.note_answer("".join(pieces))
        if self._span is not None:
            span, self._span = self._span, None
            span.end()


def start_generation(
    scope: Scope,
    name: str,
    *,
    model: str,
    alias: str,
    messages: int | None = None,
    conversation: Sequence[tuple[str, str]] | None = None,
) -> Generation:
    """Open a provider step, and label the root a generation of this model now.

    Parented on the root span from the scope, like `stage_span`, and with the
    same exception defaults off, so no provider message reaches the span. Its
    Input note names the alias and, for a chat, how many messages went out.

    `conversation` is the redacted role and text of each message, given only
    when the tenant's `trace_content` is on: it then becomes the Input here and
    on the root (`test_the_root_input_is_the_redacted_conversation`).
    """
    record = request_trace(scope)
    if record is None:
        return Generation(None)
    span = tracer().start_span(
        name, context=record.context, record_exception=False, set_status_on_exception=False
    )
    span.set_attributes(plain_notes())
    generation = Generation(span, record.span, content=conversation is not None)
    if conversation is None:
        span.set_attribute(
            INPUT_NOTE_KEY, PreviewNote.input(alias=alias, messages=messages).as_text()
        )
    else:
        generation.note_input(conversation)
    generation.report(GenerationAttributes(model=model))
    return generation


@dataclass
class RequestTrace:
    """One request's root span, the context to parent stage spans on, and its failure code.

    `error_code` is written by the error handlers, the only
    place that knows which `ErrorCode` a failure carried; the middleware reads it
    when it closes the root span. The code travels as a string rather than as the
    exception itself, because an exception on a span brings its message with it
    and a provider's message is exactly what must not reach a trace backend.
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
    root would claim a local parent no exporter ever sent.
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
    span = tracer().start_span(ROOT_SPAN_NAME, context=parent, attributes=plain_notes())
    return RequestTrace(span=span, context=set_span_in_context(span))


UNMATCHED_ROUTE = "unmatched"
"""What a trace is named after when no route matched, in place of the path."""

UNTRACED_PATHS = frozenset({"/metrics", "/healthz", "/readyz"})
"""Scrapes and probes: answered, never traced. Thousands a day would bury the
real requests and use up a trace backend's free plan."""


def root_span_name(scope: Scope) -> str:
    """The method and the matched route's template, e.g. `POST /v1/chat/completions`.

    The TEMPLATE, never the raw path: a path carries ids and whatever junk a
    caller sends, and a trace name is shown and grouped on. Pinned by
    `test_the_trace_is_named_from_the_matched_route` and
    `test_an_unmatched_path_is_named_unmatched`.
    """
    route = scope.get("route")
    template = getattr(route, "path", None)
    return (
        f"{scope.get('method', 'GET')} {template if isinstance(template, str) else UNMATCHED_ROUTE}"
    )


def end_root_span(record: RequestTrace, *, failed: bool, name: str | None = None) -> None:
    """Close the root span, recording the typed code when the request failed.

    `name` replaces the start name, because the route is only known once routing
    has run.

    The status DESCRIPTION carries the code and nothing else. An operator filters
    a trace store on "which requests failed and why" and gets the same closed
    `ErrorCode` set the wire contract uses, with no message, no provider payload
    and no exception text alongside it.
    """
    if name is not None:
        record.span.update_name(name)
    if failed:
        record.span.set_status(Status(StatusCode.ERROR, record.error_code))
    record.span.end()


def record_error_code(scope: Scope, code: str | None) -> None:
    """Note the typed code a failure carried, for the root span to report."""
    record = request_trace(scope)
    if record is not None:
        record.error_code = code


def annotate(scope: Scope, attributes: SpanAttributes) -> None:
    """Put allowlisted labels on this request's root span.

    A tenant also goes out as the trace tag `tenant:<tenant>`, so Langfuse can
    filter on it; it is the tenant auth already checked
    (`test_the_tenant_is_a_trace_tag`).
    """
    record = request_trace(scope)
    if record is not None:
        record.span.set_attributes(attributes.as_attributes())
        if attributes.tenant is not None:
            record.span.set_attribute(TAGS_KEY, (f"tenant:{attributes.tenant}",))


def note_request(scope: Scope, *, alias: str, messages: int | None = None) -> None:
    """Put the request's Input note on the root span: the alias and the message count."""
    record = request_trace(scope)
    if record is not None:
        record.span.set_attribute(
            INPUT_NOTE_KEY, PreviewNote.input(alias=alias, messages=messages).as_text()
        )


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
    without the middleware still serves, it is just not traced.

    The parent comes from the scope rather than from the ambient context on
    purpose - see this module's docstring on the worker-thread hop.

    Both of OpenTelemetry's exception defaults are turned OFF, and that is the
    security decision in this function rather than a tuning choice. Left on, a
    stage span whose body raises records an exception EVENT carrying `str(exc)`
    and sets a status description built from it - so a provider failure would
    write the provider's own message there, which the wire contract already
    refuses to disclose. What a failure is
    allowed to say is the typed code, and the root span says it once.
    """
    record = request_trace(scope)
    if record is None:
        yield None
        return
    with tracer().start_as_current_span(
        name, context=record.context, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attributes(step_notes())
        try:
            yield span
        except GatewayError as exc:
            code = getattr(exc, "code", None)
            if code is not None:
                span.set_attribute(OUTPUT_NOTE_KEY, f"{REFUSED_PREFIX}{code.value}")
            raise


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

    No endpoint means no exporter at all, not a fallback one.
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
    outage.
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
