"""`POST /v1/chat/completions`: the OpenAI wire shape, and the join it makes.

An earlier change left three things that did not touch each other: a request could be
authenticated, resolved to a provider and a model id, and a provider could be
called. This module is that join, plus the two things that have to happen around
the outbound call: inline redaction before it, and the two response cache tiers
that can replace it entirely - the exact one before redaction and the semantic
one after it. No ledger.

The OpenAI wire shape lives here and stops here. `providers/base.py` keeps the
gateway's own vocabulary, so the two mappings - a wire request into a
`ChatRequest`, a `ChatResponse` back into a `chat.completion` - are one file
rather than a shape spread through the package. That is why an OpenAI tool call
is `{"type": "function", "function": {...}}` in this module and a flat `ToolCall`
everywhere behind it.

Two published details worth knowing before reading the code:

* The body names the ALIAS the caller sent, never the provider model id. Aliases
  are the only routable names (the design), so an answer naming a provider model
  would hand back a name that cannot be sent again and disclose which provider
  served the request. The provider model id is an operator's fact and travels in
  `X-Gateway-Model` (`test_response_model_is_the_alias_the_caller_named`).
* `stream: true` answers `text/event-stream`: the same resolution and the same
  provider call, rendered as `chat.completion.chunk` frames and closed with
  `data: [DONE]`. The stream is OPENED and its first event pulled inside the
  route, before anything is returned, so a failure with nothing yet delivered is
  still a typed HTTP error rather than a 200 carrying an error frame
  (`test_a_provider_failure_before_the_first_event_is_502_with_no_stream`). Once
  the status is sent it cannot be changed, so a later failure is one error frame
  and no sentinel.
* When the tenant's rule for the request's data class carries `redact: true`,
  every message `content` is redacted before the wire request becomes a
  `ChatRequest`, so the adapter can only ever be handed placeholders
  (`test_the_chat_adapter_receives_only_what_the_class_allows`). Tool call
  ARGUMENTS are left alone: they are a machine payload the caller's own tool
  parses, and a placeholder inside that JSON is a value the tool cannot act on.

Both shapes are built here, from one set of wire models: a `chat.completion` and
the `chat.completion.chunk` frames are the same vocabulary rendered twice, so a
tool call cannot be OpenAI-shaped on one path and flat on the other.

Every rejection the wire models make is a 422 `VALIDATION` raised before any
outbound call, by construction: FastAPI validates the body before the route body
runs. The routing and provider failures - 404, 451, 503, 502 - are raised inside
the route by `resolve()` and the adapter, and reach the client through the
handlers in `gateway/errors.py`.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from decimal import Decimal
from itertools import chain
from typing import Any, Literal, assert_never, cast

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from starlette.datastructures import State
from starlette.types import Scope

from app.cache import (
    CACHE_ALLOW,
    CACHE_HEADER,
    ResponseCache,
    SemanticCache,
    cache_key,
    is_cacheable,
    probe_text,
    semantic_eligible,
    system_hash,
    tools_hash,
)
from app.core.errors import BudgetExceeded, GatewayError, ProviderError, RateLimited
from app.core.metrics import (
    EXACT_CACHE_KIND,
    SEMANTIC_CACHE_KIND,
    label_resolution,
    record_budget_ratio,
    record_cache_hit,
    record_circuit_state,
    record_cost_usd,
    record_fallback,
    record_redactions,
    record_tokens,
)
from app.core.settings import Settings
from app.core.tracing import SpanAttributes, annotate, stage_span
from app.gateway.context import Context, RequestContext
from app.gateway.errors import error_body
from app.gateway.routes.redact import (
    REDACT_SPAN,
    redact_or_reject,
    redaction_applies,
    tenant_directory,
)
from app.ledger import (
    CostLedgerRow,
    CostLedgerWriter,
    ResidencyLogWriter,
    record_cost,
    record_residency,
)
from app.limits.budget import (
    BUDGET_OVERRIDE_HEADER,
    BudgetDecision,
    BudgetStore,
    budget_ratio,
    current_month,
    decide,
    to_micro_usd,
    verify_override,
    warn_budget,
)
from app.limits.cost import COST_QUANTUM
from app.limits.prices import PriceStore
from app.limits.ratelimit import (
    SECONDS_PER_MINUTE,
    RateDimension,
    RateLimitStore,
)
from app.providers.base import (
    Adapter,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingAdapter,
    EmbeddingRequest,
    FinishReason,
    ResponseFormat,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolSpec,
    Usage,
    UsageTotal,
)
from app.providers.breaker import CircuitBreaker
from app.redaction import (
    MAP_ID_PATTERN,
    DirectoryStore,
    RedactionMapStore,
    RedactionMapStoreUnavailable,
    StreamRehydrator,
    rehydrate,
)
from app.routing.errors import AllProvidersUnavailable, IdentifierInvalid
from app.routing.policy import DataClass, Policy, PolicyStore, ResidencyMode
from app.routing.resolver import Resolved, resolve

COMPLETION_ID_PREFIX = "chatcmpl-"
"""What every completion id starts with, as OpenAI publishes it. Pinned by
`test_successful_completion_is_openai_shaped`."""

COMPLETION_ID_BYTES = 12
"""Random bytes behind each completion id, rendered as 24 hex characters.

The value is pinned exactly by `test_completion_id_is_random_and_the_pinned_length`,
which also asserts that two completions do not share an id. Both matter to a
component that has not been written yet: the ledger and the trace store key rows
by this id, so a counter, a shortened id or a per-process constant would merge
two tenants' calls into one row."""

COMPLETION_OBJECT: Literal["chat.completion"] = "chat.completion"
"""The `object` discriminator an OpenAI-compatible client branches on."""

CHUNK_OBJECT: Literal["chat.completion.chunk"] = "chat.completion.chunk"
"""The same discriminator for a streamed frame. A client reads the answer out of
`choices[].delta` rather than `choices[].message` when it sees this
(`test_every_chunk_carries_the_alias_and_one_id`)."""

SSE_MEDIA_TYPE = "text/event-stream"
"""The content type of a streamed answer."""

DONE_SENTINEL = "[DONE]"
"""The payload of the frame that ends a complete stream, spelled as OpenAI
publishes it. An SDK reads until it arrives, so its ABSENCE is what reports a
truncated stream; `test_a_stream_ends_with_done` pins the
whole final line and `test_a_failure_after_the_first_event_emits_an_error_chunk_and_no_done`
pins the absence."""

SSE_HEADERS: Mapping[str, str] = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
"""The two headers a streamed answer needs beyond the six of #9.

Both exist to stop something between the gateway and the client holding the
frames back and delivering them as one block, which turns a stream into a slow
non-streaming answer: `Cache-Control` for caches, `X-Accel-Buffering` for nginx
and the ingress controllers that read it. Pinned by
`test_headers_on_a_streamed_success`."""

FINISH_REASON_BY_STOP_REASON: Mapping[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}
"""Provider stop reasons in OpenAI's words, the whole table
(`test_finish_reason_table`).

A reason not in the table passes through unchanged rather than collapsing into
`stop`: a client branches on this field to decide whether to run a tool or to
report a truncated answer, and silently calling an unmapped reason a clean finish
is the one wrong answer that looks right.
"""

CACHE_STATUS_MISS = "miss"
CACHE_STATUS_HIT = "hit"
CACHE_STATUS_SEMANTIC = "semantic"
"""The three words `X-Gateway-Cache` answers with, and the three values the
`cache` label of `gateway_requests_total` carries.

`hit` means no adapter was called: the answer came out of the exact-match cache
under the design's key, and the caller was charged nothing for it. `semantic` means
no CHAT adapter was called: the answer was stored against a question merely
SIMILAR to this one (#23), which is a different consistency regime and therefore
a different word rather than a second spelling of `hit` -
a caller is entitled to know which one it got, and a hit rate per tier is then
one query. `miss` covers both a request that was looked up and not found and one
that was never eligible to be looked up at all - a caller cannot tell those
apart, and there is nothing it would do differently if it could.

Which one a response carries is asserted against the ADAPTER's call count rather
than against itself (`test_a_second_identical_eligible_request_calls_no_adapter`,
`test_a_differently_worded_second_request_is_served_from_the_semantic_tier`),
because a route could serve a fresh completion and label it `hit`.

The same header name carries the caller's opt-in on the way IN, where the only
accepted value is `CACHE_ALLOW` (13a decision D). That opt-in is the EXACT tier's
alone: the semantic tier opts in per ALIAS, in `routing.yaml`, and reads no
header at all."""

REDACTED_TRUE = "true"
REDACTED_FALSE = "false"
"""The two words `X-Gateway-Redacted` is written with. Lowercase, because a
console sums this string and `True` would count as neither
(`test_the_two_header_values_are_the_published_words`).

Which one a response carries is decided per REQUEST, not per route: the header is
`true` exactly when at least one value was actually replaced (an earlier change decision
C), so a redact class carrying no personal data answers `false`."""


EMBED_ALIAS = "embed"
"""The alias a semantic probe is embedded through: the design's own word, a constant
and not a setting.

Settable, it would be a way to point the tier at a model whose vectors no stored
entry shares - every search silently returning nothing, with no failure anywhere
to see. It is resolved per request like any other alias, so residency, health and
the tenant's data class apply to the embedding exactly as they apply to a
completion (`test_the_semantic_words_are_the_published_ones`,
`test_an_embed_alias_refused_by_residency_is_a_miss`)."""


COST_HEADER = "X-Gateway-Cost-USD"
"""The seventh `X-Gateway-*` header: what this request cost, in USD.

Its value is the ledger row's own `cost_usd`, rendered at exactly six decimal
places - the quantum `app.limits.cost` prices at and the scale
`cost_ledger.cost_usd` stores - so the header, the row and
`gateway_cost_usd_total` are one number computed once (an earlier change decision G,
`test_the_cost_header_is_the_booked_cost`). A float would lose the sixth place on
the way out and make a client's total disagree with the invoice.

Added by the ROUTE rather than by `gateway_headers`, and that is what keeps it
off a stream: the six headers are on the wire before the first frame, and the
usage a stream is priced from does not exist yet at that point
(`test_the_cost_header_is_absent_on_a_stream`). A cache hit does carry it, reading
`0.000000`, because zero is what a hit cost.
"""


REHYDRATE_HEADER = "X-Gateway-Rehydrate"
"""The request header a caller puts a `redaction_map_id` in to ask for its values back.

A request header rather than a body field (decision A): the body is the OpenAI
wire shape with `extra="forbid"`, so a gateway control there would be a field
every OpenAI-shaped client had to be taught, sitting next to `model` as if the
provider understood it. Read on this route and nowhere else (decision F) - it
means nothing on `/v1/redact` or on embeddings, and parsing it in the shared
`context` dependency would 422 requests that never rehydrate.

Its ABSENCE is what "off by default" means: there is no flag and no value that
means off; `tests/gateway/test_rehydration.py` covers that in
`test_no_header_answers_exactly_what_it_answers_today`, and the spelling is
pinned by every case in that module, which sends this exact name."""


class _Wire(BaseModel):
    """A model on the published API boundary: frozen, and unknown fields rejected.

    Rejected rather than ignored: a misspelled `max_token`
    becomes a 422 naming the field instead of a silently applied default
    (`test_unknown_body_field_is_422`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class FunctionSpec(_Wire):
    """The body of an OpenAI tool definition."""

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = {}


class WireTool(_Wire):
    """One tool offered to the model, in OpenAI's `{"type": "function", ...}` wrapper."""

    type: Literal["function"]
    function: FunctionSpec


class NamedFunction(_Wire):
    """The function named by an object-form `tool_choice`."""

    name: str


class NamedToolChoice(_Wire):
    """`tool_choice: {"type": "function", "function": {"name": ...}}`."""

    type: Literal["function"]
    function: NamedFunction


class WireToolCallFunction(_Wire):
    """The call the model made: a name, and its arguments as a JSON string."""

    name: str
    arguments: str


class WireToolCall(_Wire):
    """One tool call, in both directions: replayed by the caller and returned by us."""

    id: str
    type: Literal["function"] = "function"
    function: WireToolCallFunction


class WireMessage(_Wire):
    """One turn as an OpenAI client sends it, with the coherence rules enforced.

    The rules exist because the alternative is letting the provider judge the
    conversation: a tool result with no id, or a tool call hung off a user turn,
    would come back as a provider 400 the caller cannot act on instead of a 422
    naming what is wrong (`test_incoherent_message_is_422`).
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[WireToolCall, ...] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> WireMessage:
        """Reject the role and field combinations no provider can be told."""
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("a tool turn needs the tool_call_id it answers")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id belongs only on a tool turn")

        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls belong only on an assistant turn")

        if not self.content and not self.tool_calls:
            # Empty, not merely absent: Anthropic rejects an empty text block, so
            # `{"role": "user", "content": ""}` would leave as a 502 carrying the
            # provider's refusal of what is plainly the caller's own mistake
            # (`test_incoherent_message_is_422`, the empty-content rows).
            raise ValueError("content is required unless the turn carries tool_calls")
        return self


class StreamOptions(_Wire):
    """`stream_options: {"include_usage": bool}`, accepted so an ordinary SDK works.

    The OpenAI SDK sends this field on every streamed request, so refusing it
    outright through `extra="forbid"` would break a client that did nothing
    wrong. `false` is refused instead, with a 422 that says why: usage is
    returned on every stream (the design) because the ledger prices from it, so
    honouring `false` would be a promise the gateway does not keep (an earlier change
    decision D). Both halves are pinned - `test_include_usage_false_is_422` and
    `test_include_usage_true_is_accepted`.
    """

    include_usage: bool = True


class JsonSchemaSpec(_Wire):
    """The `json_schema` body of a `response_format`.

    `schema` is a Python attribute name Pydantic reserves, so the field is
    `json_schema` here and aliased back to `schema` on the wire.
    """

    name: str
    json_schema: Mapping[str, Any] = Field(alias="schema")
    strict: StrictBool | None = None


class WireResponseFormat(_Wire):
    """`response_format: {"type": "json_schema", "json_schema": {...}}`."""

    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


class ChatCompletionRequest(_Wire):
    """One chat completion request, exactly as an OpenAI-compatible client sends it.

    `model` is an alias, not a provider model id; the resolver checks its shape
    before any lookup, so a malformed one is a 422 rather than a 404 claiming the
    name does not exist (`test_a_malformed_alias_is_422_not_404`).
    """

    model: str
    messages: tuple[WireMessage, ...] = Field(min_length=1)
    tools: tuple[WireTool, ...] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    response_format: WireResponseFormat | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def _supported(self) -> ChatCompletionRequest:
        """Refuse the two requests this gateway cannot serve honestly."""
        if self.stream_options is not None and not self.stream_options.include_usage:
            raise ValueError("stream_options.include_usage: usage is reported on every stream")
        if self.tool_choice is not None and self.response_format is not None:
            # Each demands a different forced call, so whichever is applied last
            # silently wins and the caller gets an answer shaped by the field it
            # did not think was in charge.
            raise ValueError("tool_choice and response_format cannot be combined")
        return self


class CompletionMessage(BaseModel):
    """The assistant turn we answer with. `content` is null when the answer was a call."""

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: tuple[WireToolCall, ...] | None = None


class Choice(BaseModel):
    """The single choice this gateway returns. `n > 1` is not offered."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    message: CompletionMessage
    finish_reason: str


class CompletionUsage(BaseModel):
    """What the call consumed, carried through from the adapter and never zero-filled.

    The ledger prices from these three numbers, so a zero books a real, billed
    call as free (`test_usage_is_carried_through_unchanged`).
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """The `chat.completion` envelope. `model` is the alias the caller named."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion"] = COMPLETION_OBJECT
    created: int
    model: str
    choices: tuple[Choice, ...]
    usage: CompletionUsage


class WireToolCallChunk(_Wire):
    """One tool call inside a streamed delta: `WireToolCall` plus the position.

    `index` is what an OpenAI client assembles a multi-call answer by, and it is
    the provider's own position rather than this frame's: hardcoding 0 merges two
    calls of one answer into one (`test_streamed_tool_call_is_one_whole_delta`).
    """

    index: int
    id: str
    type: Literal["function"] = "function"
    function: WireToolCallFunction


class ChunkDelta(BaseModel):
    """What one frame adds to the message being assembled. Every field optional.

    Optional because a frame carries exactly one kind of news: the opening role,
    a fragment of text, a completed tool call, or - on the frame that closes the
    answer - nothing at all beside the `finish_reason` above it. `role` rides the
    first frame only; repeating it is read as a second message
    (`test_first_chunk_carries_the_assistant_role`).
    """

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: tuple[WireToolCallChunk, ...] | None = None


class ChunkChoice(BaseModel):
    """The single choice a frame carries. `n > 1` is not offered, as on the completion."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    delta: ChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """One `chat.completion.chunk` frame. `model` is the alias, as on the completion.

    `usage` is null on every frame but the last, which carries it with an EMPTY
    `choices`: that is where an OpenAI-compatible client looks for it, and it is
    the number the ledger will price from, so it is carried from the provider and
    never zero-filled
    (`test_usage_chunk_is_last_and_carries_the_fixture_numbers_exactly`).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion.chunk"] = CHUNK_OBJECT
    created: int
    model: str
    choices: tuple[ChunkChoice, ...]
    usage: CompletionUsage | None = None


ROUTE_SPAN = "route"
"""The stage span around `resolve()`. Shared with the embeddings route, which
imports it from here for the same reason it imports `gateway_headers`: one name
in one place, so two routes cannot disagree about what a stage is called. Pinned
by `test_the_emitted_span_names_are_exactly_the_stages_that_exist`."""

CACHE_SPAN = "cache"
"""The stage span around the exact-match lookup, on this route only.

Emitted on every chat request, hit or miss, by
`test_the_cache_span_is_emitted_on_a_miss_and_on_a_hit`:
a span present only on hits would make every trace that carries one a hit by
construction, and an operator could not see how long a lookup that found nothing
took. It wraps the LOOKUP and not the write, because the write happens after the
answer is on its way and its latency is not the caller's."""

SEMANTIC_SPAN = "semantic"
"""The stage span around the embedding call AND the vector search, on this route
only, and only on a request the semantic tier is eligible for.

Its own stage rather than part of `CACHE_SPAN`: the
embedding is a network call to a provider, and hidden inside the cache stage a
slow embedder reads as a slow cache, which is the wrong thing to page on.

Emitted on an eligible HIT and an eligible MISS alike, and on no ineligible
request - so its mere presence in a trace says the tier ran, not that it found
something (`test_the_semantic_span_is_emitted_on_an_eligible_miss_and_on_an_eligible_hit`,
`test_no_semantic_span_is_emitted_on_an_ineligible_request`). It wraps the
lookup and not the write, exactly as `CACHE_SPAN` does, because the write
happens after the answer is on its way."""

PROVIDER_SPAN_PREFIX = "provider:"
"""What a provider span's name starts with. The suffix is the RESOLVED provider,
never the alias the caller sent: an alias names a policy entry and a provider
names what actually served the request, which is the one question this span
exists to answer (`test_the_provider_span_names_the_resolved_provider`)."""


def provider_span_name(provider: str) -> str:
    """The stage span name for one resolved provider."""
    return f"{PROVIDER_SPAN_PREFIX}{provider}"


def health_seam(
    breaker: CircuitBreaker, has_adapter: Callable[[str], bool]
) -> Callable[[str], bool]:
    """The `is_healthy` the routes hand `resolve()`: an adapter exists AND it is allowed.

    Both routes build it here rather than each writing the conjunction, because a
    route that asked only its own half would be the bug this seam exists to
    prevent: an embeddings route still dialling an open provider while the chat
    route respected it (`test_the_embeddings_route_skips_a_provider_whose_breaker_is_open`).
    `has_adapter` is the half that differs - "an adapter exists" for a completion,
    "an EMBEDDING adapter exists" for an embedding.

    "May we dial this" is asked as `breaker.allows`, never as a state comparison
    written out here, so there is exactly one definition of it and a future
    refinement of `allows` reaches the routes rather than only the breaker's own
    tests.

    A breaker consulted HERE, and not inside the fallback loop, is what makes a
    request whose every candidate is open the resolver's 503 with no provider
    dialled at all
    (`test_every_breaker_open_is_the_resolvers_503_with_no_provider_called`).

    Publishing the gauge from here is what gives every provider in rotation a
    series rather than only the ones that have broken
    (`test_the_circuit_gauge_reads_open_for_the_failed_provider_and_closed_for_the_other`).
    The seam is called once per permitted candidate by `resolve()`, so the cost is
    one gauge set per candidate and the label is a policy provider name.
    """

    def healthy(provider: str) -> bool:
        record_circuit_state(provider=provider, state=int(breaker.state(provider)))
        return has_adapter(provider) and breaker.allows(provider)

    return healthy


def absorb_failure(breaker: CircuitBreaker, provider: str, error: ProviderError) -> None:
    """Book one candidate's failure against its breaker, for a RETRYABLE failure only.

    A 4xx and an unreadable body are not evidence about the provider - they are
    evidence about the request - so counting them would take a healthy provider out
    of rotation over a caller's own mistake
    (`test_a_non_retryable_failure_stops_at_the_first_candidate`). The caller
    re-raises a non-retryable one; this only decides what the breaker learns.
    """
    if not error.retryable:
        return
    breaker.record_failure(provider)
    record_circuit_state(provider=provider, state=int(breaker.state(provider)))


def record_served(
    scope: Scope,
    breaker: CircuitBreaker,
    *,
    tenant: str,
    alias: str,
    chosen: Resolved,
    failed: int,
    policy: Policy,
    data_class: DataClass,
    log: ResidencyLogWriter,
) -> None:
    """Close the loop on the candidate that served: the breaker, the counter, the span, the log.

    On a STREAM this runs when the first event arrives, not when the last one does,
    because that is where the route hands the response back. The consequence is
    narrow and worth naming: a half-open provider that opens a stream and then dies
    mid-answer has its trial recorded as a success and its breaker closed. Decision
    C is what makes that the honest reading - a failure after the first frame is
    not a fallback and not this loop's to absorb - and a closed breaker costs one
    more request before the next five failures re-open it.

    `failed` is how many candidates were tried and failed before this one, so it is
    the fallback count the design names and zero on an ordinary request. Both the
    counter and the attribute move once per REQUEST rather than once per attempt
    (`test_a_fallback_is_counted_once_against_the_serving_provider`,
    `test_a_fallback_is_recorded_on_the_trace_as_a_count`), and neither is touched
    at all when nothing fell back
    (`test_a_request_that_needed_no_fallback_counts_none`).

    The residency row is written here, and here is the ONLY place it is written,
    which is what makes "one row per served request" true of all three paths -
    both chat paths and `/v1/embeddings` reach the log through this function and
    through nothing else. It names `chosen`, so a request that fell back records
    the candidate that SERVED
    (`test_a_chat_request_that_fell_back_logs_the_serving_provider`). On a stream
    that is still before the first frame is on the wire, so a stream that dies
    mid-answer has already recorded where its data went. The write is guarded
    inside `record_residency`: it cannot change a status
    code, a body or a header on any of the three paths
    (`test_a_failing_writer_changes_no_status_body_or_header`).
    """
    breaker.record_success(chosen.provider)
    record_circuit_state(provider=chosen.provider, state=int(breaker.state(chosen.provider)))
    record_residency(
        log, policy=policy, tenant=tenant, data_class=data_class, alias=alias, chosen=chosen
    )
    if failed == 0:
        return
    record_fallback(tenant=tenant, alias=alias, provider=chosen.provider)
    annotate(scope, SpanAttributes(fallback_count=failed))


CHECK_COST = 1
"""What one request costs each bucket at the check.

One from the REQUEST bucket is decision C: unlike its tokens, a request is
countable before it is served, so the check and the debit are one atomic take and
a tenant's published `rpm` is exactly the number of requests it gets. One from the
TOKEN bucket is a probe, and it is what makes "this tenant has spent its minute's
tokens" expressible at all - the real usage is not known until the provider has
answered (decision B), so the check asks whether a single token is left and the
debit follows from `book_cost`. One token in a `tpm` of 120,000 is what that probe
costs (`test_the_check_takes_one_from_each_bucket_at_the_tenants_own_ceiling`).
"""


def check_rate_limit(state: State, context: RequestContext) -> None:
    """Refuse this request if the tenant has spent its minute's requests or tokens (#28).

    Called at the top of BOTH routes and immediately BEFORE `check_budget`, after
    `Context` has authenticated the tenant. The position carries two criteria at
    once. A refusal must cost no provider call, no ledger row and no cache write,
    and a streamed request must be refused before the 200 and the first frames
    are on the wire, or the client gets an error inside a stream instead of a
    typed 429 with a `Retry-After` no started response could carry
    (`test_a_refused_request_dials_no_provider_and_books_no_row`,
    `test_a_streamed_request_over_rate_is_refused_before_the_stream_opens`). And
    being before the budget check is criterion 6: both refusals are 429s, so a
    tenant over both is told to slow down rather than to top up
    (`test_the_rate_check_runs_before_the_budget_check`).

    A tenant the POLICY does not carry has no ceiling and is not this function's
    to refuse. Its limit is read with `.get` and not `policy.tenants[tenant]`,
    which would raise `KeyError` and turn the resolver's typed 451 into an
    untyped 500; inventing a ceiling of zero for it would be worse, because the
    429 would then hide the residency refusal that is the correct answer
    (`test_a_tenant_the_policy_does_not_carry_is_still_the_resolvers_451`).

    A store that cannot answer serves the request (decision E, argued in
    `limits/ratelimit.py`): a `None` from either take refuses nobody, and the
    store has already logged at ERROR by the time it is read here.

    BOTH buckets are taken from, and the request bucket first, so a tenant over
    both spends one token of each rather than being refused on a bucket it never
    reached - which keeps the two ceilings independent of the order they are
    checked in.
    """
    store: PolicyStore = state.policy
    tenant_policy = store.policy.tenants.get(context.tenant)
    if tenant_policy is None:
        return
    limits: RateLimitStore = state.rate_limit_store
    limit = tenant_policy.rate_limit
    requests = limits.take(
        context.tenant,
        RateDimension.REQUESTS,
        cost=CHECK_COST,
        capacity=limit.rpm,
        rate=limit.rpm / SECONDS_PER_MINUTE,
    )
    tokens = limits.take(
        context.tenant,
        RateDimension.TOKENS,
        cost=CHECK_COST,
        capacity=limit.tpm,
        rate=limit.tpm / SECONDS_PER_MINUTE,
    )
    for taken, ceiling, unit in (
        (requests, limit.rpm, "requests"),
        (tokens, limit.tpm, "tokens"),
    ):
        if taken is None or taken.allowed:
            continue
        raise RateLimited(
            f"tenant {context.tenant!r} has reached its rate limit of {ceiling} {unit} per minute",
            retry_after_seconds=taken.retry_after_seconds,
        )


def debit_tokens(state: State, context: RequestContext, *, row: CostLedgerRow) -> None:
    """Take one served request's tokens out of its tenant's minute.

    Called from `book_cost` and from nowhere else, beside `debit_budget`, which
    is what makes criterion 2 structural rather than a list of call sites to keep
    in step: every path that books a ledger row debits the bucket, by the tokens
    of the row it just booked, because there is one place that does both
    (`test_every_book_cost_site_debits_the_token_bucket`).

    A CACHED ROW DEBITS ZERO, and this is the one branch here. A cached row keeps
    the replayed answer's token counts on purpose
    (`test_a_cached_row_keeps_the_replayed_answers_token_counts` in
    tests/ledger/test_cost_ledger.py), so reading `input_tokens + output_tokens`
    off it would charge a tenant's `tpm` again for every hit and make the cache
    actively expensive in the one currency it saves. Zero is still a DEBIT: the
    call happens, which is what separates a path that cost nothing from a path
    the wiring missed, and no bucket level can tell those apart
    (`test_a_cache_hit_debits_zero_tokens`).

    The answer is ignored, which is decision B: the request that crosses `tpm`
    completes and the NEXT one is refused, because a token count is not known
    until the provider has answered and reserving an estimate up front would
    refuse requests on a guess. Like everything else `book_cost` calls, this
    cannot change the answer - the store swallows every `redis.RedisError`, and
    the tenant was validated by `Context` before the key was built.
    """
    limits: RateLimitStore = state.rate_limit_store
    store: PolicyStore = state.policy
    tenant_policy = store.policy.tenants.get(context.tenant)
    if tenant_policy is None:
        return
    limit = tenant_policy.rate_limit
    spent = 0 if row.cached else row.input_tokens + row.output_tokens
    limits.take(
        context.tenant,
        RateDimension.TOKENS,
        cost=spent,
        capacity=limit.tpm,
        rate=limit.tpm / SECONDS_PER_MINUTE,
    )


def check_budget(state: State, context: RequestContext, *, override: str | None) -> None:
    """Refuse this request if the tenant has already spent its month, and warn it near.

    Called at the top of BOTH routes, after `Context` has authenticated the
    tenant and before anything is resolved, redacted, embedded or dialled (plan
    15a decision D). The position is the whole point: a refusal must cost no
    provider call, no ledger row and no cache write
    (`test_a_refused_request_dials_no_provider_and_books_no_row`), and a streamed
    request must be refused before the 200 and the first frames are on the wire,
    or the client gets an error inside a stream instead of a typed 429
    (`test_a_streamed_request_over_cap_is_refused_before_the_stream_opens`). A
    middleware could not do this job at all: it runs before the `Context`
    dependency, so it has no authenticated tenant to look a cap up for.

    A tenant the POLICY does not carry has no cap and is not this function's to
    refuse. Its cap is read with `.get` and not `policy.tenants[tenant]`, which
    would raise `KeyError` and turn the resolver's typed 451 into an untyped 500;
    inventing a cap of zero for it would be worse still, because the 429 would
    then hide the residency refusal that is the correct answer
    (`test_a_tenant_the_policy_does_not_carry_is_still_the_resolvers_451`).

    A counter that cannot answer serves the request (an earlier change decision B, argued
    in `limits/budget.py`): `spend` answering `None` returns here having warned
    nobody and refused nobody.

    The warning covers the whole band at or above 80%, an overridden request
    included, because that is what criterion 2 says - it is emitted before the
    override is even considered, so a tenant running on a standing override is
    still visible in the log every request.
    """
    store: PolicyStore = state.policy
    tenant_policy = store.policy.tenants.get(context.tenant)
    if tenant_policy is None:
        return
    budget_store: BudgetStore = state.budget_store
    month = current_month()
    spent = budget_store.spend(context.tenant, month)
    if spent is None:
        return
    cap = to_micro_usd(tenant_policy.budget_usd_month)
    decision = decide(spend_micro_usd=spent, cap_micro_usd=cap)
    if decision is BudgetDecision.SERVE:
        return
    warn_budget(tenant=context.tenant, spend_micro_usd=spent, cap_micro_usd=cap)
    if decision is not BudgetDecision.REJECT:
        return
    settings: Settings = state.settings
    if verify_override(
        override, tenant=context.tenant, month=month, secret=settings.budget_override_secret
    ):
        return
    raise BudgetExceeded(
        f"tenant {context.tenant!r} has reached its monthly budget of "
        f"{tenant_policy.budget_usd_month} USD"
    )


def debit_budget(state: State, context: RequestContext, *, usd: Decimal) -> None:
    """Add one served request's cost to its tenant's month, and republish the gauge.

    Called from `book_cost` and from nowhere else, which is what makes criterion
    5 structural rather than a list to keep in step: every path that books a
    ledger row debits the counter, by the cost of the row it just booked, because
    there is one place that does both
    (`test_every_book_cost_site_debits_the_budget`,
    `test_a_stream_that_reports_no_usage_still_debits_its_estimated_row`).

    ZERO IS A DEBIT. A cache hit books a row of `0.000000` and debits it, which
    both keeps the gauge fresh on a hit and makes a missing call site visible: a
    total cannot tell a path that added nothing from one that was never wired,
    and the two cache tiers and `/v1/embeddings` against `local` all cost zero
    (`test_a_cache_hit_debits_zero`).

    The gauge is set from what `debit` ANSWERED, not from a second read, so it
    includes the request in hand rather than lagging it by one. Nothing is
    published when the counter could not answer or when the tenant has no cap:
    the last known level stands rather than being overwritten with a confident
    0.0 for the length of a Redis incident (decision B,
    `test_an_unreachable_store_serves_and_moves_no_gauge`).

    Like everything else `book_cost` calls, this cannot change the answer. The
    store swallows every `redis.RedisError`, `record_budget_ratio` is guarded like
    every other emission, and the tenant was validated by `Context` before the
    key was built.
    """
    store: PolicyStore = state.policy
    budget_store: BudgetStore = state.budget_store
    total = budget_store.debit(context.tenant, current_month(), to_micro_usd(usd))
    tenant_policy = store.policy.tenants.get(context.tenant)
    if total is None or tenant_policy is None:
        return
    record_budget_ratio(
        tenant=context.tenant,
        ratio=budget_ratio(
            spend_micro_usd=total, cap_micro_usd=to_micro_usd(tenant_policy.budget_usd_month)
        ),
    )


def book_cost(
    state: State,
    context: RequestContext,
    *,
    alias: str,
    chosen: Resolved,
    usage: Usage | None,
    cached: bool,
    prompt_chars: int = 0,
    completion_chars: int = 0,
) -> str:
    """Price one served request, file its row, count it, debit its budget, answer the header.

    The four numbers a request reports its cost through - the row in
    `cost_ledger`, the `gateway_cost_usd_total` counter, the tenant's monthly
    budget counter and `X-Gateway-Cost-USD` - come from ONE call to
    `app.ledger.record_cost` here (an earlier change decisions A and G, an earlier change
    criterion 5). Four call sites pricing separately is how a dashboard comes to
    disagree with an invoice by a rounding, and how a cap comes to refuse traffic
    on a total nobody can reconcile with the ledger; only one of the four would
    ever be audited (`test_the_counter_moves_by_exactly_the_rows_cost`,
    `test_the_cost_header_is_the_booked_cost`,
    `test_the_debit_is_the_number_the_cost_header_reports`).

    Called once per REQUEST and never once per attempt, from the same closing
    stretch as `record_served`: `chosen` is the candidate that SERVED, so a
    request that fell back books one row against the provider that charged for it
    and nothing at all against the one that failed
    (`test_a_request_that_fell_back_books_one_row_against_the_serving_provider`).

    `price_key` comes from the SERVING adapter rather than being rebuilt from the
    provider name, so the key the price was read under and the key the adapter
    publishes cannot drift (an earlier change's own argument for taking it as a
    parameter). The lookup is the same `state.adapters[chosen.provider]` the route
    dialled: the resolver's health seam is what guarantees the entry exists, and a
    cache hit reads it without dialling anything.

    `cached` is keyword-only and has NO default, exactly as `gateway_headers`'
    `redacted` has none and for the same reason: a forgotten argument would book a
    replayed answer as money that was spent. What the flag then costs is decided
    inside `record_cost` (decision C), not here.

    `usage` is what the provider reported, and `None` is the streamed request that
    ended without a usage event - the two character counts are what its row is
    estimated from then (decision E).

    Nothing here can change the answer. `record_cost` swallows a failing write,
    `record_cost_usd` and `debit_budget` are guarded like every other emission,
    and the header is formatted from a `Decimal` this function already holds
    (`test_a_failing_writer_changes_no_status_body_or_header`). The budget the
    debit moves is read at the TOP of the request by `check_budget`, never here,
    so a request is refused on what was already spent when it arrived and not on
    a total this call has just changed.
    """
    ledger: CostLedgerWriter = state.cost_ledger
    prices: PriceStore = state.prices
    adapters: Mapping[str, Adapter] = state.adapters
    row = record_cost(
        ledger,
        table=prices.table,
        tenant=context.tenant,
        alias=alias,
        provider=chosen.provider,
        model=chosen.model,
        region=chosen.region,
        usage=usage,
        prompt_chars=prompt_chars,
        completion_chars=completion_chars,
        cached=cached,
        session=context.session,
        trace_id=context.trace_id,
        price_key=adapters[chosen.provider].price_key(chosen.model),
    )
    record_cost_usd(
        tenant=context.tenant, provider=chosen.provider, model=chosen.model, usd=row.cost_usd
    )
    debit_budget(state, context, usd=row.cost_usd)
    debit_tokens(state, context, row=row)
    return cost_header_value(row.cost_usd)


def cost_header_value(usd: Decimal) -> str:
    """One cost as `X-Gateway-Cost-USD` carries it: six decimal places, always.

    `Decimal` formatting rather than a float or a rounded string: every cost this
    gateway books is already quantized to `COST_QUANTUM`, and the quantize here is
    what makes the SIX places a property of the header rather than of whatever
    reached it (`test_the_cost_header_is_the_rows_decimal_at_six_places`). A
    `float` would render `0.000105` as an approximation and drop the last place on
    some values, which is exactly the disagreement between a client's total and
    the ledger this header exists to prevent.
    """
    return f"{usd.quantize(COST_QUANTUM):f}"


def label_candidate(
    scope: Scope, alias: str, chosen: Resolved, *, cache: str = CACHE_STATUS_MISS
) -> None:
    """Attribute this request to one candidate, on the span and on the counter.

    Called for `candidates[0]` as soon as `resolve()` answers, and again for each
    candidate the fallback loop moves to, so the labels a finished request carries
    name the provider that actually served it. Both writes overwrite rather than
    accumulate.

    The FIRST call has to happen before redaction, not inside the loop, and that is
    the whole reason this is a function. Redaction can refuse a request - a message
    over `MAX_REDACTION_INPUT_CHARS` is a 422 - and labelling only inside the loop
    would book every such refusal under an empty `alias`, `provider` and `region`,
    so the design's per-alias error panel would show nothing during a redaction-side
    incident (`test_a_refused_request_is_still_attributed_to_its_resolved_candidate`).

    `cache` defaults to `miss` because that is what every caller of this function
    is describing at the moment it calls: the outcome is not known until the
    lookup has run, and a request that never reaches one missed. The chat route
    calls it once more with `hit` after a lookup finds something, which
    overwrites rather than accumulates, exactly as the fallback loop's repeated
    calls do (`test_a_hit_counts_a_cache_hit_and_labels_the_request_hit`).
    """
    annotate(scope, resolution_attributes(alias, chosen))
    label_resolution(
        scope,
        alias=alias,
        provider=chosen.provider,
        region=chosen.region,
        cache=cache,
    )


def resolution_attributes(alias: str, resolved: Resolved) -> SpanAttributes:
    """What the resolver decided, as span labels.

    Set only AFTER `resolve()` returns, which is what keeps a refused request's
    alias off its trace: a 451 names the alias and the regions a tenant may not
    reach, and a residency refusal is not the place to write either down
    (`test_a_residency_refusal_records_the_code_on_the_span`).
    """
    return SpanAttributes(alias=alias, provider=resolved.provider, region=resolved.region)


def gateway_headers(
    resolved: Resolved, trace_id: str, *, redacted: bool, cache: str = CACHE_STATUS_MISS
) -> dict[str, str]:
    """The six `X-Gateway-*` headers describing one served request (#9).

    `redacted` is keyword-only and has no default on purpose. A default would let
    a route report "nothing was redacted" for a request it did redact simply by
    forgetting the argument, and that is the one value here a caller might act on
    (`test_the_redaction_outcome_is_keyword_only_and_required`).

    `cache` does have one, and the asymmetry is deliberate: `miss` is the true
    value for every path that has no cache to consult - the embeddings route,
    which imports this function, has none - so the default cannot state something
    false the way a `redacted` default could. Only the exact-match hit passes
    anything else, and it is asserted through the adapter's call count rather
    than through this string (`test_a_hit_still_carries_the_seven_gateway_headers`).

    Six, and only six, because six is what is KNOWN here. This runs before a
    provider is dialled, and `X-Gateway-Cost-USD` cannot be priced until the
    answer's usage exists - so the route adds it to the response afterwards
    (`COST_HEADER`), and a stream, whose headers are on the wire before its first
    frame, carries six for good. The seven a non-streamed answer ends up with are
    asserted whole by `test_the_header_set_is_exactly_seven`, the six of a stream
    by `test_headers_on_a_streamed_success`, and this builder's own six by
    `test_the_header_set_is_still_exactly_six`.

    Only the success path carries them. An error answers through the handlers in
    `gateway/errors.py`, which run outside the route and have no resolver result
    or context to describe - the seam an earlier change opens for tracing.
    """
    return {
        "X-Gateway-Provider": resolved.provider,
        "X-Gateway-Region": resolved.region,
        "X-Gateway-Model": resolved.model,
        "X-Gateway-Cache": cache,
        "X-Gateway-Redacted": REDACTED_TRUE if redacted else REDACTED_FALSE,
        "X-Gateway-Trace-Id": trace_id,
    }


def _neutral_messages(messages: tuple[WireMessage, ...]) -> tuple[ChatMessage, ...]:
    """Map the conversation into the neutral contract, in the caller's order.

    The order is the mapping's whole job here: a tool result means nothing except
    immediately after the assistant turn that asked for it, and placing the
    system turn is the adapter's business rather than this layer's
    (`test_messages_reach_the_adapter_in_order`).
    """
    return tuple(
        ChatMessage(
            role=message.role,
            content=message.content if message.content is not None else "",
            tool_calls=tuple(
                ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
                for call in (message.tool_calls or ())
            ),
            tool_call_id=message.tool_call_id,
        )
        for message in messages
    )


def _neutral_tool_choice(
    choice: Literal["auto", "none", "required"] | NamedToolChoice | None,
) -> ToolChoice | None:
    """Map OpenAI's `tool_choice` union onto the neutral one.

    The string forms carry across unchanged; the object form becomes the `tool`
    mode carrying the name. `required` stays `required` - translating it into a
    provider's spelling is the adapter's job
    (`test_tool_choice_maps_into_the_neutral_contract`).
    """
    if choice is None:
        return None
    if isinstance(choice, NamedToolChoice):
        return ToolChoice(mode="tool", name=choice.function.name)
    return ToolChoice(mode=choice)


def to_neutral(body: ChatCompletionRequest, model: str) -> ChatRequest:
    """Map a wire request onto the neutral contract, against the resolved model id.

    `model` is the provider's own id from the resolver, never the alias the body
    names: an adapter is handed a name its provider knows
    (`test_the_adapter_is_handed_the_resolved_provider_model_not_the_alias`).
    """
    return ChatRequest(
        model=model,
        messages=_neutral_messages(body.messages),
        tools=tuple(
            ToolSpec(
                name=tool.function.name,
                description=tool.function.description,
                parameters=tool.function.parameters,
            )
            for tool in (body.tools or ())
        ),
        tool_choice=_neutral_tool_choice(body.tool_choice),
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        response_format=(
            ResponseFormat(
                name=body.response_format.json_schema.name,
                json_schema=body.response_format.json_schema.json_schema,
                strict=body.response_format.json_schema.strict,
            )
            if body.response_format is not None
            else None
        ),
    )


def to_wire(answer: ChatResponse, alias: str) -> ChatCompletion:
    """Map a neutral answer into the `chat.completion` envelope.

    `alias` is what the body names, not `answer.model`: see the module docstring.
    Tool calls are re-wrapped in OpenAI's `function` shape, because a client reads
    the name from `function.name` and would dispatch nothing against the flat
    neutral one (`test_returned_tool_calls_are_openai_shaped`).
    """
    calls = tuple(
        WireToolCall(
            id=call.id,
            function=WireToolCallFunction(name=call.name, arguments=call.arguments),
        )
        for call in answer.tool_calls
    )
    return ChatCompletion(
        id=f"{COMPLETION_ID_PREFIX}{secrets.token_hex(COMPLETION_ID_BYTES)}",
        created=int(time.time()),
        model=alias,
        choices=(
            Choice(
                message=CompletionMessage(
                    content=answer.content if answer.content else None,
                    tool_calls=calls if calls else None,
                ),
                finish_reason=FINISH_REASON_BY_STOP_REASON.get(
                    answer.finish_reason, answer.finish_reason
                ),
            ),
        ),
        usage=CompletionUsage(
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=answer.usage.completion_tokens,
            total_tokens=answer.usage.total_tokens,
        ),
    )


def _completion_id() -> str:
    """One fresh completion id, shared by every frame of one streamed answer."""
    return f"{COMPLETION_ID_PREFIX}{secrets.token_hex(COMPLETION_ID_BYTES)}"


def _frame(payload: str) -> str:
    """One SSE frame: `data: `, the payload, and the blank line that ends it.

    The blank line is not decoration - it is what terminates an event, so a
    client whose stream never carries one hands nothing to its parser and waits
    until its read timeout. `test_every_frame_ends_with_a_blank_line` asserts the
    raw bytes of a real response, because the helpers the other cases use split
    on lines and cannot see a separator at all.
    """
    return f"data: {payload}\n\n"


def to_chunk(
    event: StreamEvent, *, completion_id: str, created: int, alias: str
) -> ChatCompletionChunk:
    """Map one neutral stream event into the frame that carries it.

    The counterpart of `to_wire`, and it reuses the same wire models and the same
    `FINISH_REASON_BY_STOP_REASON`, so a streamed tool call cannot be shaped
    differently from a returned one and a streamed stop reason cannot be mapped
    by a second table that drifts from the first
    (`test_finish_reason_table_on_the_stream`).

    `alias` is what every frame names, never the provider model id, for the
    reason the module docstring gives
    (`test_every_chunk_carries_the_alias_and_one_id`).
    """
    envelope: dict[str, Any] = {"id": completion_id, "created": created, "model": alias}

    if isinstance(event, StreamStart):
        return ChatCompletionChunk(
            **envelope, choices=(ChunkChoice(delta=ChunkDelta(role=event.role)),)
        )
    if isinstance(event, TextDelta):
        return ChatCompletionChunk(
            **envelope, choices=(ChunkChoice(delta=ChunkDelta(content=event.text)),)
        )
    if isinstance(event, ToolCallDelta):
        return ChatCompletionChunk(
            **envelope,
            choices=(
                ChunkChoice(
                    delta=ChunkDelta(
                        tool_calls=(
                            WireToolCallChunk(
                                index=event.index,
                                id=event.call.id,
                                function=WireToolCallFunction(
                                    name=event.call.name, arguments=event.call.arguments
                                ),
                            ),
                        )
                    )
                ),
            ),
        )
    if isinstance(event, FinishReason):
        return ChatCompletionChunk(
            **envelope,
            choices=(
                ChunkChoice(
                    delta=ChunkDelta(),
                    finish_reason=FINISH_REASON_BY_STOP_REASON.get(event.reason, event.reason),
                ),
            ),
        )
    if isinstance(event, UsageTotal):
        return ChatCompletionChunk(
            **envelope,
            choices=(),
            usage=CompletionUsage(
                prompt_tokens=event.usage.prompt_tokens,
                completion_tokens=event.usage.completion_tokens,
                total_tokens=event.usage.total_tokens,
            ),
        )
    # Spelled out rather than left as a fall-through: `assert_never` makes mypy
    # fail this function the day `StreamEvent` grows a sixth member, instead of
    # letting the new event be rendered as whatever the last branch happened to
    # build. No request can reach it - the union is closed and every member is
    # handled above.
    assert_never(event)


def sse_frames(
    first: StreamEvent,
    events: Iterator[StreamEvent],
    alias: str,
    completion_id: str,
    on_usage: Callable[[Usage], None] | None = None,
    *,
    mapping: Mapping[str, str] | None = None,
    on_unreported_usage: Callable[[int], None] | None = None,
) -> Generator[str, None, None]:
    """Render the neutral events as SSE frames, ending with `[DONE]` if they finish.

    A `GatewayError` raised part way through becomes ONE frame carrying the same
    `{"error": {...}}` body every non-streaming failure uses, and the stream then
    ends with no sentinel: the status is already 200 and cannot be changed, so
    the frame says what went wrong and the missing `[DONE]` is what lets a client
    tell a truncated answer from a finished one (an earlier change decision E,
    `test_a_failure_after_the_first_event_emits_an_error_chunk_and_no_done`).

    Only a TYPED failure is rendered, and the limit is deliberate rather than an
    oversight. An untyped one is a bug, and there is nothing safe to say about a
    bug in a frame; it reaches the client as a cut stream with no sentinel, which
    is the same "truncated" report decision E gives a typed failure, minus the
    explanation. Nothing in this gateway turns it into a body: the `Exception`
    handler registered in `gateway/errors.py` is served by Starlette's
    `ServerErrorMiddleware`, which re-raises unchanged once a response has
    started, so for a crash inside this generator that handler never runs and the
    traceback is logged by the ASGI server rather than by us
    (`test_an_untyped_failure_mid_stream_is_not_rendered_as_a_frame`).

    The provider's connection is released in a `finally` rather than left to the
    garbage collector. A client that navigates away mid-answer leaves this
    generator suspended, and the streamed `httpx` response underneath it stays
    checked out of the pool until it is closed
    (`test_closing_the_frame_generator_releases_the_provider_stream`).

    Typed as a `Generator` rather than an `Iterator` because a caller has to be
    able to `close()` it: that is what releases the provider stream when the
    client goes away, and an `Iterator` does not promise the method.

    A sync generator, deliberately: FastAPI iterates it in a worker thread, so a
    provider that pauses between events holds a thread rather than the event
    loop, exactly as the non-streaming path is already handled.

    `on_unreported_usage` is called with the characters this generator actually
    SENT, and only when the stream ended without a `UsageTotal` event - a typed
    failure after the first frame, or a client that went away mid-answer. It runs
    in the `finally` beside the release above, so the abandoned stream reaches it
    too, and it is what makes an earlier change decision E true: every served request
    books a row, and the one nobody reported usage for books an estimated one
    rather than nothing at all
    (`test_a_stream_that_fails_after_its_first_frame_books_one_estimated_row`).
    A stream that DID report usage has already booked from `on_usage`, so the
    flag below is what keeps a completed stream from booking twice
    (`test_a_stream_with_usage_books_exactly_one_row`).

    The count is of text this generator EMITTED, not of what the provider sent:
    text still held in the rehydrator was never part of an answer anyone
    received, and counting it would price a fragment the client never saw.

    `mapping` is the rehydration the caller asked for, and `None` - the shape a
    request without `X-Gateway-Rehydrate` takes - leaves every frame exactly as
    it was before an earlier change (`test_a_stream_without_the_header_is_what_it_is_today`).
    With a mapping, a `TextDelta` goes through a `StreamRehydrator` and becomes
    ZERO or one frame: a fragment that could still be completing a placeholder is
    held rather than sent as an empty delta (decision D). A `ToolCallDelta`
    carries a COMPLETE call, so its `arguments` need no
    buffer and are substituted whole
    (`test_a_streamed_tool_calls_arguments_are_rehydrated`).

    The held text is flushed before EVERY non-text event, before the error frame
    and before `[DONE]` (decisions E and F), so a `finish_reason` can never
    overtake the last of the answer and nothing the provider produced is dropped
    (`test_held_text_is_flushed_before_the_finish_reason`,
    `test_held_text_is_flushed_before_an_error_frame`,
    `test_a_trailing_partial_placeholder_is_flushed_before_done`).

    The wrap runs INSIDE this generator rather than around the iterator it is
    handed (decision C): the `finally` below closes `events` by name, and closing
    a wrapper instead would leave the provider's stream to the garbage collector
    - which is what `test_closing_the_frame_generator_releases_the_provider_stream`
    exists to catch.
    """
    created = int(time.time())
    # An empty map stands in for "no rehydration asked for": both `rehydrate` and
    # `StreamRehydrator` are the identity over one, so the frames of a request
    # without the header are what they were before this parameter existed, and
    # the two paths do not need two copies of the loop below.
    restore: Mapping[str, str] = mapping if mapping is not None else {}
    rehydrator = StreamRehydrator(restore)
    sent_chars = 0
    usage_reported = False

    def rendered(event: StreamEvent) -> str:
        return _frame(
            to_chunk(
                event, completion_id=completion_id, created=created, alias=alias
            ).model_dump_json()
        )

    def held() -> Generator[str, None, None]:
        """The frame carrying whatever is still buffered, or no frame at all."""
        nonlocal sent_chars
        text = rehydrator.flush()
        if text:
            sent_chars += len(text)
            yield rendered(TextDelta(text=text))

    try:
        for event in chain((first,), events):
            if isinstance(event, TextDelta):
                emitted = rehydrator.feed(event.text)
                if not emitted and event.text:
                    # Everything the provider sent is still held: no frame at all
                    # (decision D). `event.text` guards the other way round - a
                    # provider delta that was empty to begin with is rendered
                    # exactly as it was before this parameter existed.
                    continue
                sent_chars += len(emitted)
                event = TextDelta(text=emitted)
            else:
                yield from held()
                if isinstance(event, ToolCallDelta):
                    event = ToolCallDelta(
                        index=event.index,
                        call=event.call.model_copy(
                            update={"arguments": rehydrate(event.call.arguments, restore)}
                        ),
                    )
            if isinstance(event, UsageTotal):
                usage_reported = True
                if on_usage is not None:
                    on_usage(event.usage)
            yield rendered(event)
        yield from held()
    except GatewayError as error:
        yield from held()
        yield _frame(
            json.dumps(
                error_body(
                    message=error.message,
                    code=error.code,
                    http_status=error.http_status,
                    param=error.param,
                )
            )
        )
        return
    finally:
        if isinstance(events, Generator):
            # `itertools.chain` has no `close`, so the adapter's own generator is
            # what has to be closed; `_streamed` passes it separately for that
            # reason rather than chaining the first event onto it here.
            events.close()
        if on_unreported_usage is not None and not usage_reported:
            on_unreported_usage(sent_chars)
    yield _frame(DONE_SENTINEL)


def requested_mapping(
    store: RedactionMapStore, tenant: str, header: str | None
) -> Mapping[str, str] | None:
    """The map this request asked to rehydrate through, or `None` when it asked for none.

    Everything here happens BEFORE a provider is dialled, which is the point: a
    caller holding a stale id pays for no tokens, and every refusal below is
    asserted against an adapter that recorded zero calls
    (`test_a_malformed_map_id_is_422_with_no_outbound_call`,
    `test_an_unknown_map_id_is_422_with_no_outbound_call`,
    `test_an_unreachable_map_store_is_502_with_no_outbound_call`).

    The header is tested against `None`, never for truthiness: an empty value is a
    caller that sent the header with nothing in it, and that is a malformed id
    rather than a request that never asked (the empty row of
    `test_the_map_id_rejection_table`).

    The answer is the same on a streamed request as on a returned one: an earlier change
    gave the frame renderer a boundary-safe buffer, so `stream: true` with the
    header no longer refuses and each refusal below is re-asserted on a streamed
    body (`test_a_malformed_map_id_on_a_stream_is_422_with_no_outbound_call`,
    `test_an_unknown_map_id_on_a_stream_is_422_with_no_outbound_call`,
    `test_an_id_live_for_another_tenant_is_absent_on_a_stream`,
    `test_an_unreachable_map_store_on_a_stream_is_502_with_no_outbound_call`).

    Three refusals, in the order they are cheapest to make:

    * A malformed id is 422 before the store is asked, against the store's own
      `MAP_ID_PATTERN` rather than a second copy of the rule.
      `fullmatch`, because `$` in Python also matches before a trailing newline; the inclusive boundary and the whitespace rows are
      `test_a_thirty_two_character_hex_id_is_accepted` and
      `test_the_map_id_rejection_table`.
    * An unreachable store is a 502 `PROVIDER_ERROR` in this function's OWN
      words, exactly as `filed_map_id` renders the write side: the client's text
      names an internal host and port and stays in the chained cause
      (`test_a_map_store_failure_does_not_put_the_clients_words_in_front_of_a_caller`).
    * An id with no live map is 422 (decision B). The map is read for the
      AUTHENTICATED tenant, so an id live for another tenant is simply absent and
      answers the identical body - a message naming the id, or saying whose it
      was, would hand a caller an oracle for which ids exist
      (`test_an_id_live_for_another_tenant_is_absent`,
      `test_an_unknown_map_id_is_422_with_no_outbound_call`).
    """
    if header is None:
        return None
    if MAP_ID_PATTERN.fullmatch(header) is None:
        raise IdentifierInvalid(f"{REHYDRATE_HEADER}: not a valid map id", param=REHYDRATE_HEADER)
    try:
        mapping = store.get(tenant, header)
    except RedactionMapStoreUnavailable as exc:
        raise ProviderError("the redaction map store is unavailable", retryable=True) from exc
    if mapping is None:
        raise IdentifierInvalid(f"{REHYDRATE_HEADER}: no live map", param=REHYDRATE_HEADER)
    return mapping


def allowed_by_header(header: str | None) -> bool:
    """Whether the caller opted this request into the cache, refusing anything else.

    Three states, and the middle one is the point (decision D). No header at all
    is `False` and costs nothing: the ABSENCE is how a caller says nothing, the
    same way `REHYDRATE_HEADER`'s absence is. Exactly `CACHE_ALLOW` is `True`.
    ANY other value - `yes`, `true`, `ALLOW`, `allow ` - is an
    `IdentifierInvalid`, which the handlers in `gateway/errors.py` render as a
    422 `VALIDATION` naming this header, raised here rather than treated as a
    quiet "no" because a typo that silently disables a caching regime is the
    failure a caller cannot see (`test_a_cache_header_that_is_not_the_accepted_word_is_422`,
    nine rows, each asserting the adapter was never called;
    `test_the_accepted_word_is_not_refused` is what stops a check that refuses
    everything from passing the table).

    Compared with `==` and not with `in`, `startswith`, `.strip()` or `.lower()`
    - each of those is one row of that table.
    """
    if header is None:
        return False
    if header != CACHE_ALLOW:
        raise IdentifierInvalid(f"{CACHE_HEADER}: not an accepted value", param=CACHE_HEADER)
    return True


def cached_answer(cache: ResponseCache, tenant: str, key: str | None) -> ChatResponse | None:
    """The stored answer for this request, or `None` - for any reason at all.

    `key` is `None` for a request that was never eligible, which is a miss
    without a lookup. Everything else is a lookup whose every failure is the
    same miss: an entry that is not there, a store that cannot be reached, a
    value this build cannot read.

    The `except Exception` is deliberate and is the route's own backstop rather
    than a duplicate of the store's. `RedisResponseCache` already swallows
    `redis.RedisError` and a bad stored value, so nothing the SHIPPED store does
    reaches here - but this seam is a Protocol, the wiring is settings-driven,
    and a cache is the one dependency whose failure must never change a status
    code. `test_a_cache_that_raises_still_serves_the_request` injects a store
    that raises a plain `RuntimeError` from both methods, which no `redis`-shaped
    guard would catch.
    """
    if key is None:
        return None
    try:
        return cache.get(tenant, key)
    except Exception:
        return None


def store_answer(cache: ResponseCache, tenant: str, key: str | None, answer: ChatResponse) -> None:
    """File this answer for the next identical request, or do nothing at all.

    Guarded for a sharper reason than the lookup is: this runs AFTER a provider
    has answered and been paid, so an escaping failure would lose a request that
    was completely served. `key` is `None` for an ineligible request, which is
    stored nowhere (decisions G and H).
    """
    if key is None:
        return
    try:
        cache.put(tenant, key, answer)
    except Exception:
        return


def requested_key(
    body: ChatCompletionRequest, tenant: str, chosen: Resolved, *, allow_header: bool
) -> str | None:
    """The cache key for this request, or `None` when it may not be cached (#22).

    One function for both the lookup and the write, so the two cannot disagree
    about what the key is or about which requests are eligible - the lookup uses
    `candidates[0]` and the write uses whatever candidate actually served
    (decision B), and that resolved model is the only argument that differs
    between the two calls.

    The body is the INBOUND one, before redaction (decision A). The redacted text
    depends on the tenant's directory, so keying on the outbound body would
    silently invalidate every entry the moment a name is pushed, and would give
    two callers' identical prompts two keys for nothing.
    """
    if not is_cacheable(body, allow_header=allow_header):
        return None
    dumped = body.model_dump(mode="json", exclude_none=True)
    return cache_key(
        tenant=tenant,
        alias=body.model,
        model=chosen.model,
        messages=dumped["messages"],
        tools=dumped.get("tools"),
        response_format=dumped.get("response_format"),
        temperature=dumped.get("temperature"),
    )


def embedded_probe(
    text: str | None,
    *,
    policy: Policy,
    adapters: Mapping[str, Adapter],
    breaker: CircuitBreaker,
    tenant: str,
    data_class: DataClass,
    mode: ResidencyMode,
) -> tuple[float, ...] | None:
    """The probe text as a vector, or `None` - for any reason at all (#23).

    The one place this route spends money on the semantic tier, and the one place
    it can fail. Every outcome that is not a vector is `None`, which the two
    helpers below turn into a miss and a write that does not happen: no probe
    text, a probe over the cap, an `embed` alias that does not exist, one refused
    by residency, one with no healthy embedding candidate, an adapter that raises
    and an answer carrying no embedding are the same nothing here (an earlier change
    decision B). A cache tier must never change the status code of a request a
    provider would have served, and `test_an_embedding_adapter_that_raises_is_a_miss`
    injects an adapter raising a plain `RuntimeError`, which no typed guard would
    catch.

    The cap is `MAX_EMBEDDING_INPUT_CHARS`, imported from the embeddings route
    and never restated (decision F). A longer probe is not truncated: truncating
    changes the question, and two different long prompts sharing a prefix would
    become one question (`test_the_probe_length_boundary_is_inclusive_and_never_a_422`,
    `test_a_probe_at_the_cap_is_embedded_whole_and_not_truncated`). The import is
    made HERE rather than at module scope because `routes/embeddings.py` imports
    this module at module scope, so the pair would be an import cycle (standing
    rule 32: the direction between two modules is already fixed by which one
    imports which).

    The health seam is the embeddings route's, not the chat route's - "this
    provider has an EMBEDDING adapter and its breaker allows it" - so a candidate
    that can only chat is invisible to the resolver rather than an
    `AttributeError`.

    Failures are absorbed into the breaker exactly as the embeddings route
    absorbs them, because they are real provider failures and that route would
    count them too, and the tokens are booked with `record_tokens` under the EMBED
    provider and model (decision G): they were really bought, and a tier that
    spends tokens invisibly is how a cache ends up costing more than it saves
    (`test_the_embedding_tokens_are_booked_under_the_embed_provider_and_model`).
    `record_served` is deliberately NOT called - that would write a second
    residency-log row for one client request and relabel the scope with the
    embedding provider (`test_the_embedding_writes_no_second_residency_row`,
    `test_the_embedding_does_not_relabel_the_requests_candidate`).
    """
    from app.gateway.routes.embeddings import MAX_EMBEDDING_INPUT_CHARS

    if text is None or len(text) > MAX_EMBEDDING_INPUT_CHARS:
        return None
    try:
        candidates = resolve(
            policy,
            tenant,
            EMBED_ALIAS,
            data_class.value,
            is_healthy=health_seam(
                breaker,
                lambda provider: isinstance(adapters.get(provider), EmbeddingAdapter),
            ),
            mode=mode,
        )
    except Exception:
        return None

    for chosen in candidates:
        adapter = cast(EmbeddingAdapter, adapters[chosen.provider])
        try:
            answer = adapter.embed(EmbeddingRequest(model=chosen.model, input=(text,)))
        except ProviderError as exc:
            absorb_failure(breaker, chosen.provider, exc)
            if not exc.retryable:
                return None
            continue
        except Exception:
            return None
        record_tokens(
            tenant=tenant,
            provider=chosen.provider,
            model=chosen.model,
            prompt_tokens=answer.usage.prompt_tokens,
        )
        if not answer.embeddings:
            return None
        return answer.embeddings[0].vector
    return None


def semantic_answer(
    cache: SemanticCache,
    tenant: str,
    vector: tuple[float, ...] | None,
    *,
    alias: str,
    system_hash: str,
    tools_hash: str,
) -> ChatResponse | None:
    """The stored answer to a question near enough to this one, or `None`.

    `vector` is `None` for a request that was never eligible and for one whose
    embedding did not happen, which is a miss without a lookup - the same shape
    `cached_answer` has for a `None` key, and the reason the route needs no
    second eligibility test at the write.

    The `except Exception` is the route's own backstop rather than a duplicate of
    the store's, for the reason `cached_answer` states: `app.cache`
    guarantees neither implementation raises, but this seam is a Protocol and the
    wiring is settings-driven (`test_a_store_that_raises_on_search_is_a_miss`).
    """
    if vector is None:
        return None
    try:
        return cache.search(
            tenant, vector, alias=alias, system_hash=system_hash, tools_hash=tools_hash
        )
    except Exception:
        return None


def store_probe(
    cache: SemanticCache,
    tenant: str,
    vector: tuple[float, ...] | None,
    answer: ChatResponse,
    *,
    alias: str,
    system_hash: str,
    tools_hash: str,
) -> None:
    """File this answer under the vector the lookup already paid for, or do nothing.

    The vector is the LOOKUP's, reused rather than embedded a second time (plan
    13c decision H): two calls double the cost of every miss and can disagree if
    the model behind the `embed` alias changes between them.

    Guarded for the reason `store_answer` is guarded, and it is the sharper one:
    this runs after a provider has answered and been paid, so an escaping failure
    would lose a request that was completely served
    (`test_a_store_that_raises_on_put_is_swallowed_after_the_caller_is_served`).
    """
    if vector is None:
        return
    try:
        cache.put(
            tenant, vector, answer, alias=alias, system_hash=system_hash, tools_hash=tools_hash
        )
    except Exception:
        return


def rehydrated(answer: ChatResponse, mapping: Mapping[str, str] | None) -> ChatResponse:
    """The answer with the map's placeholders restored, or the answer itself.

    Applied to the NEUTRAL response before `to_wire` (decision C), because the
    streamed path's `TextDelta` and `ToolCallDelta` are neutral too and one
    substitution covers both; on the wire model the two would need two
    implementations of the same thing. `sse_frames` is where the streamed half
    runs, over a `StreamRehydrator` rather than this function, because a
    placeholder can be split across two frames.

    Both `content` and each tool call's `arguments` (decision D). A tool call
    handed `<EMAIL_1>` is a broken tool call, and the content of an answer whose
    only placeholder sits in a tool call looks entirely correct
    (`test_a_placeholder_in_a_tool_calls_arguments_is_rehydrated`). The call is
    copied rather than rebuilt, so its `id` and `name` are what the provider sent.
    """
    if mapping is None:
        return answer
    return answer.model_copy(
        update={
            "content": rehydrate(answer.content, mapping),
            "tool_calls": tuple(
                call.model_copy(update={"arguments": rehydrate(call.arguments, mapping)})
                for call in answer.tool_calls
            ),
        }
    )


router = APIRouter()


@router.post("/v1/chat/completions", response_model=None)
def create_chat_completion(
    request: Request,
    context: Context,
    body: ChatCompletionRequest,
    response: Response,
) -> ChatCompletion | StreamingResponse:
    """Resolve the alias, then walk the allowed candidates until one answers.

    A provider with no adapter - its key unset, or its kind not yet built - and a
    provider whose breaker is open are both invisible here, through the resolver's
    own health seam rather than through a second unavailability path (an earlier change
    decision E). So a request with no usable candidate at all is the resolver's
    503 `ALL_PROVIDERS_FAILED` with nothing dialled
    (`test_no_adapter_for_any_candidate_is_503`,
    `test_every_breaker_open_is_the_resolvers_503_with_no_provider_called`).

    The loop walks the tuple `resolve()` ANSWERED, in order, and never resolves
    again. That is the whole residency argument: the list was filtered by region,
    by data class and by `enabled` before this function saw it, so there is no
    path on which a fallback reaches a provider residency excluded
    (`test_an_eu_tenants_fallback_stays_inside_the_eu_regions`).

    It moves to the next candidate on a RETRYABLE `ProviderError` and on nothing
    else. A 4xx cannot succeed anywhere, so it is returned from the first
    candidate and books nothing against its breaker
    (`test_a_non_retryable_failure_stops_at_the_first_candidate`); when every
    candidate fails retryably the caller gets the LAST one's 502, because that is
    the provider an operator would go and look at
    (`test_every_candidate_failing_retryably_is_the_last_ones_502`).

    A streamed request takes the same loop and differs only in where the window
    closes. `_streamed` pulls the first event before this route returns, so a
    failure up to that point is still un-sent and falls back like any other
    (`test_a_stream_failing_before_its_first_event_falls_back`); after it the 200
    and the first frames are on the wire, and the existing error frame is the only
    honest answer (an earlier change decision C,
    `test_a_stream_failing_after_its_first_event_is_one_error_frame_and_no_fallback`).
    The resolution happens first either way, so an unknown alias, a residency
    refusal and a candidate with no adapter are the ordinary typed errors on both
    (`test_a_routing_failure_on_a_stream_is_the_ordinary_typed_error`).

    The exact-match cache sits between the resolution and the loop (#22). One
    lookup, keyed on `candidates[0]`, and one write, keyed on the candidate that
    actually SERVED (decision B): under a healthy system those are the same
    resolution, and under degradation the next request simply misses rather than
    being handed an answer from a model the healthy path would not have chosen. A
    hit returns before `_redacted` runs, which is the point of caching at all -
    no directory is loaded, no redaction pass runs, no adapter is dialled and no
    token is booked (`test_a_second_identical_eligible_request_calls_no_adapter`,
    `test_a_hit_books_no_tokens`). It still goes through `rehydrated` and
    `to_wire`, so a replayed answer gets its OWN completion id and a caller that
    sent `X-Gateway-Rehydrate` gets its values back from ITS map (decision E), and
    it still books a cost row - `cached=true` at `0.000000`, with the replayed
    answer's own token counts (an earlier change decision B,
    `test_a_cache_hit_books_zero_with_its_tokens`). Both tiers do: a hit is a
    served request, and a ledger that skipped it could not report a hit rate or a
    saving at all.

    Only an eligible request is looked up or stored: `temperature == 0` stated in
    the body or `X-Gateway-Cache: allow`, never streamed, never with live tools
    (13a decisions G and H). Everything else answers `X-Gateway-Cache: miss`
    without touching the store at all.

    The semantic tier (#23) sits AFTER `_redacted` and not beside the exact one, which is the whole of its residency argument: the probe
    it embeds is the final USER message of the OUTBOUND body, so what reaches an
    embedding provider is what may legally leave the box, and the query and the
    stored entry are in one vocabulary
    (`test_the_embedded_probe_is_the_redacted_text_and_not_the_raw_one`). It is
    eligible only when the resolved alias carries `semantic_cache: true` and the
    request names no response schema, is not streamed and carries no live tools -
    FOUR refusals, all of them `semantic_eligible`'s, so the lookup and the write
    read one rule. An exact hit therefore returns before anything is
    embedded, an ineligible request embeds nothing at all, and a schema request is
    not stored either
    (`test_only_an_eligible_request_embeds_anything`,
    `test_a_schema_request_is_never_stored_in_the_semantic_tier`,
    `test_an_exact_hit_returns_before_anything_is_embedded`). ONE embedding per
    request, reused for the lookup and the write (decision H), and every failure
    in it is a miss served by an ordinary provider call (decision B).

    `response_model=None` because the return type is a union: FastAPI would
    otherwise try to validate a `StreamingResponse` against `ChatCompletion`. The
    non-streaming body is a built model either way, so nothing is left unchecked.

    Declared `def` rather than `async def` on purpose. The adapters are
    synchronous `httpx` clients, so FastAPI runs this in a worker thread; an
    `async def` would block the event loop for the whole provider call, and would
    hold it for the whole length of a stream.
    """
    check_rate_limit(request.app.state, context)
    check_budget(request.app.state, context, override=request.headers.get(BUDGET_OVERRIDE_HEADER))

    store: PolicyStore = request.app.state.policy
    adapters: Mapping[str, Adapter] = request.app.state.adapters
    breaker: CircuitBreaker = request.app.state.breaker
    scope = request.scope

    cache: ResponseCache = request.app.state.response_cache
    semantic_cache: SemanticCache = request.app.state.semantic_cache
    mode: ResidencyMode = request.app.state.settings.residency_mode

    allowed = allowed_by_header(request.headers.get(CACHE_HEADER))
    mapping = requested_mapping(
        request.app.state.redaction_map_store,
        context.tenant,
        request.headers.get(REHYDRATE_HEADER),
    )

    with stage_span(scope, ROUTE_SPAN):
        candidates = resolve(
            store.policy,
            context.tenant,
            body.model,
            context.data_class.value,
            is_healthy=health_seam(breaker, lambda provider: provider in adapters),
            mode=mode,
        )
    label_candidate(scope, body.model, candidates[0])

    with stage_span(scope, CACHE_SPAN):
        key = requested_key(body, context.tenant, candidates[0], allow_header=allowed)
        hit = cached_answer(cache, context.tenant, key)
    if hit is not None:
        label_candidate(scope, body.model, candidates[0], cache=CACHE_STATUS_HIT)
        record_cache_hit(tenant=context.tenant, kind=EXACT_CACHE_KIND)
        response.headers.update(
            gateway_headers(candidates[0], context.trace_id, redacted=False, cache=CACHE_STATUS_HIT)
        )
        response.headers[COST_HEADER] = book_cost(
            request.app.state,
            context,
            alias=body.model,
            chosen=candidates[0],
            usage=hit.usage,
            cached=True,
        )
        return to_wire(rehydrated(hit, mapping), body.model)

    outbound, redacted = _redacted(
        scope, store.policy, context, body, request.app.state.directory_store
    )
    annotate(scope, SpanAttributes(redacted=redacted))

    vector: tuple[float, ...] | None = None
    if semantic_eligible(body, alias=store.policy.aliases[body.model]):
        with stage_span(scope, SEMANTIC_SPAN):
            vector = embedded_probe(
                probe_text(outbound),
                policy=store.policy,
                adapters=adapters,
                breaker=breaker,
                tenant=context.tenant,
                data_class=context.data_class,
                mode=mode,
            )
            near = semantic_answer(
                semantic_cache,
                context.tenant,
                vector,
                alias=body.model,
                system_hash=system_hash(outbound),
                tools_hash=tools_hash(outbound),
            )
        if near is not None:
            label_candidate(scope, body.model, candidates[0], cache=CACHE_STATUS_SEMANTIC)
            record_cache_hit(tenant=context.tenant, kind=SEMANTIC_CACHE_KIND)
            response.headers.update(
                gateway_headers(
                    candidates[0],
                    context.trace_id,
                    redacted=redacted,
                    cache=CACHE_STATUS_SEMANTIC,
                )
            )
            response.headers[COST_HEADER] = book_cost(
                request.app.state,
                context,
                alias=body.model,
                chosen=candidates[0],
                usage=near.usage,
                cached=True,
            )
            return to_wire(rehydrated(near, mapping), body.model)

    last = len(candidates) - 1
    for attempt, chosen in enumerate(candidates):
        label_candidate(scope, body.model, chosen)
        headers = gateway_headers(chosen, context.trace_id, redacted=redacted)
        try:
            with stage_span(scope, provider_span_name(chosen.provider)):
                if body.stream:
                    streamed = _streamed(
                        adapters[chosen.provider],
                        outbound,
                        chosen,
                        headers,
                        _token_sink(request.app.state, context, alias=body.model, chosen=chosen),
                        mapping=mapping,
                        on_unreported_usage=_estimated_cost_sink(
                            request.app.state,
                            context,
                            alias=body.model,
                            chosen=chosen,
                            prompt_chars=prompt_characters(outbound),
                        ),
                    )
                else:
                    answer = adapters[chosen.provider].chat(to_neutral(outbound, chosen.model))
        except ProviderError as exc:
            absorb_failure(breaker, chosen.provider, exc)
            if not exc.retryable or attempt == last:
                raise
            continue

        record_served(
            scope,
            breaker,
            tenant=context.tenant,
            alias=body.model,
            chosen=chosen,
            failed=attempt,
            policy=store.policy,
            data_class=context.data_class,
            log=request.app.state.residency_log,
        )
        if body.stream:
            return streamed
        record_tokens(
            tenant=context.tenant,
            provider=chosen.provider,
            model=chosen.model,
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=answer.usage.completion_tokens,
        )
        store_answer(
            cache,
            context.tenant,
            requested_key(body, context.tenant, chosen, allow_header=allowed),
            answer,
        )
        store_probe(
            semantic_cache,
            context.tenant,
            vector,
            answer,
            alias=body.model,
            system_hash=system_hash(outbound),
            tools_hash=tools_hash(outbound),
        )
        response.headers.update(headers)
        # After `record_served`, and last: `book_cost` is the closing loop's other
        # half and the only place this route prices anything.
        response.headers[COST_HEADER] = book_cost(
            request.app.state,
            context,
            alias=body.model,
            chosen=chosen,
            usage=answer.usage,
            cached=False,
        )
        return to_wire(rehydrated(answer, mapping), body.model)

    # Unreachable while `resolve()` keeps its contract: it raises rather than
    # answering an empty tuple, and the last candidate's failure is re-raised
    # above. Stated as the resolver's own typed error rather than as an
    # assertion, so a future resolver that did answer empty is a 503 and not a
    # 500 with no code.
    raise AllProvidersUnavailable(
        f"no candidate was tried for tenant {context.tenant!r}, alias {body.model!r}"
    )


def _token_sink(
    state: State, context: RequestContext, *, alias: str, chosen: Resolved
) -> Callable[[Usage], None]:
    """The callback a stream books its tokens AND its cost through when the usage arrives.

    A stream reports its usage in a `UsageTotal` event, delivered from inside the
    frame generator long after this route returned - so the labels the counter
    needs have to be captured here, where they are known, rather than looked up
    there. It is a closure over the resolver's answer for the same reason the
    non-streamed call above passes `chosen.model`: the model label is the resolved
    id, never the alias and never what the provider called it
    (`test_tokens_are_recorded_from_the_streams_usage_chunk`,
    `test_the_token_model_label_is_the_resolved_model_not_the_alias`).

    The cost row is booked from the same event and through the same `book_cost`
    the three non-streamed paths use, so a streamed request is one measured row
    like any other and `estimated` on it means what it means everywhere else. The header is
    discarded here rather than never built: the six headers of a stream were on
    the wire before this event existed, which is the whole of why
    `X-Gateway-Cost-USD` is absent on a stream.
    """

    def sink(usage: Usage) -> None:
        record_tokens(
            tenant=context.tenant,
            provider=chosen.provider,
            model=chosen.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        book_cost(state, context, alias=alias, chosen=chosen, usage=usage, cached=False)

    return sink


def _estimated_cost_sink(
    state: State, context: RequestContext, *, alias: str, chosen: Resolved, prompt_chars: int
) -> Callable[[int], None]:
    """The callback a stream books an ESTIMATED row through when it reported no usage.

    Decision E: `UsageTotal` is sent by every stream that finishes, so its absence
    is a typed failure after the first frame or a client that went away - and the
    provider has already charged for what it sent either way. Booking nothing
    would make "every served request writes a row" false exactly when an operator
    is looking at an incident
    (`test_a_stream_that_fails_after_its_first_frame_books_one_estimated_row`).

    The row carries `estimated=true`, from `record_cost`'s own absent-usage path:
    the tokens are `app.limits.cost`'s four-characters-a-token
    approximation over the prompt this request sent and the characters the stream
    actually delivered, and the flag is what keeps a number nobody measured
    separable from the ones that were.

    It books a cost row and nothing else. `gateway_tokens_total` is what the
    providers REPORTED, and an estimate is not a report, so no token sample is
    emitted from here; the ledger is where the estimated row lives, flagged.
    """

    def sink(completion_chars: int) -> None:
        book_cost(
            state,
            context,
            alias=alias,
            chosen=chosen,
            usage=None,
            cached=False,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
        )

    return sink


def prompt_characters(body: ChatCompletionRequest) -> int:
    """How many characters of prompt this request sent, for the estimate above.

    The message CONTENT and nothing else: a tool specification and a tool call's
    arguments are billed too, but they are the small and highly variable part of a
    prompt, and this number is only ever the input to a stated approximation that
    the row flags as one. It is counted on the OUTBOUND body, so what it measures
    is what actually left the box - a redacted prompt is priced as the
    placeholders the provider was sent. The estimator itself is pinned on the
    COMPLETION side, where the characters sent are known exactly
    (`test_a_stream_that_reports_no_usage_books_one_estimated_row`); this half is
    a count with no branch in it and no case separates a right answer from a
    wrong one.
    """
    return sum(len(message.content or "") for message in body.messages)


def _redacted(
    scope: Scope,
    policy: Policy,
    context: RequestContext,
    body: ChatCompletionRequest,
    directory_store: DirectoryStore,
) -> tuple[ChatCompletionRequest, bool]:
    """The request the adapter should see, and whether anything was actually replaced.

    Returns the body unchanged and `False` when the tenant's rule for this data
    class does not ask for redaction, which is what keeps a `public` caller from
    getting placeholders it never asked for and keeps the 16384-character cap off
    a route that had none (`test_the_redaction_cap_does_not_apply_on_a_non_redact_class`).

    The directory is loaded INSIDE that branch, after the check, for the same
    reason: a class that is not redacted must not depend on the directory store
    at all, or a Redis incident would take down `public` traffic that never reads
    a name (`test_a_non_redact_class_is_served_while_the_directory_store_is_down`).
    Once, for the whole conversation, because the matcher is compiled from up to
    2000 names and every turn is scanned with the same one.

    Every turn goes through ONE `redact_or_reject` call, so the numbering is
    shared across the conversation: the same address in turn 0 and turn 2 is one
    placeholder, which is what #19 means by stable within a request
    (`test_every_chat_message_is_redacted_under_one_numbering`).

    A turn with no `content` (an assistant turn that is only tool calls) keeps its
    `None` rather than becoming an empty string: `_neutral_messages` reads the two
    the same way, but a body rebuilt with the wrong one would be a silent shape
    change on the way to an earlier change's admin surface.
    """
    if not redaction_applies(policy, context.tenant, context.data_class):
        return body, False

    directory = tenant_directory(directory_store, context.tenant)
    with stage_span(scope, REDACT_SPAN):
        batch = redact_or_reject(
            tuple(message.content or "" for message in body.messages),
            param="messages",
            directory=directory,
        )
    record_redactions(tenant=context.tenant, counts=batch.counts)
    rebuilt = tuple(
        message.model_copy(update={"content": content if message.content is not None else None})
        for message, content in zip(body.messages, batch.texts, strict=True)
    )
    return body.model_copy(update={"messages": rebuilt}), batch.redacted


def _streamed(
    adapter: Adapter,
    body: ChatCompletionRequest,
    chosen: Resolved,
    headers: dict[str, str],
    on_usage: Callable[[Usage], None],
    *,
    mapping: Mapping[str, str] | None = None,
    on_unreported_usage: Callable[[int], None] | None = None,
) -> StreamingResponse:
    """Open the stream, pull its first event, and only then answer 200.

    The eager first event is the whole point. `StreamingResponse` sends the
    status the moment it is returned, so anything the adapter raises after that
    can only be an error frame inside a 200 - which an SDK reports to its caller
    as a parse failure rather than as the provider outage it was. Pulling one
    event here keeps every pre-stream failure a typed HTTP error the existing
    handlers render (`test_a_provider_failure_before_the_first_event_is_502_with_no_stream`).

    A stream that opens and ends without a single event is a provider failure
    too, not a `StopIteration` for the last-resort handler to turn into a 500
    with no code (`test_a_stream_that_opens_and_yields_nothing_is_502`).
    """
    events = adapter.stream_chat(to_neutral(body, chosen.model))
    first = next(events, None)
    if first is None:
        raise ProviderError(
            f"{chosen.provider}: the provider opened a stream and sent nothing",
            retryable=False,
        )

    return StreamingResponse(
        sse_frames(
            first,
            events,
            body.model,
            _completion_id(),
            on_usage=on_usage,
            mapping=mapping,
            on_unreported_usage=on_unreported_usage,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers={**headers, **SSE_HEADERS},
    )
