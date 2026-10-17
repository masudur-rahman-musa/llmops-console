"""The chat request flow: rate limit, budget, cache tiers, redaction, cost."""

from __future__ import annotations

from collections.abc import Callable

from starlette.types import Scope

from app.cache import (
    probe_text,
    semantic_eligible,
    system_hash,
    tools_hash,
)
from app.core.context import RequestContext
from app.core.errors import ProviderError
from app.core.metrics import (
    EXACT_CACHE_KIND,
    SEMANTIC_CACHE_KIND,
    record_cache_hit,
    record_fallback,
    record_redactions,
    record_tokens,
)
from app.core.tracing import SpanAttributes, annotate, stage_span
from app.model_routing.errors import AllProvidersUnavailable
from app.model_routing.policy import DataClass, Policy, ResidencyMode
from app.model_routing.resolver import Resolved, resolve
from app.providers.breaker import CircuitBreaker
from app.redaction import DirectoryStore
from app.schemas.chat import ChatCompletionRequest
from app.schemas.provider import Usage
from app.services.chat_accounting import (
    ChatResult,
    Clients,
    check_rate_limit,
)
from app.services.chat_accounting import (
    book_cost as _book_cost,
)
from app.services.chat_accounting import (
    check_budget as _check_budget,
)
from app.services.chat_cache import (
    absorb_failure,
    cached_answer,
    embedded_probe,
    health_seam,
    requested_key,
    store_answer,
    store_probe,
)
from app.services.chat_cache import (
    semantic_answer as _semantic_answer,
)
from app.services.chat_headers import (
    CACHE_STATUS_HIT,
    CACHE_STATUS_SEMANTIC,
    allowed_by_header,
    label_candidate,
)
from app.services.chat_headers import (
    gateway_headers as _gateway_headers,
)
from app.services.ledger import record_residency
from app.services.redaction_flow import (
    REDACT_SPAN,
    redact_or_reject,
    redaction_applies,
    tenant_directory,
)
from app.services.streaming import StreamPlan, streamed
from app.services.wire_mapping import (
    rehydrated,
    requested_mapping,
    to_neutral,
    to_wire,
)
from app.storage import (
    ResidencyLogWriter,
)

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
    from app.core.metrics import record_circuit_state

    record_circuit_state(provider=chosen.provider, state=int(breaker.state(chosen.provider)))
    record_residency(
        log, policy=policy, tenant=tenant, data_class=data_class, alias=alias, chosen=chosen
    )
    if failed == 0:
        return
    record_fallback(tenant=tenant, alias=alias, provider=chosen.provider)
    annotate(scope, SpanAttributes(fallback_count=failed))


def _token_sink(
    clients: Clients, context: RequestContext, *, alias: str, chosen: Resolved
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
        _book_cost(clients, context, alias=alias, chosen=chosen, usage=usage, cached=False)

    return sink


def _estimated_cost_sink(
    clients: Clients, context: RequestContext, *, alias: str, chosen: Resolved, prompt_chars: int
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
        _book_cost(
            clients,
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


def walk_candidates[T](
    candidates: tuple[Resolved, ...],
    breaker: CircuitBreaker,
    call: Callable[[Resolved], T],
) -> T:
    """Try each candidate in order, moving on only on a retryable failure.

    A retryable failure is absorbed into the breaker and the next candidate is
    tried. A non-retryable failure or the last candidate's failure is re-raised
    so the caller gets the actual error. An empty tuple raises
    `AllProvidersUnavailable` immediately.

    This is the one fallback loop for every path: chat, streaming chat,
    embeddings, and the semantic probe all call it so the retry rule is written
    once (`test_walk_candidates_*` pins each case).
    """
    if not candidates:
        raise AllProvidersUnavailable("no candidates: the resolver answered an empty tuple")
    last = len(candidates) - 1
    for attempt, chosen in enumerate(candidates):
        try:
            return call(chosen)
        except ProviderError as exc:
            absorb_failure(breaker, chosen.provider, exc)
            if not exc.retryable or attempt == last:
                raise
    # Unreachable: the loop returns or re-raises on every path while
    # candidates is non-empty. Typed as AllProvidersUnavailable so a future
    # code path that lands here is a 503 and not an unhandled AssertionError.
    raise AllProvidersUnavailable(  # pragma: no cover
        "no candidate served (internal error)"
    )


def complete_chat(
    clients: Clients,
    context: RequestContext,
    body: ChatCompletionRequest,
    scope: Scope,
    *,
    budget_override: str | None = None,
    cache_header: str | None = None,
    rehydrate_header: str | None = None,
) -> ChatResult | StreamPlan:
    """Run the full chat pipeline and return a plain result.

    The route calls this once, then wraps the result into an HTTP response.
    All pipeline logic lives here: rate limit, budget, resolution, both cache
    tiers, redaction, the fallback loop, cost booking.  The route adds nothing
    except the HTTP layer (response headers, streaming setup).

    Returns `ChatResult` for a non-streaming body, `StreamPlan` for a streaming one
    (see `test_complete_chat_returns_a_stream_plan_...` in test_thin_routes.py).
    """
    check_rate_limit(clients, context)
    _check_budget(clients, context, override=budget_override)

    mode: ResidencyMode = clients.settings.residency_mode
    allowed = allowed_by_header(cache_header)
    mapping = requested_mapping(clients.redaction_map_store, context.tenant, rehydrate_header)

    with stage_span(scope, ROUTE_SPAN):
        candidates = resolve(
            clients.policy,
            context.tenant,
            body.model,
            context.data_class.value,
            is_healthy=health_seam(clients.breaker, lambda provider: provider in clients.adapters),
            mode=mode,
        )
    label_candidate(scope, body.model, candidates[0])

    with stage_span(scope, CACHE_SPAN):
        key = requested_key(
            body,
            context.tenant,
            candidates[0],
            allow_header=allowed,
            data_class=context.data_class.value,
        )
        hit = cached_answer(clients.response_cache, context.tenant, key)
    if hit is not None:
        label_candidate(scope, body.model, candidates[0], cache=CACHE_STATUS_HIT)
        record_cache_hit(tenant=context.tenant, kind=EXACT_CACHE_KIND)
        headers = _gateway_headers(
            candidates[0], context.trace_id, redacted=False, cache=CACHE_STATUS_HIT
        )
        headers[COST_HEADER] = _book_cost(
            clients,
            context,
            alias=body.model,
            chosen=candidates[0],
            usage=hit.usage,
            cached=True,
        )
        return ChatResult(body=to_wire(rehydrated(hit, mapping), body.model), headers=headers)

    outbound, redacted = _redacted(scope, clients.policy, context, body, clients.directory_store)
    annotate(scope, SpanAttributes(redacted=redacted))

    vector: tuple[float, ...] | None = None
    if semantic_eligible(body, alias=clients.policy.aliases[body.model]):
        with stage_span(scope, SEMANTIC_SPAN):
            vector = embedded_probe(
                probe_text(outbound),
                policy=clients.policy,
                adapters=clients.adapters,
                breaker=clients.breaker,
                tenant=context.tenant,
                data_class=context.data_class,
                mode=mode,
            )
            near = _semantic_answer(
                clients.semantic_cache,
                context.tenant,
                vector,
                alias=body.model,
                system_hash=system_hash(outbound),
                tools_hash=tools_hash(outbound),
                provider=candidates[0].provider,
                region=candidates[0].region,
                data_class=context.data_class.value,
            )
        if near is not None:
            label_candidate(scope, body.model, candidates[0], cache=CACHE_STATUS_SEMANTIC)
            record_cache_hit(tenant=context.tenant, kind=SEMANTIC_CACHE_KIND)
            headers = _gateway_headers(
                candidates[0], context.trace_id, redacted=redacted, cache=CACHE_STATUS_SEMANTIC
            )
            headers[COST_HEADER] = _book_cost(
                clients,
                context,
                alias=body.model,
                chosen=candidates[0],
                usage=near.usage,
                cached=True,
            )
            return ChatResult(body=to_wire(rehydrated(near, mapping), body.model), headers=headers)

    _call_count: list[int] = [0]

    def provider_call(chosen: Resolved) -> ChatResult | StreamPlan:
        failed_so_far = _call_count[0]
        _call_count[0] += 1
        label_candidate(scope, body.model, chosen)
        headers = _gateway_headers(chosen, context.trace_id, redacted=redacted)
        if body.stream:
            with stage_span(scope, provider_span_name(chosen.provider)):
                plan = streamed(
                    clients.adapters[chosen.provider],
                    outbound,
                    chosen,
                    headers,
                    _token_sink(clients, context, alias=body.model, chosen=chosen),
                    mapping=mapping,
                    on_unreported_usage=_estimated_cost_sink(
                        clients,
                        context,
                        alias=body.model,
                        chosen=chosen,
                        prompt_chars=prompt_characters(outbound),
                    ),
                )
            record_served(
                scope,
                clients.breaker,
                tenant=context.tenant,
                alias=body.model,
                chosen=chosen,
                failed=failed_so_far,
                policy=clients.policy,
                data_class=context.data_class,
                log=clients.residency_log,
            )
            return plan
        with stage_span(scope, provider_span_name(chosen.provider)):
            answer = clients.adapters[chosen.provider].chat(to_neutral(outbound, chosen.model))
        record_served(
            scope,
            clients.breaker,
            tenant=context.tenant,
            alias=body.model,
            chosen=chosen,
            failed=failed_so_far,
            policy=clients.policy,
            data_class=context.data_class,
            log=clients.residency_log,
        )
        record_tokens(
            tenant=context.tenant,
            provider=chosen.provider,
            model=chosen.model,
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=answer.usage.completion_tokens,
        )
        store_answer(
            clients.response_cache,
            context.tenant,
            requested_key(
                body,
                context.tenant,
                chosen,
                allow_header=allowed,
                data_class=context.data_class.value,
            ),
            answer,
        )
        store_probe(
            clients.semantic_cache,
            context.tenant,
            vector,
            answer,
            alias=body.model,
            system_hash=system_hash(outbound),
            tools_hash=tools_hash(outbound),
            provider=chosen.provider,
            region=chosen.region,
            data_class=context.data_class.value,
        )
        headers[COST_HEADER] = _book_cost(
            clients,
            context,
            alias=body.model,
            chosen=chosen,
            usage=answer.usage,
            cached=False,
        )
        return ChatResult(body=to_wire(rehydrated(answer, mapping), body.model), headers=headers)

    return walk_candidates(candidates, clients.breaker, provider_call)
