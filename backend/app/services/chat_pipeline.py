"""The chat request flow: rate limit, budget, cache tiers, redaction, cost."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from opentelemetry.trace import Span
from starlette.types import Scope

from app.cache import (
    probe_text,
    semantic_eligible,
    system_hash,
    tools_hash,
)
from app.core.context import RequestContext
from app.core.errors import GatewayError, ProviderError
from app.core.metrics import (
    EXACT_CACHE_KIND,
    SEMANTIC_CACHE_KIND,
    record_cache_hit,
    record_fallback,
    record_redactions,
    record_tokens,
)
from app.core.tracing import (
    Generation,
    GenerationAttributes,
    SpanAttributes,
    annotate,
    counted,
    note_request,
    note_step,
    stage_span,
    start_generation,
)
from app.model_routing.errors import AllProvidersUnavailable
from app.model_routing.policy import Policy
from app.model_routing.resolver import Resolved, resolve
from app.providers.breaker import CircuitBreaker
from app.redaction import DirectoryStore
from app.schemas.chat import ChatCompletionRequest
from app.schemas.provider import Usage
from app.services.chat_accounting import (
    ChatResult,
    Clients,
    book_failure,
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
    semantic_note,
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
from app.services.redaction_flow import (
    REDACT_SPAN,
    masked_line,
    redact_or_reject,
    redaction_applies,
    tenant_directory,
)
from app.services.streaming import StreamPlan, streamed
from app.services.wire_mapping import (
    FINISH_REASON_BY_STOP_REASON,
    rehydrated,
    requested_mapping,
    to_neutral,
    to_wire,
)

COST_HEADER = "X-Gateway-Cost-USD"
"""The seventh `X-Gateway-*` header: what this request cost, in USD.

Its value is the ledger row's own `cost_usd`, rendered at exactly six decimal
places - the quantum `app.limits.cost` prices at and the scale
`cost_ledger.cost_usd` stores - so the header, the row and
`gateway_cost_usd_total` are one number computed once. A float would lose the sixth place on
the way out and make a client's total disagree with the invoice.

Added by the ROUTE rather than by `gateway_headers`, and that is what keeps it
off a stream: the six headers are on the wire before the first frame, and the
usage a stream is priced from does not exist yet at that point. A cache hit does carry it, reading
`0.000000`, because zero is what a hit cost.
"""


ROUTE_SPAN = "route"
"""The stage span around `resolve()`. Shared with the embeddings route, which
imports it from here for the same reason it imports `gateway_headers`: one name
in one place, so two routes cannot disagree about what a stage is called."""


CACHE_SPAN = "cache"
"""The stage span around the exact-match lookup, on this route only.

Emitted on every chat request, hit or miss:
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
something. It wraps the
lookup and not the write, exactly as `CACHE_SPAN` does, because the write
happens after the answer is on its way."""


PROVIDER_SPAN_PREFIX = "provider:"
"""What a provider span's name starts with. The suffix is the RESOLVED provider,
never the alias the caller sent: an alias names a policy entry and a provider
names what actually served the request, which is the one question this span
exists to answer."""


def provider_span_name(provider: str) -> str:
    """The stage span name for one resolved provider."""
    return f"{PROVIDER_SPAN_PREFIX}{provider}"


def note_route(
    span: Span | None, alias: str, data_class: str, candidates: Sequence[Resolved]
) -> None:
    """The `route` step's notes: the alias in; the pick and its fallbacks out.

    Called only after `resolve()` returned, so an unknown alias puts no caller
    text on the trace.
    """
    picked, *rest = candidates
    fallback = ", then ".join(f"{c.provider} {c.model}" for c in rest) or "none"
    note_step(
        span,
        input=f"alias {alias}, data class {data_class}",
        output=f"picked {picked.provider} {picked.model}; fallback {fallback}",
    )


def record_served(
    scope: Scope,
    breaker: CircuitBreaker,
    *,
    tenant: str,
    alias: str,
    chosen: Resolved,
    failed: int,
) -> None:
    """Close the loop on the candidate that served: the breaker, the counter, the span.

    On a STREAM this runs when the first event arrives, not when the last one does,
    because that is where the route hands the response back. The consequence is
    narrow and worth naming: a half-open provider that opens a stream and then dies
    mid-answer has its trial recorded as a success and its breaker closed. Decision
    C is what makes that the honest reading - a failure after the first frame is
    not a fallback and not this loop's to absorb - and a closed breaker costs one
    more request before the next five failures re-open it.

    `failed` is how many candidates were tried and failed before this one, so it is
    the fallback count and zero on an ordinary request. Both the
    counter and the attribute move once per REQUEST rather than once per attempt,
    and neither is touched at all when nothing fell back.

    """
    breaker.record_success(chosen.provider)
    from app.core.metrics import record_circuit_state

    record_circuit_state(provider=chosen.provider, state=int(breaker.state(chosen.provider)))
    if failed == 0:
        return
    record_fallback(tenant=tenant, alias=alias, provider=chosen.provider)
    annotate(scope, SpanAttributes(fallback_count=failed))


def _token_sink(
    clients: Clients,
    context: RequestContext,
    *,
    alias: str,
    chosen: Resolved,
    generation: Generation,
    started: float,
) -> Callable[[Usage], None]:
    """The callback a stream books its tokens AND its cost through when the usage arrives.

    A stream reports its usage in a `UsageTotal` event, delivered from inside the
    frame generator long after this route returned - so the labels the counter
    needs have to be captured here, where they are known, rather than looked up
    there. It is a closure over the resolver's answer for the same reason the
    non-streamed call above passes `chosen.model`: the model label is the resolved
    id, never the alias and never what the provider called it.

    The cost row is booked from the same event and through the same `book_cost`
    the three non-streamed paths use, so a streamed request is one measured row
    like any other and `estimated` on it means what it means everywhere else. The header is
    discarded here rather than never built: the six headers of a stream were on
    the wire before this event existed, which is the whole of why
    `X-Gateway-Cost-USD` is absent on a stream.

    The same header value is what the stream's generation reports as its cost,
    so the trace and the ledger carry one number; the stream ends the span.
    """

    def sink(usage: Usage) -> None:
        record_tokens(
            tenant=context.tenant,
            provider=chosen.provider,
            model=chosen.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        cost = _book_cost(
            clients,
            context,
            alias=alias,
            chosen=chosen,
            usage=usage,
            cached=False,
            latency_ms=_elapsed_ms(clients, started),
        )
        generation.report(generation_usage(usage, cost))
        generation.note_output(tokens_out=usage.completion_tokens)

    return sink


def _estimated_cost_sink(
    clients: Clients,
    context: RequestContext,
    *,
    alias: str,
    chosen: Resolved,
    prompt_chars: int,
    started: float,
) -> Callable[[int], None]:
    """The callback a stream books an ESTIMATED row through when it reported no usage.

    `UsageTotal` is sent by every stream that finishes, so its absence
    is a typed failure after the first frame or a client that went away - and the
    provider has already charged for what it sent either way. Booking nothing
    would make "every served request writes a row" false exactly when an operator
    is looking at an incident.

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
            latency_ms=_elapsed_ms(clients, started),
        )

    return sink


def _elapsed_ms(clients: Clients, started: float) -> int:
    """Whole milliseconds from `started` to now on the clients' clock."""
    return round((clients.clock() - started) * 1000)


def generation_usage(usage: Usage, cost: str) -> GenerationAttributes:
    """The tokens a provider reported and the cost `book_cost` answered, for the trace.

    `cost` is the `X-Gateway-Cost-USD` string itself, so the generation never
    prices anything a second time.
    """
    return GenerationAttributes(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=float(cost),
    )


def traced_conversation(
    policy: Policy, tenant: str, outbound: ChatCompletionRequest
) -> list[tuple[str, str]] | None:
    """The role and redacted text of each outbound message, or `None` for the note.

    `None` unless the tenant's `trace_content` is on. Tool call
    details are not sent; a turn that is only a tool call goes as empty text.
    A test pins this.
    """
    regime = policy.tenants.get(tenant)
    if regime is None or not regime.trace_content:
        return None
    return [(message.role, message.content or "") for message in outbound.messages]


def prompt_characters(body: ChatCompletionRequest) -> int:
    """How many characters of prompt this request sent, for the estimate above.

    The message CONTENT and nothing else: a tool specification and a tool call's
    arguments are billed too, but they are the small and highly variable part of a
    prompt, and this number is only ever the input to a stated approximation that
    the row flags as one. It is counted on the OUTBOUND body, so what it measures
    is what actually left the box - a redacted prompt is priced as the
    placeholders the provider was sent. The estimator itself is pinned on the
    COMPLETION side, where the characters sent are known exactly; this half is
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
    a route that had none.

    The directory is loaded INSIDE that branch, after the check, for the same
    reason: a class that is not redacted must not depend on the directory store
    at all, or a Redis incident would take down `public` traffic that never reads
    a name.
    Once, for the whole conversation, because the matcher is compiled from up to
    2000 names and every turn is scanned with the same one.

    Every turn goes through ONE `redact_or_reject` call, so the numbering is
    shared across the conversation: the same address in turn 0 and turn 2 is one
    placeholder, which is what stable within a request means.

    A turn with no `content` (an assistant turn that is only tool calls) keeps its
    `None` rather than becoming an empty string: `_neutral_messages` reads the two
    the same way, but a body rebuilt with the wrong one would be a silent shape
    change on the way to the admin surface.
    """
    if not redaction_applies(policy, context.tenant, context.data_class):
        return body, False

    directory = tenant_directory(directory_store, context.tenant)
    with stage_span(scope, REDACT_SPAN) as span:
        batch = redact_or_reject(
            tuple(message.content or "" for message in body.messages),
            param="messages",
            directory=directory,
        )
        note_step(
            span, input=counted(len(body.messages), "message"), output=masked_line(batch.counts)
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
    once.

    A content policy refusal is a `ProviderError` that is never retryable, so it
    stops at the candidate that refused. The console's own blocks (redaction, budget,
    rate limit, a bad request) are not `ProviderError`s and
    pass straight through. Tests pin both.
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

    Returns `ChatResult` for a non-streaming body, `StreamPlan` for a streaming one.

    A request that fails with a 5xx once its alias has resolved books one zero-cost row
    carrying that status and then raises as before. Answers below 500 (a 422,
    a 429) book nothing.
    """
    started = clients.clock()
    check_rate_limit(clients, context)
    _check_budget(clients, context, override=budget_override)

    allowed = allowed_by_header(cache_header)
    mapping = requested_mapping(clients.redaction_map_store, context.tenant, rehydrate_header)

    with stage_span(scope, ROUTE_SPAN) as span:
        candidates = resolve(
            clients.policy,
            context.tenant,
            body.model,
            context.data_class.value,
            is_healthy=health_seam(clients.breaker, lambda provider: provider in clients.adapters),
        )
        note_route(span, body.model, context.data_class.value, candidates)
    try:
        return _serve(
            clients,
            context,
            body,
            scope,
            candidates,
            allowed=allowed,
            mapping=mapping,
            started=started,
        )
    except GatewayError as exc:
        status = getattr(exc, "http_status", 0)
        if status >= 500:
            book_failure(
                clients,
                context,
                alias=body.model,
                chosen=candidates[0],
                status=status,
                latency_ms=_elapsed_ms(clients, started),
            )
        raise


def _serve(
    clients: Clients,
    context: RequestContext,
    body: ChatCompletionRequest,
    scope: Scope,
    candidates: tuple[Resolved, ...],
    *,
    allowed: bool,
    mapping: Mapping[str, str] | None,
    started: float,
) -> ChatResult | StreamPlan:
    """Everything after the alias resolved: both cache tiers, redaction, the fallback loop."""
    label_candidate(scope, body.model, candidates[0])
    note_request(scope, alias=body.model, messages=len(body.messages))

    with stage_span(scope, CACHE_SPAN) as span:
        key = requested_key(
            body,
            context.tenant,
            candidates[0],
            allow_header=allowed,
            data_class=context.data_class.value,
        )
        hit = cached_answer(clients.response_cache, context.tenant, key)
        note_step(
            span,
            input=f"same request answered before, alias {body.model}",
            output="miss" if hit is None else "hit",
        )
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
            latency_ms=_elapsed_ms(clients, started),
        )
        return ChatResult(body=to_wire(rehydrated(hit, mapping), body.model), headers=headers)

    outbound, redacted = _redacted(scope, clients.policy, context, body, clients.directory_store)
    annotate(scope, SpanAttributes(redacted=redacted))

    vector: tuple[float, ...] | None = None
    if semantic_eligible(body, alias=clients.policy.aliases[body.model]):
        with stage_span(scope, SEMANTIC_SPAN) as span:
            vector = embedded_probe(
                probe_text(outbound),
                policy=clients.policy,
                adapters=clients.adapters,
                breaker=clients.breaker,
                tenant=context.tenant,
                data_class=context.data_class,
            )
            found = _semantic_answer(
                clients.semantic_cache,
                context.tenant,
                vector,
                alias=body.model,
                system_hash=system_hash(outbound),
                tools_hash=tools_hash(outbound),
                provider=candidates[0].provider,
                data_class=context.data_class.value,
            )
            note_step(
                span,
                input=f"nearest stored answer, alias {body.model}",
                output=semantic_note(found, embedded=vector is not None),
            )
        near = found.answer
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
                latency_ms=_elapsed_ms(clients, started),
            )
            return ChatResult(body=to_wire(rehydrated(near, mapping), body.model), headers=headers)

    _call_count: list[int] = [0]

    def provider_call(chosen: Resolved) -> ChatResult | StreamPlan:
        failed_so_far = _call_count[0]
        _call_count[0] += 1
        label_candidate(scope, body.model, chosen)
        headers = _gateway_headers(chosen, context.trace_id, redacted=redacted)
        generation = start_generation(
            scope,
            provider_span_name(chosen.provider),
            model=chosen.model,
            alias=body.model,
            messages=len(outbound.messages),
            conversation=traced_conversation(clients.policy, context.tenant, outbound),
        )

        def note_finish(reason: str) -> None:
            generation.note_output(finish_reason=FINISH_REASON_BY_STOP_REASON.get(reason, reason))

        if body.stream:
            try:
                plan = streamed(
                    clients.adapters[chosen.provider],
                    outbound,
                    chosen,
                    headers,
                    _token_sink(
                        clients,
                        context,
                        alias=body.model,
                        chosen=chosen,
                        generation=generation,
                        started=started,
                    ),
                    mapping=mapping,
                    on_unreported_usage=_estimated_cost_sink(
                        clients,
                        context,
                        alias=body.model,
                        chosen=chosen,
                        prompt_chars=prompt_characters(outbound),
                        started=started,
                    ),
                    on_close=generation.end,
                    on_finish=note_finish,
                    on_text=generation.add_text,
                )
            except BaseException:
                generation.end()
                raise
            record_served(
                scope,
                clients.breaker,
                tenant=context.tenant,
                alias=body.model,
                chosen=chosen,
                failed=failed_so_far,
            )
            return plan
        try:
            answer = clients.adapters[chosen.provider].chat(to_neutral(outbound, chosen.model))
        except BaseException:
            generation.end()
            raise
        record_served(
            scope,
            clients.breaker,
            tenant=context.tenant,
            alias=body.model,
            chosen=chosen,
            failed=failed_so_far,
        )
        record_tokens(
            tenant=context.tenant,
            provider=chosen.provider,
            model=chosen.model,
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=answer.usage.completion_tokens,
        )
        headers[COST_HEADER] = _book_cost(
            clients,
            context,
            alias=body.model,
            chosen=chosen,
            usage=answer.usage,
            cached=False,
            latency_ms=_elapsed_ms(clients, started),
        )
        generation.report(generation_usage(answer.usage, headers[COST_HEADER]))
        note_finish(answer.finish_reason)
        generation.note_output(tokens_out=answer.usage.completion_tokens)
        generation.note_answer(answer.content)
        generation.end()
        # Filed under the provider that answered, not candidates[0], on purpose: a
        # fallback answer is hit only while the resolver puts that provider first.
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
            data_class=context.data_class.value,
        )
        return ChatResult(body=to_wire(rehydrated(answer, mapping), body.model), headers=headers)

    return walk_candidates(candidates, clients.breaker, provider_call)
