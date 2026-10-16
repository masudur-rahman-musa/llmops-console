"""POST /v1/chat/completions: the thin route over the chat pipeline."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from app.api.deps import AppStateDep, Context
from app.cache import (
    CACHE_HEADER,
    ResponseCache,
    SemanticCache,
    probe_text,
    semantic_eligible,
    system_hash,
    tools_hash,
)
from app.core.errors import ProviderError
from app.core.metrics import EXACT_CACHE_KIND, SEMANTIC_CACHE_KIND, record_cache_hit, record_tokens
from app.core.tracing import SpanAttributes, annotate, stage_span
from app.limits.budget import BUDGET_OVERRIDE_HEADER
from app.model_routing.errors import AllProvidersUnavailable
from app.model_routing.policy import PolicyStore, ResidencyMode
from app.model_routing.resolver import resolve
from app.providers.base import Adapter
from app.providers.breaker import CircuitBreaker
from app.schemas.chat import ChatCompletion, ChatCompletionRequest
from app.services.chat_pipeline import (
    CACHE_SPAN,
    CACHE_STATUS_HIT,
    CACHE_STATUS_SEMANTIC,
    COST_HEADER,
    ROUTE_SPAN,
    SEMANTIC_SPAN,
    _estimated_cost_sink,
    _redacted,
    _token_sink,
    absorb_failure,
    allowed_by_header,
    book_cost,
    cached_answer,
    check_budget,
    check_rate_limit,
    embedded_probe,
    gateway_headers,
    health_seam,
    label_candidate,
    prompt_characters,
    provider_span_name,
    record_served,
    requested_key,
    semantic_answer,
    store_answer,
    store_probe,
)
from app.services.streaming import _streamed
from app.services.wire_mapping import (
    REHYDRATE_HEADER,
    rehydrated,
    requested_mapping,
    to_neutral,
    to_wire,
)

router = APIRouter()


@router.post("/v1/chat/completions", response_model=None)
def create_chat_completion(
    request: Request,
    context: Context,
    body: ChatCompletionRequest,
    response: Response,
    state: AppStateDep,
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
    check_rate_limit(state, context)
    check_budget(state, context, override=request.headers.get(BUDGET_OVERRIDE_HEADER))

    store: PolicyStore = state.policy
    adapters: Mapping[str, Adapter] = state.adapters
    breaker: CircuitBreaker = state.breaker
    scope = request.scope

    cache: ResponseCache = state.response_cache
    semantic_cache: SemanticCache = state.semantic_cache
    mode: ResidencyMode = state.settings.residency_mode

    allowed = allowed_by_header(request.headers.get(CACHE_HEADER))
    mapping = requested_mapping(
        state.redaction_map_store,
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
            state,
            context,
            alias=body.model,
            chosen=candidates[0],
            usage=hit.usage,
            cached=True,
        )
        return to_wire(rehydrated(hit, mapping), body.model)

    outbound, redacted = _redacted(scope, store.policy, context, body, state.directory_store)
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
                state,
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
                        _token_sink(state, context, alias=body.model, chosen=chosen),
                        mapping=mapping,
                        on_unreported_usage=_estimated_cost_sink(
                            state,
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
            log=state.residency_log,
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
            state,
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
