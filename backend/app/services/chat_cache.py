"""Cache helpers for the chat pipeline: exact-match and semantic tiers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from app.cache import (
    ResponseCache,
    SemanticCache,
    cache_key,
    is_cacheable,
)
from app.core.errors import ProviderError
from app.core.metrics import record_circuit_state, record_tokens
from app.model_routing.policy import DataClass, Policy
from app.model_routing.resolver import Resolved, resolve
from app.providers.base import Adapter, ChatResponse, EmbeddingAdapter, EmbeddingRequest
from app.providers.breaker import CircuitBreaker
from app.schemas.chat import ChatCompletionRequest
from app.schemas.embeddings import MAX_EMBEDDING_INPUT_CHARS

EMBED_ALIAS = "embed"
"""The alias a semantic probe is embedded through: a constant
and not a setting.

Settable, it would be a way to point the tier at a model whose vectors no stored
entry shares - every search silently returning nothing, with no failure anywhere
to see. It is resolved per request like any other alias, so health and
the tenant's data class apply to the embedding exactly as they apply to a
completion."""


def health_seam(
    breaker: CircuitBreaker, has_adapter: Callable[[str], bool]
) -> Callable[[str], bool]:
    """The `is_healthy` the routes hand `resolve()`: an adapter exists AND it is allowed.

    Both routes build it here rather than each writing the conjunction, because a
    route that asked only its own half would be the bug this seam exists to
    prevent: an embeddings route still dialling an open provider while the chat
    route respected it.
    `has_adapter` is the half that differs - "an adapter exists" for a completion,
    "an EMBEDDING adapter exists" for an embedding.

    "May we dial this" is asked as `breaker.allows`, never as a state comparison
    written out here, so there is exactly one definition of it and a future
    refinement of `allows` reaches the routes rather than only the breaker's own
    tests.

    A breaker consulted HERE, and not inside the fallback loop, is what makes a
    request whose every candidate is open the resolver's 503 with no provider
    dialled at all.

    Publishing the gauge from here is what gives every provider in rotation a
    series rather than only the ones that have broken.
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
    of rotation over a caller's own mistake. The caller
    re-raises a non-retryable one; this only decides what the breaker learns.
    """
    if not error.retryable:
        return
    breaker.record_failure(provider)
    record_circuit_state(provider=provider, state=int(breaker.state(provider)))


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
    code. A store that raises a plain `RuntimeError` from either method, which no
    `redis`-shaped guard would catch, must still leave the request served.
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
    body: ChatCompletionRequest,
    tenant: str,
    chosen: Resolved,
    *,
    allow_header: bool,
    data_class: str,
) -> str | None:
    """The cache key for this request, or `None` when it may not be cached (#22).

    One function for both the lookup and the write, so the two cannot disagree
    about what the key is or about which requests are eligible - the lookup uses
    `candidates[0]` and the write uses whatever candidate actually served, and that resolved model
    is the only argument that differs
    between the two calls.

    `data_class` is keyword-only and required: a default would let a call site
    quietly omit it and serve a cached answer across data-handling boundaries,
    which is the exact failure it is there to prevent.

    The body is the INBOUND one, before redaction. The redacted text
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
        provider=chosen.provider,
        data_class=data_class,
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
) -> tuple[float, ...] | None:
    """The probe text as a vector, or `None` - for any reason at all (#23).

    The one place this route spends money on the semantic tier, and the one place
    it can fail. Every outcome that is not a vector is `None`, which the two
    helpers below turn into a miss and a write that does not happen: no probe
    text, a probe over the cap, an `embed` alias that does not exist, one with no healthy embedding
    candidate, an adapter that raises and an answer carrying no embedding are the
    same nothing here. A cache tier must never change the status code of a request
    a provider would have served, so an adapter raising a plain `RuntimeError`, which
    no typed guard would catch, is a miss too.

    The cap is `MAX_EMBEDDING_INPUT_CHARS`, imported from `app.schemas.embeddings`
    and never restated. A longer probe is not truncated: truncating
    changes the question, and two different long prompts sharing a prefix would
    become one question.

    The health seam is the embeddings route's, not the chat route's - "this
    provider has an EMBEDDING adapter and its breaker allows it" - so a candidate
    that can only chat is invisible to the resolver rather than an
    `AttributeError`.

    Failures are absorbed into the breaker exactly as the embeddings route
    absorbs them, because they are real provider failures and that route would
    count them too, and the tokens are booked with `record_tokens` under the EMBED
    provider and model: they were really bought, and a tier that
    spends tokens invisibly is how a cache ends up costing more than it saves.
    `record_served` is deliberately NOT called - that would relabel the scope with
    the embedding provider.
    """

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
    provider: str,
    data_class: str,
) -> ChatResponse | None:
    """The stored answer to a question near enough to this one, or `None`.

    `vector` is `None` for a request that was never eligible and for one whose
    embedding did not happen, which is a miss without a lookup - the same shape
    `cached_answer` has for a `None` key, and the reason the route needs no
    second eligibility test at the write.

    `provider` and `data_class` are required keyword args: a
    cached answer is only served where it could have been made, so the lookup
    must filter on where and how this request would be made.

    The `except Exception` is the route's own backstop rather than a duplicate of
    the store's, for the reason `cached_answer` states: `app.cache`
    guarantees neither implementation raises, but this seam is a Protocol and the
    wiring is settings-driven.
    """
    if vector is None:
        return None
    try:
        return cache.search(
            tenant,
            vector,
            alias=alias,
            system_hash=system_hash,
            tools_hash=tools_hash,
            provider=provider,
            data_class=data_class,
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
    provider: str,
    data_class: str,
) -> None:
    """File this answer under the vector the lookup already paid for, or do nothing.

    The vector is the LOOKUP's, reused rather than embedded a second time (plan
    13c decision H): two calls double the cost of every miss and can disagree if
    the model behind the `embed` alias changes between them.

    `provider` and `data_class` are required keyword args: a
    cached answer is only served where it could have been made, so the stored
    entry must carry where and how it was made.

    Guarded for the reason `store_answer` is guarded, and it is the sharper one:
    this runs after a provider has answered and been paid, so an escaping failure
    would lose a request that was completely served.
    """
    if vector is None:
        return
    try:
        cache.put(
            tenant,
            vector,
            answer,
            alias=alias,
            system_hash=system_hash,
            tools_hash=tools_hash,
            provider=provider,
            data_class=data_class,
        )
    except Exception:
        return
