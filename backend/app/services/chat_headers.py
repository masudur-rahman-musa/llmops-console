"""Response header constants and builder functions for the chat pipeline."""

from __future__ import annotations

from starlette.types import Scope

from app.cache import CACHE_ALLOW, CACHE_HEADER
from app.core.metrics import label_resolution
from app.core.tracing import SpanAttributes, annotate
from app.core.validation import IdentifierInvalid
from app.model_routing.resolver import Resolved

CACHE_STATUS_MISS = "miss"


CACHE_STATUS_HIT = "hit"


CACHE_STATUS_SEMANTIC = "semantic"
"""The three words `X-Gateway-Cache` answers with, and the three values the
`cache` label of `gateway_requests_total` carries.

`hit` means no adapter was called: the answer came out of the exact-match cache
under its key, and the caller was charged nothing for it. `semantic` means
no CHAT adapter was called: the answer was stored against a question merely
SIMILAR to this one, which is a different consistency regime and therefore
a different word rather than a second spelling of `hit` -
a caller is entitled to know which one it got, and a hit rate per tier is then
one query. `miss` covers both a request that was looked up and not found and one
that was never eligible to be looked up at all - a caller cannot tell those
apart, and there is nothing it would do differently if it could.

Which one a response carries is asserted against the ADAPTER's call count rather
than against itself,
because a route could serve a fresh completion and label it `hit`.

The same header name carries the caller's opt-in on the way IN, where the only
accepted value is `CACHE_ALLOW`. That opt-in is the EXACT tier's
alone: the semantic tier opts in per ALIAS, in `routing.yaml`, and reads no
header at all."""


REDACTED_TRUE = "true"


REDACTED_FALSE = "false"
"""The two words `X-Gateway-Redacted` is written with. Lowercase, because a
console sums this string and `True` would count as neither.

Which one a response carries is decided per REQUEST, not per route: the header is
`true` exactly when at least one value was actually replaced,
so a redact class carrying no personal data answers `false`."""


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
    would book every such refusal under an empty `alias` and `provider`,
    so the per-alias error panel would show nothing during a redaction-side
    incident.

    `cache` defaults to `miss` because that is what every caller of this function
    is describing at the moment it calls: the outcome is not known until the
    lookup has run, and a request that never reaches one missed. The chat route
    calls it once more with `hit` after a lookup finds something, which
    overwrites rather than accumulates, exactly as the fallback loop's repeated
    calls do.
    """
    annotate(scope, resolution_attributes(alias, chosen))
    label_resolution(
        scope,
        alias=alias,
        provider=chosen.provider,
        cache=cache,
    )


def resolution_attributes(alias: str, resolved: Resolved) -> SpanAttributes:
    """What the resolver decided, as span labels.

    Set only AFTER `resolve()` returns, which is what keeps a refused request's
    alias off its trace.
    """
    return SpanAttributes(alias=alias, provider=resolved.provider)


def gateway_headers(
    resolved: Resolved, trace_id: str, *, redacted: bool, cache: str = CACHE_STATUS_MISS
) -> dict[str, str]:
    """The six `X-Gateway-*` headers describing one served request.

    `redacted` is keyword-only and has no default on purpose. A default would let
    a route report "nothing was redacted" for a request it did redact simply by
    forgetting the argument, and that is the one value here a caller might act on.

    `cache` does have one, and the asymmetry is deliberate: `miss` is the true
    value for every path that has no cache to consult - the embeddings route,
    which imports this function, has none - so the default cannot state something
    false the way a `redacted` default could. Only the exact-match hit passes
    anything else, and it is asserted through the adapter's call count rather
    than through this string.

    Six, and only six, because six is what is KNOWN here. This runs before a
    provider is dialled, and `X-Gateway-Cost-USD` cannot be priced until the
    answer's usage exists - so the route adds it to the response afterwards
    (`COST_HEADER`), and a stream, whose headers are on the wire before its first
    frame, carries six for good. The seven a non-streamed answer ends up with are
    made up of this builder's six plus the cost header.

    Only the success path carries them. An error answers through the handlers in
    `app/core/http_errors.py`, which run outside the route and have no resolver result
    or context to describe.
    """
    return {
        "X-Gateway-Provider": resolved.provider,
        "X-Gateway-Model": resolved.model,
        "X-Gateway-Cache": cache,
        "X-Gateway-Redacted": REDACTED_TRUE if redacted else REDACTED_FALSE,
        "X-Gateway-Trace-Id": trace_id,
    }


def allowed_by_header(header: str | None) -> bool:
    """Whether the caller opted this request into the cache, refusing anything else.

    Three states, and the middle one is the point. No header at all
    is `False` and costs nothing: the ABSENCE is how a caller says nothing, the
    same way `REHYDRATE_HEADER`'s absence is. Exactly `CACHE_ALLOW` is `True`.
    ANY other value - `yes`, `true`, `ALLOW`, `allow ` - is an
    `IdentifierInvalid`, which the handlers in `app/core/http_errors.py` render as a
    422 `VALIDATION` naming this header, raised here rather than treated as a
    quiet "no" because a typo that silently disables a caching regime is the
    failure a caller cannot see.

    Compared with `==` and not with `in`, `startswith`, `.strip()` or `.lower()`
    - each of those would let a near miss through.
    """
    if header is None:
        return False
    if header != CACHE_ALLOW:
        raise IdentifierInvalid(f"{CACHE_HEADER}: not an accepted value", param=CACHE_HEADER)
    return True
