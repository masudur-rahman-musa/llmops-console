"""The response caches: what makes two requests the same, and where the answer waits.

Two tiers, four files, and the same division in each: a PURE half saying what
counts as a repeat, and a STORE half saying where the answer waits.

The exact tier is `key.py` - the sha256 over the request's keyed fields and
the eligibility predicate, with no store and no HTTP surface anywhere near it -
over `store.py`, a `ResponseCache` Protocol with an in-process implementation
and a Redis one, picked by `OPSCONSOLE_REDIS_URL` in `create_app`, holding the
NEUTRAL `ChatResponse` for twenty-four hours.

The semantic tier is `probe.py` - what is embedded, the two hashes the query
filters on, the fixed 0.97 threshold and the eligibility predicate's four refusals, the
per-alias switch among them - over `vectors.py`, a `SemanticCache` Protocol with an
in-process implementation and a Qdrant one, holding the same neutral `ChatResponse` in
one collection per tenant. Nothing in it embeds anything: the vector arrives as an
argument, because an embedding is a provider call with a residency regime and a failure
mode of its own: the chat route embeds the probe, and the lookup, the headers and the
`kind="semantic"` metric are its too.

The gateway's use of all four lives in `app.services.chat_pipeline` and
`app.api.routes.chat`, not here: this package never reads a header, resolves an
alias or renders a wire body.

One property is worth carrying into any file that imports this one. Unlike
`app.redaction`'s map store, nothing here raises when the backing store
is unreachable - a cache that cannot answer has not got the entry, and turning a
Redis incident into 5xx on requests a provider would have served is the failure
this package is arranged to make impossible. A malformed tenant is the one
thing either store refuses before acting, and that is not a store failure but a
programming error at the call site.
"""

from __future__ import annotations

from app.cache.key import (
    CACHE_ALLOW,
    CACHE_HEADER,
    CACHE_ID_PATTERN,
    CACHE_KEY_PREFIX,
    NO_TOOL_CHOICE,
    cache_key,
    is_cacheable,
)
from app.cache.probe import (
    SEMANTIC_THRESHOLD,
    probe_text,
    semantic_eligible,
    system_hash,
    tools_hash,
)
from app.cache.store import (
    RESPONSE_CACHE_TTL_SECONDS,
    InProcessResponseCache,
    RedisResponseCache,
    ResponseCache,
)
from app.cache.vectors import (
    SEMANTIC_COLLECTION_PREFIX,
    InProcessSemanticCache,
    QdrantSemanticCache,
    SemanticCache,
)

__all__ = [
    "CACHE_ALLOW",
    "CACHE_HEADER",
    "CACHE_ID_PATTERN",
    "CACHE_KEY_PREFIX",
    "NO_TOOL_CHOICE",
    "RESPONSE_CACHE_TTL_SECONDS",
    "SEMANTIC_COLLECTION_PREFIX",
    "SEMANTIC_THRESHOLD",
    "InProcessResponseCache",
    "InProcessSemanticCache",
    "QdrantSemanticCache",
    "RedisResponseCache",
    "ResponseCache",
    "SemanticCache",
    "cache_key",
    "is_cacheable",
    "probe_text",
    "semantic_eligible",
    "system_hash",
    "tools_hash",
]
