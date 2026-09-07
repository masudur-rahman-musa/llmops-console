"""The exact-match response cache: what makes two requests the same, and where the answer waits.

Two files. `key.py` is pure - the sha256 over the design's seven fields and the
eligibility predicate #22 describes, with no store and no HTTP surface anywhere
near it. `store.py` is the seam: a `ResponseCache` Protocol with an in-process
implementation and a Redis one, picked by `OPSCONSOLE_REDIS_URL` in
`create_app`, holding the NEUTRAL `ChatResponse` for twenty-four hours.

The gateway's use of both lives in `opsconsole.gateway.routes.chat`, not here:
this package never reads a header, resolves an alias or renders a wire body.

One property is worth carrying into any file that imports this one. Unlike
`opsconsole.redaction`'s map store, nothing here raises when the backing store
is unreachable - a cache that cannot answer has not got the entry, and turning a
Redis incident into 5xx on requests a provider would have served is the failure
this package is arranged to make impossible.

The semantic tier (#23) is an earlier change and is not here.
"""

from __future__ import annotations

from opsconsole.cache.key import (
    CACHE_ALLOW,
    CACHE_HEADER,
    CACHE_ID_PATTERN,
    CACHE_KEY_PREFIX,
    NO_TOOL_CHOICE,
    cache_key,
    is_cacheable,
)
from opsconsole.cache.store import (
    RESPONSE_CACHE_TTL_SECONDS,
    InProcessResponseCache,
    RedisResponseCache,
    ResponseCache,
)

__all__ = [
    "CACHE_ALLOW",
    "CACHE_HEADER",
    "CACHE_ID_PATTERN",
    "CACHE_KEY_PREFIX",
    "NO_TOOL_CHOICE",
    "RESPONSE_CACHE_TTL_SECONDS",
    "InProcessResponseCache",
    "RedisResponseCache",
    "ResponseCache",
    "cache_key",
    "is_cacheable",
]
