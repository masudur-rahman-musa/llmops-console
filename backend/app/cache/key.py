r"""What makes two chat requests the same request, and which requests may be cached at all.

Two pure functions and four constants. Nothing here talks to a store, reads a
policy or knows an adapter exists, which is what lets it
run with no app and no backing service.

**The key is a digest of ten fields** the tenant, the alias the
caller named, the provider that served it, the region that provider is in, the
data class of the request, the model the resolver picked, the messages, the
tools, the response format and the temperature. A key over nine of them serves a
WRONG answer to a request that differs by the tenth, so each field is checked
on its own rather than by a sample, and the variants are checked against each
other to catch two of them sharing a slot.

Both the tenant and the resolved model are in there for reasons worth stating.
The tenant, because an entry is a tenant's data and one tenant reading another's
answer is the failure this key must not have - it is also the first segment of
the Redis key, so the scoping is enforced twice. The model, because two aliases
can resolve to the same provider model and a fallback can move an alias to a
different one; keying on both means a degraded resolution simply misses instead
of serving an answer from a model the caller's alias would not have chosen today.

**Canonicalisation is `json.dumps(sort_keys=True)`,** which makes two bodies that
differ only in the KEY ORDER of a JSON object one request, and leaves two that
differ in MESSAGE order as two. The structure is dumped whole rather
than glued together field by field, so two conversations that concatenate to the
same string still key apart.

**Eligibility is deliberately narrow.** `temperature == 0` STATED in the
body, or the `X-Gateway-Cache: allow` header. An absent temperature is not
eligible: the provider's default is not the caller's promise, and reading it as
zero would quietly make every request the gateway has ever served cacheable.
Streamed requests are out and so are requests carrying tools unless
`tool_choice` is `none`, because a tool result varies per call and
the answer is then not a function of the request.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from app.schemas.chat import ChatCompletionRequest

CACHE_KEY_PREFIX: Final[str] = "response"
"""The first segment of every Redis key a cached answer is stored under.

A running gateway's live entries are addressed by this string, so renaming it
orphans every one of them and turns a warm cache into a day of misses nobody
sees as a failure. Distinct
from `REDACTION_MAP_KEY_PREFIX` and `DIRECTORY_KEY_PREFIX`, which share the same
Redis database."""

CACHE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
r"""What a cache key must look like before it is allowed to shape a Redis key.

A sha256 rendered as lowercase hex, which is what `cache_key` mints. Used with
`fullmatch` and never with `match`: in Python `$` also matches
before a trailing newline, so `match` would let `<64 hex>\n` through. The
rejection table - 63, 65, uppercase, non-hex, a trailing newline, a trailing
CRLF, a trailing space, a colon and a glob character - is refused."""

CACHE_HEADER: Final[str] = "X-Gateway-Cache"
"""One spelling for two directions.

Inbound it is the caller's opt-in and carries `allow`; outbound it is the
outcome and carries `hit` or `miss`. The same name on both, because a caller
reading `X-Gateway-Cache: miss` after sending `X-Gateway-Cache: allow` is being
answered about the thing it asked for."""

CACHE_ALLOW: Final[str] = "allow"
"""The only value the request header accepts.

Anything else is a 422 rather than a silent miss: the header opts
into a different consistency regime, and a typo meaning "no" is the failure a
caller cannot see."""

NO_TOOL_CHOICE: Final[str] = "none"
"""The one `tool_choice` under which a request carrying tools is still cacheable.

`none` forbids a call, so the answer is a function of the request again. Absent,
`auto`, `required` and a named tool are all uncacheable - absent included,
because the provider is free to call."""


def cache_key(
    *,
    tenant: str,
    alias: str,
    provider: str,
    region: str,
    data_class: str,
    model: str,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    response_format: Mapping[str, object] | None,
    temperature: float | None,
) -> str:
    """The sha256 of the ten keyed fields, as 64 lowercase hex characters.

    Keyword-only, and every argument required. A default on any of the ten
    would let a call site key a request without one of its fields simply by
    forgetting it.

    `provider`, `region` and `data_class` are part of the key because
    a cached answer is served only where it could have been made, so the key must
    include where it was made (provider and region) and under which data-handling
    rule (data_class).

    `messages`, `tools` and `response_format` arrive as the JSON-able structures
    the route holds after `body.model_dump(mode="json", exclude_none=True)`,
    never as Pydantic models: `json.dumps` cannot render a model, and dumping
    inside here would make this function depend on the wire shape it is
    deliberately ignorant of.

    Determinism is the whole contract:
    nothing per-call - no clock, no salt, no `id()` - goes into the digest, or
    every request misses and the cache never serves anything.
    """
    canonical = json.dumps(
        {
            "tenant": tenant,
            "alias": alias,
            "provider": provider,
            "region": region,
            "data_class": data_class,
            "model": model,
            "messages": list(messages),
            "tools": None if tools is None else list(tools),
            "response_format": None if response_format is None else dict(response_format),
            "temperature": temperature,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_cacheable(body: ChatCompletionRequest, *, allow_header: bool) -> bool:
    """Whether this request may be looked up and stored.

    `allow_header` is keyword-only and has no default, for the reason
    `gateway_headers`'s `redacted` has none: a default would let a call site
    silently drop the caller's opt-in, and the request would go on being served
    correctly, so nothing but a header assertion would ever notice.

    Four rules:

    * `temperature is not None and temperature == 0`, never `not temperature`
      and never `temperature or 0`. An ABSENT temperature is not eligible.
    * `stream` is never cacheable. A response cache stores an answer, and
      replaying frames is a job of its own.
    * Tools with anything but `tool_choice: none` are not cacheable, absent
      choice included.
    * The header alone makes an otherwise-ineligible request eligible, but it
      does not override `stream` or the tool rule: those two are about whether
      the answer is a function of the request at all, which is not the caller's
      to waive.
    """
    if body.stream:
        return False
    if body.tools and body.tool_choice != NO_TOOL_CHOICE:
        return False
    if allow_header:
        return True
    return body.temperature is not None and body.temperature == 0
