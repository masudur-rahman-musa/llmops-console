r"""What makes two chat requests SIMILAR, and which requests may reach the semantic tier.

Four pure functions and one constant. Nothing here talks to a store, embeds
anything or knows an adapter exists, which is what lets it
run with no app and no backing service - the same division `key.py` holds for the
exact tier.

**The final user message and the turn just before it are embedded**.
A short reply such as "yes" means something only next to what it answers, so
two chats each answering "yes" must not sit at cosine 1.0. Turns older than the
previous one are left out, so two chats asking the same question after different
small talk still match. A final message under `MIN_PROBE_CHARS` is no candidate at
all. A body ending on an assistant turn still probes on its last question rather
than on the gateway's own previous answer. A body with no user turn at all -
legal on the wire - has no probe text and answers `None` rather than indexing
into an empty list.

**The system prompt and the tools are FILTERS, not text**.
Each is a sha256 the query matches exactly, beside the alias. A hex digest glued
into an embedded sentence degrades the vector it is attached to, and a different
system prompt has to be a MISS rather than a near miss: two callers asking one
question under "you are terse" and "you are verbose" are not asking for the same
answer. Both hashes are defined
for an absent field, because absent is what almost every request carries.

**Eligibility is four independent refusals.** The alias has to opt in (it is
off by default per alias), the request must not name a response schema, it must
not be streamed, and tools must not be live. It is deliberately NOT
`is_cacheable`: this tier does not ask for `temperature == 0`, because approximate
reuse is what the alias flag opts into.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final

from app.cache.key import NO_TOOL_CHOICE

if TYPE_CHECKING:
    from app.model_routing.policy import Alias
    from app.schemas.chat import ChatCompletionRequest

SEMANTIC_THRESHOLD: Final[float] = 0.97
"""The cosine similarity at which two probes are the same question. INCLUSIVE.

The threshold is FIXED, so it is a literal here and not a setting: a
tunable one turns a correctness property - whose answer may be served to whose
question - into an operator's mistake, and nothing about it varies per
deployment."""


MIN_PROBE_CHARS: Final[int] = 16
"""The shortest final user message, spaces trimmed, that is looked up or stored.
INCLUSIVE. A test pins this."""

PREVIOUS_TURN_CHARS: Final[int] = 1000
"""How much of the previous turn the probe keeps: its LAST characters, the part
closest to the question. A test pins this."""


def probe_text(body: ChatCompletionRequest) -> str | None:
    """The text a semantic lookup embeds, or `None`.

    The text is the previous turn, a blank line, then the FINAL user message. The
    previous turn counts only when it is a user or assistant turn with text: a
    tool result or a call-only turn adds nothing, and the final message is
    embedded alone.

    `None` means this request is not a semantic candidate at all: a body carrying
    no user turn, or a final message under `MIN_PROBE_CHARS`. Returning `""`
    instead would embed the empty string and let every such request collide with
    every other.

    A user turn whose `content` is absent cannot happen on a request that has
    reached this far - `WireMessage._coherent` refuses a turn with neither
    content nor tool calls, and `tool_calls` belong only on an assistant turn -
    but the `or None` below still answers absent rather than embedding nothing,
    because a lookup is not the place to discover a wire rule.
    """
    for index in range(len(body.messages) - 1, -1, -1):
        final = body.messages[index]
        if final.role != "user":
            continue
        if not final.content or len(final.content.strip()) < MIN_PROBE_CHARS:
            return None
        if index == 0:
            return final.content
        previous = body.messages[index - 1]
        if previous.role not in ("user", "assistant") or not previous.content:
            return final.content
        return f"{previous.content[-PREVIOUS_TURN_CHARS:]}\n\n{final.content}"
    return None


def system_hash(body: ChatCompletionRequest) -> str:
    """The sha256, as 64 lowercase hex characters, of this request's system prompts.

    A payload FILTER rather than embedded text (it is not embedded). Every system
    turn is included in order, so a body with none hashes the empty list and two
    such bodies share one filter value - the
    ordinary case, since most clients send no system turn.

    Nothing per-call goes into the digest: a salt or a clock would make every
    query filter on a value no stored entry carries, and the tier would return
    nothing forever without a single failing "these two differ" assertion.
    """
    prompts = [message.content for message in body.messages if message.role == "system"]
    return _digest(prompts)


def tools_hash(body: ChatCompletionRequest) -> str:
    """The sha256, as 64 lowercase hex characters, of the tools this request offered.

    The second payload filter. Defined for all three shapes the field takes -
    absent, empty and populated - because absent is `None` and an unguarded dump
    of it would raise out of a cache lookup on almost every request this gateway
    serves.

    The tools are dumped whole rather than counted or named: two requests each
    offering exactly one function are not the same request,
    and an answer produced
    with tools on the table is not an answer to the same question asked without
    them.
    """
    if body.tools is None:
        return _digest(None)
    return _digest([tool.model_dump(mode="json", exclude_none=True) for tool in body.tools])


def semantic_eligible(body: ChatCompletionRequest, *, alias: Alias) -> bool:
    """Whether this request may be looked up in, or stored into, the semantic tier.

    `alias` is keyword-only and has no default, for the reason `is_cacheable`'s
    `allow_header` has none: a default would let a call site drop the per-alias
    switch silently, and every request would go on being served correctly, so
    nothing but a header or a metric would ever notice.

    Four refusals, each independent:

    * the resolved alias must carry `semantic_cache: true` (it is off by default
      per alias; `Alias.semantic_cache` defaults to `False`);
    * a request naming a `response_format` is never eligible, because cosine
      distance over one user message cannot vouch that a 0.97 neighbour's answer
      satisfies a schema the neighbour never saw (the refusal is here
      rather than in the route so it gates the WRITE as well);
    * a streamed request is never eligible, exactly as the exact tier refuses one
      - a response cache stores an answer, not frames;
    * tools with anything but `tool_choice: none` are not eligible, ABSENT choice
      included, because the provider is then free to call and the answer is not a
      function of the request.

    Deliberately not a call to `is_cacheable`: that predicate also demands a
    stated `temperature == 0`, and this tier's opt-in is the alias flag rather
    than a promise about determinism.
    """
    if not alias.semantic_cache:
        return False
    if body.response_format is not None:
        return False
    if body.stream:
        return False
    return not (body.tools and body.tool_choice != NO_TOOL_CHOICE)


def _digest(value: object) -> str:
    """One canonical sha256 over a JSON-able structure, shared by both filters.

    `sort_keys=True` for the same reason `cache_key` uses it: two tool
    definitions differing only in the key order of a JSON object are one
    definition, while their ORDER in the list is the caller's and is preserved.
    """
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
