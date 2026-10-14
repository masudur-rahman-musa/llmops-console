r"""What makes two chat requests SIMILAR, and which requests may reach the semantic tier.

Four pure functions and one constant. Nothing here talks to a store, embeds
anything or knows an adapter exists, which is what lets `tests/cache/test_probe.py`
run with no app and no backing service - the same division `key.py` holds for the
exact tier.

**Only the final user message is embedded.** Similarity is measured on the
question being asked, so a caller replaying a long history asks the same question
as one asking it cold (`test_an_earlier_message_does_not_change_the_probe_text`),
and a body ending on an assistant turn still probes on its last question rather
than on the gateway's own previous answer
(`test_a_trailing_assistant_turn_is_not_the_probe_text`). A body with no user
turn at all - legal on the wire - has no probe text and answers `None` rather
than indexing into an empty list
(`test_a_request_with_no_user_turn_has_no_probe_text`).

**The system prompt and the tools are FILTERS, not text** (plan decision C).
Each is a sha256 the query matches exactly, beside the alias. A hex digest glued
into an embedded sentence degrades the vector it is attached to, and a different
system prompt has to be a MISS rather than a near miss: two callers asking one
question under "you are terse" and "you are verbose" are not asking for the same
answer (`test_a_system_prompt_changes_the_system_hash_and_not_the_probe_text`,
`test_two_different_system_prompts_hash_differently`). Both hashes are defined
for an absent field, because absent is what almost every request carries
(`test_the_tools_hash_is_defined_for_every_shape_of_tools_field`).

**Eligibility is four independent refusals.** The alias has to opt in (#23's
"off by default per alias"), the request must not name a response schema, it must
not be streamed, and tools must not be live. It is deliberately NOT
`is_cacheable`: this tier does not ask for `temperature == 0`, because approximate
reuse is what the alias flag opts into
(`test_the_semantic_eligibility_decision_table`, sixteen rows).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final

from app.cache.key import NO_TOOL_CHOICE

if TYPE_CHECKING:
    from app.gateway.routes.chat import ChatCompletionRequest
    from app.model_routing.policy import Alias

SEMANTIC_THRESHOLD: Final[float] = 0.97
"""The cosine similarity at which two probes are the same question. INCLUSIVE.

the design says a FIXED threshold, so it is a literal here and not a setting: a
tunable one turns a correctness property - whose answer may be served to whose
question - into an operator's mistake, and nothing about it varies per
deployment. Pinned to the literal by
`test_the_semantic_threshold_is_exactly_the_published_one`, and its inclusivity
by `test_a_neighbour_at_exactly_the_threshold_is_a_hit` beside
`test_a_neighbour_one_hundredth_under_the_threshold_is_a_miss` in
tests/cache/test_vectors.py."""


def probe_text(body: ChatCompletionRequest) -> str | None:
    """The text a semantic lookup embeds: the FINAL user message, or `None`.

    `None` means this request is not a semantic candidate at all, and it is the
    honest answer for a body carrying no user turn. Returning `""` instead would
    embed the empty string and let every such request collide with every other.

    A user turn whose `content` is absent cannot happen on a request that has
    reached this far - `WireMessage._coherent` refuses a turn with neither
    content nor tool calls, and `tool_calls` belong only on an assistant turn -
    but the `or None` below still answers absent rather than embedding nothing,
    because a lookup is not the place to discover a wire rule.
    """
    for message in reversed(body.messages):
        if message.role == "user":
            return message.content or None
    return None


def system_hash(body: ChatCompletionRequest) -> str:
    """The sha256, as 64 lowercase hex characters, of this request's system prompts.

    A payload FILTER rather than embedded text (plan decision C). Every system
    turn is included in order, so a body with none hashes the empty list and two
    such bodies share one filter value
    (`test_a_request_with_no_system_message_hashes_to_one_stable_value`) - the
    ordinary case, since most clients send no system turn.

    Nothing per-call goes into the digest: a salt or a clock would make every
    query filter on a value no stored entry carries, and the tier would return
    nothing forever without a single failing "these two differ" assertion
    (`test_the_system_hash_is_the_same_twice_for_the_same_prompt`).
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
    offering exactly one function are not the same request
    (`test_two_different_tool_lists_hash_differently`), and an answer produced
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

    Four refusals, each independent, and each row of
    `test_the_semantic_eligibility_decision_table` leaves the other three satisfied
    so that only its own control can decide it:

    * the resolved alias must carry `semantic_cache: true` (#23's "off by default
      per alias"; `Alias.semantic_cache` defaults to `False`);
    * a request naming a `response_format` is never eligible, because cosine
      distance over one user message cannot vouch that a 0.97 neighbour's answer
      satisfies a schema the neighbour never saw (a design note; the refusal is here
      rather than in the route so it gates the WRITE as well, which
      `tests/gateway/test_semantic_cache.py`'s
      `test_a_schema_request_is_never_stored_in_the_semantic_tier` is what pins,
      and the isolating case is
      `test_the_response_format_alone_decides_an_otherwise_eligible_request`);
    * a streamed request is never eligible, exactly as the exact tier refuses one
      (13a decision G) - a response cache stores an answer, not frames;
    * tools with anything but `tool_choice: none` are not eligible, ABSENT choice
      included, because the provider is then free to call and the answer is not a
      function of the request (the design's design note, 13a decision H).

    Deliberately not a call to `is_cacheable`: that predicate also demands a
    stated `temperature == 0`, and this tier's opt-in is the alias flag rather
    than a promise about determinism (the table's two temperature rows).
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
