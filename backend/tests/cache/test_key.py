r"""The cache key and the eligibility predicate: the design's seven fields, and #22's opt-in.

Two claims live here and each is arranged so only the control it names can decide
the outcome.

* THE KEY. One row per keyed field, each differing from the SAME baseline by
  exactly one of the seven, so a key built over six of them reddens on the row
  for the field it dropped and on no other. A sample would not do that: a key
  over the messages alone passes every "different messages, different key"
  assertion anyone would think to write first.
* CANONICALISATION. Two bodies that differ only in the KEY ORDER of a JSON object
  are the same request and share a key; two that differ in MESSAGE order are two
  different conversations and must not. Both directions are asserted, because an
  implementation that sorted the messages too would pass the first alone.

`CACHE_ID_PATTERN` gets a project rule's rejection table - a trailing newline, a
trailing CRLF, a trailing space - because the key becomes the last segment of a
Redis key and `re.match` with `[0-9a-f]{64}$` accepts the first of those.

Nothing here needs a backing service or an app: `cache_key` is a pure function and
`is_cacheable` reads a parsed wire body.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from app.cache import (
    CACHE_ALLOW,
    CACHE_HEADER,
    CACHE_ID_PATTERN,
    CACHE_KEY_PREFIX,
    cache_key,
    is_cacheable,
)
from app.schemas.chat import ChatCompletionRequest

from .conftest import (
    BASE_ALIAS,
    BASE_MESSAGES,
    BASE_MODEL,
    BASE_RESPONSE_FORMAT,
    BASE_TEMPERATURE,
    BASE_TOOLS,
    OTHER_TENANT,
    TENANT,
)

# --------------------------------------------------------------------------
# the published constants
# --------------------------------------------------------------------------


def test_the_key_prefix_is_the_published_one() -> None:
    """Catches a prefix renamed after entries exist under the old one: a running gateway's
    live entries are addressed by this string, so a rename silently orphans every one of
    them and turns a warm cache into a day of misses nobody sees as a failure."""
    assert CACHE_KEY_PREFIX == "response"


def test_the_request_and_the_response_header_are_the_same_spelling() -> None:
    """Catches an opt-in header spelled differently from the header the answer carries
    (`X-Gateway-Cache-Control`, `X-Gateway-Cache-Allow`): a caller reads the response
    header to learn the name it may send back, so two spellings is an opt-in nobody can
    discover, and #24 publishes exactly one name."""
    assert CACHE_HEADER == "X-Gateway-Cache"


def test_the_only_accepted_opt_in_word_is_allow() -> None:
    """Catches an opt-in widened to `true`, `yes` or `1` after decision D refused every
    value but one: each extra word is a value a typo can land on, and the whole point of
    the 422 is that a typo cannot silently mean "no"."""
    assert CACHE_ALLOW == "allow"


ACCEPTED_IDS = [
    pytest.param("0" * 64, id="sixty-four zeros"),
    pytest.param("0123456789abcdef" * 4, id="sixty-four lowercase hex characters"),
]


@pytest.mark.parametrize("value", ACCEPTED_IDS)
def test_the_id_pattern_accepts_a_well_formed_key(value: str) -> None:
    """Catches a pattern written one character too tight (`[0-9a-f]{63}`) or anchored so
    that nothing matches: every rejection row below would pass such a pattern, and this
    is the row that separates "checks the shape" from "refuses everything"."""
    assert CACHE_ID_PATTERN.fullmatch(value) is not None


WELL_FORMED_KEY = "0123456789abcdef" * 4

REJECTED_IDS = [
    pytest.param("", id="empty"),
    pytest.param(WELL_FORMED_KEY[:63], id="sixty-three characters, one under the boundary"),
    pytest.param(WELL_FORMED_KEY + "0", id="sixty-five characters, one over the boundary"),
    pytest.param(WELL_FORMED_KEY.upper(), id="uppercase hex"),
    pytest.param("g" * 64, id="sixty-four characters that are not hex"),
    pytest.param(WELL_FORMED_KEY + "\n", id="a trailing newline"),
    pytest.param(WELL_FORMED_KEY + "\r\n", id="a trailing CRLF"),
    pytest.param(WELL_FORMED_KEY + " ", id="a trailing space"),
    pytest.param(WELL_FORMED_KEY[:63] + ":", id="a colon, which would forge a key segment"),
    pytest.param(WELL_FORMED_KEY[:63] + "*", id="a glob character, which would widen a SCAN"),
]


@pytest.mark.parametrize("value", REJECTED_IDS)
def test_the_id_pattern_rejection_table(value: str) -> None:
    r"""Catches a pattern used with `re.match` against `[0-9a-f]{64}$` rather than with
    `fullmatch`: in Python `$` also matches before a final newline, so
    the three trailing rows would put a caller-influenced line break into a Redis key. The
    length rows keep a pattern that checks only the alphabet from passing the table."""
    assert CACHE_ID_PATTERN.fullmatch(value) is None


# --------------------------------------------------------------------------
# the key: one row per keyed field
# --------------------------------------------------------------------------


def key_for(
    *,
    tenant: str = TENANT,
    alias: str = BASE_ALIAS,
    model: str = BASE_MODEL,
    messages: Sequence[Mapping[str, object]] = BASE_MESSAGES,
    tools: Sequence[Mapping[str, object]] | None = None,
    response_format: Mapping[str, object] | None = None,
    temperature: float | None = BASE_TEMPERATURE,
) -> str:
    """The key for the baseline request with the named fields replaced.

    The arguments are the JSON-able structures the route holds after
    `body.model_dump(mode="json", exclude_none=True)`, which is what the route
    passes and therefore what these cases have to pass too: a helper that handed
    over Pydantic models would be keying something the gateway never keys.
    """
    key: str = cache_key(
        tenant=tenant,
        alias=alias,
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        temperature=temperature,
    )
    return key


def test_a_key_is_sixty_four_lowercase_hex_characters() -> None:
    """Catches a key rendered as base64 or from a shorter digest: the key becomes the last
    segment of a Redis key and is pre-checked against `CACHE_ID_PATTERN`, so a rendering
    the pattern refuses would be a store that cannot read back what it wrote. The literal
    64 is the anchor, so an implementation that shrank the digest AND the pattern still
    fails here."""
    key = key_for()

    assert len(key) == 64
    assert CACHE_ID_PATTERN.fullmatch(key) is not None
    assert key == key.lower()


def test_the_same_request_keys_the_same_twice() -> None:
    """Catches a key with anything per-call in it - a timestamp, a random salt, `id()` of
    the body: every request would miss, the cache would never serve anything, and no other
    case in this file would notice because they all compare two DIFFERENT requests."""
    assert key_for() == key_for()


FIELD_ROWS: list[tuple[str, dict[str, Any]]] = [
    ("tenant", {"tenant": OTHER_TENANT}),
    ("alias", {"alias": "agent-strong"}),
    ("model", {"model": "claude-opus-5"}),
    (
        "messages",
        {"messages": [{"role": "user", "content": "Which city is the capital of Spain?"}]},
    ),
    ("tools", {"tools": list(BASE_TOOLS)}),
    ("response_format", {"response_format": dict(BASE_RESPONSE_FORMAT)}),
    ("temperature", {"temperature": 1.0}),
]


@pytest.mark.parametrize(("field", "override"), FIELD_ROWS, ids=[row[0] for row in FIELD_ROWS])
def test_changing_one_keyed_field_changes_the_key(field: str, override: dict[str, Any]) -> None:
    """Catches a key built over a SUBSET of the design's seven fields, one field per row: the
    row for a field left out of the digest is the one that reddens, and nothing else in
    the suite can see it, because a key over the messages alone still serves the right
    answer to every request that differs by its messages. A dropped field is a WRONG
    answer served to a different request - another tenant's, another model's, another
    temperature's - which is the failure this table exists for."""
    assert key_for(**override) != key_for()


def test_the_seven_field_variants_all_key_differently_from_each_other() -> None:
    """Catches a digest that collapses two fields into one input (`tenant + alias`
    concatenated, `model` written into the same slot as `alias`): each row above only
    proves the field is read at all, and two fields sharing a slot would still differ from
    the baseline while making a request for another tenant collide with one for another
    alias."""
    keys = {field: key_for(**override) for field, override in FIELD_ROWS}

    assert len(set(keys.values())) == len(FIELD_ROWS)
    assert key_for() not in set(keys.values())


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------


def test_object_key_order_does_not_change_the_key() -> None:
    """Catches a digest over `repr(...)` or over `json.dumps` without `sort_keys`: two
    clients sending the same request with their JSON fields in a different order are the
    same request, and a key that saw them as two would halve the hit rate for a reason no
    operator could ever find."""
    ordered = [{"role": "user", "content": "Which city is the capital of Germany?"}]
    reversed_keys = [{"content": "Which city is the capital of Germany?", "role": "user"}]

    assert key_for(messages=ordered) == key_for(messages=reversed_keys)


def test_nested_object_key_order_does_not_change_the_key_either() -> None:
    """Catches a canonicalisation applied only to the TOP level - `sort_keys` on an
    outer dict built from already-serialised strings: a tool's `parameters` schema is
    where the deep nesting actually is, so a shallow sort leaves the whole tools field
    order-sensitive while the case above passes."""
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    shuffled = {
        "function": {
            "parameters": {"properties": {"city": {"type": "string"}}, "type": "object"},
            "description": "Look up the weather",
            "name": "get_weather",
        },
        "type": "function",
    }

    assert key_for(tools=[tool]) == key_for(tools=[shuffled])


def test_message_order_does_change_the_key() -> None:
    """Catches an implementation that sorted its way to canonicalisation and sorted the
    messages too: the turns are a conversation, so "you are terse / write me an essay" and
    its reverse are two different requests, and serving one the other's answer is the
    worst wrong hit this cache can produce."""
    first = {"role": "user", "content": "Which city?"}
    second = {"role": "assistant", "content": "Berlin."}

    assert key_for(messages=[first, second]) != key_for(messages=[second, first])


def test_two_conversations_that_concatenate_alike_do_not_share_a_key() -> None:
    """Catches a key built by gluing the fields together without structure
    (`tenant + alias + "".join(contents)`): both requests below flatten to the same
    string, so one caller's two-turn conversation would be served the other's one-turn
    answer. No field differs between the two rows - only the boundary between them."""
    split = [{"role": "user", "content": "Which"}, {"role": "user", "content": "city?"}]
    joined = [{"role": "user", "content": "Whichcity?"}]

    assert key_for(messages=split) != key_for(messages=joined)


# --------------------------------------------------------------------------
# eligibility (#22, decisions D and H)
# --------------------------------------------------------------------------


def wire_body(**overrides: object) -> ChatCompletionRequest:
    """One parsed wire body, exactly as FastAPI hands it to the route."""
    body: dict[str, Any] = {
        "model": "agent-default",
        "messages": [{"role": "user", "content": "Which city?"}],
    }
    body.update(overrides)
    return ChatCompletionRequest.model_validate(body)


NAMED_TOOL_CHOICE = {"type": "function", "function": {"name": "get_weather"}}

ELIGIBILITY_ROWS: list[tuple[str, dict[str, Any], bool, bool]] = [
    ("temperature zero, no header", {"temperature": 0}, False, True),
    ("temperature zero as a float", {"temperature": 0.0}, False, True),
    ("no temperature at all, no header", {}, False, False),
    ("temperature one, no header", {"temperature": 1.0}, False, False),
    ("a temperature just above zero", {"temperature": 1e-09}, False, False),
    ("no temperature, but the allow header", {}, True, True),
    ("a non-zero temperature with the allow header", {"temperature": 0.7}, True, True),
    ("temperature zero on a stream", {"temperature": 0, "stream": True}, False, False),
    ("the allow header on a stream", {"stream": True}, True, False),
    (
        "tools with no tool_choice",
        {"temperature": 0, "tools": list(BASE_TOOLS)},
        False,
        False,
    ),
    (
        "tools with tool_choice auto",
        {"temperature": 0, "tools": list(BASE_TOOLS), "tool_choice": "auto"},
        False,
        False,
    ),
    (
        "tools with tool_choice required",
        {"temperature": 0, "tools": list(BASE_TOOLS), "tool_choice": "required"},
        False,
        False,
    ),
    (
        "tools with a named tool_choice",
        {"temperature": 0, "tools": list(BASE_TOOLS), "tool_choice": NAMED_TOOL_CHOICE},
        False,
        False,
    ),
    (
        "tools with tool_choice none",
        {"temperature": 0, "tools": list(BASE_TOOLS), "tool_choice": "none"},
        False,
        True,
    ),
    (
        "tool_choice auto with no tools offered",
        {"temperature": 0, "tool_choice": "auto"},
        False,
        True,
    ),
    (
        "tools with the allow header and no tool_choice",
        {"tools": list(BASE_TOOLS)},
        True,
        False,
    ),
]


@pytest.mark.parametrize(
    ("case", "overrides", "allow_header", "expected"),
    ELIGIBILITY_ROWS,
    ids=[row[0] for row in ELIGIBILITY_ROWS],
)
def test_the_eligibility_decision_table(
    case: str, overrides: dict[str, Any], allow_header: bool, expected: bool
) -> None:
    """Catches each of the four wrong predicates this rule invites, one row group each.

    An ABSENT temperature read as the provider's default zero (`temperature or 0`, or a
    model default of 0.0) would make every request the suite sends today cacheable, which
    is the criterion's own boundary; a `<= 0` or a `not temperature` comparison would take
    the just-above-zero row; a predicate that forgot the header would refuse the two allow
    rows; one that forgot `stream` would cache an answer that is never stored in the shape
    it was served (decision G); and one that refused tools outright would take the
    `tool_choice: none` row, while one that read `tool_choice` alone would take the row
    with tools and no choice at all (decision H)."""
    assert is_cacheable(wire_body(**overrides), allow_header=allow_header) is expected
