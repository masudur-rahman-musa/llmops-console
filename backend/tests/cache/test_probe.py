r"""The semantic probe: what is embedded, what is filtered on, and who may reach the tier.

`app.cache.probe` is pure - no store, no HTTP, nothing embedded here - so
every case in this file runs with no app and no backing service, exactly as
`tests/cache/test_key.py` does for the exact tier.

Three claims, each arranged so that only the control it names can decide the
outcome:

* THE PROBE TEXT is the FINAL USER message, not the final message and not the
  whole conversation. A trailing assistant turn and a differing earlier turn are
  separate rows, because an implementation reading `messages[-1]` passes the
  "changes when the question changes" assertion anyone would write first.
* THE TWO HASHES are payload filters, never part of the embedded text. The pairing case is the one
  that separates them: two requests with the same final question and a different system prompt share
  a probe text and must NOT share a system hash.
* ELIGIBILITY is four independent refusals: the alias must opt in, the request
  must not be streamed, tools must not be live, and the request must not name a
  response schema. Each row leaves the other three controls satisfied, so the
  row for the stream reddens only when the stream check is gone and never
  because the tool check happened to catch it. The schema refusal is pinned by
  the "a named response schema" row and by
  `test_the_response_format_alone_decides_an_otherwise_eligible_request`.

Contract this file assumes (the implementation follows it):

    SEMANTIC_THRESHOLD: Final[float]
    probe_text(body: ChatCompletionRequest) -> str | None
    system_hash(body: ChatCompletionRequest) -> str
    tools_hash(body: ChatCompletionRequest) -> str
    semantic_eligible(body: ChatCompletionRequest, *, alias: Alias) -> bool

`Alias` is `app.model_routing.policy.Alias` carrying the new
`semantic_cache: bool = False` field. The alias objects below are
built with `model_validate` from a dict rather than by keyword, so this file type
checks against today's `Alias` and reddens at runtime until the field lands.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from app.cache import NO_TOOL_CHOICE
from app.cache.probe import (
    SEMANTIC_THRESHOLD,
    probe_text,
    semantic_eligible,
    system_hash,
    tools_hash,
)
from app.model_routing.policy import Alias
from app.schemas.chat import ChatCompletionRequest

HEX_64 = re.compile(r"[0-9a-f]{64}")
"""What a sha256 rendered as lowercase hex looks like, the shape both hashes
return. Used with `fullmatch`."""

SYSTEM_TURN: dict[str, Any] = {"role": "system", "content": "You are terse."}
OTHER_SYSTEM_TURN: dict[str, Any] = {"role": "system", "content": "You are verbose."}
FIRST_QUESTION: dict[str, Any] = {
    "role": "user",
    "content": "Which city is the capital of Germany?",
}
SECOND_QUESTION: dict[str, Any] = {"role": "user", "content": "Which city is the capital of Spain?"}
ASSISTANT_TURN: dict[str, Any] = {"role": "assistant", "content": "Berlin."}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

OTHER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Look up the time",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

NAMED_TOOL_CHOICE: dict[str, Any] = {"type": "function", "function": {"name": "get_weather"}}

SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        # The wire key is `schema`; `WireResponseFormat`'s nested model field is
        # `json_schema`, aliased to it, so a body must say `schema` here.
        "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
"""One named response schema in its wire shape, the value `response_format` takes."""


def wire_body(**overrides: object) -> ChatCompletionRequest:
    """One parsed wire body, exactly as FastAPI hands it to the route.

    The default conversation is a single user turn with no system message and no
    tools, so a row that overrides one field is changing exactly that field.
    """
    body: dict[str, Any] = {"model": "agent-default", "messages": [FIRST_QUESTION]}
    body.update(overrides)
    return ChatCompletionRequest.model_validate(body)


def alias_with(*, semantic_cache: bool) -> Alias:
    """A resolved alias carrying the per-alias switch.

    Built through `model_validate` rather than by keyword so this file type checks
    before `Alias.semantic_cache` exists; the candidate list is the minimum the
    model requires and no case reads it.
    """
    return Alias.model_validate(
        {
            "candidates": [{"provider": "anthropic", "model": "claude-haiku-4-5"}],
            "semantic_cache": semantic_cache,
        }
    )


# --------------------------------------------------------------------------
# the threshold
# --------------------------------------------------------------------------


def test_the_semantic_threshold_is_exactly_the_published_one() -> None:
    """Catches a threshold moved to a plausible neighbour - 0.95, 0.98, or one read from
    settings so that an operator can move it - when the published value is a FIXED 0.97. A
    band would admit every one of those, and the number is what decides whether one
    tenant's question is served another question's answer, so it is pinned to the
    literal."""
    assert SEMANTIC_THRESHOLD == 0.97


# --------------------------------------------------------------------------
# the probe text: the final USER message
# --------------------------------------------------------------------------


def test_the_probe_text_is_the_final_user_message() -> None:
    """Catches a probe built from the final message alone (two chats answering "yes, that
    one please" would collide), and one built from the whole conversation: the probe is the
    turn right before the final user message, a blank line, then the final message."""
    body = wire_body(messages=[FIRST_QUESTION, ASSISTANT_TURN, SECOND_QUESTION])

    assert probe_text(body) == f"Berlin.\n\n{SECOND_QUESTION['content']}"


def test_the_probe_text_changes_when_the_final_user_message_changes() -> None:
    """Catches a probe that ignores the messages entirely - a constant, or the alias name:
    every entry would then sit at cosine 1.0 from every query and the tier would serve one
    question's answer to every other question asked."""
    first = wire_body(messages=[FIRST_QUESTION])
    second = wire_body(messages=[SECOND_QUESTION])

    assert probe_text(first) != probe_text(second)


def test_a_trailing_assistant_turn_is_not_the_probe_text() -> None:
    """Catches a probe read off `messages[-1]` rather than the final USER turn: a client
    replaying its history ends on an assistant turn, so the vector would be built from the
    gateway's own last answer and the lookup would compare answers to answers."""
    body = wire_body(messages=[FIRST_QUESTION, ASSISTANT_TURN])

    assert probe_text(body) == FIRST_QUESTION["content"]


def test_an_earlier_message_does_not_change_the_probe_text() -> None:
    """Catches a probe that concatenates the user turns: the two bodies below end on the same
    previous turn and final question after different histories before that, which is the
    pair the tier must treat as one probe. A different PREVIOUS turn now does change the
    probe, so only the turns before it may differ."""
    first = wire_body(messages=[FIRST_QUESTION, ASSISTANT_TURN, SECOND_QUESTION])
    second = wire_body(
        messages=[{"role": "user", "content": "Anything else"}, ASSISTANT_TURN, SECOND_QUESTION]
    )

    assert probe_text(first) == probe_text(second)


def test_a_request_with_no_user_turn_has_no_probe_text() -> None:
    """Catches an implementation indexing into an empty filter result: a body carrying only
    a system turn is legal on the wire, so `messages[-1]` or `[m for m in ... ][-1]` raises
    an `IndexError` out of a cache lookup and 500s a request the provider would have
    served. Absent is the answer, and the route reads it as "not a semantic candidate"."""
    body = wire_body(messages=[SYSTEM_TURN])

    assert probe_text(body) is None


# --------------------------------------------------------------------------
# the system hash: a filter, never part of the embedded text
# --------------------------------------------------------------------------

SYSTEM_ROWS: list[tuple[str, list[dict[str, Any]]]] = [
    ("no system message at all", [FIRST_QUESTION]),
    ("a system message", [SYSTEM_TURN, FIRST_QUESTION]),
]


@pytest.mark.parametrize(("case", "messages"), SYSTEM_ROWS, ids=[row[0] for row in SYSTEM_ROWS])
def test_the_system_hash_is_sixty_four_lowercase_hex_characters(
    case: str, messages: list[dict[str, Any]]
) -> None:
    """Catches a hash that answers `None`, `""` or a raw prompt when no system message is
    present: the value becomes a Qdrant payload field a filter matches on, so a `None`
    would make the filter match everything and a raw prompt would put caller text into the
    payload. The absent row is the one that separates "hashes a prompt" from "hashes what
    is there"."""
    value = system_hash(wire_body(messages=messages))

    assert HEX_64.fullmatch(value) is not None


def test_the_system_hash_is_the_same_twice_for_the_same_prompt() -> None:
    """Catches anything per-call in the digest - a salt, `id()`, a clock: every query would
    filter on a value no entry carries and the tier would never return a hit, which no
    "these two differ" case in this file can see."""
    body = wire_body(messages=[SYSTEM_TURN, FIRST_QUESTION])

    assert system_hash(body) == system_hash(wire_body(messages=[SYSTEM_TURN, FIRST_QUESTION]))
    assert system_hash(body) == system_hash(body)


def test_a_request_with_no_system_message_hashes_to_one_stable_value() -> None:
    """Catches an absent prompt hashed through something that varies with the rest of the
    body - the whole message list, the alias: two requests with no system prompt have to
    land on the SAME filter value or they can never read each other's entries, which is
    the ordinary case for every client that sends no system turn."""
    first = wire_body(messages=[FIRST_QUESTION])
    second = wire_body(messages=[SECOND_QUESTION, ASSISTANT_TURN, FIRST_QUESTION])

    assert system_hash(first) == system_hash(second)


def test_two_different_system_prompts_hash_differently() -> None:
    """Catches a hash over a constant, or over the PRESENCE of a system turn rather than
    its content: "you are terse" and "you are verbose" produce different answers to the
    same question, so sharing a filter value means one caller is served the other's answer
    (the whole reason for the filter)."""
    terse = wire_body(messages=[SYSTEM_TURN, FIRST_QUESTION])
    verbose = wire_body(messages=[OTHER_SYSTEM_TURN, FIRST_QUESTION])

    assert system_hash(terse) != system_hash(verbose)


def test_a_system_prompt_changes_the_system_hash_and_not_the_probe_text() -> None:
    """Catches the system prompt being glued into the EMBEDDED text instead of kept as a
    payload filter. Both halves are asserted at once, because an
    implementation that embedded the prompt would still pass the hash case above: it is the
    unchanged probe text that says the prompt never reached the vector."""
    without = wire_body(messages=[FIRST_QUESTION])
    with_prompt = wire_body(messages=[SYSTEM_TURN, FIRST_QUESTION])

    assert probe_text(without) == probe_text(with_prompt)
    assert system_hash(without) != system_hash(with_prompt)


# --------------------------------------------------------------------------
# the tools hash: absent, empty, populated
# --------------------------------------------------------------------------

TOOLS_ROWS: list[tuple[str, dict[str, Any]]] = [
    ("tools absent", {}),
    ("an empty tools list", {"tools": []}),
    ("a populated tools list", {"tools": TOOLS}),
]


@pytest.mark.parametrize(("case", "overrides"), TOOLS_ROWS, ids=[row[0] for row in TOOLS_ROWS])
def test_the_tools_hash_is_defined_for_every_shape_of_tools_field(
    case: str, overrides: dict[str, Any]
) -> None:
    """Catches a hash defined only for the populated case: `tools` is `None` when absent and
    an empty tuple when sent empty, so an implementation that dumps it unguarded raises a
    `TypeError` out of the lookup for the two shapes almost every request has, and 500s a
    request that only wanted a cache probe."""
    value = tools_hash(wire_body(**overrides))

    assert HEX_64.fullmatch(value) is not None


@pytest.mark.parametrize(("case", "overrides"), TOOLS_ROWS, ids=[row[0] for row in TOOLS_ROWS])
def test_the_tools_hash_is_the_same_twice(case: str, overrides: dict[str, Any]) -> None:
    """Catches a per-call digest, the same way the system hash's determinism case does: a
    filter value that changes between the write and the read makes every query a miss, and
    no "these two differ" case can see it."""
    assert tools_hash(wire_body(**overrides)) == tools_hash(wire_body(**overrides))


def test_a_populated_tools_list_hashes_differently_from_no_tools() -> None:
    """Catches a hash over a constant: an answer produced with tools on the table is not an
    answer to the same request made without them, and one filter value for both would serve
    it anyway."""
    populated = tools_hash(wire_body(tools=TOOLS))

    assert populated != tools_hash(wire_body())
    assert populated != tools_hash(wire_body(tools=[]))


def test_two_different_tool_lists_hash_differently() -> None:
    """Catches a hash over the COUNT of tools rather than their content - `len(tools)` or a
    digest of the wrapper type: both bodies below offer exactly one function, so a counting
    implementation passes the case above and still lets a weather-tool conversation read a
    time-tool conversation's answer."""
    assert tools_hash(wire_body(tools=TOOLS)) != tools_hash(wire_body(tools=OTHER_TOOLS))


# --------------------------------------------------------------------------
# eligibility: four independent refusals
# --------------------------------------------------------------------------

ELIGIBILITY_ROWS: list[tuple[str, bool, dict[str, Any], bool]] = [
    ("the alias flag off, nothing else in the way", False, {}, False),
    ("the alias flag on, nothing else in the way", True, {}, True),
    ("the alias flag on, a streamed request", True, {"stream": True}, False),
    ("the alias flag off, a streamed request", False, {"stream": True}, False),
    ("the alias flag on, tools with no tool_choice", True, {"tools": TOOLS}, False),
    (
        "the alias flag on, tools with tool_choice auto",
        True,
        {"tools": TOOLS, "tool_choice": "auto"},
        False,
    ),
    (
        "the alias flag on, tools with tool_choice required",
        True,
        {"tools": TOOLS, "tool_choice": "required"},
        False,
    ),
    (
        "the alias flag on, tools with a named tool_choice",
        True,
        {"tools": TOOLS, "tool_choice": NAMED_TOOL_CHOICE},
        False,
    ),
    (
        "the alias flag on, tools with tool_choice none",
        True,
        {"tools": TOOLS, "tool_choice": NO_TOOL_CHOICE},
        True,
    ),
    (
        "the alias flag off, tools with tool_choice none",
        False,
        {"tools": TOOLS, "tool_choice": NO_TOOL_CHOICE},
        False,
    ),
    (
        "the alias flag on, tool_choice auto with no tools offered",
        True,
        {"tool_choice": "auto"},
        True,
    ),
    ("the alias flag on, an empty tools list", True, {"tools": []}, True),
    ("the alias flag on, a non-zero temperature", True, {"temperature": 1.0}, True),
    ("the alias flag on, no temperature at all", True, {}, True),
    (
        "the alias flag on, a named response schema",
        True,
        {"response_format": SCHEMA},
        False,
    ),
    (
        "the alias flag on, tools with no tool_choice and a named response schema",
        True,
        {"tools": TOOLS, "response_format": SCHEMA},
        False,
    ),
]


@pytest.mark.parametrize(
    ("case", "semantic_cache", "overrides", "expected"),
    ELIGIBILITY_ROWS,
    ids=[row[0] for row in ELIGIBILITY_ROWS],
)
def test_the_semantic_eligibility_decision_table(
    case: str, semantic_cache: bool, overrides: dict[str, Any], expected: bool
) -> None:
    """Catches each of the five wrong predicates this rule invites, one row group each.

    A predicate that never reads the alias takes the two `flag off` rows, and "off by
    default per alias" would be a comment. One that forgot `stream` takes the streamed row,
    which is arranged with the flag ON and no tools so nothing else can refuse it. One that
    refused tools outright takes the `tool_choice: none` row, while one that read
    `tool_choice` alone takes the row with tools and no choice at all (the same treatment the exact
    tier gets). And one that reused `is_cacheable` wholesale takes the two temperature rows: this
    tier is not the exact tier, so a hot request whose alias opts in is a semantic candidate. And
    one that never reads `response_format` takes the bare schema row, where no tool and no stream is
    in the way, so a caller owed JSON of its own shape would be handed a neighbour's prose;
    the row that also offers tools is here because that body is LEGAL on the wire -
    `_supported` forbids only `tool_choice` beside a `response_format` - and it carries two
    refusals at once, which is why it is not the isolating case."""
    body = wire_body(**overrides)

    assert semantic_eligible(body, alias=alias_with(semantic_cache=semantic_cache)) is expected


def test_the_alias_flag_alone_decides_an_otherwise_eligible_request() -> None:
    """Catches an eligibility predicate that reads the alias only in combination with
    something else - `flag and temperature == 0`, or a flag consulted after a tool check
    that already answered. One body, two aliases, two answers: nothing but the flag differs
    between the two calls, so this is the row that removing the flag check cannot survive
    while the table's other rows are all decided by a second control as well."""
    body = wire_body()

    assert semantic_eligible(body, alias=alias_with(semantic_cache=True)) is True
    assert semantic_eligible(body, alias=alias_with(semantic_cache=False)) is False


def test_the_response_format_alone_decides_an_otherwise_eligible_request() -> None:
    """Catches the absence of the statement this case protects: `if body.response_format is
    not None: return False` in `semantic_eligible`, and any version of it folded into
    another control - a schema refused only alongside tools, or only on a streamed body.
    Two bodies, one alias flag, ON both times: nothing but `response_format` differs, so
    removing that clause is the only thing that flips the second answer. The table's other
    schema row cannot say this, because it offers tools as well and the tools refusal
    decides it either way."""
    without = wire_body()
    with_schema = wire_body(response_format=SCHEMA)

    assert semantic_eligible(without, alias=alias_with(semantic_cache=True)) is True
    assert semantic_eligible(with_schema, alias=alias_with(semantic_cache=True)) is False


# --------------------------------------------------------------------------
# the minimum length and the previous turn
# --------------------------------------------------------------------------


def final_message(text: str) -> ChatCompletionRequest:
    """A body whose only turn is one user message with this text."""
    return wire_body(messages=[{"role": "user", "content": text}])


def test_a_final_message_of_15_characters_has_no_probe_text() -> None:
    """Catches a minimum of 15 or less (an off-by-one that lets "the T14, please" through
    to be matched across chats): 15 characters is below the minimum and gives `None`."""
    assert probe_text(final_message("a" * 15)) is None


def test_a_final_message_of_exactly_16_characters_is_probed() -> None:
    """Catches a minimum of 17 or more, or a `<=` where `<` belongs: 16 is the first length
    that is looked up."""
    assert probe_text(final_message("a" * 16)) == "a" * 16


def test_spaces_do_not_count_toward_the_minimum() -> None:
    """Catches a length taken before trimming: "  yes please  " is 14 characters padded
    with spaces past 16, and only a trimmed count sees it is short. The padding is what
    makes the untrimmed length pass, so the trim alone decides this row."""
    padded = "   " + "yes please" + "    "
    assert len(padded) >= 16

    assert probe_text(final_message(padded)) is None


def test_the_probe_is_the_previous_turn_then_the_final_message() -> None:
    """Catches a separator other than a blank line, or the two parts swapped: the text is
    the previous turn, `\\n\\n`, then the final user message."""
    body = wire_body(messages=[FIRST_QUESTION, ASSISTANT_TURN, SECOND_QUESTION])

    assert probe_text(body) == "Berlin.\n\nWhich city is the capital of Spain?"


def test_the_same_reply_after_different_answers_probes_differently() -> None:
    """Catches a probe that ignores the previous turn: "yes, that one please" after two
    different assistant questions must give two probes, or one chat would be served the
    other's answer."""
    reply = {"role": "user", "content": "yes, that one please"}
    after_laptop = wire_body(
        messages=[FIRST_QUESTION, {"role": "assistant", "content": "The T14?"}, reply]
    )
    after_phone = wire_body(
        messages=[FIRST_QUESTION, {"role": "assistant", "content": "The Pixel?"}, reply]
    )

    assert probe_text(after_laptop) != probe_text(after_phone)


def test_only_the_last_1000_characters_of_the_previous_turn_are_kept() -> None:
    """Catches a previous turn kept whole, or its FIRST 1,000 characters kept: the turn
    below is 1,001 characters whose first character differs from the rest, so the kept
    part must be the last 1,000 (the odd first character is dropped)."""
    previous = "X" + "y" * 1000
    body = wire_body(messages=[{"role": "assistant", "content": previous}, SECOND_QUESTION])

    assert probe_text(body) == "y" * 1000 + "\n\n" + SECOND_QUESTION["content"]


def test_a_previous_turn_of_exactly_1000_characters_is_kept_whole() -> None:
    """Catches a cut that takes 999 or drops a character at the boundary: 1,000 characters
    is the inclusive limit and passes through unchanged."""
    previous = "z" * 1000
    body = wire_body(messages=[{"role": "assistant", "content": previous}, SECOND_QUESTION])

    assert probe_text(body) == previous + "\n\n" + SECOND_QUESTION["content"]


def test_a_tool_turn_or_a_call_only_turn_before_adds_nothing() -> None:
    """Catches a previous turn taken blindly from the list: a tool result and an assistant
    turn that holds only tool calls are not text turns, so each leaves the final message
    alone (and a `None` content must not be glued in as the word "None")."""
    call = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    after_tool = wire_body(
        messages=[
            FIRST_QUESTION,
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18C, clear"},
            SECOND_QUESTION,
        ]
    )
    after_call_only = wire_body(
        messages=[
            FIRST_QUESTION,
            {"role": "assistant", "content": None, "tool_calls": [call]},
            SECOND_QUESTION,
        ]
    )

    assert probe_text(after_tool) == SECOND_QUESTION["content"]
    assert probe_text(after_call_only) == SECOND_QUESTION["content"]


def test_turns_before_the_previous_one_do_not_change_the_probe() -> None:
    """Catches a probe that reaches back past the previous turn: the two bodies share their
    last two turns and differ at the start, so they must give one probe."""
    first = wire_body(messages=[FIRST_QUESTION, ASSISTANT_TURN, SECOND_QUESTION])
    second = wire_body(
        messages=[
            SYSTEM_TURN,
            {"role": "user", "content": "Something quite different entirely"},
            {"role": "assistant", "content": "Okay."},
            ASSISTANT_TURN,
            SECOND_QUESTION,
        ]
    )

    assert probe_text(first) == probe_text(second)
