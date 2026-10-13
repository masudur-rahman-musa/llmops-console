"""Both directions of the Anthropic mapping, against recorded payloads.

`to_anthropic` and `from_anthropic` are pure, so every case here runs without a
transport; the two that must prove an ordering ("no outbound call happened
first") go through the adapter with a scripted transport that records requests.

The usage cases exist because the ledger prices from these numbers: a response
whose usage is missing must raise rather than book a zero-cost call.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from opsconsole.core.errors import ProviderError
from opsconsole.providers.anthropic import from_anthropic, to_anthropic
from opsconsole.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatMessage,
    ChatRequest,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from opsconsole.routing.errors import IdentifierInvalid

from .conftest import load_fixture, make_adapter, ok

MODEL = "claude-haiku-4-5"

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up the weather in a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
        "required": ["city"],
    },
)

TICKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "severity": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "severity"],
}


def request_with(overrides: Mapping[str, Any] | None = None) -> ChatRequest:
    """The baseline single-turn request, with the given fields changed per case."""
    fields: dict[str, Any] = {
        "model": MODEL,
        "messages": (ChatMessage(role="user", content="What is the capital of Germany?"),),
    }
    fields.update(overrides or {})
    return ChatRequest(**fields)


def test_system_message_is_lifted_to_the_top_level() -> None:
    """Catches an implementation that passes the system message through inside `messages`,
    which Anthropic rejects with a 400 because it has no `system` role."""
    request = request_with(
        {
            "messages": (
                ChatMessage(role="system", content="Answer in one word."),
                ChatMessage(role="user", content="What is the capital of Germany?"),
            )
        }
    )

    body = to_anthropic(request)

    assert body["system"] == "Answer in one word."
    assert [message["role"] for message in body["messages"]] == ["user"]


def test_no_system_message_omits_the_field_entirely() -> None:
    """Catches an implementation that always emits `system`, sending `null` or `""` and
    burning a field Anthropic validates rather than ignores."""
    body = to_anthropic(request_with())

    assert "system" not in body


MULTI_TURN = (
    ChatMessage(role="system", content="You are terse."),
    ChatMessage(role="user", content="Which city?"),
    ChatMessage(role="assistant", content="Berlin."),
    ChatMessage(role="user", content="And the country?"),
)


def test_multi_turn_order_is_preserved_after_the_system_lift() -> None:
    """Catches an implementation that groups messages by role or rebuilds the list from a
    dict, which would reorder the turns and change what the model is answering."""
    body = to_anthropic(request_with({"messages": MULTI_TURN}))

    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "Which city?"),
        ("assistant", "Berlin."),
        ("user", "And the country?"),
    ]


def test_model_and_defaults_are_carried_through() -> None:
    """Catches a mapper that sends an alias instead of the resolved provider model id,
    which would make Anthropic 404 on a name only our policy knows."""
    body = to_anthropic(request_with())

    assert body["model"] == MODEL


MAX_TOKENS_CASES: list[tuple[str, int | None, int]] = [
    ("omitted falls back to the default", None, DEFAULT_MAX_TOKENS),
    ("the caller's value is honoured", 77, 77),
    ("a caller value equal to the default", DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOKENS),
]


@pytest.mark.parametrize(
    ("case", "requested", "expected"),
    MAX_TOKENS_CASES,
    ids=[case for case, *_ in MAX_TOKENS_CASES],
)
def test_max_tokens_default_and_override(case: str, requested: int | None, expected: int) -> None:
    """Catches both halves: a mapper that omits `max_tokens` (Anthropic requires it, so
    every call 400s) and one that always sends 4096 and ignores the caller's ceiling."""
    body = to_anthropic(request_with({"max_tokens": requested}))

    assert body["max_tokens"] == expected, case


TEMPERATURE_CASES: list[tuple[str, float | None, bool]] = [
    ("unset is omitted", None, False),
    ("zero is still sent", 0.0, True),
    ("a set value is sent", 0.4, True),
]


@pytest.mark.parametrize(
    ("case", "temperature", "present"),
    TEMPERATURE_CASES,
    ids=[case for case, *_ in TEMPERATURE_CASES],
)
def test_temperature_is_sent_only_when_the_caller_set_it(
    case: str, temperature: float | None, present: bool
) -> None:
    """Catches a mapper testing `if request.temperature:` rather than `is not None`, which
    silently drops an explicit temperature of 0 - the setting an eval run depends on."""
    body = to_anthropic(request_with({"temperature": temperature}))

    assert ("temperature" in body) is present, case
    if present:
        assert body["temperature"] == temperature


def test_tools_render_in_the_anthropic_shape() -> None:
    """Catches a mapper that forwards the OpenAI tool shape (`parameters`, or a
    `{"type": "function", "function": {...}}` wrapper), which Anthropic rejects."""
    body = to_anthropic(request_with({"tools": (WEATHER_TOOL,)}))

    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "Look up the weather in a city.",
            "input_schema": dict(WEATHER_TOOL.parameters),
        }
    ]
    assert "tool_choice" not in body, "plain tools must not force a tool"


def test_tool_use_block_becomes_a_tool_call_with_json_string_arguments() -> None:
    """Catches an implementation that returns the raw dict as `arguments`: the published
    API is OpenAI-shaped, where `arguments` is a JSON *string* the client parses."""
    payload = load_fixture("tool_use")
    block = payload["content"][1]

    response = from_anthropic(payload)

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == block["id"]
    assert call.name == block["name"]
    assert isinstance(call.arguments, str), "arguments must be a JSON string, not a dict"
    assert json.loads(call.arguments) == block["input"]


def test_tools_round_trip_from_spec_to_call() -> None:
    """Catches a name mismatch across the two directions: a mapper that renames or
    namespaces the tool on the way out would return a `tool_calls` name the caller never
    declared and cannot dispatch."""
    body = to_anthropic(request_with({"tools": (WEATHER_TOOL,)}))
    sent_names = [tool["name"] for tool in body["tools"]]

    response = from_anthropic(load_fixture("tool_use"))

    assert sent_names == [WEATHER_TOOL.name]
    assert [call.name for call in response.tool_calls] == sent_names


def test_text_blocks_are_concatenated_into_content() -> None:
    """Catches an implementation that returns only the first text block, which truncates
    every answer Anthropic happens to split across blocks."""
    response = from_anthropic(load_fixture("multi_turn_system"))

    assert response.content == "Berlin is the capital."


def test_tool_use_answer_keeps_only_the_text_in_content() -> None:
    """Catches a mapper that stringifies the tool_use block into `content`, leaking the
    provider's block shape into the client-facing answer text."""
    response = from_anthropic(load_fixture("tool_use"))

    assert response.content == "Checking the weather."


def test_a_tool_only_answer_has_empty_content() -> None:
    """Catches a mapper that returns `None` for a tool-only answer, which would break the
    `content: str` contract the response model declares."""
    response = from_anthropic(load_fixture("json_schema_tool"))

    assert response.content == ""


def test_finish_reason_is_the_providers_stop_reason() -> None:
    """Catches a mapper that hardcodes "stop", losing the `tool_use` signal the caller
    needs to know an answer is a tool call rather than a final message."""
    assert from_anthropic(load_fixture("plain")).finish_reason == "end_turn"
    assert from_anthropic(load_fixture("tool_use")).finish_reason == "tool_use"


RESPONSE_FORMAT = ResponseFormat(name="ticket_summary", json_schema=TICKET_SCHEMA)


def test_response_format_renders_one_forced_tool() -> None:
    """Catches a mapper that sends the OpenAI `response_format` object straight through,
    or that adds the tool without `tool_choice`, leaving the model free to answer in prose
    and the JSON contract unenforced."""
    body = to_anthropic(request_with({"response_format": RESPONSE_FORMAT}))

    assert body["tool_choice"] == {"type": "tool", "name": "ticket_summary"}
    tools = body["tools"]
    assert len(tools) == 1, "the forced tool is the only tool on this request"
    assert tools[0]["name"] == "ticket_summary"
    assert tools[0]["input_schema"] == TICKET_SCHEMA
    assert "response_format" not in body, "our own field name must not travel to Anthropic"


def test_strict_never_changes_the_anthropic_body() -> None:
    """Catches an Anthropic mapper that starts carrying the caller's `strict` flag into the
    forced tool (Anthropic has no such field and would 400), or that branches on it: the
    outbound body must be byte-identical for true, false and absent."""
    bodies = {
        strict: json.dumps(
            to_anthropic(
                request_with(
                    {
                        "response_format": ResponseFormat.model_validate(
                            {
                                "name": "ticket_summary",
                                "json_schema": TICKET_SCHEMA,
                                "strict": strict,
                            }
                        )
                    }
                )
            )
        )
        for strict in (True, False, None)
    }

    assert bodies[True] == bodies[None]
    assert bodies[False] == bodies[None]
    assert "strict" not in bodies[None]


def test_response_format_is_added_alongside_caller_tools() -> None:
    """Catches a mapper that overwrites `tools` with the forced tool, which would drop the
    caller's own tools from a request that used both."""
    body = to_anthropic(
        request_with({"tools": (WEATHER_TOOL,), "response_format": RESPONSE_FORMAT})
    )

    assert [tool["name"] for tool in body["tools"]] == ["get_weather", "ticket_summary"]
    assert body["tool_choice"] == {"type": "tool", "name": "ticket_summary"}


def test_forced_tool_answer_maps_back_to_a_tool_call() -> None:
    """With no `forced_tool`, the block stays a tool call. Catches a mapper that turns a
    `tool_use` block into `content` by its shape alone: a caller that asked for no schema
    must get its call back, with arguments that parse to the recorded object."""
    payload = load_fixture("json_schema_tool")

    response = from_anthropic(payload)

    assert [call.name for call in response.tool_calls] == ["ticket_summary"]
    assert json.loads(response.tool_calls[0].arguments) == payload["content"][0]["input"]


# --- a response_format answer read back as content -----------------
#
# `to_anthropic` turns `response_format` into one forced tool, so Anthropic answers
# a schema request with a `tool_use` block. An OpenAI-compatible caller reads that
# answer from `message.content`; `forced_tool` names the block that is the answer.

FORCED_NAME = "ticket_summary"

# The recorded `json_schema_tool.json` input, written out so a fixture edited
# into agreement with a wrong mapper is visible here.
FORCED_INPUT: dict[str, Any] = {"title": "Card declined", "severity": 3, "tags": ["billing", "eu"]}
FORCED_CONTENT = json.dumps(FORCED_INPUT, sort_keys=True)


def test_a_forced_tool_answer_becomes_json_content() -> None:
    """Catches a mapper that ignores `forced_tool` and hands back the forced block as a
    tool call with empty content and `tool_use`, which an OpenAI-compatible caller reads
    as no answer at all."""
    payload = load_fixture("json_schema_tool")
    assert payload["content"][0]["input"] == FORCED_INPUT

    response = from_anthropic(payload, forced_tool=FORCED_NAME)

    assert response.content == FORCED_CONTENT
    assert json.loads(response.content) == FORCED_INPUT
    assert response.tool_calls == ()
    assert response.finish_reason == "end_turn"


def test_text_beside_a_forced_block_is_dropped() -> None:
    """Catches a mapper that joins the model's preamble text onto the JSON (decision B):
    content `"Here it is.{...}"` does not parse, and the caller asked for JSON."""
    payload = load_fixture("json_schema_tool")
    payload["content"].insert(0, {"type": "text", "text": "Here it is."})

    response = from_anthropic(payload, forced_tool=FORCED_NAME)

    assert response.content == FORCED_CONTENT
    assert response.tool_calls == ()
    assert response.finish_reason == "end_turn"


def test_a_forced_tool_without_its_block_keeps_the_text() -> None:
    """Catches a mapper that drops text whenever `forced_tool` is set rather than only when
    the forced block is present: a plain-text refusal would vanish, and the empty answer
    would then raise as unreadable."""
    response = from_anthropic(load_fixture("plain"), forced_tool=FORCED_NAME)

    assert response.content == "Ship it."
    assert response.tool_calls == ()
    assert response.finish_reason == "end_turn"


def test_a_caller_tool_beside_the_forced_block_stays_a_tool_call() -> None:
    """Catches a mapper that turns EVERY `tool_use` block into content once `forced_tool`
    is set, or that rewrites `tool_use` to `end_turn` while a real call remains: the
    caller would never run the tool the model asked for."""
    payload = load_fixture("json_schema_tool")
    weather = load_fixture("tool_use")["content"][1]
    payload["content"].append(weather)

    response = from_anthropic(payload, forced_tool=FORCED_NAME)

    assert response.content == FORCED_CONTENT
    assert response.tool_calls == (
        ToolCall(
            id="toolu_01WeatherCall00000000",
            name="get_weather",
            arguments=json.dumps({"city": "Berlin", "units": "metric"}, sort_keys=True),
        ),
    )
    assert response.finish_reason == "tool_use"


def test_a_forced_name_matching_no_block_changes_nothing() -> None:
    """Catches a mapper that converts whichever `tool_use` block it finds (by position, not
    by name) once `forced_tool` is set: the caller's `get_weather` call would come back as
    content and its preamble text would be dropped."""
    payload = load_fixture("tool_use")

    response = from_anthropic(payload, forced_tool=FORCED_NAME)

    assert response == from_anthropic(load_fixture("tool_use"))
    assert response.content == "Checking the weather."
    assert [call.name for call in response.tool_calls] == ["get_weather"]
    assert response.finish_reason == "tool_use"


def test_a_forced_answer_cut_at_max_tokens_keeps_its_stop_reason() -> None:
    """Catches a mapper that sets `end_turn` on every converted answer instead of rewriting
    only `tool_use`: a truncated JSON answer would reach the caller as `stop` rather than
    `length`, and it would never learn to raise its ceiling."""
    payload = load_fixture("json_schema_tool")
    payload["stop_reason"] = "max_tokens"

    response = from_anthropic(payload, forced_tool=FORCED_NAME)

    assert response.content == FORCED_CONTENT
    assert response.tool_calls == ()
    assert response.finish_reason == "max_tokens"


def test_forced_usage_is_unchanged() -> None:
    """Catches a forced path that builds its response on a side road and loses or swaps
    the usage numbers: the ledger prices from them, so a schema answer would bill wrong."""
    usage = from_anthropic(load_fixture("json_schema_tool"), forced_tool=FORCED_NAME).usage

    assert usage.prompt_tokens == 210
    assert usage.completion_tokens == 64
    assert usage.total_tokens == 274


TICKET_TOOL = ToolSpec(
    name=FORCED_NAME, description="File a ticket summary.", parameters=TICKET_SCHEMA
)

# Both rows send the SAME forced `tool_choice` to Anthropic: `to_anthropic` renders
# a `response_format` and a caller's named `tool_choice` identically on the wire.
# So only `request.response_format` can tell them apart.
CHAT_FORCED_CASES: list[tuple[str, dict[str, Any], bool]] = [
    ("a response_format request", {"response_format": RESPONSE_FORMAT}, True),
    (
        "a caller's own forced tool, no response_format",
        {"tools": (TICKET_TOOL,), "tool_choice": ToolChoice(mode="tool", name=FORCED_NAME)},
        False,
    ),
]


@pytest.mark.parametrize(
    ("case", "overrides", "as_content"),
    CHAT_FORCED_CASES,
    ids=[case for case, *_ in CHAT_FORCED_CASES],
)
def test_chat_passes_the_response_format_name_as_forced_tool(
    case: str, overrides: dict[str, Any], as_content: bool
) -> None:
    """Catches a `chat` that never passes `forced_tool` (the first row stays a tool call),
    and one that reads it from the outbound `tool_choice` instead of `response_format`:
    a caller who forced its own tool would get the call back as content."""
    adapter, scripted = make_adapter([ok("json_schema_tool")])

    response = adapter.chat(request_with(overrides))

    assert scripted.request_count == 1
    assert scripted.body()["tool_choice"] == {"type": "tool", "name": FORCED_NAME}, case
    if as_content:
        assert response.content == FORCED_CONTENT, case
        assert response.tool_calls == (), case
        assert response.finish_reason == "end_turn", case
    else:
        assert response.content == "", case
        assert [call.name for call in response.tool_calls] == [FORCED_NAME], case
        assert response.finish_reason == "tool_use", case


USAGE_CASES: list[tuple[str, str, int, int]] = [
    ("a plain answer", "plain", 41, 7),
    ("a multi-turn answer", "multi_turn_system", 128, 19),
    ("a tool_use answer", "tool_use", 88, 33),
    ("a forced-tool answer", "json_schema_tool", 210, 64),
]


@pytest.mark.parametrize(
    ("case", "fixture", "prompt", "completion"),
    USAGE_CASES,
    ids=[case for case, *_ in USAGE_CASES],
)
def test_usage_maps_with_total_as_the_sum(
    case: str, fixture: str, prompt: int, completion: int
) -> None:
    """Catches a mapper that copies Anthropic's numbers into the wrong fields, or that
    reports `total_tokens` as the input count alone - either way the ledger bills wrong."""
    usage = from_anthropic(load_fixture(fixture)).usage

    assert usage.prompt_tokens == prompt, case
    assert usage.completion_tokens == completion, case
    assert usage.total_tokens == prompt + completion, case


def missing_usage_payloads() -> list[tuple[str, dict[str, Any]]]:
    """The three ways usage can be unusable, built from the recorded payloads."""
    without_input = load_fixture("plain")
    del without_input["usage"]["input_tokens"]
    without_output = load_fixture("plain")
    del without_output["usage"]["output_tokens"]
    return [
        ("no usage block at all", load_fixture("no_usage")),
        ("no input_tokens", without_input),
        ("no output_tokens", without_output),
    ]


@pytest.mark.parametrize(
    ("case", "payload"),
    missing_usage_payloads(),
    ids=[case for case, _ in missing_usage_payloads()],
)
def test_a_response_without_usage_raises_provider_error(case: str, payload: dict[str, Any]) -> None:
    """Catches the silent-zero implementation (`payload.get("usage", {}).get(...) or 0`),
    which would book a real, billed call into the ledger at zero cost."""
    with pytest.raises(ProviderError) as raised:
        from_anthropic(payload)

    assert raised.value.retryable is False, f"{case}: a malformed body is not worth a retry"


BAD_TOOL_NAME_CASES: list[tuple[str, dict[str, Any]]] = [
    (
        "a caller tool with a trailing newline",
        {"tools": (ToolSpec(name="get_weather\n"),)},
    ),
    (
        "a caller tool with an interior space",
        {"tools": (ToolSpec(name="get weather"),)},
    ),
    (
        "a caller tool one over the length cap",
        {"tools": (ToolSpec(name="a" * 129),)},
    ),
    (
        # The forced tool name is caller-supplied too, and is the one an
        # implementation that only walks `request.tools` forgets to check.
        "a response_format name with a trailing space",
        {"response_format": ResponseFormat(name="ticket summary ", json_schema=TICKET_SCHEMA)},
    ),
]


@pytest.mark.parametrize(
    ("case", "overrides"), BAD_TOOL_NAME_CASES, ids=[case for case, _ in BAD_TOOL_NAME_CASES]
)
def test_bad_tool_name_raises_before_any_outbound_call(
    case: str, overrides: dict[str, Any]
) -> None:
    """Catches validation done after the request is sent (or not at all, leaving Anthropic
    to 400): the transport is scripted to answer 200, so the only thing that can make this
    raise is the pre-check, and `request_count == 0` proves it ran first."""
    adapter, scripted = make_adapter([ok("plain")])

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_with(overrides))

    assert raised.value.http_status == 422, case
    assert scripted.request_count == 0, f"{case}: the provider was called before validation"


def test_a_good_tool_name_still_reaches_the_provider() -> None:
    """Catches a pre-check so strict it rejects everything: without this row the case
    above would pass against `checked_tool_name` raising unconditionally."""
    adapter, scripted = make_adapter([ok("tool_use")])

    response = adapter.chat(request_with({"tools": (WEATHER_TOOL,)}))

    assert scripted.request_count == 1
    assert [call.name for call in response.tool_calls] == ["get_weather"]


def test_an_unknown_block_type_is_ignored() -> None:
    """Catches a mapper that raises on, or stringifies, a block type it does not know:
    Anthropic adds block types over time (`thinking`), and an answer whose text we can
    read must keep working when one appears beside it."""
    payload = load_fixture("plain")
    payload["content"].insert(0, {"type": "thinking", "thinking": "internal reasoning"})

    response = from_anthropic(payload)

    assert response.content == "Ship it."
    assert response.tool_calls == ()


def test_an_answer_with_no_readable_block_raises() -> None:
    """Catches the silent empty answer: a response of only unknown blocks would otherwise
    return `content=""` with real, billed usage, which is the zero-usage failure one field
    over - the tenant pays for an answer the mapper dropped."""
    payload = load_fixture("plain")
    payload["content"] = [{"type": "thinking", "thinking": "internal reasoning"}]

    with pytest.raises(ProviderError) as raised:
        from_anthropic(payload)

    assert raised.value.retryable is False


# --- the tool loop, request side ----------------------------------
#
# Anthropic carries a tool RESULT in a `user` turn holding a `tool_result` block,
# and a tool CALL in an `assistant` turn holding a `tool_use` block. Neither has
# a role of its own, so this is the mapping that makes an OpenAI-shaped tool loop
# expressible at all - and it is the one an implementation is most likely to get
# right in the response direction and forget in the request direction.

TOOL_CALL_ID = "toolu_01WeatherCall00000000"

WEATHER_CALL = ToolCall(id=TOOL_CALL_ID, name="get_weather", arguments='{"city": "Berlin"}')

TOOL_LOOP = (
    ChatMessage(role="system", content="You are terse."),
    ChatMessage(role="user", content="Weather in Berlin?"),
    ChatMessage(role="assistant", tool_calls=(WEATHER_CALL,)),
    ChatMessage(role="tool", content="18C, clear", tool_call_id=TOOL_CALL_ID),
)


def test_tool_result_turn_renders_as_a_user_turn_in_order() -> None:
    """Catches a mapper that passes the `tool` role straight through (Anthropic has no such
    role and 400s on it), or that appends the result somewhere other than immediately after
    the assistant turn that asked for it, which breaks the pairing the model reads."""
    body = to_anthropic(request_with({"messages": TOOL_LOOP}))

    assert body["system"] == "You are terse."
    assert body["messages"] == [
        {"role": "user", "content": "Weather in Berlin?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": TOOL_CALL_ID,
                    "name": "get_weather",
                    "input": {"city": "Berlin"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": TOOL_CALL_ID, "content": "18C, clear"}
            ],
        },
    ]


def test_a_tool_call_only_assistant_turn_has_no_text_block() -> None:
    """Catches a mapper that always prepends a text block: Anthropic rejects an empty
    `text`, so an assistant turn whose only answer was a tool call would 400 on every
    second request of a tool loop."""
    body = to_anthropic(request_with({"messages": (TOOL_LOOP[2],)}))

    assert body["messages"][0]["content"] == [
        {"type": "tool_use", "id": TOOL_CALL_ID, "name": "get_weather", "input": {"city": "Berlin"}}
    ]


def test_an_assistant_turn_with_text_and_a_call_keeps_both_in_order() -> None:
    """Catches a mapper that drops the assistant's text when tool calls are present, which
    loses the model's own words from the transcript it is shown next turn."""
    turn = ChatMessage(role="assistant", content="Checking.", tool_calls=(WEATHER_CALL,))

    body = to_anthropic(request_with({"messages": (turn,)}))

    assert [block["type"] for block in body["messages"][0]["content"]] == ["text", "tool_use"]
    assert body["messages"][0]["content"][0] == {"type": "text", "text": "Checking."}


def test_a_plain_assistant_turn_stays_a_plain_string() -> None:
    """Catches a mapper that wraps every assistant turn in a block list once tool calls
    exist: the existing multi-turn expectation is a string, and changing it would be a
    silent change to what every non-tool request sends."""
    body = to_anthropic(request_with({"messages": MULTI_TURN}))

    assert body["messages"][1] == {"role": "assistant", "content": "Berlin."}


# --- the ids on the way out -------------------------

AT_CAP_ID = "a" * 128
OVER_CAP_ID = "a" * 129


def test_tool_call_id_at_128_is_accepted() -> None:
    """Catches an exclusive boundary (`len(value) < 128`), which would refuse a tool call
    id that is exactly at the documented cap: the id is not ours to shorten - it is the one
    the provider issued and the client sent back."""
    adapter, scripted = make_adapter([ok("plain")])
    messages = (
        ChatMessage(
            role="assistant",
            tool_calls=(ToolCall(id=AT_CAP_ID, name="get_weather", arguments="{}"),),
        ),
        ChatMessage(role="tool", content="18C", tool_call_id=AT_CAP_ID),
    )

    adapter.chat(request_with({"messages": messages}))

    assert scripted.request_count == 1
    sent = scripted.body()
    assert sent["messages"][1]["content"][0]["tool_use_id"] == AT_CAP_ID


def test_tool_call_id_at_129_is_rejected() -> None:
    """Catches an id pre-check with no upper bound: a 129-character id would travel to the
    provider and come back as its 400, and the id is a client-supplied string that becomes
    an identifier on an outbound call."""
    adapter, scripted = make_adapter([ok("plain")])
    messages = (ChatMessage(role="tool", content="18C", tool_call_id=OVER_CAP_ID),)

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_with({"messages": messages}))

    assert raised.value.http_status == 422
    assert scripted.request_count == 0


# Each row carries exactly one bad id, with every other id on the request valid,
# so the row can only be rejected by the pre-check on the field it names. The
# newline and CRLF rows are the ones `re.match` with `^...$` would let through.
BAD_ID_CASES: list[tuple[str, tuple[ChatMessage, ...], str]] = [
    (
        "a tool_call_id with a trailing newline",
        (ChatMessage(role="tool", content="18C", tool_call_id=f"{TOOL_CALL_ID}\n"),),
        "tool_call_id",
    ),
    (
        "a tool_call_id with a trailing CRLF",
        (ChatMessage(role="tool", content="18C", tool_call_id=f"{TOOL_CALL_ID}\r\n"),),
        "tool_call_id",
    ),
    (
        "a tool_call_id with a trailing space",
        (ChatMessage(role="tool", content="18C", tool_call_id=f"{TOOL_CALL_ID} "),),
        "tool_call_id",
    ),
    (
        "a tool_call_id with a slash",
        (ChatMessage(role="tool", content="18C", tool_call_id="calls/1"),),
        "tool_call_id",
    ),
    (
        "a tool_call_id one over the cap",
        (ChatMessage(role="tool", content="18C", tool_call_id=OVER_CAP_ID),),
        "tool_call_id",
    ),
    (
        "a tool call id with a trailing newline",
        (
            ChatMessage(
                role="assistant",
                tool_calls=(ToolCall(id=f"{TOOL_CALL_ID}\n", name="get_weather", arguments="{}"),),
            ),
        ),
        "tool_calls",
    ),
    (
        "a tool call name with a trailing space",
        (
            ChatMessage(
                role="assistant",
                tool_calls=(ToolCall(id=TOOL_CALL_ID, name="get_weather ", arguments="{}"),),
            ),
        ),
        "tool_calls",
    ),
    (
        "a tool call name one over the cap",
        (
            ChatMessage(
                role="assistant",
                tool_calls=(ToolCall(id=TOOL_CALL_ID, name=OVER_CAP_ID, arguments="{}"),),
            ),
        ),
        "tool_calls",
    ),
]


@pytest.mark.parametrize(
    ("case", "messages", "field"), BAD_ID_CASES, ids=[case for case, *_ in BAD_ID_CASES]
)
def test_bad_tool_ids_raise_before_any_outbound_call(
    case: str, messages: tuple[ChatMessage, ...], field: str
) -> None:
    """Catches ids checked with `re.match` and `^...$`, checked after the request is built,
    or not checked at all: the transport is scripted to answer 200, so the pre-check is the
    only thing that can raise, and `request_count == 0` proves it ran first."""
    adapter, scripted = make_adapter([ok("plain")])

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_with({"messages": messages}))

    assert raised.value.http_status == 422, case
    assert scripted.request_count == 0, f"{case}: the provider was called before validation"
    named = f"{raised.value.message} {raised.value.param}"
    assert field in named, f"{case}: the error does not name the offending field"


def test_a_tool_turn_without_a_tool_call_id_raises() -> None:
    """Catches a mapper that renders `tool_use_id: null` when the caller omitted the id:
    Anthropic cannot pair the result with a call, and the failure would arrive as a
    provider 400 rather than as the caller's own 422."""
    adapter, scripted = make_adapter([ok("plain")])
    messages = (ChatMessage(role="tool", content="18C"),)

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_with({"messages": messages}))

    assert raised.value.http_status == 422
    assert scripted.request_count == 0


def test_tool_call_arguments_that_are_not_json_raise() -> None:
    """Catches a mapper that passes the arguments string through as Anthropic's `input`:
    `input` is an object, so a string would be rejected upstream - and a mapper that
    swallowed the parse failure would send `{}` and change what the model is told it did."""
    adapter, scripted = make_adapter([ok("plain")])
    messages = (
        ChatMessage(
            role="assistant",
            tool_calls=(ToolCall(id=TOOL_CALL_ID, name="get_weather", arguments="not json"),),
        ),
    )

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_with({"messages": messages}))

    assert raised.value.http_status == 422
    assert scripted.request_count == 0


# --- tool_choice -------------------------------------------------------------

TOOL_CHOICE_CASES: list[tuple[str, ToolChoice | None, dict[str, Any] | None]] = [
    ("unset sends no tool_choice", None, None),
    ("auto", ToolChoice(mode="auto"), {"type": "auto"}),
    ("none", ToolChoice(mode="none"), {"type": "none"}),
    # Anthropic spells "required" as "any". This row is the whole reason the
    # neutral contract keeps its own word for it.
    ("required becomes any", ToolChoice(mode="required"), {"type": "any"}),
    (
        "a named tool",
        ToolChoice(mode="tool", name="get_weather"),
        {"type": "tool", "name": "get_weather"},
    ),
]


@pytest.mark.parametrize(
    ("case", "choice", "expected"),
    TOOL_CHOICE_CASES,
    ids=[case for case, *_ in TOOL_CHOICE_CASES],
)
def test_tool_choice_maps_to_the_anthropic_table(
    case: str, choice: ToolChoice | None, expected: dict[str, Any] | None
) -> None:
    """Catches a mapper that forwards our own mode strings: Anthropic has no `required`,
    so a caller demanding a tool call would get a 400, or worse a free-form answer if the
    field were dropped. The `unset` row catches a mapper that always sends one."""
    body = to_anthropic(request_with({"tools": (WEATHER_TOOL,), "tool_choice": choice}))

    assert body.get("tool_choice") == expected, case


def test_tool_choice_name_is_checked_before_any_outbound_call() -> None:
    """Catches a forced tool name reaching Anthropic unchecked: it is caller-supplied and
    becomes an identifier on the outbound call, the same class of input as `tools[].name`
    and `response_format.name`."""
    adapter, scripted = make_adapter([ok("plain")])
    choice = ToolChoice(mode="tool", name="get_weather\n")

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_with({"tools": (WEATHER_TOOL,), "tool_choice": choice}))

    assert raised.value.http_status == 422
    assert scripted.request_count == 0


def test_a_named_tool_choice_with_no_name_raises_before_any_outbound_call() -> None:
    """Catches a mapper that renders `{"type": "tool", "name": null}` when the mode demands
    a named tool and no name was given: the contract allows the pair (`name` is optional,
    because it is meaningless for the other three modes), so the adapter is the only place
    it can be refused, and Anthropic would answer its own 400."""
    adapter, scripted = make_adapter([ok("plain")])

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(
            request_with({"tools": (WEATHER_TOOL,), "tool_choice": ToolChoice(mode="tool")})
        )

    assert raised.value.http_status == 422
    assert scripted.request_count == 0


def test_response_format_wins_over_tool_choice() -> None:
    """Catches an adapter that lets a plain `tool_choice` override the schema demand: the
    forced tool is what makes a JSON schema binding, and applying the caller's `auto` last
    would leave the model free to answer in prose. The wire boundary refuses the pair
    outright (`tests/gateway/test_chat_completions.py`); this is the adapter's tie-break."""
    body = to_anthropic(
        request_with(
            {
                "tools": (WEATHER_TOOL,),
                "tool_choice": ToolChoice(mode="auto"),
                "response_format": RESPONSE_FORMAT,
            }
        )
    )

    assert body["tool_choice"] == {"type": "tool", "name": "ticket_summary"}
