"""Both directions of the Anthropic mapping, against recorded payloads.

`to_anthropic` and `from_anthropic` are pure, so every case here runs without a
transport; the two that must prove an ordering ("no outbound call happened
first") go through the adapter with a scripted transport that records requests.

The usage cases exist because the ledger prices from these numbers: a response
whose usage is missing must raise rather than book a zero-cost call.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any

import pytest

from app.core.errors import ProviderError
from app.core.validation import IdentifierInvalid
from app.providers import anthropic as anthropic_module
from app.providers.anthropic import from_anthropic, from_anthropic_events, to_anthropic
from app.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatMessage,
    ChatRequest,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    ToolSpec,
)

from .conftest import SleepRecorder, fail, load_fixture, make_adapter, ok

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


CACHE_MARK: dict[str, str] = {"type": "ephemeral"}
"""The marker Anthropic reads as "cache the request up to and including this block"."""

TIME_TOOL = ToolSpec(
    name="get_time",
    description="Look up the local time in a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def marked_text(text: str) -> list[dict[str, Any]]:
    """The `system` value the adapter sends: one text block carrying the cache marker."""
    return [{"type": "text", "text": text, "cache_control": dict(CACHE_MARK)}]


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

    assert body["system"] == marked_text("Answer in one word.")
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
        ("user", marked_text("And the country?")),
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
            "cache_control": dict(CACHE_MARK),
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

# What a `response_format` becomes on the wire: Anthropic's native
# structured outputs, carrying the caller's schema exactly as sent.
NATIVE_OUTPUT_CONFIG: dict[str, Any] = {
    "format": {"type": "json_schema", "schema": TICKET_SCHEMA},
}


def test_response_format_renders_output_config() -> None:
    """Catches a mapper that still fakes the schema with a forced tool, that forwards the
    OpenAI `response_format` object as it is, or that rewrites the caller's schema (adding
    `additionalProperties`, say) against the spec: `TICKET_SCHEMA` has no such key, so
    any rewrite shows up as an unequal dict."""
    body = to_anthropic(request_with({"response_format": RESPONSE_FORMAT}))

    assert body["output_config"] == NATIVE_OUTPUT_CONFIG
    assert body["output_config"]["format"]["schema"] == TICKET_SCHEMA
    assert "response_format" not in body, "our own field name must not travel to Anthropic"


def test_response_format_sends_no_tools_or_tool_choice() -> None:
    """Catches a mapper that still adds the schema as a tool or still forces `tool_choice`
    beside `output_config`, and one that sends an empty `tools: []`: the request names no
    caller tools, so either key on the wire can only have come from `response_format`."""
    body = to_anthropic(request_with({"response_format": RESPONSE_FORMAT}))

    assert "tools" not in body
    assert "tool_choice" not in body


def test_response_format_keeps_only_the_callers_tools() -> None:
    """Catches a mapper that still appends the schema tool to the caller's own tools, or
    that forces a `tool_choice` the caller never sent: the caller sent `get_weather` and no
    `tool_choice`, so that one tool and no `tool_choice` is the whole correct answer."""
    body = to_anthropic(
        request_with({"tools": (WEATHER_TOOL,), "response_format": RESPONSE_FORMAT})
    )

    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "Look up the weather in a city.",
            "input_schema": dict(WEATHER_TOOL.parameters),
            "cache_control": dict(CACHE_MARK),
        }
    ]
    assert "tool_choice" not in body
    assert body["output_config"] == NATIVE_OUTPUT_CONFIG


def test_a_bad_response_format_name_is_refused_before_any_call() -> None:
    """Catches the name check dropped along with the schema tool: `output_config` carries
    no name, so an implementation deleting the forced tool can easily delete its
    `checked_tool_name` too. The transport answers 200 and the name is the only bad field,
    so only that check can raise, and `param` names it."""
    adapter, scripted = make_adapter([ok("plain")])
    bad = ResponseFormat(name="ticket_summary\n", json_schema=TICKET_SCHEMA)

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_with({"response_format": bad}))

    assert raised.value.http_status == 422
    assert raised.value.param == "response_format.name"
    assert scripted.request_count == 0, "the provider was called before validation"


NATIVE_JSON_FIXTURES = ["json_output_opus5", "json_output_haiku45"]

# The keys of the schema the three live fixtures were recorded with.
RECORDED_JSON_KEYS = {"calls", "reason"}


@pytest.mark.parametrize("fixture", NATIVE_JSON_FIXTURES, ids=NATIVE_JSON_FIXTURES)
def test_a_native_json_answer_becomes_content(fixture: str) -> None:
    """Catches a reader that sends a native JSON answer anywhere but `content` (into a
    tool call, or joined with a thinking block's text), one that rewrites `end_turn`, and
    one that misreads usage: the expected numbers are read off the recording itself, and
    the ledger prices from them."""
    payload = load_fixture(fixture)
    texts = [block["text"] for block in payload["content"] if block["type"] == "text"]
    assert len(texts) == 1, f"{fixture}: the recording must hold exactly one text block"
    recorded_input = payload["usage"]["input_tokens"]
    recorded_output = payload["usage"]["output_tokens"]

    response = from_anthropic(payload)

    assert response.content == texts[0]
    assert set(json.loads(response.content)) == RECORDED_JSON_KEYS
    assert response.tool_calls == ()
    assert response.finish_reason == "end_turn"
    assert response.usage.prompt_tokens == recorded_input
    assert response.usage.completion_tokens == recorded_output
    assert response.usage.total_tokens == recorded_input + recorded_output


def test_an_output_config_400_is_not_retried(sleeps: SleepRecorder) -> None:
    """Catches an adapter that answers a refused `output_config` by trying again with the
    old forced tool: the script serves a 200 after the 400, so any second call would turn
    the raise into a success, and every body that went out is checked for a forced tool."""
    adapter, scripted = make_adapter([fail(400, "error_400"), ok("plain")], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(request_with({"response_format": RESPONSE_FORMAT}))

    assert raised.value.retryable is False
    assert scripted.request_count == 1, "a 400 is not retried, with or without a forced tool"
    assert sleeps.calls == []
    for index in range(scripted.request_count):
        sent = scripted.body(index)
        assert sent["output_config"] == NATIVE_OUTPUT_CONFIG
        assert "tools" not in sent
        assert "tool_choice" not in sent


def test_chat_sends_no_beta_header_for_response_format() -> None:
    """Catches an adapter that adds an `anthropic-beta` structured-outputs header: the
    feature is generally available, and a beta header pins the request to a preview
    contract. The body check makes sure this really was a native-output request."""
    adapter, scripted = make_adapter([ok("plain")])

    adapter.chat(request_with({"response_format": RESPONSE_FORMAT}))

    assert scripted.request_count == 1
    assert "anthropic-beta" not in scripted.requests[0].headers
    assert scripted.body()["output_config"] == NATIVE_OUTPUT_CONFIG


def test_the_readers_take_no_forced_tool() -> None:
    """Catches the unwrap left in place behind its keyword or its helpers: any
    caller still passing a name would have a real tool call turned into content, and the
    dead path would keep a second meaning of `tool_use` alive in the reader."""
    assert "forced_tool" not in inspect.signature(from_anthropic).parameters
    assert "forced_tool" not in inspect.signature(from_anthropic_events).parameters
    assert not hasattr(anthropic_module, "_forced_tool")
    assert not hasattr(anthropic_module, "_forced_stop_reason")


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
        # The response_format name is caller-supplied too, and is the one an
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

    assert body["system"] == marked_text("You are terse.")
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
                {
                    "type": "tool_result",
                    "tool_use_id": TOOL_CALL_ID,
                    "content": "18C, clear",
                    "cache_control": dict(CACHE_MARK),
                }
            ],
        },
    ]


def test_a_tool_call_only_assistant_turn_has_no_text_block() -> None:
    """Catches a mapper that always prepends a text block: Anthropic rejects an empty
    `text`, so an assistant turn whose only answer was a tool call would 400 on every
    second request of a tool loop."""
    body = to_anthropic(request_with({"messages": (TOOL_LOOP[2],)}))

    assert body["messages"][0]["content"] == [
        {
            "type": "tool_use",
            "id": TOOL_CALL_ID,
            "name": "get_weather",
            "input": {"city": "Berlin"},
            "cache_control": dict(CACHE_MARK),
        }
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


CALLER_CHOICE_CASES: list[tuple[str, ToolChoice, dict[str, Any]]] = [
    ("auto", ToolChoice(mode="auto"), {"type": "auto"}),
    ("required", ToolChoice(mode="required"), {"type": "any"}),
    (
        "a named caller tool",
        ToolChoice(mode="tool", name="get_weather"),
        {"type": "tool", "name": "get_weather"},
    ),
]


@pytest.mark.parametrize(
    ("case", "choice", "expected"),
    CALLER_CHOICE_CASES,
    ids=[case for case, *_ in CALLER_CHOICE_CASES],
)
def test_a_callers_tool_choice_passes_through_beside_response_format(
    case: str, choice: ToolChoice, expected: dict[str, Any]
) -> None:
    """Catches an adapter that still lets `response_format` override the caller's
    `tool_choice` with a forced schema tool: the schema now travels in `output_config`, so
    the caller's choice is the only `tool_choice` there is. The wire refuses the pair
    (`tests/gateway/test_chat_completions.py`); this is the adapter's own mapping."""
    body = to_anthropic(
        request_with(
            {
                "tools": (WEATHER_TOOL,),
                "tool_choice": choice,
                "response_format": RESPONSE_FORMAT,
            }
        )
    )

    assert body["tool_choice"] == expected, case
    assert [tool["name"] for tool in body["tools"]] == ["get_weather"], case
    assert body["output_config"] == NATIVE_OUTPUT_CONFIG, case


# --- prompt caching ---------------------------------------------------
#
# `system` and the tool list are the long, repeated start of every agent turn.
# Anthropic caches tools then system as one prefix, so two markers cover both: one
# on the system block and one on the LAST tool.


def cached_prefix_request() -> ChatRequest:
    """The request `request_cached_prefix.json` was written for: a system turn, one user
    turn and two tools."""
    return request_with(
        {
            "messages": (
                ChatMessage(role="system", content="You are terse."),
                ChatMessage(role="user", content="Weather in Berlin?"),
            ),
            "tools": (WEATHER_TOOL, TIME_TOOL),
        }
    )


def test_a_request_with_system_and_tools_matches_the_recorded_shape() -> None:
    """Catches any drift of the whole body from the recorded expectation: a marker on every
    tool, a marker missing from the system block, a `system` left as a string, or an extra
    field. The fixture is the one place the wire shape is written down in full."""
    body = to_anthropic(cached_prefix_request())

    assert body == load_fixture("request_cached_prefix")


def test_the_system_prompt_is_one_text_block_marked_for_caching() -> None:
    """Catches an implementation that leaves `system` a plain string (nothing is cached),
    or that puts the marker on a block of its own beside the text: the marker must sit on
    the one text block that carries the prompt."""
    body = to_anthropic(cached_prefix_request())

    assert body["system"] == [
        {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}
    ]


def test_several_system_messages_still_make_one_marked_block() -> None:
    """Catches an implementation that emits one marked block per system message: Anthropic
    allows four markers in all, so a long list of system turns would pass the limit, and
    the joined text of the existing mapping would be split."""
    body = to_anthropic(
        request_with(
            {
                "messages": (
                    ChatMessage(role="system", content="First rule."),
                    ChatMessage(role="system", content="Second rule."),
                    ChatMessage(role="user", content="Go."),
                )
            }
        )
    )

    assert body["system"] == marked_text("First rule.\n\nSecond rule.")


def test_only_the_last_tool_is_marked_for_caching() -> None:
    """Catches an implementation that marks every tool (four tools would already use all
    four markers Anthropic allows) and one that marks the first tool, which would cache
    only part of the list."""
    body = to_anthropic(cached_prefix_request())

    assert [tool["name"] for tool in body["tools"]] == ["get_weather", "get_time"]
    assert "cache_control" not in body["tools"][0]
    assert body["tools"][1]["cache_control"] == {"type": "ephemeral"}


def test_no_system_and_no_tools_sends_no_cache_marker() -> None:
    """Catches an implementation that always emits `system` and `tools` with a marker, or
    that sends a marker with nothing under it: Anthropic refuses an empty text block, so a
    plain question with no system prompt would fail on every call. The one mark left is
    the newest message's, so a second mark anywhere is caught too."""
    body = to_anthropic(request_with())

    assert "system" not in body
    assert "tools" not in body
    assert json.dumps(body).count("cache_control") == 1
    assert body["messages"][0]["content"] == marked_text("What is the capital of Germany?")


def test_cache_counts_are_read_from_the_reply() -> None:
    """Catches a reader that ignores `cache_read_input_tokens` (the ledger then prices a
    cached prefix as nothing), one that adds the cached tokens into `prompt_tokens` (they
    would be billed at the full input rate as well as their own), and one that swaps the
    read and write fields."""
    usage = from_anthropic(load_fixture("cache_hit")).usage

    assert usage.cache_read_tokens == 2048
    assert usage.cache_write_tokens == 0
    assert usage.prompt_tokens == 12


def test_a_reply_without_cache_fields_reads_zero() -> None:
    """Catches a reader that requires the cache fields and raises on every reply that does
    not carry them (most replies, and every older recording), or that fills them with
    something other than zero."""
    usage = from_anthropic(load_fixture("plain")).usage

    assert usage.cache_read_tokens == 0
    assert usage.cache_write_tokens == 0


NEGATIVE_CACHE_FIELDS = ["cache_read_input_tokens", "cache_creation_input_tokens"]


@pytest.mark.parametrize("field", NEGATIVE_CACHE_FIELDS, ids=NEGATIVE_CACHE_FIELDS)
def test_a_negative_cache_count_is_unreadable(field: str) -> None:
    """Catches a reader that passes a negative count on (the ledger would book a credit) or
    clamps it to zero (a malformed body would book a real call without saying so), and one
    that lets the schema's own error escape instead of the typed, non-retryable
    `ProviderError`. Both fields have a row, so a check on only one is visible."""
    payload = load_fixture("cache_hit")
    payload["usage"][field] = -1

    with pytest.raises(ProviderError) as raised:
        from_anthropic(payload)

    assert raised.value.retryable is False


# --- the newest message is marked too -----------------------


def test_the_newest_message_carries_the_cache_mark() -> None:
    """Catches an implementation that leaves the chat history unmarked (it is re-billed at
    full price every call), or marks every turn (older turns would use up the four marks
    Anthropic allows): a plain string newest turn becomes one marked text block and the
    earlier turns stay as they were."""
    body = to_anthropic(request_with({"messages": MULTI_TURN}))

    assert body["messages"][-1] == {"role": "user", "content": marked_text("And the country?")}
    assert body["messages"][0] == {"role": "user", "content": "Which city?"}
    assert body["messages"][1] == {"role": "assistant", "content": "Berlin."}


def test_a_newest_tool_result_marks_its_tool_result_block() -> None:
    """Catches an implementation that wraps a tool result in an extra text block to mark it,
    or marks the turn rather than the block: the mark must sit on the `tool_result` block
    itself."""
    body = to_anthropic(request_with({"messages": TOOL_LOOP}))

    assert body["messages"][-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": TOOL_CALL_ID,
            "content": "18C, clear",
            "cache_control": dict(CACHE_MARK),
        }
    ]
    assert "cache_control" not in body["messages"][1]["content"][0]


def test_a_newest_text_and_call_turn_marks_only_its_last_block() -> None:
    """Catches an implementation that marks every block of the newest turn (two marks for
    one turn) or the first one: the text block stays plain and only the closing `tool_use`
    carries the mark."""
    turn = ChatMessage(role="assistant", content="Checking.", tool_calls=(WEATHER_CALL,))

    body = to_anthropic(request_with({"messages": (turn,)}))

    text_block, call_block = body["messages"][0]["content"]
    assert text_block == {"type": "text", "text": "Checking."}
    assert call_block["type"] == "tool_use"
    assert call_block["cache_control"] == CACHE_MARK


def test_system_tools_and_newest_message_make_three_marks() -> None:
    """Catches an implementation that adds the newest-message mark and drops or doubles
    another (system, last tool), or marks a system turn as the newest message: the body
    carries exactly three marks, one per place."""
    body = to_anthropic(cached_prefix_request())

    assert json.dumps(body).count("cache_control") == 3
    assert body["messages"][-1]["content"][-1]["cache_control"] == CACHE_MARK
