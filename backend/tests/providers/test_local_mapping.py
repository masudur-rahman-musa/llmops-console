"""Both directions of the OpenAI-compatible mapping, against hand-written payloads.

`to_openai` and `from_openai` are pure, so every case here runs without a
transport; the ones that must prove an ordering ("no outbound call happened
first") go through the adapter with a scripted transport that records requests.

The fixtures under `tests/providers/fixtures/local/` are HAND-WRITTEN to the
published OpenAI chat-completions wire shape, not recorded from a llama.cpp
server: none was available when these tests were written. That is a deliberate
deviation from the plan's section 5, and it is stated here and in each `.sse`
file so the next reader does not take them for a recording.

The mapping is deliberately NOT the Anthropic one. A `system` turn stays a
`system` role message in place, `tools` carry a `{"type": "function"}` wrapper,
a `response_format` stays a `response_format`, and a tool call's `arguments`
travel as the JSON string the caller sent. Every case below that looks obvious
is there because the adapter next to this one does the opposite.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from app.core.errors import ProviderError
from app.core.validation import IdentifierInvalid
from app.providers.base import (
    DEFAULT_MAX_TOKENS,
    MAX_TOOL_NAME_LENGTH,
    ChatMessage,
    ChatRequest,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from app.providers.openai_compatible import (
    OPENAI_TOOL_CHOICE_BY_MODE,
    from_openai,
    to_openai,
)

from .conftest import load_local_fixture, local_ok, make_local_adapter

MODEL = "qwen2.5-7b-instruct"

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

RESPONSE_FORMAT = ResponseFormat(name="ticket_summary", json_schema=TICKET_SCHEMA)

TOOL_CALL_ID = "call_local_weather_0001"

# Deliberately not what `json.dumps` produces: two spaces after the colon. A
# mapper that parses and re-serialises the arguments normalises them away, which
# is how `test_tool_call_arguments_travel_as_the_caller_wrote_them` sees it.
ODD_ARGUMENTS = '{"city":  "Berlin"}'

WEATHER_CALL = ToolCall(id=TOOL_CALL_ID, name="get_weather", arguments=ODD_ARGUMENTS)


def request_with(overrides: Mapping[str, Any] | None = None) -> ChatRequest:
    """The baseline single-turn request, with the given fields changed per case."""
    fields: dict[str, Any] = {
        "model": MODEL,
        "messages": (ChatMessage(role="user", content="What is the capital of Germany?"),),
    }
    fields.update(overrides or {})
    return ChatRequest(**fields)


# --- the request direction ---------------------------------------------------


def test_the_system_turn_stays_a_system_role_message_in_place() -> None:
    """Catches a mapper copied from the Anthropic adapter, which lifts the system turn out
    of `messages` into a top-level `system` field: an OpenAI-compatible server has a
    `system` role and would silently answer with no instruction at all."""
    request = request_with(
        {
            "messages": (
                ChatMessage(role="system", content="Answer in one word."),
                ChatMessage(role="user", content="What is the capital of Germany?"),
            )
        }
    )

    body = to_openai(request)

    assert "system" not in body, "the system turn must not become a top-level field"
    assert body["messages"] == [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "What is the capital of Germany?"},
    ]


MULTI_TURN = (
    ChatMessage(role="system", content="You are terse."),
    ChatMessage(role="user", content="Which city?"),
    ChatMessage(role="assistant", content="Berlin."),
    ChatMessage(role="user", content="And the country?"),
)


def test_multi_turn_order_is_preserved_including_the_system_turn() -> None:
    """Catches a mapper that groups messages by role, sorts them, or moves the system turn
    to the front or the back: the model would be answering a different conversation from
    the one the caller sent."""
    body = to_openai(request_with({"messages": MULTI_TURN}))

    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("system", "You are terse."),
        ("user", "Which city?"),
        ("assistant", "Berlin."),
        ("user", "And the country?"),
    ]


def test_the_resolved_provider_model_id_is_sent() -> None:
    """Catches a mapper that sends the alias instead of the resolved provider model id,
    which the local server answers with a 404 on a name only our policy knows."""
    body = to_openai(request_with())

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
    """Catches both halves: a mapper that omits `max_tokens` when the caller did (an
    OpenAI-compatible server treats that as unbounded, and an unbounded generation on a
    CPU box holds a worker for minutes) and one that always sends the default, ignoring
    the caller's own ceiling."""
    body = to_openai(request_with({"max_tokens": requested}))

    assert body["max_tokens"] == expected, case


def test_the_default_max_tokens_is_exactly_4096() -> None:
    """Catches the cap drifting: the rows above are written in terms of the constant, so
    they follow it rather than pinning it, and this is the acceptance criterion's number."""
    assert DEFAULT_MAX_TOKENS == 4096


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
    body = to_openai(request_with({"temperature": temperature}))

    assert ("temperature" in body) is present, case
    if present:
        assert body["temperature"] == temperature


def test_tools_render_inside_the_function_wrapper() -> None:
    """Catches a mapper that sends the neutral tool shape (a bare `name`, `description`
    and `parameters`) or Anthropic's `input_schema`: an OpenAI-compatible server rejects
    both, so every request carrying a tool would 400."""
    body = to_openai(request_with({"tools": (WEATHER_TOOL,)}))

    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather in a city.",
                "parameters": dict(WEATHER_TOOL.parameters),
            },
        }
    ]
    assert "tool_choice" not in body, "plain tools must not force a tool"


def test_no_tools_omits_the_field_entirely() -> None:
    """Catches a mapper that always emits `tools: []`, which some OpenAI-compatible
    servers validate rather than ignore, turning every plain completion into a 400."""
    body = to_openai(request_with())

    assert "tools" not in body


def response_format_with(strict: object) -> ResponseFormat:
    """`RESPONSE_FORMAT` with the caller's own `strict` flag set.

    Built through `model_validate` so the case reads the same whether or not the
    type checker has seen the field yet; a contract without `strict` rejects it
    here, since `_Contract` forbids unknown fields.
    """
    return ResponseFormat.model_validate(
        {"name": "ticket_summary", "json_schema": TICKET_SCHEMA, "strict": strict}
    )


def test_response_format_without_strict_renders_a_plain_json_schema() -> None:
    """Catches a mapper that turns the schema into a forced tool the way the Anthropic
    adapter has to (the answer would come back as a tool call the caller never declared),
    and catches a mapper that adds `strict: true` on the caller's behalf, which OpenAI
    answers with a 400 for any schema with an optional field."""
    body = to_openai(request_with({"response_format": RESPONSE_FORMAT}))

    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "ticket_summary", "schema": TICKET_SCHEMA},
    }
    assert "tools" not in body, "a response format is not a tool on this wire"
    assert "tool_choice" not in body


def test_response_format_forwards_strict_true() -> None:
    """Catches a mapper that drops the caller's `strict: true`, which turns a schema the
    caller made binding into a suggestion the model may ignore."""
    body = to_openai(request_with({"response_format": response_format_with(True)}))

    assert body["response_format"]["json_schema"] == {
        "name": "ticket_summary",
        "schema": TICKET_SCHEMA,
        "strict": True,
    }


def test_response_format_forwards_strict_false() -> None:
    """Catches a mapper that writes `strict` only when it is truthy (`if strict:`), so a
    caller's explicit `false` is dropped instead of sent as given."""
    body = to_openai(request_with({"response_format": response_format_with(False)}))

    assert "strict" in body["response_format"]["json_schema"]
    assert body["response_format"]["json_schema"]["strict"] is False


def test_response_format_without_strict_sends_no_strict_key() -> None:
    """Catches a mapper that sends `strict: null` or a default value when the caller left
    the flag out, instead of leaving the key off and letting OpenAI apply its own default."""
    body = to_openai(request_with({"response_format": RESPONSE_FORMAT}))

    assert "strict" not in body["response_format"]["json_schema"]


def test_response_format_leaves_the_callers_tools_alone() -> None:
    """Catches a mapper that appends the schema to `tools` or overwrites `tool_choice`
    when both are set: the caller's own tools would either vanish or stop being callable
    on the one request that used both."""
    body = to_openai(request_with({"tools": (WEATHER_TOOL,), "response_format": RESPONSE_FORMAT}))

    assert [tool["function"]["name"] for tool in body["tools"]] == ["get_weather"]
    assert body["response_format"]["json_schema"]["name"] == "ticket_summary"


# --- tool_choice -------------------------------------------------------------

TOOL_CHOICE_CASES: list[tuple[str, ToolChoice | None, str | dict[str, Any] | None]] = [
    ("unset sends no tool_choice", None, None),
    ("auto", ToolChoice(mode="auto"), "auto"),
    ("none", ToolChoice(mode="none"), "none"),
    # Unlike Anthropic, this wire spells `required` the same way our contract
    # does. The row exists so a mapper that translated it (to `any`, the way the
    # neighbouring adapter must) is caught rather than assumed away.
    ("required stays required", ToolChoice(mode="required"), "required"),
    (
        "a named tool",
        ToolChoice(mode="tool", name="get_weather"),
        {"type": "function", "function": {"name": "get_weather"}},
    ),
]


@pytest.mark.parametrize(
    ("case", "choice", "expected"),
    TOOL_CHOICE_CASES,
    ids=[case for case, *_ in TOOL_CHOICE_CASES],
)
def test_tool_choice_maps_to_the_openai_table(
    case: str, choice: ToolChoice | None, expected: str | dict[str, Any] | None
) -> None:
    """Catches a mapper that sends the named mode as the bare string `"tool"` (the server
    reads it as an unknown mode and answers a 400), one that sends Anthropic's `any` for
    `required`, and - through the unset row - one that always sends a `tool_choice` the
    caller never asked for."""
    body = to_openai(request_with({"tools": (WEATHER_TOOL,), "tool_choice": choice}))

    assert body.get("tool_choice") == expected, case


def test_the_nameless_tool_choice_table_is_exactly_the_three_modes() -> None:
    """Catches a table that grew a `tool` row: that mode carries a name and is built as an
    object, so a lookup finding it here would send a bare string and lose the name."""
    assert OPENAI_TOOL_CHOICE_BY_MODE == {"auto": "auto", "none": "none", "required": "required"}


# --- the tool loop, request side ---------------------------------------------


def test_an_assistant_turn_with_calls_renders_openai_tool_calls() -> None:
    """Catches a mapper that renders Anthropic's `tool_use` content blocks, or that puts
    the call in `content`: an OpenAI-compatible server reads only `message.tool_calls`, so
    the model would be replayed a transcript in which it never called anything."""
    turn = ChatMessage(role="assistant", content="Checking.", tool_calls=(WEATHER_CALL,))

    body = to_openai(request_with({"messages": (turn,)}))

    assert body["messages"] == [
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": TOOL_CALL_ID,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ODD_ARGUMENTS},
                }
            ],
        }
    ]


def test_tool_call_arguments_travel_as_the_caller_wrote_them() -> None:
    """Catches a mapper that parses `arguments` and re-serialises it (which the Anthropic
    adapter must do, because its `input` is an object): the string is the model's own
    output replayed back, and normalising it changes what the model is told it said."""
    turn = ChatMessage(role="assistant", tool_calls=(WEATHER_CALL,))

    body = to_openai(request_with({"messages": (turn,)}))

    arguments = body["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert arguments == ODD_ARGUMENTS
    assert arguments != json.dumps(json.loads(ODD_ARGUMENTS)), "the fixture must be re-serialisable"


def test_a_tool_result_turn_keeps_the_tool_role_and_its_id() -> None:
    """Catches a mapper that rewrites the result into a `user` turn the way the Anthropic
    adapter has to: the server would lose the pairing between the call and its result and
    charge for a transcript the model cannot follow."""
    messages = (
        ChatMessage(role="user", content="Weather in Berlin?"),
        ChatMessage(role="assistant", tool_calls=(WEATHER_CALL,)),
        ChatMessage(role="tool", content="18C, clear", tool_call_id=TOOL_CALL_ID),
    )

    body = to_openai(request_with({"messages": messages}))

    assert body["messages"][2] == {
        "role": "tool",
        "tool_call_id": TOOL_CALL_ID,
        "content": "18C, clear",
    }
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "tool"]


# --- the response direction --------------------------------------------------


def test_content_and_finish_reason_come_from_the_first_choice() -> None:
    """Catches a mapper reading `payload["content"]` or `payload["message"]` (the shape of
    the neighbouring provider) rather than `choices[0].message.content`, which would make
    every answer unreadable."""
    response = from_openai(load_local_fixture("plain"))

    assert response.content == "Ship it."
    assert response.finish_reason == "stop"
    assert response.model == "qwen2.5-7b-instruct"
    assert response.tool_calls == ()


def test_a_null_content_with_tool_calls_becomes_an_empty_string() -> None:
    """Catches a mapper that passes `None` through into `content`, which the frozen
    response model rejects, turning a perfectly good tool-only answer into a 502."""
    response = from_openai(load_local_fixture("tool_call_only"))

    assert response.content == ""
    assert [call.name for call in response.tool_calls] == ["ticket_summary"]


def test_tool_calls_keep_their_id_name_and_argument_string() -> None:
    """Catches a mapper that reads the call's `name` and `arguments` from the wrapper
    instead of from `function`, or that parses `arguments` into a dict: the published API
    hands its clients a JSON string, and a dict breaks every OpenAI client."""
    payload = load_local_fixture("tool_call")
    wire = payload["choices"][0]["message"]["tool_calls"][0]

    response = from_openai(payload)

    assert response.content == "Checking the weather."
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == wire["id"]
    assert call.name == wire["function"]["name"]
    assert isinstance(call.arguments, str), "arguments must be a JSON string, not a dict"
    assert call.arguments == wire["function"]["arguments"]


USAGE_CASES: list[tuple[str, str, int, int, int]] = [
    ("a plain answer", "plain", 41, 7, 48),
    ("a multi-turn answer", "multi_turn", 128, 19, 147),
    ("a tool call answer", "tool_call", 88, 33, 121),
    ("a tool-only answer", "tool_call_only", 210, 64, 274),
]


@pytest.mark.parametrize(
    ("case", "fixture", "prompt", "completion", "total"),
    USAGE_CASES,
    ids=[case for case, *_ in USAGE_CASES],
)
def test_usage_maps_field_by_field(
    case: str, fixture: str, prompt: int, completion: int, total: int
) -> None:
    """Catches a mapper that copies the two counts into the wrong fields (the ledger bills
    a prompt at the completion price) or reads Anthropic's `input_tokens` names, which
    would leave every number at zero."""
    usage = from_openai(load_local_fixture(fixture)).usage

    assert usage.prompt_tokens == prompt, case
    assert usage.completion_tokens == completion, case
    assert usage.total_tokens == total, case


def test_total_tokens_is_the_providers_own_number_not_a_recomputed_sum() -> None:
    """Catches a mapper that always reports `prompt + completion`: a server counting a
    cached or a reused prefix reports a total that is not the sum, and the ledger is
    supposed to bill what the provider says it served. The payload is built here rather
    than in a fixture because a real server's numbers agree, which is exactly why a
    recomputing mapper would look right against one."""
    payload = load_local_fixture("plain")
    payload["usage"] = {"prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 61}

    usage = from_openai(payload).usage

    assert usage.total_tokens == 61
    assert usage.prompt_tokens + usage.completion_tokens == 48, "the row must not be a sum"


def test_a_usage_block_without_a_total_falls_back_to_the_sum() -> None:
    """Catches a mapper that raises (a 502 for an answer that was served and billed) or
    reports zero when the server omits `total_tokens`, which llama.cpp builds have done."""
    payload = load_local_fixture("plain")
    del payload["usage"]["total_tokens"]

    usage = from_openai(payload).usage

    assert usage.total_tokens == 48


def missing_usage_payloads() -> list[tuple[str, dict[str, Any]]]:
    """The three ways usage can be unusable, built from the hand-written payloads."""
    without_prompt = load_local_fixture("plain")
    del without_prompt["usage"]["prompt_tokens"]
    without_completion = load_local_fixture("plain")
    del without_completion["usage"]["completion_tokens"]
    return [
        ("no usage block at all", load_local_fixture("no_usage")),
        ("no prompt_tokens", without_prompt),
        ("no completion_tokens", without_completion),
    ]


@pytest.mark.parametrize(
    ("case", "payload"),
    missing_usage_payloads(),
    ids=[case for case, _ in missing_usage_payloads()],
)
def test_a_response_without_usage_raises_provider_error(case: str, payload: dict[str, Any]) -> None:
    """Catches the silent-zero implementation (`payload.get("usage", {}).get(...) or 0`),
    which would book a real call into the ledger at zero cost - and the local provider is
    the one whose per-token price is lowest, so a zero there is the easiest to overlook."""
    with pytest.raises(ProviderError) as raised:
        from_openai(payload)

    assert raised.value.retryable is False, f"{case}: a malformed body is not worth a retry"


def test_an_answer_with_usage_but_nothing_to_render_raises() -> None:
    """Catches an adapter that serves `content: ""` with real usage: the tenant pays for
    an empty answer, and the non-streaming Anthropic path already refuses the same reply,
    so serving it here would be two behaviours for one situation."""
    with pytest.raises(ProviderError) as raised:
        from_openai(load_local_fixture("no_content"))

    assert raised.value.retryable is False
    assert raised.value.http_status == 502


def test_a_response_with_no_choices_raises() -> None:
    """Catches an `IndexError` escaping the mapper as a 500 with a traceback: a server
    that answers 200 with an empty `choices` (llama.cpp does, when every slot is dropped)
    must leave as the `ProviderError` the routes' fallback loop reads."""
    payload = load_local_fixture("plain")
    payload["choices"] = []

    with pytest.raises(ProviderError) as raised:
        from_openai(payload)

    assert raised.value.retryable is False


# --- the six request-path identifiers ---------------
#
# Each row below carries exactly ONE bad identifier, with every other identifier
# on the request valid, so a row can only be rejected by the pre-check on the
# field it names. The transport is scripted to answer 200, so the pre-check is
# the only thing that can raise, and `request_count == 0` proves it ran first.

GOOD_ID = "call_local_weather_0001"

SITES = (
    "messages[].tool_calls[].id",
    "messages[].tool_calls[].name",
    "messages[].tool_call_id",
    "tools[].name",
    "tool_choice.name",
    "response_format.name",
)


def request_for(site: str, value: str) -> ChatRequest:
    """The baseline request with `value` planted at exactly one identifier site."""
    if site == "messages[].tool_calls[].id":
        call = ToolCall(id=value, name="get_weather", arguments="{}")
        return request_with({"messages": (ChatMessage(role="assistant", tool_calls=(call,)),)})
    if site == "messages[].tool_calls[].name":
        call = ToolCall(id=GOOD_ID, name=value, arguments="{}")
        return request_with({"messages": (ChatMessage(role="assistant", tool_calls=(call,)),)})
    if site == "messages[].tool_call_id":
        return request_with(
            {"messages": (ChatMessage(role="tool", content="18C", tool_call_id=value),)}
        )
    if site == "tools[].name":
        return request_with({"tools": (ToolSpec(name=value),)})
    if site == "tool_choice.name":
        return request_with(
            {"tools": (WEATHER_TOOL,), "tool_choice": ToolChoice(mode="tool", name=value)}
        )
    if site == "response_format.name":
        return request_with({"response_format": ResponseFormat(name=value, json_schema={})})
    raise AssertionError(f"unknown identifier site {site!r}")


AT_CAP = "a" * MAX_TOOL_NAME_LENGTH
OVER_CAP = "a" * (MAX_TOOL_NAME_LENGTH + 1)

# The rejection table a project rule asks for. The newline and CRLF rows are the
# ones `re.match` with `^...$` lets through; the space row is the one a `.strip()`
# before validation lets through.
BAD_VALUES: list[tuple[str, str]] = [
    ("a trailing newline", f"{GOOD_ID}\n"),
    ("a trailing CRLF", f"{GOOD_ID}\r\n"),
    ("a trailing space", f"{GOOD_ID} "),
    ("an interior space", "get weather"),
    ("one character over the cap", OVER_CAP),
]

REJECTION_TABLE = [(site, case, value) for site in SITES for case, value in BAD_VALUES]


@pytest.mark.parametrize(
    ("site", "case", "value"),
    REJECTION_TABLE,
    ids=[f"{site} with {case}" for site, case, _ in REJECTION_TABLE],
)
def test_every_request_path_identifier_is_checked_before_any_outbound_call(
    site: str, case: str, value: str
) -> None:
    """Catches an identifier validated with `re.match` and `^...$` (a trailing newline
    passes), validated after the body is built, or - the likeliest miss - checked at some
    sites and not at others: each row plants one bad value at one site, so only that
    site's own check can reject it, and `request_count == 0` proves nothing was sent."""
    adapter, scripted = make_local_adapter([local_ok("plain")])

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_for(site, value))

    assert raised.value.http_status == 422, f"{site}: {case}"
    assert scripted.request_count == 0, f"{site}: {case}: the provider was called first"


@pytest.mark.parametrize("site", SITES)
def test_an_identifier_of_exactly_128_characters_is_accepted(site: str) -> None:
    """Catches an exclusive boundary (`len(value) < 128`), which would refuse an id at
    exactly the documented cap - and it is not ours to shorten, it is the one the provider
    issued and the client sent back. Without this row the rejection table above would pass
    against a pre-check that rejected everything."""
    adapter, scripted = make_local_adapter([local_ok("plain")])

    adapter.chat(request_for(site, AT_CAP))

    assert scripted.request_count == 1
    assert len(AT_CAP) == MAX_TOOL_NAME_LENGTH
    assert MAX_TOOL_NAME_LENGTH == 128


def test_a_tool_turn_without_a_tool_call_id_raises_before_any_outbound_call() -> None:
    """Catches a mapper that renders `tool_call_id: null` when the caller omitted it: the
    server cannot pair the result with a call, and the failure would arrive as its 400
    rather than as the caller's own 422 naming the field."""
    adapter, scripted = make_local_adapter([local_ok("plain")])

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(request_with({"messages": (ChatMessage(role="tool", content="18C"),)}))

    assert raised.value.http_status == 422
    assert scripted.request_count == 0


def test_a_named_tool_choice_with_no_name_raises_before_any_outbound_call() -> None:
    """Catches a mapper that renders `{"type": "function", "function": {"name": null}}`
    when the mode demands a named tool and no name was given: the contract allows the pair
    (`name` is meaningless for the other three modes), so the adapter is the only place it
    can be refused."""
    adapter, scripted = make_local_adapter([local_ok("plain")])

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(
            request_with({"tools": (WEATHER_TOOL,), "tool_choice": ToolChoice(mode="tool")})
        )

    assert raised.value.http_status == 422
    assert scripted.request_count == 0


def test_a_valid_tool_request_still_reaches_the_provider() -> None:
    """Catches a pre-check so strict it rejects everything: without this the rejection
    table would pass against `checked_tool_name` raising unconditionally, and every tool
    request would be a 422 no caller could satisfy."""
    adapter, scripted = make_local_adapter([local_ok("tool_call")])

    response = adapter.chat(request_with({"tools": (WEATHER_TOOL,)}))

    assert scripted.request_count == 1
    assert [tool["function"]["name"] for tool in scripted.body()["tools"]] == ["get_weather"]
    assert [call.name for call in response.tool_calls] == ["get_weather"]
