"""The provider-neutral adapter contract: the tool name pre-check and the pinned constants.

The tool name is the only client-supplied string in an earlier change that becomes an
identifier on an outbound call, so a project rule applies: a typed pre-check
against the store's own constant, with the inclusive boundary asserted here and
the "before any effect" half asserted in `test_anthropic_mapping.py`. Standing
rule 4 is why the rejection table below carries a trailing newline, a trailing
CRLF and a trailing space: `re.match` with `^...$` accepts the first of those,
`re.fullmatch` does not, and the table is what tells the two apart.

`DEFAULT_MAX_TOKENS`, `MAX_TOOL_NAME_LENGTH` and the `price_key` format are
pinned exactly: each is a value another component reads back -
Anthropic requires `max_tokens`, and the ledger stores `price_key` verbatim.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from opsconsole.providers import base
from opsconsole.providers.base import (
    DEFAULT_MAX_TOKENS,
    MAX_TOOL_NAME_LENGTH,
    Adapter,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Embedding,
    EmbeddingAdapter,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    ToolSpec,
    Usage,
    checked_tool_name,
)
from opsconsole.routing.errors import IdentifierInvalid

from .conftest import local_ok, make_adapter, make_local_adapter, ok


def test_default_max_tokens_is_exactly_4096() -> None:
    """Catches an implementation that picks its own default (1024 is Anthropic's docs
    example), which would silently truncate every answer where the caller omitted it."""
    assert DEFAULT_MAX_TOKENS == 4096


def test_max_tool_name_length_is_exactly_128() -> None:
    """Catches a cap drifting from the 128 the pattern encodes, which would let the
    boundary test below pass against a pattern that accepts a longer name."""
    assert MAX_TOOL_NAME_LENGTH == 128


def test_tool_name_accepted_at_exactly_the_maximum_length() -> None:
    """Catches an exclusive boundary (`{1,127}`, or `len(value) < MAX`), which rejects a
    name that is exactly at the documented cap and is therefore legal."""
    name = "a" * MAX_TOOL_NAME_LENGTH

    assert checked_tool_name(name, "tools[0].name") == name


def test_tool_name_rejected_one_over_the_maximum_length() -> None:
    """Catches a pre-check with no upper bound at all (`[a-zA-Z0-9_-]+`), which would
    hand a 129-character name to Anthropic and turn our bug into its 400."""
    with pytest.raises(IdentifierInvalid):
        checked_tool_name("a" * (MAX_TOOL_NAME_LENGTH + 1), "tools[0].name")


# Every row is a name the pre-check must refuse. The trailing-newline and
# trailing-CRLF rows are the ones `^...$` with `.match` would wrongly accept.
REJECTED_TOOL_NAMES: list[tuple[str, str]] = [
    ("a trailing newline", "get_weather\n"),
    ("a trailing CRLF", "get_weather\r\n"),
    ("a trailing carriage return", "get_weather\r"),
    ("a trailing space", "get_weather "),
    ("a leading space", " get_weather"),
    ("an interior space", "get weather"),
    ("empty", ""),
    ("a dot", "get.weather"),
    ("a slash", "tools/get_weather"),
    ("one over the length cap", "a" * (MAX_TOOL_NAME_LENGTH + 1)),
]


@pytest.mark.parametrize(
    ("case", "name"), REJECTED_TOOL_NAMES, ids=[case for case, _ in REJECTED_TOOL_NAMES]
)
def test_tool_name_rejection_table(case: str, name: str) -> None:
    """Catches a pre-check written as `re.match(r"^[a-zA-Z0-9_-]{1,128}$", ...)`: in
    Python `$` also matches before a trailing newline, so the CRLF and newline rows
    would pass validation and travel to the provider."""
    with pytest.raises(IdentifierInvalid) as raised:
        checked_tool_name(name, "tools[0].name")

    assert raised.value.code == "VALIDATION", f"{case}: wrong error code"
    assert raised.value.http_status == 422, f"{case}: wrong status"


ACCEPTED_TOOL_NAMES: list[tuple[str, str]] = [
    ("uppercase", "GetWeather"),
    ("mixed case", "getWeather"),
    ("digits", "tool42"),
    ("underscore", "get_weather"),
    ("hyphen", "get-weather"),
    ("a single character", "t"),
]


@pytest.mark.parametrize(
    ("case", "name"), ACCEPTED_TOOL_NAMES, ids=[case for case, _ in ACCEPTED_TOOL_NAMES]
)
def test_tool_name_acceptance_table(case: str, name: str) -> None:
    """Catches a pre-check that over-tightens - lowercasing the pattern, or reusing the
    routing identifier rule - which would reject the legal names Anthropic documents."""
    assert checked_tool_name(name, "tools[0].name") == name, case


def test_rejection_message_names_the_field() -> None:
    """Catches a pre-check that raises a bare `VALIDATION` with no `param`, leaving an
    OpenAI-compatible client unable to point the caller at the offending tool."""
    with pytest.raises(IdentifierInvalid) as raised:
        checked_tool_name("bad name", "tools[0].name")

    error = raised.value
    assert "tools[0].name" in error.message or error.param == "tools[0].name"


PRICE_KEY_CASES: list[tuple[str, str, str, str]] = [
    ("the pinned pair", "anthropic", "claude-haiku-4-5", "anthropic/claude-haiku-4-5"),
    # The policy provider name, not the vendor: the factory key, the ledger key
    # and the metric label must be the same string.
    ("a policy name", "anthropic-eu", "claude-haiku-4-5", "anthropic-eu/claude-haiku-4-5"),
]


@pytest.mark.parametrize(
    ("case", "name", "model", "expected"),
    PRICE_KEY_CASES,
    ids=[case for case, *_ in PRICE_KEY_CASES],
)
def test_price_key_is_provider_name_slash_model(
    case: str, name: str, model: str, expected: str
) -> None:
    """Catches a `price_key` built from the vendor ("anthropic") or from the adapter's
    class instead of the policy provider name, which would make the ledger row for
    `anthropic-eu` indistinguishable from one for a second Anthropic account."""
    adapter, scripted = make_adapter([ok("plain")], name=name)

    assert adapter.price_key(model) == expected, case
    assert scripted.request_count == 0, "price_key must not call the provider"


def test_adapter_satisfies_the_runtime_protocol() -> None:
    """Catches an adapter that drops or renames a protocol member (`complete` instead of
    `chat`), which the routes' fallback loop would only discover at runtime."""
    adapter, _ = make_adapter([ok("plain")])

    assert isinstance(adapter, Adapter)
    assert adapter.name == "anthropic-eu"


def test_healthcheck_is_true_without_calling_the_provider() -> None:
    """Catches a healthcheck that pings the provider: it would need a live key in CI and
    would put a network call on the readiness path. The breaker did NOT replace this seam:
    it learns from the outcomes of real requests through the routes' `is_healthy` and
    calls `healthcheck()` nowhere, so this method is still always-true by design."""
    adapter, scripted = make_adapter([ok("plain")])

    assert adapter.healthcheck() is True
    assert scripted.request_count == 0


# Provider vocabulary that must never appear in the neutral contract, so the
# OpenAI, Azure and llama.cpp adapters of later plans implement it unchanged.
FORBIDDEN_VOCABULARY: list[str] = [
    "anthropic",
    "claude",
    "azure",
    "input_schema",
    "stop_reason",
    "tool_use",
    "x_api_key",
]


@pytest.mark.parametrize("token", FORBIDDEN_VOCABULARY)
def test_base_contract_carries_no_provider_vocabulary(token: str) -> None:
    """Catches a base contract that leaks Anthropic's field names (an `input_schema` on
    ToolSpec, a `stop_reason` on ChatResponse), which would force every later adapter to
    translate into Anthropic's shape rather than into ours."""
    names = [name.lower() for name in vars(base) if not name.startswith("__")]
    for model in (
        ChatMessage,
        ToolSpec,
        ResponseFormat,
        ChatRequest,
        ToolCall,
        ToolChoice,
        Usage,
        ChatResponse,
    ):
        names.extend(field.lower() for field in model.model_fields)

    offenders = [name for name in names if token in name]
    assert offenders == [], f"provider vocabulary in providers/base.py: {offenders}"


MODEL_CASES: list[tuple[str, type[Any], dict[str, Any]]] = [
    ("ChatMessage", ChatMessage, {"role": "user", "content": "hi"}),
    ("ToolSpec", ToolSpec, {"name": "get_weather"}),
    ("ResponseFormat", ResponseFormat, {"name": "summary", "json_schema": {"type": "object"}}),
    (
        "ChatRequest",
        ChatRequest,
        {"model": "claude-haiku-4-5", "messages": (ChatMessage(role="user", content="hi"),)},
    ),
    ("ToolCall", ToolCall, {"id": "toolu_1", "name": "get_weather", "arguments": "{}"}),
    ("ToolChoice", ToolChoice, {"mode": "auto"}),
    ("Usage", Usage, {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
    (
        "ChatResponse",
        ChatResponse,
        {
            "model": "claude-haiku-4-5",
            "content": "hi",
            "finish_reason": "end_turn",
            "usage": Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        },
    ),
]


@pytest.mark.parametrize(
    ("case", "model", "kwargs"), MODEL_CASES, ids=[case for case, *_ in MODEL_CASES]
)
def test_contract_models_forbid_unknown_fields(
    case: str, model: type[Any], kwargs: dict[str, Any]
) -> None:
    """Catches a model left on Pydantic's default `extra="ignore"`, which would swallow a
    misspelled `max_token` from a caller and answer with the 4096 default instead."""
    model(**kwargs)  # the same kwargs without the extra key must construct

    with pytest.raises(ValidationError, match=r"(?i)extra"):
        model(**kwargs, definitely_not_a_field="x")


@pytest.mark.parametrize(
    ("case", "model", "kwargs"), MODEL_CASES, ids=[case for case, *_ in MODEL_CASES]
)
def test_contract_models_are_frozen(case: str, model: type[Any], kwargs: dict[str, Any]) -> None:
    """Catches a mutable request model: the adapter mutating the caller's request during
    mapping would make a fallback to a second provider send a different body."""
    instance = model(**kwargs)
    field = next(iter(model.model_fields))

    with pytest.raises(ValidationError, match=r"(?i)frozen|immutable"):
        setattr(instance, field, getattr(instance, field))


# --- the tool loop the contract has to be able to express ---------
#
# A tool loop whose REQUEST half cannot be said in the neutral contract is not
# tool support: the caller gets a `tool_calls` answer back and then has nowhere
# to put the result. These cases pin the three field defaults that make the loop
# expressible, and the two closed sets that keep it typed.


def test_a_tool_call_only_assistant_turn_needs_no_content() -> None:
    """Catches a `ChatMessage` that still requires `content`: an assistant turn whose
    whole answer was a tool call has no text, so a required field would force the caller
    to invent one and change what the model is shown of its own last turn."""
    call = ToolCall(id="toolu_1", name="get_weather", arguments='{"city": "Berlin"}')

    turn = ChatMessage(role="assistant", tool_calls=(call,))

    assert turn.content == ""
    assert turn.tool_calls == (call,)
    assert turn.tool_call_id is None


def test_a_plain_turn_carries_no_tool_fields() -> None:
    """Catches defaults that make the new fields required, or that default `tool_calls` to
    a shared mutable list: an ordinary user turn must construct from role and content
    alone, exactly as every existing caller writes it."""
    turn = ChatMessage(role="user", content="Which city?")

    assert turn.tool_calls == ()
    assert turn.tool_call_id is None


def test_a_tool_result_turn_carries_the_id_it_answers() -> None:
    """Catches a contract with no `tool_call_id`: with several calls in one assistant turn
    there is no other way to say which result answers which call, and a provider would be
    handed results it cannot pair up."""
    turn = ChatMessage(role="tool", content="18C, clear", tool_call_id="toolu_1")

    assert turn.role == "tool"
    assert turn.tool_call_id == "toolu_1"


ACCEPTED_ROLES: list[str] = ["system", "user", "assistant", "tool"]


@pytest.mark.parametrize("role", ACCEPTED_ROLES)
def test_role_set_accepts_the_four_turns(role: str) -> None:
    """Catches a contract that adds `tool` to the response side only, leaving a tool result
    inexpressible on the way in - the half of the loop an earlier change exists to close."""
    assert ChatMessage(role=role, content="x").role == role


REJECTED_ROLES: list[tuple[str, str]] = [
    ("OpenAI's legacy function role", "function"),
    ("OpenAI's developer role", "developer"),
    ("a capitalised role", "Tool"),
    ("a trailing newline", "tool\n"),
    ("empty", ""),
]


@pytest.mark.parametrize(("case", "role"), REJECTED_ROLES, ids=[case for case, _ in REJECTED_ROLES])
def test_role_set_stays_closed(case: str, role: str) -> None:
    """Catches a `role: str` widened while `tool` was added: an unknown role would then be
    forwarded to a provider that rejects it, turning a caller's typo into a provider 400
    the caller cannot read."""
    with pytest.raises(ValidationError):
        ChatMessage(role=role, content="x")


ACCEPTED_TOOL_CHOICE_MODES: list[str] = ["auto", "none", "required", "tool"]


@pytest.mark.parametrize("mode", ACCEPTED_TOOL_CHOICE_MODES)
def test_tool_choice_modes_are_the_four_neutral_ones(mode: str) -> None:
    """Catches a `ToolChoice` that omits one of the four - dropping `required` would leave
    a caller who demands a tool call no way to say so except through `response_format`,
    which is a different request."""
    assert ToolChoice(mode=mode).mode == mode
    assert ToolChoice(mode=mode).name is None


REJECTED_TOOL_CHOICE_MODES: list[tuple[str, str]] = [
    ("Anthropic's own word for required", "any"),
    ("OpenAI's object form as a string", "function"),
    ("a trailing newline", "auto\n"),
    ("empty", ""),
]


@pytest.mark.parametrize(
    ("case", "mode"),
    REJECTED_TOOL_CHOICE_MODES,
    ids=[case for case, _ in REJECTED_TOOL_CHOICE_MODES],
)
def test_tool_choice_mode_set_stays_closed(case: str, mode: str) -> None:
    """Catches a `mode: str`: `"any"` is Anthropic's spelling of `required`, and letting it
    through the neutral contract would put one provider's vocabulary in the gateway's own
    request type - the thing this module exists to prevent."""
    with pytest.raises(ValidationError):
        ToolChoice(mode=mode)


def test_chat_request_tool_choice_defaults_to_unset() -> None:
    """Catches a `ChatRequest` that defaults `tool_choice` to `auto`: every existing caller
    would start sending an explicit choice, and a provider that treats "no field" and
    "auto" differently would change its behaviour with no caller having asked."""
    request = ChatRequest(
        model="claude-haiku-4-5", messages=(ChatMessage(role="user", content="hi"),)
    )

    assert request.tool_choice is None


# --- the embedding contract ---------------------------------------
#
# A separate protocol beside `Adapter`, not an `embed()` on it (decision B). The
# two cases that matter are the two ends of that choice: the OpenAI-compatible
# adapter satisfies `EmbeddingAdapter`, and the Anthropic one does not - which is
# what the embeddings route turns into the resolver's health seam, so an alias
# pointing at a chat-only provider is a typed 503 rather than an AttributeError
# served as a 500.
#
# The models get their own `MODEL_CASES` table rather than rows on the existing
# one, so the chat contract's cases are untouched.

EMBEDDING_MODEL_CASES: list[tuple[str, type[Any], dict[str, Any]]] = [
    ("EmbeddingRequest", EmbeddingRequest, {"model": "bge-small-en-v1.5", "input": ("Berlin",)}),
    ("EmbeddingUsage", EmbeddingUsage, {"prompt_tokens": 11, "total_tokens": 13}),
    ("Embedding", Embedding, {"index": 0, "vector": (0.125, -0.375)}),
    (
        "EmbeddingResponse",
        EmbeddingResponse,
        {
            "model": "bge-small-en-v1.5",
            "embeddings": (Embedding(index=0, vector=(0.125, -0.375)),),
            "usage": EmbeddingUsage(prompt_tokens=11, total_tokens=13),
        },
    ),
]


@pytest.mark.parametrize(
    ("case", "model", "kwargs"),
    EMBEDDING_MODEL_CASES,
    ids=[case for case, *_ in EMBEDDING_MODEL_CASES],
)
def test_embedding_contract_models_forbid_unknown_fields(
    case: str, model: type[Any], kwargs: dict[str, Any]
) -> None:
    """Catches an embedding model left on Pydantic's default `extra="ignore"`, which would
    swallow a `dimensions` or an `encoding_format` a caller set on the wrong model and
    serve the request as though it had never been asked for."""
    model(**kwargs)  # the same kwargs without the extra key must construct

    with pytest.raises(ValidationError, match=r"(?i)extra"):
        model(**kwargs, definitely_not_a_field="x")


@pytest.mark.parametrize(
    ("case", "model", "kwargs"),
    EMBEDDING_MODEL_CASES,
    ids=[case for case, *_ in EMBEDDING_MODEL_CASES],
)
def test_embedding_contract_models_are_frozen(
    case: str, model: type[Any], kwargs: dict[str, Any]
) -> None:
    """Catches a mutable embedding request: a fallback sends the same request to a
    second provider, and an adapter that normalised the input list in place would send the
    second provider something the caller never asked for."""
    instance = model(**kwargs)
    field = next(iter(model.model_fields))

    with pytest.raises(ValidationError, match=r"(?i)frozen|immutable"):
        setattr(instance, field, getattr(instance, field))


def test_embedding_usage_has_exactly_two_counts() -> None:
    """Catches an `EmbeddingUsage` that reused the chat contract's three-field `Usage`
    (decision F): this wire has no completion tokens for an embedding, so the third field
    could only ever be written as a zero - and a zero-filled count is exactly what the
    "no usage raises" rule exists to prevent, hidden behind a field that always looks
    populated."""
    assert set(EmbeddingUsage.model_fields) == {"prompt_tokens", "total_tokens"}


def test_the_openai_compatible_adapter_satisfies_the_embedding_protocol() -> None:
    """Catches an `embed()` that drifted from the protocol - a `create_embeddings` name, or
    a method on a helper rather than on the adapter - which the embeddings route's health
    seam would read as "this provider cannot embed", turning the one provider that CAN into
    a permanent 503."""
    adapter, scripted = make_local_adapter([local_ok("embeddings_single")])

    assert isinstance(adapter, EmbeddingAdapter)
    assert scripted.request_count == 0, "a capability check must not call the provider"


def test_the_anthropic_adapter_does_not_satisfy_the_embedding_protocol() -> None:
    """Catches `embed()` being put on `Adapter` after all, or on a shared base class: the
    Anthropic wire has no embeddings endpoint, so an `embed()` there could only raise - and
    the route's health seam would then offer a chat-only provider an embedding request and
    answer the caller a 500 instead of the resolver's 503 (decision B)."""
    adapter, _ = make_adapter([ok("plain")])

    assert isinstance(adapter, Adapter)
    assert not isinstance(adapter, EmbeddingAdapter)
