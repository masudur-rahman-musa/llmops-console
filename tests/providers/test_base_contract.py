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
    ResponseFormat,
    ToolCall,
    ToolSpec,
    Usage,
    checked_tool_name,
)
from opsconsole.routing.errors import IdentifierInvalid

from .conftest import make_adapter, ok


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
    `chat`), which an earlier change's fallback would only discover at runtime."""
    adapter, _ = make_adapter([ok("plain")])

    assert isinstance(adapter, Adapter)
    assert adapter.name == "anthropic-eu"


def test_healthcheck_is_true_without_calling_the_provider() -> None:
    """Catches a healthcheck that pings the provider: it would need a live key in CI and
    would put a network call on a code path an earlier change replaces with a breaker."""
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
    mapping would make a an earlier change retry against a second provider send a different body."""
    instance = model(**kwargs)
    field = next(iter(model.model_fields))

    with pytest.raises(ValidationError, match=r"(?i)frozen|immutable"):
        setattr(instance, field, getattr(instance, field))
