"""The provider-neutral adapter contract.

Everything a provider adapter takes and returns is defined here, in the
gateway's own vocabulary. The models are OpenAI-shaped because that is the API
this product publishes, but they are ours: no provider SDK is imported, and no
provider's field names appear, so the OpenAI, Azure and OpenAI-compatible
adapters of later plans implement this unchanged rather than translating into
one provider's dialect. `test_base_contract_carries_no_provider_vocabulary`
holds that by walking this module's names and every model's fields.

The models are frozen and reject unknown fields. Frozen because an adapter must
not mutate the caller's request while mapping it - an earlier change retries the same
request against a second provider, and it has to be the same request; unknown
fields rejected because a misspelled `max_token` should be an error rather than
a silently applied default. Both are pinned by
`test_contract_models_are_frozen` and `test_contract_models_forbid_unknown_fields`.

`embed()` is deliberately absent: Anthropic has no embeddings endpoint, and a
method that exists only to raise is worse than a second protocol. An earlier change adds
an embedding protocol next to this one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from opsconsole.routing.errors import IdentifierInvalid

DEFAULT_MAX_TOKENS = 4096
"""What `max_tokens` becomes when the caller omits it.

Anthropic requires the field, so there has to be a default; 4096 is large enough
that it is not felt as a cap on an ordinary answer. Pinned exactly by
`test_default_max_tokens_is_exactly_4096`.
"""

MAX_TOOL_NAME_LENGTH = 128
"""Longest tool name accepted. Pinned by `test_max_tool_name_length_is_exactly_128`."""

TOOL_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,128}")
"""The character set a tool name may use, matched with `re.fullmatch`.

`.match` with `^...$` would accept a trailing newline, and this string is sent
to a provider as the identifier the model calls back by name. The rejection
table in `test_tool_name_rejection_table` covers a trailing newline, a trailing
CRLF and a trailing space; the accepted set (uppercase included) is held by
`test_tool_name_acceptance_table`.
"""


def checked_tool_name(value: str, field: str) -> str:
    """Validate one caller-supplied tool name before it reaches a provider.

    Raises `IdentifierInvalid` (code VALIDATION, status 422) naming `field`, so a
    bad name is the caller's error with a pointer to it rather than a provider
    400 the caller cannot act on. Adapters call this before the outbound request
    is built, which `test_bad_tool_name_raises_before_any_outbound_call` proves by
    asserting no request was made.
    """
    if len(value) > MAX_TOOL_NAME_LENGTH:
        raise IdentifierInvalid(
            f"{field}: longer than {MAX_TOOL_NAME_LENGTH} characters", param=field
        )
    if TOOL_NAME_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{field}: must match {TOOL_NAME_PATTERN.pattern}", param=field)
    return value


class _Contract(BaseModel):
    """Frozen, and unknown fields rejected. See the module docstring for why."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ChatMessage(_Contract):
    """One turn. `system` is a role here; adapters place it wherever their provider wants it."""

    role: Literal["system", "user", "assistant"]
    content: str


class ToolSpec(_Contract):
    """A tool the model may call. `parameters` is a JSON schema, passed through as data."""

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = {}


class ResponseFormat(_Contract):
    """A demand that the answer conform to a JSON schema.

    `name` is caller-supplied and becomes an identifier on the outbound call, so
    adapters run it through `checked_tool_name` like any other tool name.
    """

    name: str
    json_schema: Mapping[str, Any]


class ChatRequest(_Contract):
    """One completion request, already resolved to a provider model id.

    `model` is the provider's own id from the routing policy, never an alias: the
    resolver has already run by the time an adapter sees this.
    """

    model: str
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...] = ()
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: ResponseFormat | None = None


class ToolCall(_Contract):
    """A tool the model asked to call. `arguments` is a JSON string, as OpenAI publishes it."""

    id: str
    name: str
    arguments: str


class Usage(_Contract):
    """What the call consumed. The ledger prices from these three numbers."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(_Contract):
    """One completion. `content` is empty rather than absent when the answer is only tool calls."""

    model: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str
    usage: Usage


@runtime_checkable
class Adapter(Protocol):
    """What every provider adapter offers the gateway.

    `name` is the policy provider name (the key in `routing.yaml`'s `providers`
    and in the factory's mapping), not the vendor: two accounts with the same
    vendor are two providers, and the ledger, the metric label and the resolver
    all have to address them apart. `price_key` is built from it for that reason,
    pinned by `test_price_key_is_provider_name_slash_model`.
    """

    name: str

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run one non-streaming completion. Every failure is a `ProviderError`."""
        ...

    def healthcheck(self) -> bool:
        """Whether this provider should be offered traffic."""
        ...

    def price_key(self, model: str) -> str:
        """The stable `"<provider>/<model>"` string the ledger stores."""
        ...
