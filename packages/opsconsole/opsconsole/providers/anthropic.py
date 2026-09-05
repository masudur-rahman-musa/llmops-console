"""The Anthropic adapter: the Messages API behind the neutral contract.

Three separable pieces, on purpose. `to_anthropic` and `from_anthropic` are pure
functions over dictionaries, so both directions of the mapping are tested against
recorded payloads without a transport; `AnthropicAdapter` adds the HTTP, the
fixed timeouts and the retry loop around them, with `sleep` and `jitter` injected
so the retry table costs microseconds to test and produces exact delays rather
than a range. No provider SDK is imported: the mapping is the thing we own, and
it stops here, so no Anthropic field name reaches the rest of the gateway.

Every PROVIDER failure that leaves this module is a `ProviderError` carrying
`retryable`, including a timeout and a 200 whose body we cannot read, so plan
10's fallback reads one flag instead of re-deriving a verdict from a status it
can no longer see. The caller's own bad input is the deliberate exception: an
invalid tool name leaves as `IdentifierInvalid` (422) before any call is made,
because that is the caller's error to fix and not a provider outage to fall back
from.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from opsconsole.core.errors import ProviderError
from opsconsole.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    ToolCall,
    Usage,
    checked_tool_name,
)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
MESSAGES_PATH = "/v1/messages"

CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 60.0
"""The fixed timeouts (the design). Not caller-settable: a caller able to raise the
read timeout could hold a worker for as long as it liked. Pinned on the constants
by `test_timeout_constants_are_exactly_the_spec_values` and on the built client by
`test_client_timeouts_are_fixed_at_five_and_sixty`. The 120s stream timeout
arrives with streaming in an earlier change."""

MAX_RETRIES = 2
RETRY_BASE_DELAY = 0.5
"""Two retries, so three attempts at most, with the delay doubling and jittered.

Retries are restricted to 429, any 5xx and a timeout: those are the failures a
second attempt can fix. Every other status is raised on the first attempt,
because retrying a 400 bills the tenant three times for a request that cannot
succeed. `test_retry_table_by_status` holds the whole table, request counts
included, and `test_backoff_doubles_and_is_jittered` pins the two delays."""

PROVIDER_MESSAGE_LIMIT = 500
"""How much of a provider's error body is kept for the log, so a proxy answering
with a megabyte of HTML cannot write a megabyte per failure. Pinned by
`test_provider_message_limit_is_exactly_500` and
`test_an_oversized_provider_body_is_truncated`; it never reaches the client
(`test_provider_message_never_reaches_the_client_message`)."""


def _unreadable(exc: Exception) -> ProviderError:
    """The one answer to a body we cannot parse: a 502, not a traceback.

    Not retryable: a well-formed failure would have carried a status, so a body
    we cannot read is a shape change or a bug, and a second identical request
    produces the same unreadable answer.
    """
    return ProviderError(
        "the provider answered with a body the adapter could not read",
        retryable=False,
        provider_message=f"{type(exc).__name__}: {exc}",
    )


def to_anthropic(request: ChatRequest) -> dict[str, Any]:
    """Render a normalised request as an Anthropic Messages request body.

    Every caller-supplied tool name is checked first, so an invalid one raises
    before any outbound call is built
    (`test_bad_tool_name_raises_before_any_outbound_call`).

    The system turn is lifted out of `messages` into the top-level `system`
    field, because Anthropic has no `system` role and rejects one; the remaining
    turns keep the caller's order. Several system messages join with a blank
    line, which is the only reading that neither drops one nor runs two
    instructions together.
    """
    tools: list[dict[str, Any]] = [
        {
            "name": checked_tool_name(tool.name, f"tools[{index}].name"),
            "description": tool.description,
            "input_schema": dict(tool.parameters),
        }
        for index, tool in enumerate(request.tools)
    ]

    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_TOKENS,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role != "system"
        ],
    }

    system = "\n\n".join(
        message.content for message in request.messages if message.role == "system"
    )
    if system:
        body["system"] = system

    if request.temperature is not None:
        # `is not None`, not truthiness: an explicit temperature of 0 is the
        # setting an eval run depends on, and dropping it would change the answer.
        body["temperature"] = request.temperature

    if request.response_format is not None:
        # A JSON schema becomes one forced tool. Anthropic has no response_format,
        # and a tool the model is required to call is the mechanism that makes the
        # schema binding rather than a suggestion the prose can ignore.
        forced = checked_tool_name(request.response_format.name, "response_format.name")
        tools.append(
            {
                "name": forced,
                "description": "Return the answer as this schema.",
                "input_schema": dict(request.response_format.json_schema),
            }
        )
        body["tool_choice"] = {"type": "tool", "name": forced}

    if tools:
        # Omitted entirely rather than sent empty: an empty `tools` is a field
        # Anthropic validates, not one it ignores.
        body["tools"] = tools

    return body


def from_anthropic(payload: Mapping[str, Any]) -> ChatResponse:
    """Read an Anthropic Messages response into the normalised response.

    Text blocks concatenate into `content` and `tool_use` blocks become
    `tool_calls` whose `arguments` is a JSON *string*, which is the shape the
    published OpenAI-compatible API promises its clients. A block of any other
    type is passed over rather than guessed at, and an answer that leaves nothing
    behind - no text and no tool call - is a `ProviderError` rather than an empty
    string a tenant is billed for (`test_an_unknown_block_type_is_ignored` and
    `test_an_answer_with_no_readable_block_raises`).

    Usage is required. Anthropic's `input_tokens` and `output_tokens` become
    `prompt_tokens` and `completion_tokens` with `total_tokens` their sum, and a
    response missing any of it raises rather than defaulting to zero: the ledger
    prices from these numbers, so a silent zero books a real, billed call as free
    (`test_a_response_without_usage_raises_provider_error`).
    """
    try:
        text: list[str] = []
        calls: list[ToolCall] = []
        for block in payload["content"]:
            if block["type"] == "text":
                text.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=json.dumps(block["input"], sort_keys=True),
                    )
                )
            # Any other block type is deliberately not rendered. Anthropic adds
            # block types over time (`thinking` is one), and a block we do not
            # know is not one we can claim is answer text. Dropping it silently
            # would be the same failure as a zero usage, so the guard below
            # refuses an answer that ends up carrying nothing at all.

        usage = payload["usage"]
        prompt_tokens = usage["input_tokens"]
        completion_tokens = usage["output_tokens"]

        content = "".join(text)
        if not content and not calls:
            # Not `_unreadable`: the body parsed fine, it simply carried nothing
            # this adapter can render. Saying so is what stops an operator
            # chasing a parse bug that is not there.
            raise ProviderError(
                "the provider answered with no readable content block",
                retryable=False,
                provider_message=(
                    f"stop_reason={payload.get('stop_reason')!r}, "
                    f"block types={[block.get('type') for block in payload['content']]}"
                ),
            )

        return ChatResponse(
            model=payload["model"],
            content=content,
            tool_calls=tuple(calls),
            finish_reason=payload["stop_reason"],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _unreadable(exc) from exc


class AnthropicAdapter:
    """One Anthropic account, addressed by its policy provider name.

    `name` is that policy name rather than the vendor, so two Anthropic accounts
    in one policy stay distinguishable everywhere they are counted; `price_key`
    is built from it (`test_price_key_is_provider_name_slash_model`).

    The key travels only in the `x-api-key` header
    (`test_the_key_travels_only_in_the_x_api_key_header`). This module logs
    nothing at all; what reaches a log on a failure is
    `ProviderError.provider_message`, which holds the provider's answer or the
    transport's own exception text, never the request body or its headers.
    """

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str = ANTHROPIC_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.name = name
        self._sleep = sleep
        self._jitter = jitter
        self.client = httpx.Client(
            base_url=base_url,
            headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
            transport=transport,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run one non-streaming completion."""
        body = to_anthropic(request)
        response = self._send(body)
        try:
            payload = response.json()
        except ValueError as exc:
            # A 200 that is not JSON at all - a proxy's HTML error page, a
            # truncated body. Pinned by `test_a_non_json_success_body_is_a_provider_error`.
            raise _unreadable(exc) from exc
        return from_anthropic(payload)

    def healthcheck(self) -> bool:
        """Always true, and it calls nothing.

        The seam exists so an earlier change's circuit breaker has somewhere to answer
        from. Probing the provider here would put a network call - and a live key
        in CI - on the readiness path, which
        `test_healthcheck_is_true_without_calling_the_provider` forbids.
        """
        return True

    def price_key(self, model: str) -> str:
        """The stable `"<provider>/<model>"` string the ledger stores."""
        return f"{self.name}/{model}"

    def close(self) -> None:
        """Release the connection pool."""
        self.client.close()

    def _send(self, body: Mapping[str, Any]) -> httpx.Response:
        """POST the body, retrying per the table, and raise `ProviderError` otherwise.

        The body is built once and resent unchanged, so a retry cannot ask the
        provider something the caller did not (`test_retry_resends_the_same_body`).
        """
        attempt = 0
        while True:
            try:
                response = self.client.post(MESSAGES_PATH, json=body)
            except httpx.RequestError as exc:
                error = self._transport_error(exc)
            else:
                if response.status_code == httpx.codes.OK:
                    return response
                error = self._status_error(response)

            if error.retryable and attempt < MAX_RETRIES:
                # A 429's `retry-after` is deliberately not read: honouring it
                # would let a provider hold a gateway worker for as long as it
                # asked, past the read timeout this adapter fixes. Two short
                # jittered waits, then the caller is told; an earlier change's breaker is
                # what a sustained 429 is supposed to trip.
                self._sleep(RETRY_BASE_DELAY * (2**attempt) * (0.5 + 0.5 * self._jitter()))
                attempt += 1
                continue
            raise error

    def _status_error(self, response: httpx.Response) -> ProviderError:
        """Turn a failure status into the typed error, keeping the provider's words out.

        `message` is client-facing and carries the provider name and the status
        only; the body goes to `provider_message` for the log.
        """
        status = response.status_code
        return ProviderError(
            f"{self.name}: the provider answered {status}",
            retryable=status == httpx.codes.TOO_MANY_REQUESTS or status >= 500,
            provider_message=response.text[:PROVIDER_MESSAGE_LIMIT],
        )

    def _transport_error(self, exc: httpx.RequestError) -> ProviderError:
        """Turn a request that never completed into the typed error.

        Only a timeout is retryable. Any other transport failure is reported as
        it is rather than tried twice, which keeps the retry table to the three
        cases the plan allows.
        """
        return ProviderError(
            f"{self.name}: the request did not complete ({type(exc).__name__})",
            retryable=isinstance(exc, httpx.TimeoutException),
            provider_message=str(exc),
        )
