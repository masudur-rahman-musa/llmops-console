"""The OpenAI-compatible adapter: llama.cpp, vLLM and anything else on that wire shape.

One class serves every provider whose `kind` is `openai_compatible`, because
`local` and `vllm` differ only in the endpoint the policy gives them. What makes
this adapter worth having is not a second vendor: it is the box we run
ourselves. A tenant whose only candidate is that box has no other route, so its
failures are final: the routes' fallback loop walks the resolver's own tuple,
and for such a tenant that tuple has one entry.

The mapping and the transport are kept apart, as in the adapter next door. Every
module-level mapping function here is pure. `grep -nE "^def [a-z]"` on this file
returns five, in this order: `to_openai`, `from_openai`, `from_openai_events`,
`to_openai_embedding`, `from_openai_embedding` - over dictionaries for the
completion pair and the embedding pair, over the decoded SSE lines for the
stream - so every direction is tested against payloads on disk with no transport
at all. `OpenAICompatibleAdapter` adds only the endpoints, the auth header and
the streamed body, because the client, the fixed timeouts, the retry table and
the provider-message truncation are `providers.transport`, shared with every
other HTTP adapter. No provider SDK is imported.

Two endpoints are served: `/chat/completions` and `/embeddings`.
The second serves embeddings as well as completions from the local box.

This wire shape is the one the gateway itself publishes, so the mapping is close
to an identity - and that is the trap. The adapter next door lifts the system
turn, renames `required`, rewrites a tool result as a user turn and re-serialises
a call's arguments; every one of those would be wrong here, so each is asserted
rather than assumed.

Every PROVIDER failure that leaves this module is a `ProviderError` carrying
`retryable`. The caller's own bad input is the deliberate exception: it leaves as
`IdentifierInvalid` (422) before any call is made. Six inputs reach that path,
the six `checked_tool_name` call sites this module has:
`grep -n checked_tool_name backend/app/providers/openai_compatible.py`
returns 9 lines - these two prose lines, the import, and those six calls - which
are a tool call's id and its name, the `tool_call_id` on a result turn, a
`tools[].name`, a forced `tool_choice.name` and a `response_format.name`.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.errors import ProviderError
from app.providers.base import (
    DEFAULT_MAX_TOKENS,
    HttpAdapter,
    checked_tool_name,
)
from app.providers.transport import ProviderTransport, unreadable
from app.schemas.provider import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FinishReason,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageTotal,
)

CHAT_COMPLETIONS_PATH = "/chat/completions"
"""Appended to the policy's `base_url`, which already ends in `/v1`.

A path carrying its own `/v1` would send the request to `/v1/v1/chat/completions`.
The tests build the whole
URL against a base url ending in `/v1`, mirroring the shipped policy's. Nothing
validates that suffix at load time, so a policy edited to drop the `/v1` 404s at
request time with no config-time signal. The admin reload is where that check
belongs."""

EMBEDDINGS_PATH = "/embeddings"
"""Appended to the policy's `base_url`, which already ends in `/v1`.

The same rule `CHAT_COMPLETIONS_PATH` follows and for the same reason: a path
carrying its own `/v1` would send the request to `/v1/v1/embeddings`.
The whole URL is built
against a base url ending in `/v1`, mirroring the shipped policy's."""

OPENAI_BASE_URL = "https://api.openai.com/v1"
"""The vendor's own endpoint, for a `kind: openai` provider that names no `base_url`.

It sits here rather than in `providers/factory.py` because it belongs to the wire
shape this module implements, beside the two paths that are appended to it: the
`/v1` is part of the constant precisely because `CHAT_COMPLETIONS_PATH` and
`EMBEDDINGS_PATH` carry none of their own. The factory's job is to choose which
endpoint a provider gets, not to know what the vendor's is.

Only the `openai` kind falls back to it. An `openai_compatible` provider naming
no `base_url` is dropped instead, because there is no such thing as a default box
of our own."""

FLOAT32_MAX = 3.4028234663852886e38
"""The largest finite value a float32 can hold, and therefore the largest
component a vector may carry.

The gateway renders base64 by packing components as float32
(`services/embedding_pipeline.py`), and `struct.pack("<f", 1e300)` raises
`OverflowError` - uncaught, that is a 500 with no code for what is really a
provider defect."""

OUTBOUND_ENCODING_FORMAT = "float"
"""What this adapter always asks the provider for, whatever the caller asked the
gateway for.

The gateway renders base64 itself from the float vector, so forwarding a
caller's `base64` would hand this reader a string where it expects numbers - and
a reader that passed it through unchanged would return a DOUBLE-encoded blob,
which decodes cleanly to nonsense."""

OPENAI_TOOL_CHOICE_BY_MODE = {"auto": "auto", "none": "none", "required": "required"}
"""The three nameless `ToolChoice` modes, in this wire's words.

Unlike Anthropic's table next door, every row here is the identity: this wire
spells `required` the same way the neutral contract does. The table exists so
that stays a fact a test can hold rather than an assumption. The `tool`
mode is absent because it carries a name and is built as an object."""

DONE_SENTINEL = "[DONE]"
"""The line that ends an OpenAI-compatible stream.

It is not JSON, so a reader that parses every `data:` line fails on every single
stream; and a reader that ignores it keeps parsing whatever a proxy appended
after it."""

SSE_DATA_PREFIX = "data: "
"""The only SSE field this reader looks at. Comments, `id:` lines and the blank
separators carry nothing the `data:` payload does not."""


def _tool_call_wire(call: ToolCall, field_name: str) -> dict[str, Any]:
    """Render one tool call the model made, as this wire replays it.

    Both ids are checked here rather than at the contract, so an invalid one
    raises before any outbound request is built.

    `arguments` is passed through as the JSON STRING it already is. The adapter
    next door has to parse it, because Anthropic's `input` is an object; doing
    the same here would re-serialise the model's own output and change what the
    model is told it said.
    """
    return {
        "id": checked_tool_name(call.id, f"{field_name}.id"),
        "type": "function",
        "function": {
            "name": checked_tool_name(call.name, f"{field_name}.name"),
            "arguments": call.arguments,
        },
    }


def _rendered_turn(index: int, message: ChatMessage) -> dict[str, Any]:
    """Render one turn. Every role survives as itself, including `system` and `tool`.

    That is the whole difference from the Anthropic mapping, which has neither
    role and has to rewrite a tool result as a `user` turn.
    """
    if message.role == "tool":
        return {
            "role": "tool",
            # `or ""` rather than a branch: an absent id fails the same pre-check
            # as a malformed one, with the same 422 naming the same field.
            "tool_call_id": checked_tool_name(
                message.tool_call_id or "", f"messages[{index}].tool_call_id"
            ),
            "content": message.content,
        }

    turn: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        turn["tool_calls"] = [
            _tool_call_wire(call, f"messages[{index}].tool_calls[{position}]")
            for position, call in enumerate(message.tool_calls)
        ]
    return turn


def to_openai(request: ChatRequest) -> dict[str, Any]:
    """Render a normalised request as an OpenAI-compatible chat completions body.

    `max_tokens` is always sent: this wire treats an omitted value as unbounded,
    and an unbounded generation on a CPU box holds a worker for minutes. `stream` and
    `stream_options`
    are deliberately NOT set here - they belong to the streamed call alone.
    """
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [
            _rendered_turn(index, message) for index, message in enumerate(request.messages)
        ],
        "max_tokens": request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_TOKENS,
    }

    if request.temperature is not None:
        # `is not None`, not truthiness: an explicit temperature of 0 is the
        # setting an eval run depends on, and dropping it would change the answer.
        body["temperature"] = request.temperature

    if request.tools:
        # Omitted entirely rather than sent empty: some OpenAI-compatible servers
        # validate an empty `tools` rather than ignoring it.
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": checked_tool_name(tool.name, f"tools[{index}].name"),
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            for index, tool in enumerate(request.tools)
        ]

    if request.tool_choice is not None:
        choice = request.tool_choice
        if choice.mode == "tool":
            # `or ""` for the same reason as the tool result id above: a `tool`
            # mode with no name fails the pre-check rather than sending `null`.
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": checked_tool_name(choice.name or "", "tool_choice.name")},
            }
        else:
            body["tool_choice"] = OPENAI_TOOL_CHOICE_BY_MODE[choice.mode]

    if request.response_format is not None:
        # A first-class field on this wire, so it neither becomes a tool nor
        # touches `tool_choice` - both of which the Anthropic mapping has to do,
        # and both of which would take the caller's own tools away here. `strict` is
        # the caller's to set: forwarded as given, and absent when they left it
        # out, because a strict schema must close every object and require every
        # field, and most callers' schemas do neither.
        json_schema: dict[str, Any] = {
            "name": checked_tool_name(request.response_format.name, "response_format.name"),
            "schema": dict(request.response_format.json_schema),
        }
        if request.response_format.strict is not None:
            json_schema["strict"] = request.response_format.strict
        body["response_format"] = {"type": "json_schema", "json_schema": json_schema}

    return body


def _usage(block: Mapping[str, Any]) -> Usage:
    """Read one usage block, keeping the provider's own total where it gave one.

    `total_tokens` is not recomputed: a server counting a cached or reused prefix
    reports a total that is not the sum, and the ledger books what the provider
    says it served.
    The two counts are required - a missing one raises rather than defaulting to
    zero, which would book a real, billed call as free - while an absent
    total falls back to the sum, because some builds omit it.

    OpenAI counts its cached prompt tokens INSIDE `prompt_tokens`, under
    `prompt_tokens_details.cached_tokens`. They become `cache_read_tokens`, and
    `prompt_tokens` keeps only the rest, the meaning Anthropic's count has, so the
    input rate never bills a cached token twice. Absent or `null` details read 0;
    a cached count above the prompt count is unreadable. Pinned by
    `test_openai_cache_tokens.py`.
    """
    prompt_tokens = block["prompt_tokens"]
    completion_tokens = block["completion_tokens"]
    total = block.get("total_tokens")
    details = block.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    if not isinstance(cached, int) or isinstance(cached, bool) or cached < 0:
        raise ValueError("cached_tokens: not a count of tokens")
    if cached > prompt_tokens:
        raise ValueError("cached_tokens: more than prompt_tokens")
    return Usage(
        prompt_tokens=prompt_tokens - cached,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens if total is None else total,
        cache_read_tokens=cached,
    )


def from_openai(payload: Mapping[str, Any]) -> ChatResponse:
    """Read an OpenAI-compatible chat completion into the normalised response.

    Everything is read from `choices[0]`: a `content` of `null` becomes the empty
    string the frozen contract requires, and `tool_calls` keep the JSON *string*
    the published API promises its clients rather than being parsed into a dict.

    An answer with real usage but nothing to render is refused rather than served
    as an empty string a tenant is billed for, which is the same verdict the
    Anthropic path reaches for the same reply. An empty
    `choices` is refused the same way rather than escaping as an `IndexError`.
    """
    try:
        choices = payload["choices"]
        if not choices:
            raise ProviderError(
                "the provider answered with no choices",
                retryable=False,
                provider_message=f"id={payload.get('id')!r}, choices={choices!r}",
            )
        choice = choices[0]
        message = choice["message"]

        content = message.get("content") or ""
        calls = tuple(
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
            )
            for call in message.get("tool_calls") or ()
        )

        usage = _usage(payload["usage"])

        if not content and not calls:
            # Not `unreadable`: the body parsed fine, it simply carried nothing
            # this adapter can render. Saying so is what stops an operator
            # chasing a parse bug that is not there.
            raise ProviderError(
                "the provider answered with no readable content",
                retryable=False,
                provider_message=f"finish_reason={choice.get('finish_reason')!r}",
            )

        return ChatResponse(
            model=payload["model"],
            content=content,
            tool_calls=calls,
            finish_reason=choice["finish_reason"],
            usage=usage,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise unreadable(exc) from exc


@dataclass
class _OpenCall:
    """One tool call being assembled from its `delta.tool_calls[]` fragments.

    Kept in a dict keyed by the call's own `index`, never by "the call that
    opened last": this wire interleaves two calls' fragments, so matching on the
    index is what stops one call's JSON being appended to another's.
    """

    id: str = ""
    name: str = ""
    fragments: list[str] = field(default_factory=list)


def _sse_payload(line: str) -> dict[str, Any] | None:
    """The JSON object on one `data:` line, `None` for a line that carries none.

    The `[DONE]` sentinel is not JSON and is handled by the caller, so it never
    reaches this function.
    """
    if not line.startswith(SSE_DATA_PREFIX):
        return None
    payload: Any = json.loads(line[len(SSE_DATA_PREFIX) :])
    if not isinstance(payload, dict):
        raise TypeError(f"an SSE data line carried {type(payload).__name__}, not an object")
    return payload


def from_openai_events(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Read OpenAI-compatible chat completion chunks into the neutral stream events.

    Pure, and pure on purpose: it takes the decoded lines `iter_lines()` yields
    and never touches a transport, so every shape it has to handle is a `.sse`
    file rather than a mocked socket.

    Text arrives as `delta.content` and is forwarded in the provider's order; the
    opening chunk's empty string is not an answer fragment and is not forwarded.
    A tool call is ASSEMBLED: its `function.arguments` fragments are buffered per
    `index` and every buffered call is emitted, in index order, when the finish
    reason arrives. Flushing a call when the NEXT index opens would
    be a guess about provider ordering, and this wire interleaves them.

    Usage rides a final chunk whose `choices` is empty and is required, which is
    why the adapter always asks for it. A stream that ends without it raises
    rather than yielding a total the ledger would price as zero,
    and so does one that renders nothing at all.
    """
    started = False
    rendered = False
    finish_reason: str | None = None
    usage: Usage | None = None
    open_calls: dict[int, _OpenCall] = {}

    try:
        for line in lines:
            if line.startswith(SSE_DATA_PREFIX) and line[len(SSE_DATA_PREFIX) :] == DONE_SENTINEL:
                break
            payload = _sse_payload(line)
            if payload is None:
                continue

            if not started:
                # On the first chunk rather than on the first text delta: an
                # answer that is only a tool call has no text delta and still
                # has to open.
                started = True
                yield StreamStart()

            if payload.get("usage"):
                usage = _usage(payload["usage"])

            choices = payload.get("choices") or ()
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            content = delta.get("content")
            if content:
                rendered = True
                yield TextDelta(text=content)

            for fragment in delta.get("tool_calls") or ():
                open_call = open_calls.setdefault(fragment["index"], _OpenCall())
                function = fragment.get("function") or {}
                if fragment.get("id"):
                    open_call.id = fragment["id"]
                if function.get("name"):
                    open_call.name = function["name"]
                if function.get("arguments"):
                    open_call.fragments.append(function["arguments"])

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
                for index in sorted(open_calls):
                    closed = open_calls[index]
                    rendered = True
                    yield ToolCallDelta(
                        index=index,
                        call=ToolCall(
                            id=closed.id,
                            name=closed.name,
                            arguments="".join(closed.fragments),
                        ),
                    )
                open_calls.clear()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise unreadable(exc) from exc

    if finish_reason is None or usage is None:
        raise ProviderError(
            "the provider ended the stream before it finished",
            retryable=False,
            provider_message=f"finish_reason={finish_reason!r}, usage={usage!r}",
        )

    if not rendered:
        # The streaming twin of `from_openai`'s no-readable-content guard, and it
        # exists for the same reason: finishing the stream with a usage chunk
        # would bill the tenant for an empty answer. Second, not first: a stream
        # cut off before its first delta renders nothing EITHER, and reporting
        # that as an empty answer would point an operator at a content bug
        # instead of the dropped connection it was.
        raise ProviderError(
            "the provider streamed no readable content",
            retryable=False,
            provider_message=f"finish_reason={finish_reason!r}",
        )

    yield FinishReason(reason=finish_reason)
    yield UsageTotal(usage=usage)


def to_openai_embedding(request: EmbeddingRequest) -> dict[str, Any]:
    """Render a normalised embedding request as an OpenAI-compatible embeddings body.

    Three fields, and only three: this endpoint validates what it is sent rather
    than ignoring it, so a `max_tokens` or an invented `dimensions` copied in
    from the chat body would make every embedding request a provider 400.

    `input` is always a list, even for a single text. The wire accepts a bare
    string, so a server would answer 200 either way and nothing downstream would
    notice the shape differed by batch size.
    """
    return {
        "model": request.model,
        "input": list(request.input),
        "encoding_format": OUTBOUND_ENCODING_FORMAT,
    }


def _embedding_usage(block: Mapping[str, Any]) -> EmbeddingUsage:
    """Read an embedding usage block. `prompt_tokens` is required, the total is not.

    The two counts are handled differently on purpose. A missing `prompt_tokens`
    raises, because zero-filling it would book a real, billed call as free. A missing
    `total_tokens` falls back to the prompt count rather than raising, because
    some llama.cpp builds omit it and there is no second count on this wire to
    add - the prompt count IS the whole total.
    """
    prompt_tokens = block["prompt_tokens"]
    total = block.get("total_tokens")
    return EmbeddingUsage(
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens if total is None else total,
    )


def _checked_vector(vector: Sequence[float]) -> None:
    """Refuse a vector whose NUMBERS are unusable, not merely a container that is.

    Annotated `Sequence[float]` although the payload is `Any`: that is what a
    vector on this wire is supposed to be, and anything else - a list of strings
    is the shape a release change would produce - fails inside on a `TypeError`
    that `from_openai_embedding` turns into the same `unreadable` 502 as any
    other unparseable body.

    The other guards in this module read the envelope - is `data` empty, are the
    indices `0..n-1`, is there a usage block. None of them looks inside a vector,
    and all three failures below produce a well-formed envelope:

    * an EMPTY vector renders as `[]`, or as the empty string on the base64 path,
      and answers 200. The caller stores a zero-length vector against a real row
      and every similarity computed from it afterwards is quietly wrong - the
      same verdict `from_openai`'s no-readable-content guard reaches, for the
      same reason;
    * a `NaN` or an infinity, which a badly quantised local model does produce,
      cannot be serialised by the JSON renderer (it runs with `allow_nan=False`),
      so it reaches the client as a 500 with no code rather than the 502 a retry
      loop is written against;
    * a finite component beyond `FLOAT32_MAX` raises `OverflowError` inside the
      base64 packer, with the same untyped 500 as the result.

    Refusing all three here means the route can pack any vector it is handed.
    """
    if not vector:
        raise ProviderError(
            "the provider answered an embedding with no components",
            retryable=False,
            provider_message=f"vector={vector!r}",
        )
    for component in vector:
        if not math.isfinite(component) or abs(component) > FLOAT32_MAX:
            raise ProviderError(
                "the provider answered an embedding this gateway cannot represent",
                retryable=False,
                provider_message=f"component={component!r}",
            )


def from_openai_embedding(payload: Mapping[str, Any]) -> EmbeddingResponse:
    """Read an OpenAI-compatible embeddings answer into the normalised response.

    The vectors are ordered by each item's OWN `index`, never by the order the
    provider serialised them, and the sorted indices must be exactly `0..n-1`.
    Both halves are load bearing, and neither is paranoia about a malicious
    server: the caller matches vectors back to its own rows by index and by
    nothing else, so a positional reader, or a sorted reader that accepts a
    duplicated or a one-based numbering, returns real vectors attached to the
    wrong texts - plausible numbers, no error anywhere.

    An empty `data` is refused rather than answered as an empty list, which is
    the twin of `from_openai`'s no-choices guard and reaches the caller as a 502
    the routes' fallback loop can move past. The
    vectors themselves are checked by `_checked_vector`, because every guard in
    this function reads the ENVELOPE and a well-formed envelope can carry numbers
    no client can use.
    """
    try:
        data = payload["data"]
        if not data:
            raise ProviderError(
                "the provider answered with no embeddings",
                retryable=False,
                provider_message=f"model={payload.get('model')!r}, data={data!r}",
            )

        for item in data:
            _checked_vector(item["embedding"])

        embeddings = tuple(
            sorted(
                (Embedding(index=item["index"], vector=tuple(item["embedding"])) for item in data),
                key=lambda embedding: embedding.index,
            )
        )
        if [embedding.index for embedding in embeddings] != list(range(len(embeddings))):
            raise ProviderError(
                "the provider numbered its embeddings in a way the caller cannot use",
                retryable=False,
                provider_message=f"indices={[item.index for item in embeddings]!r}",
            )

        return EmbeddingResponse(
            model=payload["model"],
            embeddings=embeddings,
            usage=_embedding_usage(payload["usage"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise unreadable(exc) from exc


class OpenAICompatibleAdapter(HttpAdapter):
    """One OpenAI-compatible endpoint, addressed by its policy provider name.

    `name` is that policy name, never the kind: `local` and `vllm` are the same
    kind on two different boxes, and the ledger, the metric label and the
    resolver all have to address them apart.

    `api_key` is optional, and the reason is that the shipped `local` and `vllm`
    providers sit on a private network and declare no `key_env`, so no
    `Authorization` header is sent at all rather than an empty bearer. A key given travels only
    in that header.

    Two methods are SEAMS rather than behaviour: `_auth_headers` and `_path`,
    added for `providers/azure_openai.py`, whose adapter differs from
    this one in the auth header and the request URL and in nothing else. Both
    defaults are identity here, so the two kinds this class serves directly -
    `openai_compatible` and `openai` - send exactly what they
    sent before.

    This module logs nothing. What reaches a log on a failure is
    `ProviderError.provider_message`, which holds the provider's answer or the
    transport's own exception text, never the request body or its headers.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.name = name
        self.http = ProviderTransport(
            name=name,
            base_url=base_url,
            headers=self._auth_headers(api_key),
            transport=transport,
            sleep=sleep,
            jitter=jitter,
        )

    def _auth_headers(self, api_key: str | None) -> dict[str, str]:
        """The headers that carry the key on this wire: a bearer token, or none.

        A seam. Called once, from `__init__`, so the key is set on the client and
        never rebuilt per request. Overridden by `AzureOpenAIAdapter`, which sends
        an `api-key` header instead.

        Empty rather than an empty bearer when there is no key, because the shipped
        `local` and `vllm` sit on a private network and declare no `key_env`.
        """
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _path(self, leaf: str, model: str) -> str:
        """Where `leaf` is addressed on this wire, given the model being called.

        A seam, and on this wire the identity: `model` travels in the BODY here,
        so the path is the leaf unchanged and the argument is unused.
        Overridden by `AzureOpenAIAdapter`, where the model names a deployment in
        the URL path and an `api-version` query is mandatory.
        """
        return leaf

    @property
    def client(self) -> httpx.Client:
        """The underlying HTTP client, so the fixed timeouts stay observable here."""
        return self.http.client

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run one non-streaming completion."""
        body = to_openai(request)
        response = self.http.send(self._path(CHAT_COMPLETIONS_PATH, request.model), body)
        try:
            payload = response.json()
        except ValueError as exc:
            # A 200 that is not JSON at all - a reverse proxy's HTML error page,
            # a truncated body.
            raise unreadable(exc) from exc
        return from_openai(payload)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Run one embedding call.

        The same three lines `chat()` is, over the same `ProviderTransport`, so
        the timeouts, the retry table and the provider-message truncation are the
        shared ones rather than a second copy that would drift.
        A 200 that is not JSON - a reverse proxy's HTML error page - is the same
        `unreadable` 502 here as there.

        The one thing this method does that `chat()` does not is COUNT. Nothing
        `from_openai_embedding` can see settles whether the answer is complete:
        a server that drops a slot answers 63 well-formed vectors numbered 0..62
        to a 64-item request, which satisfies every guard that reads the payload
        alone. Only here are both the request and the answer in hand, which is
        why the completeness half of the protocol's guarantee is checked here and
        the numbering half is checked there.
        """
        body = to_openai_embedding(request)
        response = self.http.send(self._path(EMBEDDINGS_PATH, request.model), body)
        try:
            payload = response.json()
        except ValueError as exc:
            raise unreadable(exc) from exc

        answer = from_openai_embedding(payload)
        if len(answer.embeddings) != len(request.input):
            # Not retryable, and deliberately not for the usual reason - a server
            # that dropped a slot under load probably would answer completely on
            # a second attempt. `retryable` is what the routes' fallback loop
            # reads to decide whether to try a DIFFERENT provider, and for an embedding
            # that is the one recovery that must never happen: another provider
            # means another vector space, so the caller gets plausible numbers
            # that do not compare against anything it has already stored
            # An error it can act on is the better answer.
            raise ProviderError(
                f"{self.http.name}: the provider answered "
                f"{len(answer.embeddings)} embeddings for {len(request.input)} inputs",
                retryable=False,
                provider_message=f"model={answer.model!r}",
            )
        return answer

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Run one streamed completion, yielding neutral events as they arrive.

        `stream_options.include_usage` is always sent, never left to the caller:
        without it this wire sends no usage chunk, and the reader would refuse
        every streamed answer rather than book one at zero.

        The open and the read are separate. `ProviderTransport.open_stream` owns
        the retry loop, so a retry cannot reach around the iteration: once one
        event has been yielded the client has already been shown part of an
        answer, and nothing that cannot be un-sent is resent.
        """
        body = to_openai(request)
        # Copied rather than mutated in place: `to_openai` is the shape the
        # non-streaming path sends, and the two extra fields belong to this call.
        streamed = {**body, "stream": True, "stream_options": {"include_usage": True}}
        path = self._path(CHAT_COMPLETIONS_PATH, request.model)
        with self.http.open_stream(path, streamed) as response:
            try:
                yield from from_openai_events(response.iter_lines())
            except httpx.RequestError as exc:
                raise self.http.transport_error(exc) from exc
