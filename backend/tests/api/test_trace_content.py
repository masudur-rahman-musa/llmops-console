"""What a tenant with `trace_content` on sends to Langfuse: the redacted talk and the answer.

An earlier change and a design note. Every case is a real request through `TestClient`, read back from
the in-memory span exporter. `evals` is the tenant under test: it is switched on in the
shipped policy, and the off case runs on a copy where it is switched off.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.cache import InProcessResponseCache
from app.core.config import Settings
from app.providers.base import Adapter

from .conftest import (
    CHAT,
    TEST_KEYS,
    EmbeddingStubAdapter,
    StubAdapter,
    app_of,
    auth_headers,
    build_chat_client,
    chat_body,
    embeddings_body,
    evals_switched_off,
    post_chat,
    post_embeddings,
    stream_events,
    stub_response,
)

INPUT_KEY = "langfuse.observation.input"
OUTPUT_KEY = "langfuse.observation.output"
FINISH_KEY = "langfuse.observation.metadata.finish_reason"

EMAIL = "ana@example.com"
SENTINEL = "zqx-trace-content-sentinel-106"
ANSWER_WITH_PLACEHOLDER = "Reply to <EMAIL_1> soon."
STREAM_PIECES = ("Reply to ", "<EMAIL_1>", " soon.")
CUT_LIMIT = 8000
CUT_MARK = " [cut]"
REHYDRATE_HEADER = "X-Gateway-Rehydrate"


def finished(spans: InMemorySpanExporter) -> tuple[ReadableSpan, ...]:
    """Every span this request finished; fails when there are none."""
    recorded = tuple(spans.get_finished_spans())
    assert recorded, "the request emitted no span at all"
    return recorded


def root_of(recorded: Sequence[ReadableSpan]) -> ReadableSpan:
    """The one span with no local parent."""
    roots = [span for span in recorded if span.parent is None or span.parent.is_remote]
    assert len(roots) == 1, [span.name for span in roots]
    return roots[0]


def provider_span(recorded: Sequence[ReadableSpan]) -> ReadableSpan:
    """The one `provider:` span."""
    matches = [span for span in recorded if span.name.startswith("provider:")]
    assert len(matches) == 1, [span.name for span in recorded]
    return matches[0]


def attribute(span: ReadableSpan, key: str) -> str:
    """A string attribute of a span; fails when it is missing."""
    value = dict(span.attributes or {}).get(key)
    assert isinstance(value, str), f"{span.name} has no {key}, got {value!r}"
    return value


def everything_on(recorded: Sequence[ReadableSpan]) -> str:
    """All attribute values of all spans in one string."""
    return " ".join(
        f"{key}={value}" for span in recorded for key, value in dict(span.attributes or {}).items()
    )


def both_providers(**stub_arguments: object) -> dict[str, Adapter]:
    """The same scripted stub under both chat providers, so the route may pick either."""
    return {
        name: StubAdapter(name=name, **stub_arguments)  # type: ignore[arg-type]
        for name in ("anthropic", "openai")
    }


@pytest.fixture(scope="module")
def plain_adapters() -> dict[str, Adapter]:
    """Stubs that answer with a placeholder in the text."""
    return both_providers(response=stub_response(content=ANSWER_WITH_PLACEHOLDER))


@pytest.fixture(scope="module")
def plain_client(settings: Settings, plain_adapters: Mapping[str, Adapter]) -> Iterator[TestClient]:
    """One app for the plain-answer rows."""
    with build_chat_client(settings, plain_adapters) as client:
        yield client


@pytest.fixture(scope="module")
def stream_client(settings: Settings) -> Iterator[TestClient]:
    """One app whose stubs stream the pieces with a placeholder in the middle."""
    adapters = both_providers(events=stream_events(fragments=STREAM_PIECES))
    with build_chat_client(settings, adapters) as client:
        yield client


@pytest.fixture(scope="module")
def cache_client(settings: Settings) -> Iterator[TestClient]:
    """One app for the cache-hit row."""
    with build_chat_client(settings, both_providers(response=stub_response())) as client:
        yield client


def used_adapter(client: TestClient, response_provider: str) -> StubAdapter:
    """The stub that served the last request."""
    adapter = app_of(client).state.adapters[response_provider]
    assert isinstance(adapter, StubAdapter)
    return adapter


def post_rehydrating(client: TestClient, **extra: object) -> Response:
    """POST under `pii` as `evals`, asking for the placeholders to be put back in the answer."""
    store = app_of(client).state.redaction_map_store
    map_id = store.put("evals", {"<EMAIL_1>": EMAIL})
    headers = auth_headers(key=TEST_KEYS["evals"], tenant="evals", data_class="pii")
    headers[REHYDRATE_HEADER] = map_id
    response: Response = client.post(
        CHAT, json=chat_body(model="agent-default", messages=CONVERSATION, **extra), headers=headers
    )
    return response


def send_pii(client: TestClient, messages: list[dict[str, str]], **extra: object) -> str:
    """POST under data class `pii` as `evals`; returns the provider that served it."""
    response = post_chat(
        client, chat_body(model="agent-default", messages=messages, **extra), data_class="pii"
    )
    assert response.status_code == 200, response.text
    return response.headers["x-gateway-provider"]


CONVERSATION = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": f"write to {EMAIL} about the invoice"},
]


def test_a_switched_on_tenant_sends_the_redacted_conversation(
    plain_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an implementation that sends the request as the client wrote it (the address
    would reach Langfuse), sends only the last message, or drops the roles. The Input must
    equal, message for message, what the provider received."""
    served = send_pii(plain_client, CONVERSATION)
    received = used_adapter(plain_client, served).requests[-1].messages
    assert EMAIL not in " ".join(str(message.content) for message in received), (
        "redaction did not run, so the check below proves nothing"
    )

    recorded = finished(spans)
    raw = attribute(provider_span(recorded), INPUT_KEY)

    assert json.loads(raw) == [
        {"role": message.role, "content": message.content} for message in received
    ]
    assert "EMAIL_1" in raw
    assert EMAIL not in everything_on(recorded)


def test_a_switched_on_tenant_sends_the_answer(
    plain_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an Output taken after the placeholders are put back (the address would
    reach Langfuse), a bare string instead of the role and content object, and an Output
    set on the provider step only."""
    response = post_rehydrating(plain_client)
    assert response.status_code == 200, response.text
    assert EMAIL in response.json()["choices"][0]["message"]["content"], (
        "the client must get the address back, or the check below proves nothing"
    )

    recorded = finished(spans)
    expected = {"role": "assistant", "content": ANSWER_WITH_PLACEHOLDER}

    assert json.loads(attribute(provider_span(recorded), OUTPUT_KEY)) == expected
    assert json.loads(attribute(root_of(recorded), OUTPUT_KEY)) == expected


def test_a_streamed_answer_is_gathered_into_the_output(
    stream_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a stream whose Output stays the note, holds only the last piece, or is
    joined after the placeholders are put back."""
    response = post_rehydrating(stream_client, stream=True)
    assert response.status_code == 200, response.text
    assert EMAIL in response.text, "the client must get the address back"

    recorded = finished(spans)
    expected = {"role": "assistant", "content": "".join(STREAM_PIECES)}

    assert json.loads(attribute(provider_span(recorded), OUTPUT_KEY)) == expected
    assert json.loads(attribute(root_of(recorded), OUTPUT_KEY)) == expected
    assert EMAIL not in everything_on(recorded)


def test_the_root_input_is_the_redacted_conversation(
    plain_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a root that keeps the note, or one set before redaction ran, which would put
    the raw address on the trace list while the provider step is clean."""
    send_pii(plain_client, CONVERSATION)

    recorded = finished(spans)
    root_input = attribute(root_of(recorded), INPUT_KEY)

    assert json.loads(root_input) == json.loads(attribute(provider_span(recorded), INPUT_KEY))
    assert json.loads(root_input)[1]["content"].startswith("write to ")
    assert EMAIL not in root_input


def sent_content(client: TestClient, spans: InMemorySpanExporter, text: str) -> str:
    """Send one `internal` message of `text`; return its content as the Input shows it."""
    response = post_chat(
        client,
        chat_body(model="agent-default", messages=[{"role": "user", "content": text}]),
    )
    assert response.status_code == 200, response.text
    sent = json.loads(attribute(provider_span(finished(spans)), INPUT_KEY))
    assert len(sent) == 1
    content: str = sent[0]["content"]
    return content


def test_a_message_at_8000_characters_is_kept_whole(
    plain_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a cut limit one too low (7999), which would mark a whole message as cut."""
    text = "a" * CUT_LIMIT

    assert sent_content(plain_client, spans, text) == text


def test_a_message_at_8001_characters_is_cut(
    plain_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a limit one too high (8001 kept) and a cut with no mark, which would pass
    a long message off as a whole one."""
    content = sent_content(plain_client, spans, "a" * (CUT_LIMIT + 1))

    assert content == "a" * CUT_LIMIT + CUT_MARK


def test_a_cut_applies_to_each_message_alone(
    plain_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches one cut over the whole conversation: a long first message must not eat the
    short second one."""
    response = post_chat(
        plain_client,
        chat_body(
            model="agent-default",
            messages=[
                {"role": "user", "content": "b" * (CUT_LIMIT + 5)},
                {"role": "user", "content": "short"},
            ],
        ),
    )
    assert response.status_code == 200, response.text

    sent = json.loads(attribute(provider_span(finished(spans)), INPUT_KEY))

    assert [message["content"] for message in sent] == ["b" * CUT_LIMIT + CUT_MARK, "short"]


def test_the_finish_reason_goes_as_metadata(
    plain_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a finish reason lost once the note is gone, sent as the stub's raw
    `end_turn` instead of the OpenAI word, or set on the provider step only."""
    send_pii(plain_client, CONVERSATION)

    recorded = finished(spans)

    assert attribute(provider_span(recorded), FINISH_KEY) == "stop"
    assert attribute(root_of(recorded), FINISH_KEY) == "stop"


def test_a_streamed_finish_reason_goes_as_metadata(
    stream_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a stream that never reports the finish reason as metadata."""
    response = post_chat(stream_client, chat_body(model="agent-default", stream=True))
    assert response.status_code == 200, response.text

    assert attribute(provider_span(finished(spans)), FINISH_KEY) == "stop"


def test_a_switched_off_tenant_sends_only_the_note(
    plain_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches an implementation that ignores the switch and sends the text anyway. The
    same request is sent first with the switch on, so the sentinel is known to reach a
    span when allowed and its absence afterwards is the switch's doing."""
    messages = [{"role": "user", "content": SENTINEL}]
    send_pii(plain_client, messages)
    assert SENTINEL in everything_on(finished(spans)), "the switched-on control sent no text"

    spans.clear()
    with evals_switched_off(tmp_path, plain_client):
        send_pii(plain_client, messages)
    recorded = finished(spans)
    provider = provider_span(recorded)

    assert SENTINEL not in everything_on(recorded)
    assert ANSWER_WITH_PLACEHOLDER not in everything_on(recorded)
    assert attribute(provider, INPUT_KEY) == (
        "not sent: prompts stay inside the gateway (alias agent-default, 1 message)"
    )
    assert attribute(provider, OUTPUT_KEY).startswith("not sent (finish reason stop")


def test_a_cache_hit_and_an_embedding_keep_the_note(
    cache_client: TestClient,
    embeddings_client: TestClient,
    embedding_stub: EmbeddingStubAdapter,
    spans: InMemorySpanExporter,
) -> None:
    """Catches content sent for a cache hit (no provider answered, so there is no answer)
    or for an embedding call (its input is text to embed, not a conversation), when the
    switch is on."""
    for client in (cache_client, embeddings_client):
        assert app_of(client).state.policy.policy.tenants["evals"].trace_content is True

    app_of(cache_client).state.response_cache = InProcessResponseCache()
    body = chat_body(
        model="agent-default", temperature=0, messages=[{"role": "user", "content": SENTINEL}]
    )
    assert post_chat(cache_client, body).headers["x-gateway-cache"] == "miss"
    spans.clear()
    hit = post_chat(cache_client, body)
    assert hit.headers["x-gateway-cache"] == "hit"

    root = root_of(finished(spans))
    assert attribute(root, INPUT_KEY) == (
        "not sent: prompts stay inside the gateway (alias agent-default, 1 message)"
    )
    assert attribute(root, OUTPUT_KEY) == "not sent"

    spans.clear()
    response = post_embeddings(embeddings_client, embeddings_body(input=SENTINEL))
    assert response.status_code == 200, response.text
    recorded = finished(spans)

    assert attribute(provider_span(recorded), INPUT_KEY) == (
        "not sent: prompts stay inside the gateway (alias embed)"
    )
    assert SENTINEL not in everything_on(recorded)
