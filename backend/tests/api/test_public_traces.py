"""Public traces: every span of a request carries `langfuse.trace.public` when it may.

An earlier change, a design note and 0062. Every case is a real request through `TestClient`,
read back from the in-memory span exporter. The gateway switch is
`Settings.public_trace_langfuse`; a tenant can opt out with `trace_public: false` in the
policy, and cannot turn it on alone.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.config import Settings
from app.model_routing.policy import PolicyStore
from app.providers.base import Adapter

from .conftest import (
    CHAT,
    SHIPPED_POLICY_PATH,
    SHIPPED_PROVIDERS_DIR,
    EmbeddingStubAdapter,
    StubAdapter,
    app_of,
    auth_headers,
    build_chat_client,
    chat_body,
    embeddings_body,
    post_chat,
    post_embeddings,
    stub_response,
)

PUBLIC_KEY = "langfuse.trace.public"


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


def carriers(recorded: Sequence[ReadableSpan]) -> list[ReadableSpan]:
    """The spans that carry the public key at all."""
    return [span for span in recorded if PUBLIC_KEY in dict(span.attributes or {})]


def adapters() -> dict[str, Adapter]:
    """The same scripted stub under both chat providers."""
    return {
        name: StubAdapter(name=name, response=stub_response()) for name in ("anthropic", "openai")
    }


def policy_store_with_evals_public(directory: Path, value: bool) -> PolicyStore:
    """A store over a copy of the shipped policy where `evals` has the given `trace_public`."""
    document = yaml.safe_load(SHIPPED_POLICY_PATH.read_text(encoding="utf-8"))
    document["tenants"]["evals"]["trace_public"] = value
    copy = directory / "routing-evals-public.yaml"
    copy.write_text(yaml.safe_dump(document), encoding="utf-8")
    return PolicyStore(copy, SHIPPED_PROVIDERS_DIR)


@contextmanager
def evals_trace_public(directory: Path, client: TestClient, value: bool) -> Iterator[None]:
    """Run the block with `evals` carrying `trace_public: value`, then put the policy back."""
    store = policy_store_with_evals_public(directory, value)
    original = app_of(client).state.policy
    app_of(client).state.policy = store
    try:
        yield
    finally:
        app_of(client).state.policy = original


@pytest.fixture(scope="module")
def public_settings(settings: Settings) -> Settings:
    """The shared settings with the switch on."""
    return settings.model_copy(update={"public_trace_langfuse": True})


@pytest.fixture(scope="module")
def on_client(public_settings: Settings) -> Iterator[TestClient]:
    """One app with the switch on."""
    with build_chat_client(public_settings, adapters()) as client:
        yield client


@pytest.fixture(scope="module")
def off_client(settings: Settings) -> Iterator[TestClient]:
    """One app with the default settings."""
    with build_chat_client(settings, adapters()) as client:
        yield client


@pytest.fixture(scope="module")
def on_embeddings_client(
    public_settings: Settings, embedding_stub: EmbeddingStubAdapter
) -> Iterator[TestClient]:
    """One app with the switch on and an embedding stub."""
    with build_chat_client(public_settings, {embedding_stub.name: embedding_stub}) as client:
        yield client


def send_chat(client: TestClient) -> None:
    """One chat as `evals`; it must succeed."""
    response = post_chat(client, chat_body())
    assert response.status_code == 200, response.text


def test_the_switch_on_marks_every_span_public(
    on_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a label left off any span (root only, or `auth` before the decision), a string
    "true" instead of a bool, or a provider step opened without the copy."""
    send_chat(on_client)
    recorded = finished(spans)

    assert len(recorded) > 1, "a single span would make the every-span check empty"
    names = {span.name for span in recorded}
    assert "auth" in names
    assert any(name.startswith("provider:") for name in names)
    for span in recorded:
        assert dict(span.attributes or {}).get(PUBLIC_KEY) is True, span.name


def test_every_span_of_an_embedding_request_is_public(
    on_embeddings_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a mark placed in the chat route only, which leaves embedding traces private."""
    response = post_embeddings(on_embeddings_client, embeddings_body())
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    assert len(recorded) > 1
    for span in recorded:
        assert dict(span.attributes or {}).get(PUBLIC_KEY) is True, span.name


def test_the_last_started_span_carries_the_mark(
    on_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Langfuse keeps the value of the span that started last, so that span must be public."""
    send_chat(on_client)

    last = max(finished(spans), key=lambda span: span.start_time or 0)
    assert dict(last.attributes or {}).get(PUBLIC_KEY) is True, last.name


def test_the_switch_off_marks_nothing(off_client: TestClient, spans: InMemorySpanExporter) -> None:
    """Catches an implementation that marks every trace whatever the setting (default on)."""
    send_chat(off_client)

    assert carriers(finished(spans)) == []


def test_a_tenant_that_opts_out_stays_private(
    on_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches an implementation that ignores `trace_public: false`. The same app marks the
    request first without the opt-out, so the key's absence afterwards is the opt-out's doing."""
    send_chat(on_client)
    assert carriers(finished(spans)), "the control request was not marked"

    spans.clear()
    with evals_trace_public(tmp_path, on_client, False):
        send_chat(on_client)

    assert carriers(finished(spans)) == []


def test_a_tenant_cannot_turn_it_on_alone(
    off_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches an implementation where a tenant's `trace_public: true` overrides the switch
    being off, which would make a trace public the operator never allowed."""
    with evals_trace_public(tmp_path, off_client, True):
        send_chat(off_client)

    assert carriers(finished(spans)) == []


def test_an_unauthorized_request_is_never_public(
    on_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a mark set before auth runs (for example in the middleware that opens the root
    span), which would publish a trace of a request that was refused."""
    response = on_client.post(
        CHAT,
        json=chat_body(),
        headers=auth_headers(key=None, tenant="evals", data_class="internal"),
    )
    assert response.status_code == 401, response.text

    assert PUBLIC_KEY not in dict(root_of(finished(spans)).attributes or {})
    assert carriers(finished(spans)) == []
