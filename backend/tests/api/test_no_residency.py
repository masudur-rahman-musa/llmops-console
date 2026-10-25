"""The gateway's wire surface after residency is removed.

No `X-Gateway-Region` header, no `residency_mode` on `/readyz`, no
`/console/residency` page, no mode setting read from the environment, and no
residency or region series on `/metrics`. Everything goes through `TestClient`
against an app `create_app` built, with stubs behind the adapters.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings

from .conftest import (
    ADMIN_KEY,
    API_KEYS,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    StubAdapter,
    build_chat_client,
    chat_body,
    embeddings_body,
    post_chat,
    post_embeddings,
)

GATEWAY_PREFIX = "x-gateway-"
REGION_HEADER = "x-gateway-region"
SIX_HEADERS = {
    "x-gateway-provider",
    "x-gateway-model",
    "x-gateway-cache",
    "x-gateway-redacted",
    "x-gateway-trace-id",
    "x-gateway-cost-usd",
}


def _gateway_headers(headers: dict[str, str]) -> set[str]:
    return {name.lower() for name in headers if name.lower().startswith(GATEWAY_PREFIX)}


def test_no_response_carries_a_region_header(
    chat_client: TestClient, embeddings_client: TestClient
) -> None:
    """Catches a header builder that still adds `X-Gateway-Region` (an empty or a made-up
    value is still a claim about where data went). The other six are asserted whole in the
    same case, so a builder that dropped every header cannot be what passes."""
    chat = post_chat(chat_client, chat_body())
    embedding = post_embeddings(embeddings_client, embeddings_body())

    assert chat.status_code == 200, chat.text
    assert embedding.status_code == 200, embedding.text
    assert _gateway_headers(dict(chat.headers)) == SIX_HEADERS
    assert _gateway_headers(dict(embedding.headers)) == SIX_HEADERS
    assert REGION_HEADER not in {name.lower() for name in chat.headers}
    assert REGION_HEADER not in {name.lower() for name in embedding.headers}


def test_readyz_has_no_residency_mode(gateway_client: TestClient) -> None:
    """Catches a readiness body that still reports the mode: the setting is gone, so a key
    named for it would be a stale claim an operator might act on. `checks` is asserted
    present, so an empty body is not what passes."""
    response = gateway_client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert "checks" in body
    assert "residency_mode" not in body
    assert "residency_mode" not in body["checks"]


def test_the_residency_console_path_is_gone(admin_client: TestClient) -> None:
    """Catches a residency page that survives the removal. The same admin key reaches
    `/console/home` in the same case (not 404), so a wrong key or a console that was never
    mounted cannot be what makes the residency path 404."""
    headers = {"Authorization": f"Bearer {ADMIN_KEY}"}

    assert admin_client.get("/console/home", headers=headers).status_code != 404
    assert admin_client.get("/console/residency", headers=headers).status_code == 404


def test_the_residency_mode_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a setting that is still read: with `OPSCONSOLE_RESIDENCY_MODE=relaxed`
    exported, the app must start and resolve exactly as it does without it. Compared on
    the same tenant, alias and data class, so a relaxed widening would show as a
    different candidate list."""

    def served() -> tuple[int, str | None]:
        settings = Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH)
        stub = StubAdapter(name="anthropic")
        with build_chat_client(settings, {stub.name: stub}) as client:
            reply = post_chat(client, chat_body(), tenant="nordlicht", key=TEST_KEYS["nordlicht"])
        return reply.status_code, reply.headers.get("x-gateway-provider")

    without = served()
    monkeypatch.setenv("OPSCONSOLE_RESIDENCY_MODE", "relaxed")
    with_relaxed = served()

    assert without[0] != 401
    assert with_relaxed == without
    assert not hasattr(Settings(), "residency_mode")


def test_the_metrics_page_has_no_residency_or_region(settings: Settings) -> None:
    """Catches a leftover `gateway_residency_violations_total` family or a `region=` label
    on the request series. One chat request is sent first, so the labelled series really
    exist when the page is read; the request is asserted to have succeeded."""
    stub = StubAdapter(name="anthropic")
    with build_chat_client(settings, {stub.name: stub}) as client:
        assert post_chat(client, chat_body()).status_code == 200
        page = client.get("/metrics")

    assert page.status_code == 200
    assert "gateway_requests_total{" in page.text
    assert "residency" not in page.text.lower()
    assert re.search(r"\bregion=", page.text) is None
