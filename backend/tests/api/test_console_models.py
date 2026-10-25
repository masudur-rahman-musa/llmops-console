"""`GET /console/models`: the live routing policy, read through console-api.

No database and no Prometheus: the app comes from `create_app` over the shipped policy,
and its policy store is swapped for one with a fixed clock so `loaded_at` can be pinned.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes import dashboard
from app.core.config import Settings
from app.main import create_app
from app.model_routing.policy import PolicyStore

from .conftest import (
    ADMIN_KEY,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    SHIPPED_PROVIDERS_DIR,
    TEST_KEYS,
    assert_error_body,
)

MODELS = "/console/models"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}
LOADED = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)


@pytest.fixture
def client(admin_settings: Settings) -> Iterator[TestClient]:
    """The admin app with a policy store whose clock is fixed."""
    built = create_app(admin_settings, NO_PROVIDER_KEYS)
    with TestClient(built) as test_client:
        built.state.policy = PolicyStore(
            SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR, clock=lambda: LOADED
        )
        yield test_client


@pytest.fixture
def body(client: TestClient) -> dict[str, Any]:
    """The 200 body of one authorised GET."""
    response = client.get(MODELS, headers=ADMIN_HEADERS)
    assert response.status_code == 200, response.text
    parsed: dict[str, Any] = response.json()
    return parsed


def test_models_lists_aliases_in_policy_order(body: dict[str, Any]) -> None:
    """Catches a body that sorts the aliases, drops the purpose, or reads another source
    than the live policy: the order and the four purposes are the file's, exactly."""
    assert set(body) == {"loaded_at", "aliases", "provider_files"}
    assert [(a["name"], a["purpose"]) for a in body["aliases"]] == [
        ("agent-default", "Everyday help desk chat replies on the cheap model."),
        ("agent-strong", "Harder chat turns that need the strong model."),
        ("judge", "Grades eval answers against the reference labels."),
        ("embed", "Turns text into vectors for search and the semantic cache."),
    ]
    assert "sources" not in body


def test_models_keeps_fallback_order(body: dict[str, Any]) -> None:
    """Catches a body that sorts candidates by provider name (openai before anthropic is
    wrong here: the first is the default, the rest are fallbacks)."""
    default = next(a for a in body["aliases"] if a["name"] == "agent-default")

    assert default["candidates"] == [
        {"provider": "anthropic", "model": "claude-haiku-5-5"},
        {"provider": "openai", "model": "gpt-5-mini"},
    ]


def test_models_lists_loaded_provider_files(body: dict[str, Any]) -> None:
    """Catches a list that names providers instead of files, leaves out the kind or the
    enabled flag, or reads the folder in an unstable order."""
    assert body["provider_files"] == [
        {"name": "anthropic.yaml", "provider": "anthropic", "kind": "anthropic", "enabled": True},
        {
            "name": "local_embed.yaml",
            "provider": "local_embed",
            "kind": "openai_compatible",
            "enabled": True,
        },
        {"name": "openai.yaml", "provider": "openai", "kind": "openai", "enabled": True},
    ]


def test_models_reports_loaded_at(body: dict[str, Any]) -> None:
    """Catches a `loaded_at` taken from the request clock (or the process start) instead of
    the policy store's own time, or one that is not UTC."""
    parsed = datetime.fromisoformat(body["loaded_at"])

    assert parsed == LOADED
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


def test_models_is_401_without_admin_key(
    client: TestClient, no_admin_key_client: TestClient
) -> None:
    """Catches a route left off the `admin_key` dependency, or checked against the tenant
    key store: no key, a working tenant key, and an app with no admin key set all get 401."""
    assert_error_body(client.get(MODELS), status=401, code="UNAUTHORIZED")
    tenant = client.get(MODELS, headers={"Authorization": f"Bearer {TEST_KEYS['nordlicht']}"})
    assert_error_body(tenant, status=401, code="UNAUTHORIZED")
    unset = no_admin_key_client.get(MODELS, headers=ADMIN_HEADERS)
    assert_error_body(unset, status=401, code="UNAUTHORIZED")


def test_models_route_reads_no_app_state(client: TestClient) -> None:
    """Catches a route that reaches into `app.state` itself instead of taking the policy
    through the `PolicyDep` dependency, which the other routes and the tests swap."""
    assert client.get(MODELS, headers=ADMIN_HEADERS).status_code == 200

    assert "app.state" not in inspect.getsource(dashboard.models)
