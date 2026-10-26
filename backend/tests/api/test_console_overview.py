"""`GET /console/home` as the Overview tab: the live routing, the Grafana link, four tabs.

No database and no Prometheus: the app comes from `create_app` over the shipped policy, with
neither source configured (the home route answers 200 with both flagged `unconfigured`, which
tests/api/test_console_routes.py pins). The policy store is swapped for one with a fixed clock
so `routing.loaded_at` can be pinned. The routing is not a source; it is read from the policy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.model_routing.policy import PolicyStore

from .conftest import (
    ADMIN_KEY,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    SHIPPED_PROVIDERS_DIR,
)

HOME = "/console/home"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}
LOADED = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
GRAFANA = "https://grafana.example.com/d/ops"
INCIDENT_ID = "0af7651916cd43dd8448eb211c80319c"

MakeClient = Callable[[str | None], TestClient]


@pytest.fixture
def make_client(admin_settings: Settings) -> Iterator[MakeClient]:
    """Builds the admin app per case with a chosen `grafana_url` and a fixed policy clock."""
    entered: list[Any] = []

    def build(grafana_url: str | None) -> TestClient:
        settings = admin_settings.model_copy(update={"grafana_url": grafana_url})
        built = create_app(settings, NO_PROVIDER_KEYS)
        context = TestClient(built)
        client = context.__enter__()
        entered.append(context)
        built.state.policy = PolicyStore(
            SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR, clock=lambda: LOADED
        )
        return client

    yield build
    for context in entered:
        context.__exit__(None, None, None)


def home_body(client: TestClient) -> dict[str, Any]:
    """The 200 body of one authorised GET of the home route."""
    response = client.get(HOME, headers=ADMIN_HEADERS)
    assert response.status_code == 200, response.text
    parsed: dict[str, Any] = response.json()
    return parsed


def test_home_carries_the_live_routing_in_fallback_order(make_client: MakeClient) -> None:
    """Catches a home body that sorts the aliases or the candidates, drops the purpose, takes
    `loaded_at` from the request clock instead of the policy store's own time, or reads a
    stale copy of the policy instead of the live one the store holds."""
    client = make_client(None)
    body = home_body(client)

    routing = body["routing"]
    assert set(routing) == {"loaded_at", "aliases"}
    assert [(a["name"], a["purpose"]) for a in routing["aliases"]] == [
        ("agent-default", "Everyday help desk chat replies on the cheap model."),
        ("agent-strong", "Harder chat turns that need the strong model."),
        ("judge", "Grades eval answers against the reference labels."),
        ("embed", "Turns text into vectors for search and the semantic cache."),
    ]
    default = next(a for a in routing["aliases"] if a["name"] == "agent-default")
    assert default["candidates"] == [
        {"provider": "anthropic", "model": "claude-haiku-5-5"},
        {"provider": "openai", "model": "gpt-5-mini"},
    ]
    live = client.app.state.policy.policy  # type: ignore[attr-defined]
    for alias in routing["aliases"]:
        assert alias["candidates"] == [
            {"provider": c.provider, "model": c.model}
            for c in live.aliases[alias["name"]].candidates
        ]
    loaded = datetime.fromisoformat(routing["loaded_at"])
    assert loaded == LOADED
    offset = loaded.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


GRAFANA_ROWS: list[tuple[str, str | None]] = [
    ("set", GRAFANA),
    ("unset", None),
]


@pytest.mark.parametrize(("case", "setting"), GRAFANA_ROWS, ids=[r[0] for r in GRAFANA_ROWS])
def test_home_carries_the_grafana_url_or_null(
    case: str, setting: str | None, make_client: MakeClient
) -> None:
    """Catches a home body that hard-codes the link, shows a URL while the setting is unset,
    drops the key when it is unset (the page needs an explicit null to show "not set"), or
    still carries the removed `open_incidents` key."""
    body = home_body(make_client(setting))

    assert "grafana_url" in body
    assert body["grafana_url"] == setting
    assert "open_incidents" not in body


def test_only_four_console_routes_remain(make_client: MakeClient) -> None:
    """Catches a removed tab whose route stays mounted (each of the four is a 404 with the
    admin key, so only an unmounted route passes) and a kept tab that went with them."""
    client = make_client(None)

    for removed in ("reliability", "incidents", "releases", "models"):
        assert client.get(f"/console/{removed}", headers=ADMIN_HEADERS).status_code == 404, removed
    for kept in ("home", "cost", "quality", "requests"):
        response = client.get(f"/console/{kept}", headers=ADMIN_HEADERS)
        assert response.status_code == 200, (kept, response.text)


def test_the_incident_admin_api_is_gone(make_client: MakeClient) -> None:
    """Catches an incident write or resolve route left mounted (a 201 or 200 would show) and
    catches the whole admin releases router being unmounted with it: `POST /admin/releases`
    must still reach its handler, so it answers a validation error for the empty body and
    neither 404 nor 405."""
    client = make_client(None)

    created = client.post("/admin/incidents", json={}, headers=ADMIN_HEADERS)
    resolved = client.patch(f"/admin/incidents/{INCIDENT_ID}", json={}, headers=ADMIN_HEADERS)
    releases = client.post("/admin/releases", json={}, headers=ADMIN_HEADERS)

    assert created.status_code in (404, 405)
    assert resolved.status_code in (404, 405)
    assert releases.status_code not in (404, 405)
