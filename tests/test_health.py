"""Liveness and readiness endpoints.

Values are pinned exactly: 200 with `{"status": "ok"}` for
liveness, 200 for readiness while every probe passes, 503 the moment one fails.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from opsconsole.gateway.app import create_app
from opsconsole.gateway.health import ProbeRegistry


def passing() -> bool:
    return True


def failing() -> bool:
    return False


def raising() -> bool:
    raise RuntimeError("probe blew up")


def test_healthz_ok(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Decision table: readiness is 200 only when every registered probe passes.
# Boundary rows: the empty registry (zero probes) and a single failing probe.
OK_CASES: list[tuple[str, list[tuple[str, Callable[[], bool]]]]] = [
    ("empty registry", []),
    ("one passing probe", [("redis", passing)]),
    ("all probes passing", [("redis", passing), ("postgres", passing), ("qdrant", passing)]),
]


@pytest.mark.parametrize(("case", "probes"), OK_CASES, ids=[c[0] for c in OK_CASES])
def test_readyz_ok_when_all_probes_pass(
    case: str,
    probes: list[tuple[str, Callable[[], bool]]],
    registry: ProbeRegistry,
    client: TestClient,
) -> None:
    for name, probe in probes:
        registry.register(name, probe)

    response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["checks"], dict)
    assert list(body["checks"]) == [name for name, _ in probes]
    assert all(body["checks"].values())


FAIL_CASES: list[tuple[str, list[tuple[str, Callable[[], bool]]], str]] = [
    ("single failing probe", [("redis", failing)], "redis"),
    ("failing probe among passing ones", [("redis", passing), ("postgres", failing)], "postgres"),
    ("a raising probe counts as failed", [("qdrant", raising)], "qdrant"),
]


@pytest.mark.parametrize(
    ("case", "probes", "failed_name"), FAIL_CASES, ids=[c[0] for c in FAIL_CASES]
)
def test_readyz_503_when_a_probe_fails(
    case: str,
    probes: list[tuple[str, Callable[[], bool]]],
    failed_name: str,
    registry: ProbeRegistry,
    client: TestClient,
) -> None:
    for name, probe in probes:
        # `register` returns None: call it, never assert on its result.
        registry.register(name, probe)

    response = client.get("/readyz")

    assert response.status_code == 503
    # The body must name the probe that failed, not merely mention it: a report
    # that lists every probe as passing while returning 503 is a lie the plain
    # substring check would not catch.
    assert response.json()["checks"][failed_name] is False


def test_registries_are_per_app(registry: ProbeRegistry, client: TestClient) -> None:
    """A probe registered on one app is invisible to another built by the factory.

    Also pins that liveness ignores probes entirely: a dependency outage must
    make the pod unready, never restart it.
    """
    registry.register("redis", failing)
    other = create_app(environ={})

    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 200
    with TestClient(other) as other_client:
        response = other_client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"checks": {}}


def test_registering_the_same_name_replaces_the_probe(
    registry: ProbeRegistry, client: TestClient
) -> None:
    """The second registration wins, and the name keeps its original position."""
    registry.register("redis", passing)
    registry.register("postgres", passing)
    registry.register("redis", failing)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["checks"] == {"redis": False, "postgres": True}
    assert list(response.json()["checks"]) == ["redis", "postgres"]
