"""Liveness and readiness endpoints.

Values are pinned exactly: 200 with `{"status": "ok"}` for
liveness, 200 for readiness while every probe passes, 503 the moment one fails.

These cases are about the REGISTRY: what an app does with the probes it is
given. Since an earlier change, `create_app` registers probes of its own - `provider`
always, and one per configured backing service (none here, because the root
conftest clears those variables for the whole suite) - so every case measures
what it registered against `baseline_names`, the set the factory put there
first. Asserting an empty registry instead would pin the factory's probe set in
a file that is not about the factory, and `tests/gateway/test_probes.py` is
where that set belongs.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.gateway.app import create_app
from app.gateway.health import ProbeRegistry
from app.model_routing.policy import ResidencyMode


def passing() -> bool:
    return True


def failing() -> bool:
    return False


def raising() -> bool:
    raise RuntimeError("probe blew up")


def baseline_names(registry: ProbeRegistry) -> list[str]:
    """The probes `create_app` registered, in order, before a case adds its own."""
    return [name for name, _ in registry]


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
    baseline = baseline_names(registry)
    for name, probe in probes:
        registry.register(name, probe)

    response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["checks"], dict)
    assert list(body["checks"]) == [*baseline, *(name for name, _ in probes)]
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
    # The second app carries the factory's own probes and NOT the failing one
    # registered above, which is the leak this case exists to catch.
    assert "redis" not in response.json()["checks"]
    assert list(response.json()["checks"]) == baseline_names(other.state.probes)


def test_registering_the_same_name_replaces_the_probe(
    registry: ProbeRegistry, client: TestClient
) -> None:
    """The second registration wins, and the name keeps its original position."""
    baseline = baseline_names(registry)
    registry.register("redis", passing)
    registry.register("postgres", passing)
    registry.register("redis", failing)

    response = client.get("/readyz")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["redis"] is False
    assert checks["postgres"] is True
    # Registered once, under its first position: a second registration that
    # appended would put "redis" after "postgres" here.
    assert list(checks) == [*baseline, "redis", "postgres"]


# The residency mode on `/readyz`. A top-level key beside `checks`
# rather than inside it (decision H): `checks` means "a probe ran and this is its
# verdict", and the mode is a setting, not a probe. Both cases build their own app
# with the mode stated, because the root conftest clears the variable for the
# suite and a relaxed run must not be inferred from a developer's shell.

RESIDENCY_MODE_KEY = "residency_mode"


def mode_app(mode: ResidencyMode) -> FastAPI:
    return create_app(Settings(residency_mode=mode), environ={})


def test_readyz_reports_the_residency_mode() -> None:
    """Catches a readiness body that reports no mode at all, and one that files it inside
    `checks`, where `all(checks.values())` would read the string as a passing probe and a
    probe-name sweep would report a probe nothing ran.

    The strict row is the one every deployment serves, so a field present only on the
    relaxed path would leave an operator unable to tell a strict gateway from an old
    build."""
    with TestClient(mode_app(ResidencyMode.STRICT)) as mode_client:
        response = mode_client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body[RESIDENCY_MODE_KEY] == "strict"
    assert RESIDENCY_MODE_KEY not in body["checks"]


def test_readyz_reports_a_relaxed_residency_mode() -> None:
    """Catches a hard-coded `"strict"` in the body - which is what the case above alone would
    accept - and a field rendered from the enum member rather than its value, which would
    answer `ResidencyMode.RELAXED` to a scraper that can only read a string.

    Read from `app.state.settings`, so an app built relaxed reports relaxed however the
    process environment reads at request time."""
    with TestClient(mode_app(ResidencyMode.RELAXED)) as mode_client:
        response = mode_client.get("/readyz")

    assert response.status_code == 200
    assert response.json()[RESIDENCY_MODE_KEY] == "relaxed"
