"""Shared fixtures for the gateway bootstrap tests.

Fixtures here are cheap (an in-process FastAPI app, no backing services), so they
are function-scoped; a project rule applies only to expensive backing-service
fixtures, and the one that exists (the session-scoped Redis client in
`tests/redaction/test_redis_store.py`) lives beside the cases that need it.

The autouse fixture below is the suite's one piece of environment isolation, and
it exists because three settings now read the environment and change what an app
IS: with `OPSCONSOLE_REDIS_URL` exported, every `create_app` in the suite would
file its redaction maps in Redis and register a `redis` readiness probe, so a
developer with a local stack up would run a different suite from one without.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opsconsole.gateway.app import create_app
from opsconsole.gateway.health import ProbeRegistry

BACKING_SERVICE_VARIABLES = (
    "OPSCONSOLE_REDIS_URL",
    "OPSCONSOLE_DATABASE_URL",
    "OPSCONSOLE_QDRANT_URL",
    "OPSCONSOLE_PROMETHEUS_URL",
)
"""The settings whose PRESENCE decides what `create_app` builds (an earlier change
decisions B and C). Cleared for the whole session below. The Redis suite reads
`OPSCONSOLE_TEST_REDIS_URL` instead, which nothing in `Settings` looks at."""


@pytest.fixture(scope="session", autouse=True)
def _no_backing_service_environment() -> Iterator[None]:
    """Run the whole suite as if no backing service were configured.

    Session-scoped and autouse, so it covers apps built at module import and in
    other directories' fixtures too. A test that wants one of these set says so
    with `monkeypatch.setenv` (`tests/test_settings.py`) or by passing the value
    to `Settings` directly (`tests/gateway/test_probes.py`), which is what makes
    those cases assert their own arrangement rather than the developer's shell.
    """
    with pytest.MonkeyPatch.context() as patched:
        for variable in BACKING_SERVICE_VARIABLES:
            patched.delenv(variable, raising=False)
        yield


@pytest.fixture
def app() -> FastAPI:
    """A freshly built app, so probe registrations never leak between tests."""
    # No provider key is read, whatever the shell exports (#66); see the note in
    # `tests/gateway/conftest.py`.
    built: FastAPI = create_app(environ={})
    return built


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def registry(app: FastAPI) -> ProbeRegistry:
    """The readiness probe registry the app exposes on `app.state.probes`."""
    probes = app.state.probes
    assert isinstance(probes, ProbeRegistry)
    return probes
