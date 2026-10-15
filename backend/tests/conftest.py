"""Shared fixtures for the gateway bootstrap tests.

Fixtures here are cheap (an in-process FastAPI app, no backing services), so they
are function-scoped; a project rule applies only to expensive backing-service
fixtures, and the one that exists (the session-scoped Redis client in
`tests/redaction/test_redis_store.py`) lives beside the cases that need it.

The autouse fixture below is the suite's one piece of environment isolation, and
it exists because several settings read the environment and change what an app
IS: with `OPSCONSOLE_REDIS_URL` exported, every `create_app` in the suite would
file its redaction maps in Redis and register a `redis` readiness probe, so a
developer with a local stack up would run a different suite from one without.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import httpx
import pytest
import redis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.health import ProbeRegistry
from app.main import create_app

CLEARED_VARIABLES = (
    "OPSCONSOLE_REDIS_URL",
    "OPSCONSOLE_DATABASE_URL",
    "OPSCONSOLE_QDRANT_URL",
    "OPSCONSOLE_PROMETHEUS_URL",
    "OPSCONSOLE_RESIDENCY_MODE",
)
"""Every setting whose value in the developer's shell would change what the suite
IS, cleared for the whole session below.

The four URLs are the ones whose PRESENCE decides what `create_app` builds (plan
09a decisions B and C). `OPSCONSOLE_RESIDENCY_MODE` joined them in an earlier change for the
same reason and a sharper one: exported as `relaxed` it would widen the region set
for every `resolve()` in the suite, so the residency cases would pass against a
gateway enforcing something else. A case that wants a value states it with
`monkeypatch.setenv` or passes it to `Settings` directly. The Redis suite reads
`OPSCONSOLE_TEST_REDIS_URL` instead, which nothing in `Settings` looks at."""


@pytest.fixture(scope="session", autouse=True)
def _no_backing_service_environment() -> Iterator[None]:
    """Run the whole suite as if no backing service and no mode were configured.

    Session-scoped and autouse, so it covers apps built at module import and in
    other directories' fixtures too. A test that wants one of these set says so
    with `monkeypatch.setenv` (`tests/test_settings.py`) or by passing the value
    to `Settings` directly (`tests/gateway/test_probes.py`), which is what makes
    those cases assert their own arrangement rather than the developer's shell.
    """
    with pytest.MonkeyPatch.context() as patched:
        for variable in CLEARED_VARIABLES:
            patched.delenv(variable, raising=False)
        # Bound the OTLP export retry for the whole session. A tracing test builds
        # an app pointed at an endpoint nothing listens on (an earlier change,
        # `tests/api/test_tracing.py`), which attaches a `BatchSpanProcessor` to the
        # process-wide provider that can never be detached. Every later app shutdown
        # then calls `flush_tracing()` -> `force_flush()`, and the OTLP/HTTP exporter
        # retries the dead endpoint with exponential backoff - about seven seconds a
        # flush, a third of the old run. The exporter bounds its own retry loop by
        # this timeout, so one second turns each of those flushes into under a second.
        patched.setenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "1")
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


# --- shared backing-service clients ------------------------
#
# One Redis client and one Qdrant client for the whole run, built once here so no
# test file builds its own (the own-client guard, `test_own_client_guard.py`,
# scans every non-conftest file for a real build and holds the allowlist empty).
# Under `make test`'s `-n auto`, session scope means one client per xdist worker;
# per-test isolation is the uuid `key_prefix` and `collection_prefix` below, never
# a rebuild and never a flush that would delete a neighbour's keys.
#
# The skip is an explicit conditional on the `OPSCONSOLE_TEST_*` variable's NAME
#, never a swallowed connection error: a cache answers absent on
# every failure, so a suite that could not reach its store would otherwise report
# success having checked nothing. Deliberately NOT the `OPSCONSOLE_*` names
# `Settings` reads, which the session fixture above clears.

REDIS_URL_VAR = "OPSCONSOLE_TEST_REDIS_URL"
QDRANT_URL_VAR = "OPSCONSOLE_TEST_QDRANT_URL"

QDRANT_SESSION_PREFIX = f"opsconsole-test-{uuid.uuid4().hex[:8]}-"
"""Every collection this run creates begins with it, so the teardown names its own
collections and touches nothing else a shared Qdrant may hold."""


@pytest.fixture(scope="session")
def redis_client() -> Iterator[redis.Redis]:
    """One Redis client for the whole session, built once."""
    url = os.environ.get(REDIS_URL_VAR)
    if url is None:
        pytest.skip(f"{REDIS_URL_VAR} is not set: start Redis and export it to run these")
    client: redis.Redis = redis.Redis.from_url(url)
    yield client
    client.close()


@pytest.fixture
def key_prefix() -> str:
    """A Redis namespace unique to one test, which is the whole of its isolation.

    Never `FLUSHDB`: a run may share a database with anything else, and a flush
    would delete keys it does not own. Every key written under this prefix expires
    on its own TTL.
    """
    return f"test-{uuid.uuid4().hex}"


@pytest.fixture(scope="session")
def qdrant_client() -> Iterator[httpx.Client]:
    """One Qdrant client for the whole session, built once.

    The teardown removes the collections THIS run created, found by
    `QDRANT_SESSION_PREFIX`. It is not a sweep: a shared Qdrant may hold
    collections this suite does not own.
    """
    url = os.environ.get(QDRANT_URL_VAR)
    if url is None:
        pytest.skip(f"{QDRANT_URL_VAR} is not set: start Qdrant and export it to run these")
    client = httpx.Client(base_url=url, timeout=httpx.Timeout(5.0))
    yield client
    listed = client.get("/collections")
    if listed.status_code < 300:
        for entry in listed.json()["result"]["collections"]:
            if str(entry["name"]).startswith(QDRANT_SESSION_PREFIX):
                client.delete(f"/collections/{entry['name']}")
    client.close()


@pytest.fixture
def collection_prefix() -> str:
    """A Qdrant collection namespace unique to one test, the whole of its isolation."""
    return f"{QDRANT_SESSION_PREFIX}{uuid.uuid4().hex[:8]}-cache_"
