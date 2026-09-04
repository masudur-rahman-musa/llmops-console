"""Shared fixtures for the gateway bootstrap tests.

Fixtures here are cheap (an in-process FastAPI app, no backing services), so they
are function-scoped; a project rule applies only to expensive backing-service
fixtures, of which this plan has none.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opsconsole.gateway.app import create_app
from opsconsole.gateway.health import ProbeRegistry


@pytest.fixture
def app() -> FastAPI:
    """A freshly built app, so probe registrations never leak between tests."""
    built: FastAPI = create_app()
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
