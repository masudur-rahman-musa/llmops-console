"""The uvicorn entrypoint exposes an app built by the package factory."""

from __future__ import annotations

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

import services.gateway.main as entrypoint
from services.gateway.main import app as entrypoint_app


def test_entrypoint_exposes_app() -> None:
    assert isinstance(entrypoint_app, FastAPI)

    with TestClient(entrypoint_app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_serve_uses_the_configured_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """`serve` takes its port from Settings, so `OPSCONSOLE_PORT` reaches the socket."""
    calls: list[dict[str, object]] = []

    def fake_run(target: str, **kwargs: object) -> None:
        calls.append({"target": target, **kwargs})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setenv("OPSCONSOLE_PORT", "9000")

    entrypoint.serve()

    assert calls == [{"target": "services.gateway.main:app", "port": 9000, "reload": False}]
