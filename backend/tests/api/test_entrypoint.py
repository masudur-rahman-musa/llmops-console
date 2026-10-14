"""The uvicorn entrypoint exposes an app built by the package factory."""

from __future__ import annotations

import importlib

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as entrypoint
from app.main import app as entrypoint_app


def test_the_entrypoint_imports_and_builds_the_app() -> None:
    """`app.main` is the one entrypoint; the `services.gateway.main` shim is gone."""
    assert isinstance(entrypoint.app, FastAPI)
    assert callable(entrypoint.serve)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("services.gateway.main")


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

    assert calls == [{"target": "app.main:app", "port": 9000, "reload": False}]
