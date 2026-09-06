"""Process settings, read from the environment at construction time."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ROUTING_POLICY_PATH = Path(__file__).resolve().parents[4] / "routing.yaml"
"""The shipped `routing.yaml`, found from this file rather than from the working
directory. A deployment that does not carry the repository layout - the container
image, a wheel install - sets `OPSCONSOLE_ROUTING_POLICY_PATH` instead."""


class Settings(BaseSettings):
    """Gateway settings.

    Every field has a default, so the process starts with no environment at all.
    Values come from `OPSCONSOLE_`-prefixed variables; presence decides an
    override, not truthiness, so `OPSCONSOLE_ENV=""` yields an empty `env`.
    Pinned by `test_settings_env_override`.
    """

    model_config = SettingsConfigDict(env_prefix="OPSCONSOLE_")

    env: str = "local"
    port: int = 8080

    api_keys: str = ""
    """Comma-separated `tenant:sha256hex` pairs. Digests only: an issued tenant
    key lives in `.env` locally and in a SOPS-encrypted secret in the cluster, and
    no issued key is in this repository (the worthless strings in
    `tests/gateway/conftest.py` are test data, not anyone's key). Empty is a
    running gateway that authorises nobody."""

    routing_policy_path: Path = DEFAULT_ROUTING_POLICY_PATH

    otlp_endpoint: str | None = None
    """Where traces are sent, or `None` for a gateway that builds spans and drops
    them. Langfuse EU is an OTLP/HTTP endpoint, so this is a
    full traces URL rather than a host. Absent by default so a developer, a test
    run and a CI job export nothing without saying so; pinned by
    `test_no_exporter_is_installed_without_an_endpoint`."""

    otlp_headers: str | None = None
    """`key=value,key=value`, the auth the endpoint needs. Langfuse wants a basic
    `Authorization` header built from a project key pair, which is a secret: it
    lives in `.env` locally and in a SOPS-encrypted secret in the cluster, never
    in this repository."""

    otel_service_name: str = "opsconsole-gateway"
    """What this process is called in a trace viewer. One trace carries spans from
    the Helpdesk and from the gateway, so the two need different names to be
    told apart at all."""
