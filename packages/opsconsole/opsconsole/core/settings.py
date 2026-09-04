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
