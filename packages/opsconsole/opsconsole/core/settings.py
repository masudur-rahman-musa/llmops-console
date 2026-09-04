"""Process settings, read from the environment at construction time."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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
