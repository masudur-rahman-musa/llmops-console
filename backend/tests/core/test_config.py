"""Settings read from the environment under the `OPSCONSOLE_` prefix."""

from __future__ import annotations

import pytest

from app.core.config import (
    DEFAULT_PROVIDERS_DIR,
    DEFAULT_ROUTING_POLICY_PATH,
    Settings,
)


def test_routing_and_providers_default_to_backend_config() -> None:
    """Both defaults resolve inside `backend/config/`, not a parent-folder climb."""
    for default, name in (
        (DEFAULT_ROUTING_POLICY_PATH, "routing.yaml"),
        (DEFAULT_PROVIDERS_DIR, "providers"),
    ):
        assert default.name == name
        assert default.parent.name == "config"
        assert default.parent.parent.name == "backend"
        assert default.exists()
    assert DEFAULT_PROVIDERS_DIR.is_dir()


ENV_CASES: list[tuple[str, str | None, str]] = [
    ("unset falls back to the default", None, "local"),
    ("override wins", "prod", "prod"),
    ("empty string is still an override", "", ""),
]


@pytest.mark.parametrize(
    ("case", "env_value", "expected"), ENV_CASES, ids=[c[0] for c in ENV_CASES]
)
def test_settings_env_override(
    case: str,
    env_value: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if env_value is None:
        monkeypatch.delenv("OPSCONSOLE_ENV", raising=False)
    else:
        monkeypatch.setenv("OPSCONSOLE_ENV", env_value)
    monkeypatch.delenv("OPSCONSOLE_PORT", raising=False)

    settings = Settings()

    assert settings.env == expected
    assert settings.port == 8080


# The three backing-service URLs. Absent by default, because
# absence is what decides that a probe is not registered and that the redaction
# map store stays in process (decision B and decision C): a field defaulting to a
# localhost URL would register a probe on every developer's machine and make
# `/readyz` red for a service nobody asked for.

BACKING_SERVICE_FIELDS: list[tuple[str, str, str]] = [
    ("redis", "redis_url", "redis://redis:6379/0"),
    ("postgres", "database_url", "postgresql://opsconsole@postgres:5432/opsconsole"),
    ("qdrant", "qdrant_url", "http://qdrant:6333"),
]


@pytest.mark.parametrize(
    ("case", "field", "value"),
    BACKING_SERVICE_FIELDS,
    ids=[c[0] for c in BACKING_SERVICE_FIELDS],
)
def test_a_backing_service_url_is_absent_until_the_environment_sets_it(
    case: str, field: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a field given a localhost default, and one wired to a variable outside the
    `OPSCONSOLE_` prefix: both rows run against the same field, so a default that happens
    to equal the value under test cannot pass the pair."""
    variable = f"OPSCONSOLE_{field.upper()}"
    monkeypatch.delenv(variable, raising=False)

    assert getattr(Settings(), field) is None

    monkeypatch.setenv(variable, value)

    assert getattr(Settings(), field) == value
