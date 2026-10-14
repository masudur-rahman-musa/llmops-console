"""Settings read from the environment under the `OPSCONSOLE_` prefix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import (
    DEFAULT_PRICES_PATH,
    DEFAULT_ROUTING_POLICY_PATH,
    Settings,
)
from app.model_routing.policy import ResidencyMode


def test_routing_and_prices_default_to_backend_config() -> None:
    """Both defaults resolve inside `backend/config/`, not a parent-folder climb."""
    for default, name in (
        (DEFAULT_ROUTING_POLICY_PATH, "routing.yaml"),
        (DEFAULT_PRICES_PATH, "prices.yaml"),
    ):
        assert default.name == name
        assert default.parent.name == "config"
        assert default.parent.parent.name == "backend"
        assert default.is_file()


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


# The residency mode. A closed enum, so an unknown value stops the
# process at construction rather than turning into a routing decision nobody
# chose. Stated here rather than in a routing test because the refusal belongs to
# `Settings`, before any app exists.

RESIDENCY_MODE_VARIABLE = "OPSCONSOLE_RESIDENCY_MODE"


def test_the_residency_mode_defaults_to_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a field defaulting to relaxed, and one with no default at all: every
    deployment that sets nothing - the cluster, CI, a bare `uvicorn` - must enforce
    residency exactly as it did before this field existed."""
    monkeypatch.delenv(RESIDENCY_MODE_VARIABLE, raising=False)

    assert Settings().residency_mode is ResidencyMode.STRICT


def test_the_residency_mode_accepts_relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a field typed `str`, which would carry `"relaxed"` through to a resolver
    comparing it against an enum member and silently never matching - a relaxed run that
    behaves strictly, with the startup warning and the `/readyz` field saying otherwise."""
    monkeypatch.setenv(RESIDENCY_MODE_VARIABLE, "relaxed")

    mode = Settings().residency_mode

    assert mode is ResidencyMode.RELAXED
    # A StrEnum member equals its string; mypy's narrowing to the member Literal
    # cannot see that, and this line exists to prove it at run time.
    assert mode == "relaxed"  # type: ignore[comparison-overlap]


# `loose` is a plausible typo for the real value; the empty string is the row the
# other settings' "presence decides an override, not truthiness" rule produces,
# and a field that fell back to the default for it would make
# `OPSCONSOLE_RESIDENCY_MODE=` look like a deliberate strict run.
UNKNOWN_MODES: list[tuple[str, str]] = [
    ("a plausible typo", "loose"),
    ("the empty string", ""),
]


@pytest.mark.parametrize(("case", "value"), UNKNOWN_MODES, ids=[c[0] for c in UNKNOWN_MODES])
def test_an_unknown_residency_mode_is_refused_at_construction(
    case: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a field that coerces an unknown value to a default: the two values differ by
    whether personal data may leave the box, so the only safe answer to a value nobody
    recognises is to refuse to start. The message must name the field, or an operator reads
    a stack trace with no clue which variable they mistyped."""
    monkeypatch.setenv(RESIDENCY_MODE_VARIABLE, value)

    with pytest.raises(ValidationError, match="residency_mode"):
        Settings()
