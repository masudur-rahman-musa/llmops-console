"""Settings read from the environment under the `OPSCONSOLE_` prefix."""

from __future__ import annotations

import pytest

from opsconsole.core.settings import Settings

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
