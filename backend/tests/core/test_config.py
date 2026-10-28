"""Settings read from the environment under the `OPSCONSOLE_` prefix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
# map store stays in process: a field defaulting to a
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


GRAFANA_ACCEPTED = ["https://grafana.example.com", "http://localhost:3001"]
GRAFANA_REFUSED = [
    ("a javascript link", "javascript:alert(1)"),
    ("ftp", "ftp://grafana.example.com"),
    ("a bare host with no scheme", "grafana.example.com"),
]


def test_grafana_url_accepts_http_and_https_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a `grafana_url` that is stored unchecked (so a `javascript:` value reaches the
    Overview link), one that accepts any scheme (`ftp://`), or one that insists on https and
    refuses the local `http://localhost:3001`. The unset case is asserted first so a
    check that refuses everything cannot pass the two accepted rows by accident."""
    monkeypatch.delenv("OPSCONSOLE_GRAFANA_URL", raising=False)
    assert Settings().grafana_url is None

    for value in GRAFANA_ACCEPTED:
        monkeypatch.setenv("OPSCONSOLE_GRAFANA_URL", value)
        assert Settings().grafana_url == value

    for _case, value in GRAFANA_REFUSED:
        monkeypatch.setenv("OPSCONSOLE_GRAFANA_URL", value)
        with pytest.raises(ValidationError) as caught:
            Settings()
        assert "grafana_url" in str(caught.value), value


def test_public_trace_langfuse_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a default of on (every trace public with nobody choosing it), an env name
    without the `OPSCONSOLE_` prefix, and text such as `maybe` read as a bool."""
    monkeypatch.delenv("OPSCONSOLE_PUBLIC_TRACE_LANGFUSE", raising=False)
    assert Settings().public_trace_langfuse is False

    monkeypatch.setenv("OPSCONSOLE_PUBLIC_TRACE_LANGFUSE", "true")
    assert Settings().public_trace_langfuse is True

    monkeypatch.setenv("OPSCONSOLE_PUBLIC_TRACE_LANGFUSE", "maybe")
    with pytest.raises(ValidationError):
        Settings()


BUDGET_CONTACT = "OPSCONSOLE_BUDGET_CONTACT"
BUDGET_CONTACT_MAX = 254


def an_email_of(length: int) -> str:
    """One address of exactly `length` characters, with a dot in the domain."""
    domain = "@b.com"
    return "a" * (length - len(domain)) + domain


def test_budget_contact_accepts_an_email_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a setting that insists on an address (the empty default, which means "name no
    contact", would stop the gateway starting), one that is not read from
    `OPSCONSOLE_BUDGET_CONTACT`, and a default that is not empty. The default is asserted
    with the variable removed, so no environment can decide it."""
    monkeypatch.delenv(BUDGET_CONTACT, raising=False)
    assert Settings().budget_contact == ""

    monkeypatch.setenv(BUDGET_CONTACT, "budget-owner@example.com")
    assert Settings().budget_contact == "budget-owner@example.com"

    monkeypatch.setenv(BUDGET_CONTACT, "")
    assert Settings().budget_contact == ""


BUDGET_CONTACT_REFUSED = [
    ("no at sign", "musa"),
    ("no dot in the domain", "a@b"),
    ("two addresses", "a@b.com, c@d.com"),
    ("a comma", "a@b.com,c@d.com"),
    ("a space inside", "a b@c.com"),
    ("a trailing newline", "a@b.com\n"),
    ("a trailing CRLF", "a@b.com\r\n"),
    ("a trailing space", "a@b.com "),
    ("a leading space", " a@b.com"),
    ("an embedded newline", "a@b.com\nc@d.com"),
    ("an empty local part", "@b.com"),
    ("an empty domain", "a@"),
]


@pytest.mark.parametrize(
    ("label", "value"), BUDGET_CONTACT_REFUSED, ids=[row[0] for row in BUDGET_CONTACT_REFUSED]
)
def test_budget_contact_refuses_what_is_not_one_email(
    label: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a check that lets a malformed contact into the message shown to users (a
    list of addresses, a value with a line break) and a pattern that ends in `$` and so lets
    a trailing newline through. Each row is refused at start, naming the field."""
    monkeypatch.setenv(BUDGET_CONTACT, value)

    with pytest.raises(ValidationError) as caught:
        Settings()

    assert "budget_contact" in str(caught.value), label


def test_budget_contact_length_limit_is_254_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a limit of 253 or 255, or none. 254 characters loads; 255 is refused."""
    at_limit = an_email_of(BUDGET_CONTACT_MAX)
    over = an_email_of(BUDGET_CONTACT_MAX + 1)
    assert len(at_limit) == 254
    assert len(over) == 255

    monkeypatch.setenv(BUDGET_CONTACT, at_limit)
    assert Settings().budget_contact == at_limit

    monkeypatch.setenv(BUDGET_CONTACT, over)
    with pytest.raises(ValidationError) as caught:
        Settings()
    assert "budget_contact" in str(caught.value)
