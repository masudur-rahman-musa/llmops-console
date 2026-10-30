"""`backend/.env.example`: every name it lists is read by something, provider keys ship
empty, and the tracing names are listed as comments."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

TRACING_NAMES = ("OPSCONSOLE_OTLP_ENDPOINT", "OPSCONSOLE_OTLP_HEADERS")


@pytest.mark.parametrize("name", TRACING_NAMES)
def test_env_example_lists_the_tracing_names(name: str) -> None:
    """Catches a file that leaves a tracing name out (the list to fill when going live is
    then incomplete), and one that writes it as a live line such as `NAME=`: an empty
    endpoint is not "off" and would be read as a setting, so the name may only appear as a
    comment line."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    commented = [line for line in lines if re.match(rf"^#\s*{name}=", line)]
    live = [line for line in lines if re.match(rf"^\s*{name}\s*=", line)]

    assert len(commented) == 1, f"{name} must appear once as a commented line"
    assert live == []


def test_env_example_lists_the_grafana_link() -> None:
    """Catches a file that leaves `OPSCONSOLE_GRAFANA_URL` out, writes it only as a comment, or
    ships a value in it: the list to fill when going live must carry the name with no value."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    assert [line for line in lines if line.startswith("OPSCONSOLE_GRAFANA_URL")] == [
        "OPSCONSOLE_GRAFANA_URL="
    ]


PUBLIC_SWITCH = "OPSCONSOLE_PUBLIC_TRACE_LANGFUSE"
ENV_DEMO = ENV_EXAMPLE.with_name(".env.demo")


def test_env_example_lists_the_public_trace_switch() -> None:
    """Catches a file that leaves the switch out, or writes it as a live line: an empty value
    is not true or false and would stop the gateway starting. The comment must show `true`."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    commented = [line for line in lines if re.match(rf"^#\s*{PUBLIC_SWITCH}=", line)]
    live = [line for line in lines if re.match(rf"^\s*{PUBLIC_SWITCH}\s*=", line)]

    assert commented == [f"# {PUBLIC_SWITCH}=true"]
    assert live == []


def test_env_demo_turns_public_traces_on() -> None:
    """Catches a demo env that leaves the switch off or out, so the local console's
    "Open trace" would land on a login page."""
    if not ENV_DEMO.exists():
        pytest.skip(".env.demo is gitignored and absent in CI")
    lines = ENV_DEMO.read_text(encoding="utf-8").splitlines()

    assert f"{PUBLIC_SWITCH}=true" in lines


REPO_ROOT = ENV_EXAMPLE.parents[1]
BACKEND = ENV_EXAMPLE.parent

#: Where a name the gateway or compose reads is spelled out, apart from the settings class.
#: Azure ships no provider file; its two names are the ones its documented file reads.
READERS = (
    *sorted((BACKEND / "config" / "providers").glob("*.yaml")),
    *sorted((BACKEND / "app").rglob("*.py")),
    REPO_ROOT / "docker-compose.yml",
    REPO_ROOT / "docs" / "azure.md",
)


def listed_names() -> list[str]:
    """Every `NAME=` in the file, live or commented out."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    return [m.group(1) for line in lines if (m := re.match(r"^#?\s*([A-Z][A-Z0-9_]*)=", line))]


def settings_names() -> set[str]:
    from app.core.config import Settings

    prefix = Settings.model_config.get("env_prefix", "")
    return {f"{prefix}{field}".upper() for field in Settings.model_fields}


@pytest.mark.parametrize("name", listed_names())
def test_every_listed_name_is_read_by_something(name: str) -> None:
    """Catches a name nothing reads, such as a bare `DATABASE_URL` beside the prefixed one the
    gateway really reads: a reader fills it in and nothing changes."""
    if name in settings_names():
        return
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])")
    readers = [path for path in READERS if path.exists()]
    assert any(pattern.search(path.read_text(encoding="utf-8")) for path in readers), name


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
def test_provider_keys_ship_empty(name: str) -> None:
    """Catches a stand-in such as `replace-me` as a provider key: copied as it is, the gateway
    builds an adapter that every request fails on, where an empty key leaves the provider off."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    assert [line for line in lines if line.startswith(f"{name}=")] == [f"{name}="]
