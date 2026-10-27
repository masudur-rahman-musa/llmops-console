"""`backend/.env.example`: the tracing names are listed, and listed as comments."""

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
