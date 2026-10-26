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
