"""An earlier change: the budget contact reaches the gateway, and `.env.example` names it.

The contact is the address a refused request's message tells the user to ask for more budget.
the `gateway-config` must carry it, or the live message names no one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from infra.test_k8s_gateway_prometheus import GATEWAY_CONFIG_MAP, gateway_config

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_EXAMPLE = REPO_ROOT / "backend" / ".env.example"
KEY = "OPSCONSOLE_BUDGET_CONTACT"
CONTACT = "budget-owner@example.com"


def test_server_gateway_config_names_the_budget_contact(
    server_built: list[dict[str, Any]],
) -> None:
    """Catches a the server build that leaves the contact out (the live refusal names no one to
    ask), one that sets another address, and a name that drifts from what `Settings` reads.
    Also catches `.env.example` without the name (the list to fill when going live would
    miss it) or with a value in it (that file never holds values)."""
    data = gateway_config(server_built)

    assert data.get(KEY) == CONTACT, (
        f"{GATEWAY_CONFIG_MAP} on the server sets {KEY} to {data.get(KEY)!r}"
    )

    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    assert f"{KEY}=" in lines, f"{ENV_EXAMPLE} has no bare `{KEY}=` line"
