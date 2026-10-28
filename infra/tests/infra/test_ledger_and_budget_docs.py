"""The ledger-drop alert.

Text reads over committed files; no build, no service. `rules.yml` is read straight from
its file (the same text the built `prometheus-config` carries), so no `kubectl` is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: This file's own location, never the working directory - `pytest` runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[3]

RULES = REPO_ROOT / "infra" / "k8s" / "monitoring" / "alerts" / "rules.yml"

ALERT = "LedgerWriteDropped"
DROPPED_SELECTOR = 'gateway_ledger_writes_total{result=~"dropped_.*"}'


def alert_named(name: str) -> dict[str, Any]:
    """The one alerting rule called `name`, whichever group holds it."""
    parsed = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    found = [
        rule
        for group in parsed["groups"]
        for rule in group["rules"]
        if isinstance(rule, dict) and rule.get("alert") == name
    ]
    assert len(found) == 1, f"{name} appears {len(found)} times in rules.yml"
    return found[0]


def test_the_ledger_drop_alert_reads_every_dropped_result() -> None:
    """Catches an alert that watches one `dropped_*` value (a full buffer fires, a possibly
    double counted commit does not), one that waits `for` minutes before warning, one at
    the wrong severity. The selector is
    asserted whole, regex included, so `dropped` alone or `dropped_full` alone fails."""
    rule = alert_named(ALERT)

    expr = " ".join(str(rule["expr"]).split())
    assert DROPPED_SELECTOR in expr
    assert "increase(" in expr or "sum(increase(" in expr
    assert expr.endswith("> 0")
    assert str(rule["for"]) == "0m"
    assert rule["labels"]["severity"] == "warning"
