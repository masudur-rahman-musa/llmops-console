"""An earlier change: the ledger-drop alert, its runbook section, and the budget overshoot decision.

Text reads over committed files; no build, no service. `rules.yml` is read straight from
its file (the same text the built `prometheus-config` carries), so no `kubectl` is needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

#: This file's own location, never the working directory - `pytest` runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[3]

RULES = REPO_ROOT / "infra" / "k8s" / "monitoring" / "alerts" / "rules.yml"
RUNBOOK = REPO_ROOT / "dev" / "ops" / "runbook.md"
BUDGET_ADR = REPO_ROOT / "dev" / "adr" / "0042-budget-cap-can-be-overshot.md"

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


def section(text: str, heading: str, level: str) -> str:
    """The lines under `heading` up to the next heading of the same or a higher level."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.rstrip() == f"{level} {heading}"]
    assert len(starts) == 1, f"{level} {heading} occurs {len(starts)} times"
    body: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if re.match(rf"^#{{1,{len(level)}}} ", line):
            break
        body.append(line)
    return "\n".join(body)


def test_the_ledger_drop_alert_reads_every_dropped_result() -> None:
    """Catches an alert that watches one `dropped_*` value (a full buffer fires, a possibly
    double counted commit does not), one that waits `for` minutes before warning, one at
    the wrong severity, and one whose runbook link points at no anchor. The selector is
    asserted whole, regex included, so `dropped` alone or `dropped_full` alone fails."""
    rule = alert_named(ALERT)

    expr = " ".join(str(rule["expr"]).split())
    assert DROPPED_SELECTOR in expr
    assert "increase(" in expr or "sum(increase(" in expr
    assert expr.endswith("> 0")
    assert str(rule["for"]) == "0m"
    assert rule["labels"]["severity"] == "warning"
    assert rule["annotations"]["runbook"] == "dev/ops/runbook.md#ledgerwritedropped"


def test_the_runbook_has_a_ledger_write_dropped_section() -> None:
    """Catches an alert whose runbook link lands nowhere (no `### LedgerWriteDropped`
    heading, so the anchor `#ledgerwritedropped` does not exist) and a section that names
    no query, which leaves the responder with nothing to run. The body must carry the
    metric it reads."""
    body = section(RUNBOOK.read_text(encoding="utf-8"), ALERT, "###")

    assert "gateway_ledger_writes_total" in body
    assert "dropped_" in body
    assert "```" in body, "the section has no command or query block"


def test_the_budget_adr_names_the_overshoot_and_the_switch() -> None:
    """Catches a missing a design note, one without the four ADR headings, and one that records
    the overshoot but not the rejected alternative (reserving the estimated cost before
    each call) or when the console would switch to it."""
    assert BUDGET_ADR.is_file(), f"{BUDGET_ADR.name} does not exist"
    text = BUDGET_ADR.read_text(encoding="utf-8")

    headings = {line.rstrip() for line in text.splitlines() if line.startswith("## ")}
    for needed in ("## Context", "## Decision", "## Alternatives considered", "## Consequences"):
        assert needed in headings, f"{BUDGET_ADR.name} lacks {needed}"
    lowered = text.lower()
    assert "reserv" in lowered and "estimated cost" in lowered
    alternatives = section(text, "Alternatives considered", "##").lower()
    assert "reserv" in alternatives and "estimated cost" in alternatives
    assert "switch" in lowered
