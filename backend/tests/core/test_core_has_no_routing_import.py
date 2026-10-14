"""The import cycle is gone and the old names no longer resolve.

`app.routing` is renamed to `app.model_routing` and `app.core.settings` to
`app.core.config`, so the old import names are dead. Nothing under
`app/core/` imports the routing package any more, which is what lets `core`
hold the shared identifier check without a cycle.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[2] / "app" / "core"


@pytest.mark.parametrize("name", ["app.routing", "app.core.settings"])
def test_the_old_tree_is_gone(name: str) -> None:
    """Catches a leftover compatibility shim re-exporting the old names:
    `app.routing` and `app.core.settings` are renamed, so importing either must
    fail outright rather than quietly resolving to the renamed module."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_core_imports_nothing_from_model_routing() -> None:
    """Catches a `core` module that still reaches up into the routing package:
    an import of `app.model_routing` or `app.routing` on any import line under
    `app/core/` re-creates the cycle this move removes, so no such line may
    exist."""
    offenders: list[str] = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            if "app.model_routing" in stripped or "app.routing" in stripped:
                offenders.append(f"{path.name}: {stripped}")

    assert offenders == [], f"core imports the routing package: {offenders}"
