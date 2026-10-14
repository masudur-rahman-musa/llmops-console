"""Structure tests for an earlier change: the chat pipeline moves out of the fat route modules."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import app

_OLD_ROUTE_MODULES = (
    "app.gateway.routes.chat",
    "app.gateway.routes.embeddings",
    "app.gateway.routes.redact",
)


def _imported_modules(path: Path) -> list[str]:
    """Return every `from X import ...` target module named in the file at path."""
    tree = ast.parse(path.read_text())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def test_the_old_route_modules_are_gone() -> None:
    """Catches an impl that leaves the fat chat/embeddings/redact route modules importable."""
    for name in _OLD_ROUTE_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_no_route_imports_another_route_and_cache_imports_no_route() -> None:
    """Catches a route importing a sibling route, or a cache module importing any route."""
    app_dir = Path(app.__file__).parent
    routes_dir = app_dir / "api" / "routes"
    cache_dir = app_dir / "cache"

    assert routes_dir.is_dir(), f"expected new route package at {routes_dir}"
    for leaf in ("chat.py", "embeddings.py", "redact.py"):
        assert (routes_dir / leaf).is_file(), f"expected thin route file {routes_dir / leaf}"

    for path in routes_dir.rglob("*.py"):
        for module in _imported_modules(path):
            assert not (module == "app.api.routes" or module.startswith("app.api.routes.")), (
                f"route {path} imports sibling route module {module}; "
                "routes never import sibling routes"
            )

    for path in cache_dir.rglob("*.py"):
        for module in _imported_modules(path):
            assert not (module == "app.api.routes" or module.startswith("app.api.routes.")), (
                f"cache {path} imports route module {module}; "
                "the cache reads schema shapes, never a route"
            )
            assert not (
                module == "app.gateway.routes" or module.startswith("app.gateway.routes.")
            ), (
                f"cache {path} imports route module {module}; "
                "the cache reads schema shapes, never a route"
            )
