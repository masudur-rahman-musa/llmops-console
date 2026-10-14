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

_GONE_GATEWAY_MODULES = (
    "app.gateway.app",
    "app.gateway.auth",
    "app.gateway.context",
    "app.gateway.errors",
    "app.gateway.health",
    "app.gateway.probes",
    "app.gateway.prometheus",
    "app.gateway.routes",
)

_GONE_STORAGE_AND_CLI_MODULES = (
    "app.ledger",
    "app.ledger.cli",
    "app.ledger.db",
    "app.ledger.residency",
    "app.evals.store",
    "app.evals.metrics",
    "app.evals.metrics_push",
    "app.evals.residency",
    "app.evals.cli",
    "app.redaction.directory",
)

_NEW_STORAGE_AND_CLI_MODULES = (
    "app.storage.db",
    "app.storage.cost_ledger",
    "app.storage.console_events",
    "app.storage.residency_log",
    "app.storage.eval_store",
    "app.storage.dashboard_reads",
    "app.cli.ledger",
    "app.cli.evals",
    "app.evals.scores",
    "app.evals.push",
    "app.evals.region_check",
    "app.redaction.names_store",
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


def test_the_gateway_api_modules_are_gone() -> None:
    """Catches an impl that leaves a gateway app/support/route module importable after the split."""
    for name in _GONE_GATEWAY_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_the_gateway_package_is_gone() -> None:
    """Catches an impl that left the gateway namespace importable after console_reads moved."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.gateway")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.gateway.console_reads")


def test_the_old_storage_and_cli_modules_are_gone() -> None:
    """Catches an impl that left an old storage/cli/renamed module importable."""
    for name in _GONE_STORAGE_AND_CLI_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_the_storage_and_cli_packages_exist() -> None:
    """Catches an impl that did not create the new storage/cli packages and renamed modules."""
    for name in _NEW_STORAGE_AND_CLI_MODULES:
        importlib.import_module(name)
    app_dir = Path(app.__file__).parent
    for pkg in ("storage", "cli"):
        pkg_dir = app_dir / pkg
        assert pkg_dir.is_dir(), f"expected new package at {pkg_dir}"
        assert (pkg_dir / "__init__.py").is_file(), f"expected {pkg_dir / '__init__.py'}"


def test_the_admin_routes_are_split() -> None:
    """Catches an impl that did not split the admin surface into its own package and files."""
    app_dir = Path(app.__file__).parent
    admin_dir = app_dir / "api" / "routes" / "admin"
    assert admin_dir.is_dir(), f"expected admin route package at {admin_dir}"
    for leaf in ("routing.py", "incidents.py", "releases.py", "directory.py"):
        assert (admin_dir / leaf).is_file(), f"expected admin route file {admin_dir / leaf}"
    routes_dir = app_dir / "api" / "routes"
    for leaf in ("dashboard.py", "health.py", "model_list.py"):
        assert (routes_dir / leaf).is_file(), f"expected route file {routes_dir / leaf}"
