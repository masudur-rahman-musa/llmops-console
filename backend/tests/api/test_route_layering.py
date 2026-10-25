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
    "app.evals.store",
    "app.evals.metrics",
    "app.evals.metrics_push",
    "app.evals.cli",
    "app.redaction.directory",
)

_NEW_STORAGE_AND_CLI_MODULES = (
    "app.storage.db",
    "app.storage.cost_ledger",
    "app.storage.console_events",
    "app.storage.eval_store",
    "app.storage.dashboard_reads",
    "app.cli.ledger",
    "app.cli.evals",
    "app.evals.scores",
    "app.evals.push",
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


# ---------------------------------------------------------------------------
# An earlier change: storage-only-data
# ---------------------------------------------------------------------------

_COST_NAMES = ("record_cost",)


def test_storage_holds_no_cost_decision() -> None:
    """Catches an impl that leaves record_cost importable from app.storage or its
    sub-modules."""
    cost_ledger = importlib.import_module("app.storage.cost_ledger")
    storage = importlib.import_module("app.storage")

    assert not hasattr(cost_ledger, "record_cost"), (
        "app.storage.cost_ledger must not expose record_cost"
    )
    for name in _COST_NAMES:
        assert not hasattr(storage, name), f"app.storage must not re-export {name}"


def test_the_cost_recording_lives_in_the_service() -> None:
    """Catches an impl that did not move record_cost into app.services.ledger."""
    ledger = importlib.import_module("app.services.ledger")
    for name in _COST_NAMES:
        assert hasattr(ledger, name), f"app.services.ledger must define {name}"


def test_the_ledger_cli_runs_no_sql() -> None:
    """Catches an impl where app/cli/ledger.py still contains raw SQL strings or
    direct connection calls instead of delegating to the storage modules."""
    app_dir = Path(app.__file__).parent
    cli_ledger_path = app_dir / "cli" / "ledger.py"
    source = cli_ledger_path.read_text()
    tree = ast.parse(source)

    # Check no raw SQL patterns appear as string literals in the source text.
    for banned in ("DROP TABLE", "REFRESH"):
        assert banned not in source, f"app/cli/ledger.py must not contain raw SQL string {banned!r}"

    # Check no sql.SQL(...) attribute calls exist.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sql"
            and node.attr == "SQL"
        ):
            raise AssertionError(
                "app/cli/ledger.py must not call sql.SQL; delegate to storage modules"
            )

    # Check no conn.execute(...) calls exist.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "conn"
            and node.attr == "execute"
        ):
            raise AssertionError(
                "app/cli/ledger.py must not call conn.execute; delegate to storage modules"
            )

    refresh_from_cost = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "app.storage.cost_ledger":
            for alias in node.names:
                if alias.name == "refresh_daily_rollup":
                    refresh_from_cost = True

    assert refresh_from_cost, (
        "app/cli/ledger.py must import refresh_daily_rollup from app.storage.cost_ledger"
    )


_DASHBOARD_MODEL_NAMES = (
    "LastEval",
    "CostDay",
    "Saving",
    "AliasCost",
    "EvalRunSummary",
    "ReleaseMarker",
    "CategoryCount",
    "TagCount",
    "Regression",
    "Incident",
    "Release",
)


def test_the_console_read_models_live_in_schemas() -> None:
    """Catches an impl that left the console read models defined in
    app.storage.dashboard_reads instead of moving them to app.schemas.dashboard."""
    schemas_mod = importlib.import_module("app.schemas.dashboard")
    storage_mod = importlib.import_module("app.storage.dashboard_reads")

    for name in _DASHBOARD_MODEL_NAMES:
        # Must be reachable from schemas.
        assert hasattr(schemas_mod, name), f"{name} must be defined in app.schemas.dashboard"
        obj = getattr(schemas_mod, name)
        assert getattr(obj, "__module__", "").startswith("app.schemas.dashboard"), (
            f"{name}.__module__ must be app.schemas.dashboard, got {obj.__module__!r}"
        )

        # Must NOT be defined in storage.
        if hasattr(storage_mod, name):
            obj_storage = getattr(storage_mod, name)
            assert not getattr(obj_storage, "__module__", "").startswith(
                "app.storage.dashboard_reads"
            ), f"{name} must not be defined in app.storage.dashboard_reads"


def test_schemas_dashboard_imports_no_storage() -> None:
    """Catches an impl where app/schemas/dashboard.py still imports from app.storage,
    meaning the read models were not moved out of storage."""
    app_dir = Path(app.__file__).parent
    schemas_dashboard_path = app_dir / "schemas" / "dashboard.py"
    imported = _imported_modules(schemas_dashboard_path)
    for module in imported:
        assert not (module == "app.storage" or module.startswith("app.storage.")), (
            f"app/schemas/dashboard.py must not import from app.storage; found: {module}"
        )
