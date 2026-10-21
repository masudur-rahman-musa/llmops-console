"""Structural tests that pin the module layout changes from an earlier change.

Each test checks where a name lives (or must not live) so that the split,
move and cleanup steps are proven by import rather than by convention.

Test 3 (probe) is here as a standalone case; the old parametrised table in
test_probes.py is replaced by the new test added to that file directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).parent.parent / "app"


def app_py_files(*, exclude_tests: bool = True) -> list[Path]:
    """All .py files under backend/app/, optionally skipping tests/."""
    return [p for p in APP_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


# ---------------------------------------------------------------------------
# test 3 — provider probe: non-empty mapping → True, empty → False,
#           no adapter method called
# ---------------------------------------------------------------------------


def test_the_provider_probe_is_ready_when_an_adapter_exists() -> None:
    """Catches an implementation that kept calling adapter.healthcheck() instead
    of checking only whether the mapping is non-empty, or one that returns False
    for a non-empty mapping / True for an empty one."""
    from app.api.routes.health import provider_probe

    class _NoMethodAdapter:
        """A plain object with NO healthcheck — probe must not call any method."""

        name = "fake"

        def chat(self, request: object) -> object:  # pragma: no cover
            raise AssertionError("chat must not be called by the probe")

        def stream_chat(self, request: object) -> object:  # pragma: no cover
            raise AssertionError("stream_chat must not be called by the probe")

        def price_key(self, model: str) -> str:  # pragma: no cover
            raise AssertionError("price_key must not be called by the probe")

    fake = _NoMethodAdapter()

    assert provider_probe({"fake": fake})() is True  # type: ignore[dict-item]
    assert provider_probe({})() is False


# ---------------------------------------------------------------------------
# test 4 — chat_pipeline helpers live in their new sibling modules
# ---------------------------------------------------------------------------


def test_the_chat_pipeline_helpers_live_in_their_new_modules() -> None:
    """Catches an implementation that left check_budget, book_cost,
    semantic_answer or gateway_headers defined inside chat_pipeline instead of
    moving them to their new homes (chat_accounting, chat_cache, chat_headers)."""
    from app.services import chat_accounting, chat_cache, chat_headers

    assert hasattr(chat_accounting, "check_budget"), (
        "check_budget must be defined in chat_accounting"
    )
    assert hasattr(chat_accounting, "book_cost"), "book_cost must be defined in chat_accounting"
    assert hasattr(chat_cache, "semantic_answer"), "semantic_answer must be defined in chat_cache"
    assert hasattr(chat_headers, "gateway_headers"), (
        "gateway_headers must be defined in chat_headers"
    )

    # None of the four must be defined (owned) in chat_pipeline itself.
    import app.services.chat_pipeline as pipeline_mod

    for name in ("check_budget", "book_cost", "semantic_answer", "gateway_headers"):
        assert name not in pipeline_mod.__dict__, (
            f"{name} must not be defined in chat_pipeline (it belongs in its new module)"
        )


# ---------------------------------------------------------------------------
# test 5 — eval CLI handlers live in evals_commands
# ---------------------------------------------------------------------------


def test_the_eval_cli_handlers_live_in_the_commands_module() -> None:
    """Catches an implementation that left _conversations, _label, _gate, _tradeoff,
    _calibrate or _baseline_guard defined in cli.evals instead of moving them to
    cli.evals_commands, or one that removed main from cli.evals. `_run` is gone with the
    single-reply command, and a leftover would keep the old command reachable."""
    from app.cli import evals, evals_commands

    for handler in (
        "_conversations",
        "_label",
        "_gate",
        "_tradeoff",
        "_calibrate",
        "_baseline_guard",
    ):
        assert hasattr(evals_commands, handler), f"{handler} must be defined in cli.evals_commands"
    assert not hasattr(evals_commands, "_run"), "the old `run` handler must be gone"

    assert hasattr(evals, "main"), "cli.evals must still expose main"


# ---------------------------------------------------------------------------
# test 6 — no app module imports a private name from another app module
# ---------------------------------------------------------------------------


def test_no_app_module_imports_a_private_name_from_another_app_module() -> None:
    """Catches an implementation that left a cross-module import of a _name
    (e.g. `from app.services.dashboard import _postgres`) instead of making
    that name public first."""
    violations: list[str] = []

    for py_file in app_py_files():
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if not module.startswith("app."):
                continue
            for alias in node.names:
                imported_name = alias.name
                if imported_name.startswith("_"):
                    rel = py_file.relative_to(APP_ROOT.parent.parent)
                    violations.append(f"{rel}: from {module} import {imported_name}")

    assert violations == [], "Private names imported across app modules:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# test 7 — neutral contracts live in schemas.provider, not providers.base
# ---------------------------------------------------------------------------


def test_the_neutral_contracts_live_in_schemas_provider() -> None:
    """Catches an implementation that left ChatRequest, ChatResponse, StreamStart
    or EmbeddingRequest defined in providers.base instead of moving them to
    schemas.provider, or one that forgot to add them to schemas.provider."""
    import app.providers.base as providers_base
    from app.schemas import provider as schemas_provider

    for name in ("ChatRequest", "ChatResponse", "StreamStart", "EmbeddingRequest"):
        assert hasattr(schemas_provider, name), f"{name} must be defined in app.schemas.provider"
        # Must NOT be defined (owned) in providers.base
        obj_in_base = getattr(providers_base, name, None)
        if obj_in_base is not None:
            # It is fine for providers.base to RE-IMPORT the name so existing
            # code keeps working, but it must not be the DEFINING module.
            defining_module = getattr(obj_in_base, "__module__", None)
            assert defining_module != "app.providers.base", (
                f"{name}.__module__ is still app.providers.base; "
                "move the definition to app.schemas.provider"
            )


# ---------------------------------------------------------------------------
# test 8 — no production module re-exports a name only tests use
# ---------------------------------------------------------------------------


def test_no_production_module_reexports_a_name_only_tests_use() -> None:
    """Catches an implementation that left transport constants (e.g.
    CONNECT_TIMEOUT, MAX_RETRIES) in a production __all__ even though nothing
    outside tests/ imports them from that module."""
    tests_root = Path(__file__).parent

    violations: list[str] = []

    for py_file in app_py_files():
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        # Collect names in __all__ for this module.
        all_names: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
                and isinstance(node.value, ast.List)
            ):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        all_names.append(elt.value)

        if not all_names:
            continue

        # Derive the dotted module name from the file path.
        rel_parts = py_file.relative_to(APP_ROOT.parent).with_suffix("").parts
        module_dotted = ".".join(rel_parts)

        for exported_name in all_names:
            # Search for any non-test importer of this name from this module.
            found_non_test_importer = False
            search_root = APP_ROOT.parent  # backend/

            for candidate in search_root.rglob("*.py"):
                if "__pycache__" in candidate.parts:
                    continue
                # Skip the module itself.
                if candidate == py_file:
                    continue
                # Skip test files for this check — we want non-test importers.
                try:
                    candidate.relative_to(tests_root)
                    is_test = True
                except ValueError:
                    is_test = False

                if is_test:
                    continue

                try:
                    csource = candidate.read_text(encoding="utf-8")
                except OSError:
                    continue

                # Quick text scan: `from <module> import ... <name> ...`
                # or `import <module>` followed by `<module>.<name>`.
                # A simple substring check is good enough here; we only need
                # to know whether ANY non-test file references it.
                if exported_name in csource and module_dotted in csource:
                    found_non_test_importer = True
                    break

            if not found_non_test_importer:
                # Double-check: is there at least one test importer?
                # (If nothing imports it at all it may just be unused, which
                #  is a different problem; we only flag test-only re-exports.)
                has_test_importer = False
                for candidate in tests_root.rglob("*.py"):
                    if "__pycache__" in candidate.parts:
                        continue
                    try:
                        csource = candidate.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    if exported_name in csource and module_dotted in csource:
                        has_test_importer = True
                        break

                if has_test_importer:
                    rel = py_file.relative_to(APP_ROOT.parent.parent)
                    violations.append(
                        f"{rel}: {module_dotted}.__all__ exports '{exported_name}' "
                        "but only tests/ imports it from there"
                    )

    assert violations == [], "Names re-exported only for tests:\n" + "\n".join(violations)
