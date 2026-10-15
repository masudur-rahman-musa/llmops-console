"""The backend test tree stands on its own: no backend test imports `infra`.

`infra/` is a separate tree with its own tests. A backend test that imports it
only works when the repo root happens to be on `sys.path`, so it goes red the
moment either tree moves.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
SKIP_DIRS = frozenset({"__pycache__", ".venv"})
FORBIDDEN_TOP = "infra"


def _imports_infra(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == FORBIDDEN_TOP for alias in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and node.module.split(".")[0] == FORBIDDEN_TOP
        ):
            return True
    return False


def _test_files() -> list[Path]:
    return sorted(
        path
        for path in TESTS_ROOT.rglob("*.py")
        if not SKIP_DIRS.intersection(path.relative_to(TESTS_ROOT).parts)
    )


def test_no_backend_test_imports_the_infra_tree() -> None:
    """Every `.py` under `backend/tests/` is parsed and none imports `infra`.

    Catches a backend test that reaches into the `infra/` tree with
    `import infra...` or `from infra... import ...`, such as
    `evals/test_tradeoff_policy.py` borrowing helpers from `infra.test_hetzner_module`.
    """
    files = _test_files()
    assert files, f"no test files found under {TESTS_ROOT}"
    offenders = [
        path.relative_to(TESTS_ROOT).as_posix()
        for path in files
        if _imports_infra(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    ]
    assert offenders == [], f"backend tests importing the infra tree: {offenders}"
