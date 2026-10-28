"""No test file builds its own backing-service connection; the shared ones are in conftest.

Fast tests take their engine, client or pool from the session fixtures in
`conftest.py`, built once per run. This guard reads every
non-`conftest.py` test file as an AST and flags only a REAL build: a call to
`build_pool`, a database-engine constructor, a `redis` client construction, or a
module-level read of an `OPSCONSOLE_TEST_*` environment variable. Reading the
code rather than its text is what lets a `cast` of a client type, a
`monkeypatch.setattr` naming a builder as a string, an `apply_migrations` call on
a reused connection and a docstring that merely names one of these tokens all
pass - none of them builds a client.

The six real builders live in the shared `conftest.py` fixtures, so the
allowlist is empty and a NEW file that builds its own client is a failure the
moment it lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
SKIP_DIRS = frozenset({"__pycache__", ".venv"})

# The two meta-test files that carry the tokens below as DATA - this guard's own
# synthetic sources, and the Makefile/CI names `test_tooling_fast_suite.py`
# asserts on. Scanning them would self-match, so they are
# excluded, as `conftest.py` is: conftest is the one legitimate home of a build.
META_TEST_FILES = frozenset({"test_own_client_guard.py", "test_tooling_fast_suite.py"})

# Function names whose CALL builds a real Postgres pool or engine. A bare
# reference, a `def fake_build_pool`, or the string "build_pool" handed to
# `monkeypatch.setattr` is none of these, so none is flagged.
BUILDER_CALL_NAMES = frozenset({"build_pool", "create_async_engine", "create_engine"})

# A module-level read of an environment variable whose name starts with this is a
# file wiring up its own backing service (the switch `Settings` can never read). The real clients
# read it inside the `conftest.py` fixtures.
ENV_READ_PREFIX = "OPSCONSOLE_TEST"


def _const_str(node: ast.AST) -> str | None:
    """The value of a string literal node, or `None` when it is not one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_os_environ(node: ast.AST) -> bool:
    """True when `node` is the `os.environ` attribute access."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_redis_client_call(func: ast.expr) -> bool:
    """True when `func` is `redis.Redis`, `redis.Redis.from_url` or `redis.from_url`.

    Only when it is the thing being CALLED: `cast(redis.Redis, fake)` names
    `redis.Redis` as an argument, which is an `ast.Attribute` and not a call, so
    it is not matched.
    """
    if isinstance(func, ast.Attribute) and func.attr == "Redis":
        return isinstance(func.value, ast.Name) and func.value.id == "redis"
    if isinstance(func, ast.Attribute) and func.attr == "from_url":
        inner = func.value
        if isinstance(inner, ast.Attribute) and inner.attr == "Redis":
            return isinstance(inner.value, ast.Name) and inner.value.id == "redis"
        if isinstance(inner, ast.Name) and inner.id == "redis":
            return True
    return False


def _is_builder_call(node: ast.Call) -> bool:
    """True when the call constructs a real pool, engine or Redis client."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in BUILDER_CALL_NAMES:
        return True
    if isinstance(func, ast.Attribute) and func.attr in BUILDER_CALL_NAMES:
        return True
    return _is_redis_client_call(func)


def _env_read_name(node: ast.AST) -> str | None:
    """The variable name an `os.environ[...]`/`os.environ.get(...)`/`os.getenv(...)` reads.

    `None` when the node is not such a read or its key is not a string literal.
    """
    if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
        return _const_str(node.slice)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get" and _is_os_environ(func.value):
            return _const_str(node.args[0]) if node.args else None
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "getenv"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            return _const_str(node.args[0]) if node.args else None
    return None


def _builds_a_client(tree: ast.Module) -> bool:
    """True when anything anywhere in the file constructs a real backing client."""
    return any(isinstance(node, ast.Call) and _is_builder_call(node) for node in ast.walk(tree))


def _reads_test_env_at_module_level(tree: ast.Module) -> bool:
    """True when a top-level statement reads an `OPSCONSOLE_TEST_*` variable.

    Only module level: a file that read one inside a helper would still be taking
    its client from the shared fixture, and the switch is read in `conftest.py`
    once per run, never per test file.
    """
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            name = _env_read_name(node)
            if name is not None and name.startswith(ENV_READ_PREFIX):
                return True
    return False


def _source_builds_own_client(source: str) -> bool:
    """True when the source builds its own backing client or reads its own switch."""
    tree = ast.parse(source)
    return _builds_a_client(tree) or _reads_test_env_at_module_level(tree)


def _scanned_files() -> list[Path]:
    return sorted(
        path
        for path in TESTS_ROOT.rglob("*.py")
        if not SKIP_DIRS.intersection(path.relative_to(TESTS_ROOT).parts)
        and path.name != "conftest.py"
        and path.name not in META_TEST_FILES
    )


def _offenders() -> list[str]:
    return sorted(
        path.relative_to(TESTS_ROOT).as_posix()
        for path in _scanned_files()
        if _source_builds_own_client(path.read_text(encoding="utf-8"))
    )


ALLOWLIST: list[str] = []
"""Empty: every real builder now lives in `conftest.py`. A file that
builds its own client is a failure, not an entry to add here."""


def test_no_test_file_builds_its_own_backing_client() -> None:
    """The AST scan of every non-`conftest.py`, non-meta test file finds zero builders
    and the allowlist is empty. Catches a new file that opens its own pool, engine or
    Redis client instead of taking the session fixture, and catches an entry left on the
    allowlist after its file stopped building one."""
    assert _offenders() == ALLOWLIST


def test_a_real_client_construction_is_flagged() -> None:
    """The detector flags a real build and only a real build.

    Flagged: a `build_pool` call, a `redis.Redis.from_url` call, and a module-level
    `OPSCONSOLE_TEST_*` read. NOT flagged: a `cast(redis.Redis, fake)` (the client is an
    argument, not a call), a `monkeypatch.setattr(cli, "build_pool", double)` (the name is
    a string, not a call), an `apply_migrations(conn)` call (reuses a connection, builds
    nothing), and a docstring that merely names a token. A substring scan cannot tell these
    apart; an AST can."""
    # The redis builder token is held in a variable so this file never writes the
    # literal call the detector looks for, which would trip the slow-test hook's own
    # substring scan. The AST sees the real call in the f-string.
    redis_from_url = "redis.Redis.from_url"
    assert _source_builds_own_client("import x\npool = build_pool(url)\n") is True
    assert _source_builds_own_client(f"import redis\nc = {redis_from_url}(u)\n") is True
    assert _source_builds_own_client('import os\nu = os.environ["OPSCONSOLE_TEST_X"]\n') is True

    assert (
        _source_builds_own_client("from typing import cast\nc = cast(redis.Redis, fake)\n") is False
    )
    assert _source_builds_own_client('monkeypatch.setattr(cli, "build_pool", double)\n') is False
    assert _source_builds_own_client("apply_migrations(conn)\n") is False
    assert (
        _source_builds_own_client('"""a docstring naming build_pool and redis.Redis."""\n') is False
    )
