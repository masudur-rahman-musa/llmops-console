"""The set of test files that build their own backing-service connection is frozen.

Fast tests must take their engine, client or pool from the session fixtures in
`conftest.py`. The files that still build their own are listed here as a
committed allowlist, so a NEW file that builds its own client is caught the
moment it lands. An earlier change empties the allowlist.
"""

from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
SKIP_DIRS = frozenset({"__pycache__", ".venv"})

# The tokens that mark a file as building its own backing client. These strings
# also live in this guard file and in test_tooling_fast_suite.py as DATA (the
# patterns searched for / the Makefile names asserted on), so scanning those two
# meta-test files would self-match. They are excluded from the
# scan below; see `_scanned_files`.
OWN_CLIENT_TOKENS = (
    "OPSCONSOLE_TEST",
    "build_pool",
    "apply_migrations",
    "redis.Redis",
    "redis.from_url",
)

# The two meta-test files that carry the tokens as data and must not be scanned.
META_TEST_FILES = frozenset({"test_own_client_guard.py", "test_tooling_fast_suite.py"})

ALLOWLIST = sorted(
    (
        "api/test_app_wiring.py",
        "api/test_budget_enforcement.py",
        "api/test_cost_wiring.py",
        "api/test_exact_cache.py",
        "api/test_rate_limit_enforcement.py",
        "api/test_residency_log.py",
        "cache/test_qdrant_cache.py",
        "cache/test_redis_cache.py",
        "evals/test_cli.py",
        "evals/test_store.py",
        "limits/test_redis_budget.py",
        "limits/test_redis_ratelimit.py",
        "redaction/test_redis_directory.py",
        "redaction/test_redis_store.py",
        "storage/test_cli.py",
        "storage/test_console_events.py",
        "storage/test_console_events_postgres.py",
        "storage/test_cost_ledger.py",
        "storage/test_cost_ledger_postgres.py",
        "storage/test_dashboard_reads_postgres.py",
        "storage/test_eval_store_postgres.py",
        "storage/test_migrations.py",
        "storage/test_postgres_writer.py",
        "storage/test_residency_writer.py",
    )
)


def _source_builds_own_client(source: str) -> bool:
    """True when the source text names any backing-client token."""
    return any(token in source for token in OWN_CLIENT_TOKENS)


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


def test_only_the_allowlisted_files_build_their_own_backing_client() -> None:
    """Catches a new test file that builds its own backing client without being
    added to the frozen allowlist, and catches silently deleting an entry the
    allowlist still names.
    """
    offenders = _offenders()
    extra = sorted(set(offenders) - set(ALLOWLIST))
    missing = sorted(set(ALLOWLIST) - set(offenders))
    assert offenders == ALLOWLIST, (
        f"files building their own backing client but not on the allowlist: {extra}; "
        f"allowlist entries no longer detected: {missing}"
    )


def test_a_new_offender_file_is_caught() -> None:
    """Catches a detector that fails to flag a source building its own client
    (so a real offender would slip the allowlist check), or one that flags clean
    source (so every file would look like an offender).
    """
    assert _source_builds_own_client("import x\npool = build_pool(url)\n") is True
    assert _source_builds_own_client("x = 1\n") is False
