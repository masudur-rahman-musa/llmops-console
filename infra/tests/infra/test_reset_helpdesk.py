"""`infra/evals/reset_helpdesk.sh` and `reset_helpdesk.py`: the per-play reset of the help desk.

The Python file runs in the help desk's own Python, so here it is loaded by path and only its
parts that import nothing from the help desk are called: the check that says whether policy
search can work, and the statements that keep the policy rows across the reset. The shell
script is EXECUTED with a fake `uv` first on `PATH` that only writes down its arguments, so
nothing reaches a database.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALS = REPO_ROOT / "infra" / "evals"
RESET_SH = EVALS / "reset_helpdesk.sh"
RESET_PY = EVALS / "reset_helpdesk.py"
PREPARE_SH = EVALS / "prepare_helpdesk.sh"


def load_reset() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reset_helpdesk", RESET_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reset = load_reset()


def test_the_last_runs_state_is_reported_as_broken() -> None:
    """Catches the setup the last eval run played against unseen: Qdrant still held every
    tenant's vectors, but the reset had dropped every policy_documents row, so all 58 policy
    searches failed. Both the empty table and each tenant whose vectors lost their rows are
    named."""
    vectors = {"nordlicht": {"n1", "n2"}, "cuan": {"c1"}, "demo": {"d1"}}
    rows: dict[str, set[str]] = {"nordlicht": set(), "cuan": set(), "demo": set()}

    problems = reset.policy_problems(vectors, rows)

    assert any("no policy_documents rows" in line for line in problems), problems
    for slug in ("nordlicht", "cuan", "demo"):
        assert any(line.startswith(f"{slug}:") for line in problems), (slug, problems)


def test_matching_vectors_and_rows_are_not_a_problem() -> None:
    """Catches a check that stops a good run: every vector names a row that exists, and a
    tenant with rows but no collection is fine."""
    vectors = {"nordlicht": {"n1", "n2"}, "cuan": {"c1"}}
    rows = {"nordlicht": {"n1", "n2", "n-old"}, "cuan": {"c1"}, "demo": {"d1"}}

    assert reset.policy_problems(vectors, rows) == []


def test_one_missing_row_is_enough_to_stop() -> None:
    """Catches a check that only looks for an empty table: one vector whose row is gone
    still fails every search that finds it."""
    vectors = {"nordlicht": {"n1", "n2"}, "cuan": {"c1"}}
    rows = {"nordlicht": {"n1"}, "cuan": {"c1"}}

    problems = reset.policy_problems(vectors, rows)

    assert len(problems) == 1, problems
    assert problems[0].startswith("nordlicht: 1 documents")


def test_the_rows_are_copied_aside_into_a_schema_the_reset_does_not_drop() -> None:
    """Catches a copy kept inside `public`, which the reset drops with everything else, a
    copy that never replaces the last one, and chunks copied before their documents."""
    statements = reset.keep_statements()

    assert reset.KEEP_SCHEMA != "public"
    assert statements[0] == f"CREATE SCHEMA IF NOT EXISTS {reset.KEEP_SCHEMA}"
    creates = [s for s in statements if s.startswith("CREATE TABLE")]
    assert creates == [
        f"CREATE TABLE {reset.KEEP_SCHEMA}.policy_documents AS TABLE public.policy_documents",
        f"CREATE TABLE {reset.KEEP_SCHEMA}.policy_chunks AS TABLE public.policy_chunks",
    ]
    drops = [s for s in statements if s.startswith("DROP")]
    assert drops and all(f"{reset.KEEP_SCHEMA}." in s for s in drops)
    assert statements.index(drops[-1]) < statements.index(creates[0])


def test_the_rows_go_back_by_column_name() -> None:
    """Catches a `SELECT *` copy back, which breaks as soon as a migration adds a column."""
    statement = reset.restore_statement("policy_documents", ["id", "tenant_id", "slug"])

    assert statement == (
        'INSERT INTO public.policy_documents ("id", "tenant_id", "slug") '
        f'SELECT "id", "tenant_id", "slug" FROM {reset.KEEP_SCHEMA}.policy_documents'
    )


def run_reset(tmp_path: Path, *args: str) -> list[str]:
    """Run `reset_helpdesk.sh` with a fake `uv`; answer the arguments `uv` was given."""
    helpdesk = tmp_path / "helpdesk"
    (helpdesk / "backend").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv.log"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$PWD" "$@" > "{log}"\n')
    fake_uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

    subprocess.run(["bash", str(RESET_SH), *args, str(helpdesk)], env=env, check=True)
    return log.read_text().splitlines()


def test_the_shell_script_runs_the_python_reset_with_the_check(tmp_path: Path) -> None:
    """Catches a per-play reset that goes back to the help desk's own reset with no policy
    rows kept, or one that leaves the check off."""
    called = run_reset(tmp_path)

    assert called[0].endswith("helpdesk/backend")
    assert called[1:] == ["run", "--env-file", ".env.demo", "python", str(RESET_PY)]


def test_the_check_is_skipped_only_when_asked(tmp_path: Path) -> None:
    """Catches `--no-check` lost on the way to the Python reset, which would stop every
    eval run at its first reset, before the ingest has written anything."""
    called = run_reset(tmp_path, "--no-check")

    assert called[-2:] == [str(RESET_PY), "--no-check"]


def test_only_the_reset_before_the_ingest_skips_the_check() -> None:
    """Catches the prepare step's reset left checking (every run would stop before its
    ingest), and the `make eval` per-play reset given `--no-check`."""
    prepare = PREPARE_SH.read_text(encoding="utf-8")
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert 'reset_helpdesk.sh" --no-check "$1"' in prepare
    assert "--reset-command infra/evals/reset_helpdesk.sh --out" in makefile


def test_a_tenant_the_eval_never_ingests_is_not_checked() -> None:
    """Catches a check that stops a run over a tenant no case plays: `demo` has old vectors
    and no rows, but only the tenants with a policy folder are ingested and played, so only
    their gaps count. A gap in a played tenant still stops the run."""
    vectors = {"nordlicht": {"n1"}, "cuan": {"c1"}, "demo": {"d1"}}
    rows: dict[str, set[str]] = {"nordlicht": {"n1"}, "cuan": {"c1"}, "demo": set()}
    played = {"nordlicht", "cuan"}

    assert reset.policy_problems(vectors, rows, played) == []
    broken = reset.policy_problems(vectors, {**rows, "cuan": set()}, played)
    assert [line.split(":")[0] for line in broken] == ["cuan"]
