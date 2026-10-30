"""`make tenant-key`: a fresh local tenant key and the env line that holds its digest.

Runs the target itself, so it skips only when `make` or `openssl` is not on `PATH`. No
case needs a backing service, and none writes anywhere.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from infra.test_hetzner_module import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTING = REPO_ROOT / "backend" / "config" / "routing.yaml"


def run_target(*args: str) -> tuple[str, str]:
    """The printed key and env line of one `make tenant-key` run."""
    if shutil.which("make") is None or shutil.which("openssl") is None:
        pytest.skip("make or openssl is not on PATH")
    completed = subprocess.run(
        ["make", "-s", "tenant-key", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 2, lines
    key_line, env_line = lines
    match = re.fullmatch(r"key: ([0-9a-f]{64})", key_line)
    assert match is not None, key_line
    return match.group(1), env_line


def test_the_env_line_holds_the_sha256_of_the_printed_key() -> None:
    """Catches a line that carries the key itself, a digest of something else (a trailing
    newline hashed in), or a tenant other than the one asked for."""
    key, env_line = run_target("TENANT=cuan")
    digest = hashlib.sha256(key.encode()).hexdigest()

    assert env_line == f"OPSCONSOLE_API_KEYS=cuan:{digest}"
    assert key not in env_line


def test_the_default_tenant_is_one_the_routing_policy_names() -> None:
    """Catches a default tenant the gateway does not know, whose key would be refused."""
    _, env_line = run_target()
    tenant = env_line.removeprefix("OPSCONSOLE_API_KEYS=").split(":")[0]

    assert tenant in load_yaml(ROUTING)["tenants"]


def test_two_runs_make_two_different_keys() -> None:
    """Catches a fixed or reused key."""
    assert run_target()[0] != run_target()[0]
