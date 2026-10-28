#!/usr/bin/env python3
"""Keep the repo root to the names in .github/root-allowlist.

  check_root.py   check every tracked or new (not ignored) file; exit 1 on a stray

Anything else belongs inside one of the allowed folders.
"""

import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    ).stdout.strip()
    or "."
)
HELP = (
    "It belongs inside one of the allowed folders. "
    "A new kind of top-level folder is added to .github/root-allowlist."
)


def allowed() -> list[str]:
    lines = (ROOT / ".github" / "root-allowlist").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def ok(name: str, is_dir: bool) -> bool:
    for pat in allowed():
        if pat.endswith("/"):
            if is_dir and name == pat[:-1]:
                return True
        elif fnmatch.fnmatch(name, pat):
            return True
    return False


def check() -> int:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    stray = sorted(
        {
            f.split("/")[0] + ("/" if "/" in f else "")
            for f in out
            if not ok(f.split("/")[0], "/" in f)
        }
    )
    # git ignores `.env`, so the file list above never shows it. The real values
    # live in `backend/.env`; a root copy is a stray even though `.*` would allow it.
    if (ROOT / ".env").exists():
        stray.append(".env")
    for s in stray:
        print(f"not allowed at the repo root: {s}")
    if stray:
        print(HELP)
    return 1 if stray else 0


if __name__ == "__main__":
    sys.exit(check())
