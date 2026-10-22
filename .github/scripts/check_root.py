#!/usr/bin/env python3
"""Keep the repo root to the names in .claude/root-allowlist.

  check_root.py                 check every tracked or new (not ignored) file; exit 1 on a stray
  check_root.py --claude-hook   PreToolUse hook for Write, Edit and Bash (mkdir, touch, > file)

Anything else belongs inside one of the allowed folders, by KIND, as the
`layout` skill's "Root folders" table says.
"""

import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(
    os.environ.get("CLAUDE_PROJECT_DIR")
    or subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    ).stdout.strip()
    or "."
)
HELP = (
    "It belongs inside an allowed folder by kind: "
    "see 'Root folders' in .claude/skills/layout/SKILL.md. "
    "A truly new kind of folder needs a plan the owner approves, "
    "which adds it to .claude/root-allowlist. "
    "Under auto mode never widen the list: "
    "put it by kind and log it (.claude/guides/auto-mode.md)."
)


def allowed() -> list[str]:
    lines = (ROOT / ".claude" / "root-allowlist").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def ok(name: str, is_dir: bool) -> bool:
    for pat in allowed():
        if pat.endswith("/"):
            if is_dir and name == pat[:-1]:
                return True
        elif fnmatch.fnmatch(name, pat):
            return True
    return False


def top(path: str, base: Path | None = None) -> tuple[str, bool] | None:
    """The root entry a path lands in, and whether it is a folder; None if outside the repo.
    A relative path is read from `base` (the folder a leading `cd` moved to), else the repo root."""
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    p = ((base or ROOT) / p) if not p.is_absolute() else p
    try:
        rel = p.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    return parts[0], len(parts) > 1


def hook() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    tool = payload.get("tool_name", "")
    data = payload.get("tool_input") or {}
    targets: list[tuple[str, bool]] = []
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = data.get("file_path") or data.get("notebook_path") or ""
        if path:
            t = top(path)
            if t:
                targets.append(t)
    elif tool == "Bash":
        # Heredoc bodies are text being written, not commands being run.
        cmd = re.sub(
            r"<<-?\s*(['\"]?)(\w+)\1([^\n]*)\n.*?\n\s*\2\s*(?=\n|$)",
            r"<<\2\3",
            data.get("command", ""),
            flags=re.S,
        )
        # Quoted strings can hold a ">" that is not a redirect.
        cmd = re.sub(r"'[^']*'|\"[^\"]*\"", "''", cmd)
        base = ROOT
        for part in re.split(r"[;&|\n]+", cmd):
            try:
                words = shlex.split(part)
            except ValueError:
                words = part.split()
            if words[:1] == ["cd"] and len(words) > 1:
                nb = Path(os.path.expandvars(os.path.expanduser(words[1])))
                base = nb if nb.is_absolute() else base / nb
                continue
            if words[:1] in (["mkdir"], ["touch"]):
                for w in words[1:]:
                    if not w.startswith("-"):
                        t = top(w, base)
                        if t:
                            targets.append((t[0], words[0] == "mkdir" or t[1]))
            for m in re.finditer(r"(?<![<>&\d=-])>>?\s*([\w./~-][^\s;&|`'\"]*)", part):
                t = top(m.group(1), base)
                if t:
                    targets.append(t)
    for name, is_dir in targets:
        if not ok(name, is_dir):
            print(
                f"BLOCKED by .github/scripts/check_root.py: `{name}` is not allowed "
                "at the repo root. "
                f"{HELP}",
                file=sys.stderr,
            )
            return 2
    return 0


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
    sys.exit(hook() if "--claude-hook" in sys.argv else check())
