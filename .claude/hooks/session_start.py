#!/usr/bin/env python3
"""SessionStart hook: print project-management/CURRENT.md so a session starts knowing where
things stand, without spending a tool call. In a linked worktree of a private-mode repo the PM
files live in the main folder; then one more line says where (`PM files: <path>`). The folder
is found by find_pm() in the cleanup skill's pm_check.py, the one copy of that rule. Silent
when there is no CURRENT.md; never fails."""
import importlib.util
import os
import sys
from pathlib import Path

PM_CHECK = Path(__file__).resolve().parents[1] / "skills" / "cleanup" / "pm_check.py"


def find_pm(root: Path) -> tuple[Path, bool]:
    try:
        spec = importlib.util.spec_from_file_location("pm_check", PM_CHECK)
        pm_check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pm_check)
        return pm_check.find_pm(root)
    except Exception:  # no checker installed: the repo's own folder only
        return root / "project-management", False


def main() -> int:
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or ".")
    pm, from_main = find_pm(root)
    current = pm / "CURRENT.md"
    if current.is_file():
        text = current.read_bytes()
        sys.stdout.buffer.write(text if text.endswith(b"\n") else text + b"\n")
        if from_main:
            sys.stdout.buffer.write(f"PM files: {pm.as_posix()}\n".encode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
