#!/usr/bin/env python3
"""SessionStart hook: print project-management/CURRENT.md so a session starts knowing where
things stand, without spending a tool call. Silent when there is no CURRENT.md; never fails."""
import os
import subprocess
import sys
from pathlib import Path


def find_current(root: Path) -> Path | None:
    p = root / "project-management" / "CURRENT.md"
    if p.is_file():
        return p
    # Private PM mode inside a linked worktree: project-management/ exists only in the main folder.
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=root, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return None
    if common:
        p = Path(common).parent / "project-management" / "CURRENT.md"
        if p.is_file():
            return p
    return None


def main() -> int:
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or ".")
    p = find_current(root)
    if p:
        sys.stdout.buffer.write(p.read_bytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
