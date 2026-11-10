#!/usr/bin/env python3
"""SessionStart hook: print project-management/CURRENT.md so a session starts knowing where
things stand, without spending a tool call. Silent when there is no CURRENT.md; never fails.

In a linked worktree of a private-mode repo it first gives the worktree its own PM checkout: a
linked worktree of the main folder's nested PM repo at <worktree>/project-management, on a PM
branch named after the host branch (new from the PM repo's main, or the existing one). PM
commits made there land with the host branch (.claude/rules/git.md, "Landing a worktree"). A
worktree that can't have one reads the main folder's PM files, and one more line says where
(`PM files: <path>`). The rules come from find_pm() and pm_checkout_plan() in the cleanup
skill's pm_check.py.

A session that moved into its worktree after it started runs this by hand from there:
    python .claude/hooks/session_start.py .
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PM_CHECK = Path(__file__).resolve().parents[1] / "skills" / "cleanup" / "pm_check.py"


def load_pm_check():
    try:
        spec = importlib.util.spec_from_file_location("pm_check", PM_CHECK)
        pm_check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pm_check)
        return pm_check
    except Exception:  # no checker installed: the repo's own folder only
        return None


def git(cwd: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                       timeout=30)
    return p.returncode, (p.stdout + p.stderr).strip()


def add_checkout(pm_check, root: Path) -> str:
    """Create the worktree's PM checkout if it should have one: "created", or why not."""
    main_pm, branch = pm_check.pm_checkout_plan(root)
    if main_pm is None:
        return branch  # the reason
    path = root / "project-management"
    base = "main" if git(main_pm, "rev-parse", "--verify", "-q", "refs/heads/main")[0] == 0 \
        else "HEAD"
    exists = git(main_pm, "rev-parse", "--verify", "-q", f"refs/heads/{branch}")[0] == 0
    rc, out = git(main_pm, "worktree", "add", "-q",
                  *([str(path), branch] if exists else ["-b", branch, str(path), base]))
    if rc != 0:
        return out.splitlines()[-1] if out else f"git worktree add failed ({rc})"
    if exists:  # an older PM branch of the same name catches up with main when it can
        git(path, "merge", "--ff-only", "-q", base)
    return "created"


def where(pm_check, root: Path) -> tuple[Path, list[str]]:
    """The PM folder, and the lines that say whose it is when it isn't the plain one."""
    made = add_checkout(pm_check, root)
    pm, from_main = pm_check.find_pm(root)
    if pm_check.is_pm_checkout(pm):
        _, branch = git(pm, "symbolic-ref", "--short", "-q", "HEAD")
        return pm, [f"PM files: project-management/ is this worktree's own checkout of the PM "
                    f"repo, on PM branch '{branch}'{' (created now)' if made == 'created' else ''}"
                    ". Commit PM changes in it as you go; they land with this branch "
                    "(.claude/rules/git.md, before taking a new ID)."]
    if from_main:
        return pm, ([f"No PM checkout for this worktree: {made}"] if made else []) + [
            f"PM files: {pm.as_posix()}"]
    return pm, []


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLAUDE_PROJECT_DIR") or ".")
    root = root.resolve()
    pm, lines = root / "project-management", []
    pm_check = load_pm_check()
    if pm_check:
        try:
            pm, lines = where(pm_check, root)
        except Exception:  # a checker from another version: print CURRENT.md all the same
            pass
    current = pm / "CURRENT.md"
    if current.is_file():
        text = current.read_bytes()
        sys.stdout.buffer.write((text if text.endswith(b"\n") else text + b"\n")
                                + "".join(f"{ln}\n" for ln in lines).encode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
