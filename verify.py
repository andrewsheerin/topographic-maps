#!/usr/bin/env python3
"""One command to answer "is this repo healthy?" — run: python verify.py

Runs every check that applies and skips (without failing) whatever isn't set up yet:
  PM        .claude/skills/cleanup/pm_check.py: project-management/ on the v3 standard
  Hooks     pytest .claude/hooks: the git guard's behaviour matrix
  Backend   ruff format --check, ruff check, pytest
  Frontend  npm run lint / typecheck / test, when package.json defines them
Python checks and ruff use the repo's .venv when there is one; ruff that is configured but
can't be found fails rather than skips. In a linked worktree of a private-mode repo, pm_check
reads the worktree's own PM checkout, or the main folder's project-management/ when it has
none. A linked worktree that lacks a .venv or frontend/node_modules the main folder has fails
those checks with a pointer to .claude/rules/git.md, "Setting up a worktree", instead of
running them with the wrong tools. Exit 0 = every check that ran passed; anything
else = fix it (or register it) before wrapping, pushing, or opening a PR.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV = [ROOT / ".venv/Scripts/python.exe", ROOT / ".venv/bin/python"]
PY = str(next((p for p in VENV if p.exists()), sys.executable))
VENV_RUFF = [ROOT / ".venv/Scripts/ruff.exe", ROOT / ".venv/bin/ruff"]
RUFF = next((str(p) for p in VENV_RUFF if p.exists()), shutil.which("ruff"))
SETUP = '.claude/rules/git.md, "Setting up a worktree"'
results = []


def main_folder():
    """The main folder when ROOT is a linked worktree (its .git is a file), else None."""
    if not (ROOT / ".git").is_file():
        return None
    try:
        p = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                           cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return Path(p.stdout.strip()).parent if p.returncode == 0 and p.stdout.strip() else None


MAIN = main_folder()


def not_set_up(folder):
    """A linked worktree without its own copy of a folder the main folder has (.venv,
    frontend/node_modules): the checks would run with the wrong tools, or none."""
    return MAIN is not None and (MAIN / folder).is_dir() and not (ROOT / folder).is_dir()


def run(name, args, cwd):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        results.append((name, f"ERROR ({e})"))
        return False
    ok = p.returncode == 0
    results.append((name, "PASS" if ok else "FAIL"))
    if not ok:
        tail = (p.stdout + p.stderr).strip().splitlines()[-15:]
        print(f"\n--- {name} output (tail) ---")
        print("\n".join(tail))
    return ok


def skip(name, why):
    results.append((name, f"SKIP ({why})"))


def has_pytest():
    return subprocess.run([PY, "-c", "import pytest"], capture_output=True).returncode == 0


def ruff_configured(*folders):
    for d in folders:
        if (d / "ruff.toml").exists() or (d / ".ruff.toml").exists():
            return True
        pyproject = d / "pyproject.toml"
        if pyproject.exists() and "[tool.ruff" in pyproject.read_text(encoding="utf-8"):
            return True
    return False


def factory():
    ok = True
    checker = ROOT / ".claude/skills/cleanup/pm_check.py"
    if checker.exists():
        ok &= run("pm_check", [PY, str(checker), str(ROOT)], ROOT)
    else:
        skip("pm_check", "checker not installed")
    if not any((ROOT / ".claude/hooks").glob("*_test.py")):
        skip("hook tests", "none")
    elif has_pytest():
        ok &= run("hook tests", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                 ".claude/hooks"], ROOT)
    else:
        skip("hook tests", "pytest not installed")
    return ok


def backend():
    be = ROOT / "backend"
    if not be.exists() or not any(be.rglob("*.py")):
        skip("backend", "no python files yet")
        return True
    if not_set_up(".venv"):
        results.append(("backend", f"FAIL (this worktree has no .venv; {SETUP})"))
        return False
    ok = True
    if RUFF:
        ok &= run("ruff format --check", [RUFF, "format", "--check", "."], be)
        ok &= run("ruff check", [RUFF, "check", "."], be)
    elif ruff_configured(be, ROOT):
        results.append(("ruff", "FAIL (configured, but not in .venv or on PATH)"))
        ok = False
    else:
        skip("ruff", "not installed")
    has_tests = any(be.rglob("test_*.py")) or any(be.rglob("*_test.py")) or (be / "tests").exists()
    if not has_tests:
        skip("pytest", "no tests yet")
    else:
        ok &= run("pytest", [PY, "-m", "pytest", "-q"], be)
    return ok


def frontend():
    fe = ROOT / "frontend"
    pkg = fe / "package.json"
    if not pkg.exists():
        skip("frontend", "no package.json yet")
        return True
    scripts = json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {})
    checks = ("lint", "typecheck", "test")
    if not_set_up("frontend/node_modules") and any(s in scripts for s in checks):
        results.append(("frontend", f"FAIL (this worktree has no frontend/node_modules; {SETUP})"))
        return False
    npm = shutil.which("npm")
    if not npm:
        skip("frontend", "npm not installed")
        return True
    ok = True
    for s in checks:
        if s in scripts:
            args = [npm, "run", s]
            if s == "test" and "vitest" in scripts["test"]:
                args += ["--", "--run"]
            ok &= run(f"npm run {s}", args, fe)
        else:
            skip(f"npm run {s}", "script not defined")
    return ok


def main() -> int:
    ok = factory() & backend() & frontend()
    print("\n=== verify summary ===")
    for name, status in results:
        print(f"  {status:<28} {name}")
    print("=> " + ("ALL GOOD" if ok else "FAILURES — fix or register them before wrapping"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
