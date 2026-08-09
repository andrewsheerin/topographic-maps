#!/usr/bin/env python3
"""One command to answer "is the codebase healthy?" — run: python verify.py

Runs every check that applies and skips (without failing) anything not installed
or not present yet. Backend: ruff format --check, ruff check, pytest.
Frontend: npm run lint / typecheck / test when package.json defines them.
Exit 0 = all run checks passed. Non-zero = something failed (fix before wrapping).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
results = []


def run(name, args, cwd):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=600)
        ok = p.returncode == 0
        results.append((name, "PASS" if ok else "FAIL"))
        if not ok:
            tail = (p.stdout + p.stderr).strip().splitlines()[-15:]
            print(f"\n--- {name} output (tail) ---")
            print("\n".join(tail))
        return ok
    except Exception as e:
        results.append((name, f"ERROR ({e})"))
        return False


def skip(name, why):
    results.append((name, f"SKIP ({why})"))


def backend():
    be = ROOT / "backend"
    if not any(be.rglob("*.py")) if be.exists() else True:
        skip("backend", "no python files yet")
        return True
    ok = True
    if shutil.which("ruff"):
        ok &= run("ruff format --check", ["ruff", "format", "--check", "."], be)
        ok &= run("ruff check", ["ruff", "check", "."], be)
    else:
        skip("ruff", "not installed")
    has_tests = any(be.rglob("test_*.py")) or any(be.rglob("*_test.py")) or (be / "tests").exists()
    if not has_tests:
        skip("pytest", "no tests yet")
    elif shutil.which("pytest") or shutil.which("python"):
        ok &= run("pytest", [sys.executable, "-m", "pytest", "-q"], be)
    return ok


def frontend():
    fe = ROOT / "frontend"
    pkg = fe / "package.json"
    if not pkg.exists():
        skip("frontend", "no package.json yet")
        return True
    npm = shutil.which("npm")
    if not npm:
        skip("frontend", "npm not installed")
        return True
    scripts = json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {})
    ok = True
    for s in ("lint", "typecheck", "test"):
        if s in scripts:
            args = [npm, "run", s]
            if s == "test":
                args += ["--", "--run"] if "vitest" in scripts.get("test", "") else []
            ok &= run(f"npm run {s}", args, fe)
        else:
            skip(f"npm run {s}", "script not defined")
    return ok


def main() -> int:
    ok = backend() & frontend()
    print("\n=== verify summary ===")
    for name, status in results:
        print(f"  {status:<28} {name}")
    print("=> " + ("ALL GOOD" if ok else "FAILURES — fix before wrapping the session"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
