#!/usr/bin/env python3
"""PostToolUse formatter for Edit|Write. Cross-platform. Never blocks (always exit 0);
silently no-ops when a formatter isn't installed."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

PRETTIER_EXTS = {".ts", ".tsx", ".js", ".jsx", ".css", ".json", ".md"}


def run(args):
    try:
        subprocess.run(args, capture_output=True, timeout=25)
    except Exception:
        pass


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    fp = (data.get("tool_input") or {}).get("file_path") or ""
    p = Path(fp)
    if not fp or not p.is_file():
        return 0
    ext = p.suffix.lower()
    if ext == ".py" and shutil.which("ruff"):
        run(["ruff", "format", str(p)])
        run(["ruff", "check", "--fix", str(p)])
    elif ext in PRETTIER_EXTS:
        if shutil.which("prettier"):
            run(["prettier", "--write", str(p)])
        elif shutil.which("npx"):
            run(["npx", "--no-install", "prettier", "--write", str(p)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
