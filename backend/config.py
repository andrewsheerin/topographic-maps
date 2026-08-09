"""Runtime configuration and secret loading.

The OpenTopography API key is the only secret. It is loaded from the environment
(populated from a gitignored `.env` at the repo root via python-dotenv), with a
legacy file fallback. No secret ever lives in the repo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = parent of the backend/ package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the repo root (no-op if the file is absent). Real environment
# variables already set take precedence over .env values.
load_dotenv(REPO_ROOT / ".env")

# Preferred env var name, plus a common alternative spelling.
_ENV_KEYS = ("OPEN_TOPO_API_KEY", "OPENTOPOGRAPHY_API_KEY")


def _read_text_file(path: Path) -> str:
    try:
        return (path.read_text(encoding="utf-8") or "").strip()
    except OSError:
        return ""


def get_opentopo_api_key() -> str:
    """Return the OpenTopography API key, or "" if not configured.

    Resolution order (first non-empty wins):
      1. env OPEN_TOPO_API_KEY
      2. env OPENTOPOGRAPHY_API_KEY
      3. file <repo-root>/API_KEY.txt   (legacy, gitignored)
    """
    for name in _ENV_KEYS:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return _read_text_file(REPO_ROOT / "API_KEY.txt")


def require_opentopo_api_key() -> str:
    key = get_opentopo_api_key()
    if not key:
        raise RuntimeError(
            "Missing OpenTopography API key. Set OPEN_TOPO_API_KEY in a .env file "
            "at the repo root (see .env.example), or export it in your shell."
        )
    return key
