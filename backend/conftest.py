"""Ensure backend/ is importable as the app root during tests, so tests can do
`from core.mesh import ...` regardless of the working directory pytest picks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
