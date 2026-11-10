---
paths:
  - "**/tests/**"
  - "**/test_*.py"
  - "**/*_test.py"
  - "**/*.test.*"
  - "verify.py"
---

# Testing and verification

- `python verify.py` is the one health check: code (ruff, pytest, npm lint/typecheck/test) and
  PM (`pm_check.py`). It skips whatever isn't set up yet. Green is the bar for a wrap, a push
  under the `main` policy, and a PR under `pr`.
- Use the project's own interpreter. If the repo has a `.venv`, run
  `.venv/Scripts/python.exe -m pytest` (Windows) rather than a bare `python`, which may be a
  different install missing the dependencies. A `ModuleNotFoundError` for an installed package
  usually means the wrong interpreter.
- What the tier requires: calculation and data-transformation logic gets tests at every tier,
  against a worked example the user has checked (science-integrity). L2 adds happy-path API
  tests and a boot smoke test; L3 adds integration and E2E on critical paths. Every bug fix at
  L2+ ships with a regression test.
- Passing tests are necessary, not sufficient. For a user-visible change, exercise the flow too.
