# Current

**Phase:** A — Factory migration & as-built baseline
**Register item:** F-1 (WIP)
**Status:** Migration complete; verifying + committing

## Right now
App-factory installed at L1. Backend restructured into `backend/` (api/core/models/constants/config/
tests); frontend rewritten as React + Vite. Backend verify (ruff format+check, pytest) is green.

## Blocked on
Nothing. Open decisions D-1 (road-width defaults) and D-2 (carve overlap rule) are queued for
Phase B, not blocking. Chore C-1 (rotate leaked API key) is the user's.

## Next
Run full `python verify.py` (incl. frontend build/lint), smoke-test the server serves the built
app, commit F-1, then mark F-1 DONE with the hash.

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
