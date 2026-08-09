---
description: Switch the project's tier (L1/L2/L3) with a proper walkthrough
---

I want to change the tier. Target: $ARGUMENTS (ask if not given).

**1. Read** `.claude/TIER.md` (current) and both tier rule files. Summarize what actually changes
for THIS project in two short lists: autonomy differences, rigor differences.

**2. If upgrading** (L1→L2, L2→L3): audit the gap — auth, secrets handling, tests, CI, migrations,
docs — against the target tier's table. Propose a **hardening phase** for `ROADMAP.md` with the
gap items as register rows. The upgrade isn't done when the file says L2; it's done when that
phase closes.

**3. If downgrading:** confirm intent explicitly. L3 → anything for a deployed app with real
users: challenge me on it — deployed users don't care what our file says.

**4. On my confirmation:** update `.claude/TIER.md`, load the new tier rules, note the switch in
the Decision Log with the reason, update `CURRENT.md`, commit.
