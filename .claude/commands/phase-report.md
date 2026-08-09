---
description: Close out a phase — report, audit, prune, propose next
---

Close the current phase.

**1. Write `reports/phases/phase-<X>.md`:** goal vs outcome; what was built (by F-n, brief — git
has the detail); decisions made (both lanes) and what they closed off; what was deferred and to
where; **what I'd do differently, honestly**; open risks.

**2. Audit before claiming done:** run `python verify.py` (red = phase isn't closable); then dead code, duplicated components, off-token CSS values,
files past the review-trigger sizes, TODO comments, anything built above the tier. Fix trivial finds; register the
rest. Report what you found either way.

**3. Prune the living docs:** register statuses + commits current; `ROADMAP.md` actual vs planned;
`ARCHITECTURE.md` / `DATABASE.md` still telling the truth.

**4. Next phase:** L1 — announce it and continue. L2/L3 — present the plan and stop.
