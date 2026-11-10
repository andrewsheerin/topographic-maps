---
description: Close out a phase — report, audit, prune, propose the next one
---

Close the current phase.

1. **Write `project-management/reports/phases/phase-<X>.md`:** goal vs outcome; what was built
   (by F-n, briefly — git has the detail); decisions made in both lanes and what they closed
   off; what was deferred and to where; **what you'd do differently, honestly**; open risks.
2. **Audit before claiming done:** `python verify.py` (red = the phase can't close); then dead
   code, duplicated components, off-token CSS, files past the review-trigger sizes, TODO
   comments, anything built above the tier. Fix trivial finds, register the rest, and report
   what you found either way.
3. **Prune the living docs:** register statuses and Refs current; PLAN.md's Roadmap actual vs
   planned, its Architecture and Data model still true; answered decisions archived.
4. **Next phase:** L1 — announce it in `CURRENT.md` and continue. L2/L3 — present the plan
   and stop.
