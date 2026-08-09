---
description: Add a feature — registered, scoped to tier, then built
---

I want to add a feature. Before code:

**1. Restate it** in one sentence. Ask now if ambiguous — features are always my call, every tier.

**2. Register it.** Row in `FEATURE_REGISTER.md` (next F-n, type, phase, PLANNED). If it belongs
in a later phase, set Phase = later, tell me, and ask whether to pull it forward.

**3. Blast radius.** Which files change; whether it touches schema, tokens, architecture, or
domain logic — and route those per the current tier's decision rules (L1: mostly decide-and-log;
L2: reversibility test; L3: stop).

**4. Propose the smallest version that delivers the value**, and name what you're leaving out.
Check leftovers against the tier — don't build above it.

**5. Build** (per tier: L1 proceeds; L2/L3 wait if a gate applies) — on a fresh branch:
`<type>/F-n-short-slug` off `main`. One primary item WIP, `CURRENT.md`
live, commits at logical points. Second problem discovered → new register row, keep moving.

**Done means:** status DONE + commit hash(es) in the register, `DATABASE.md` updated if schema
moved, no dead code, no off-token CSS, no new dependency that skipped its gate.
