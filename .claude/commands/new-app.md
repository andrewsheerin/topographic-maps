---
description: Bootstrap a new app from the factory template — interview, tier, plans, register
---

Brand-new project from the app-factory template. Work in order; no application code during this
command. The interview happens at EVERY tier — autonomy starts after it, not instead of it.

**1. Interview me.** What I'm building, who it's for, what it must do on day one, the feel I want.
One batch of questions, not a drip. If I pasted a scope brief, mine it first and ask only gaps.

**2. Confirm the tier.** Recommend L1/L2/L3 with one line of reasoning; wait for my confirmation.
Write it to `.claude/TIER.md` and load that tier's rules — they govern everything after this.

**3. Draft the scope → `plans/PLAN.md`.** Purpose, in-scope, **explicitly out of scope** (be
aggressive), design direction, domain assumptions to get from me. Sign-off gate at every tier.

**4. Propose design tokens.** Palette (4–6 values), type pairing, spacing scale, justified against
what the app is. Sign-off gate at every tier — design is always mine.

**5. Write the plans.** `ROADMAP.md` (phases with goals + done-conditions; Phase A ends at
something runnable), `ARCHITECTURE.md` (stack + system shape), `DATABASE.md` (initial schema).
Seed `FEATURE_REGISTER.md` from the scope with IDs, phases, statuses. Log open items to
`DECISIONS.md` and `CHORES.md`.

**6. Initialize.** Fill the Project block in `CLAUDE.md`, write `CURRENT.md`, commit.

**7. Then:** at L1 — announce Phase A in `CURRENT.md` and start building. At L2/L3 — present the
Phase A plan and stop for sign-off.
