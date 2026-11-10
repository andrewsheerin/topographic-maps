---
description: Bootstrap a new repo from the app-factory template — interview, profile and policies, plans, seeded register
---

A brand-new repo made from `app-factory/template/`. Work in order; no application code during
this command. The interview happens at every tier: autonomy starts after it, not instead of it.

**1. Interview the user**, in one batch (if they pasted a scope brief, mine it and ask only the
gaps):
- what this is, who it's for, what it must do on day one, the feel they want;
- **profile**: app (default), paper, engagement (a client's code), partner (a partner's repo) or
  site (a public website) — see `C:/Users/flow/CursorProjects/app-factory/profiles/`;
- **git policy**: `main` (default) or `pr` (when merging deploys); **Deploys on push** yes/no;
- **PM mode**: `committed` (default) or `private` (when others can see the repo);
- the path of this repo's **vault note**, or none.

**2. Confirm the tier.** Recommend L1/L2/L3 with one line of reasoning; wait for the answer.

**3. Write the settings:** `.claude/TIER.md` (`TIER:`, the `@tiers/Ln.md` import, `Git policy:`,
`PM mode:`, `Deploys on push:`); the Project block in `CLAUDE.md` (name, purpose, profile, vault
note); `CURRENT.md`'s frontmatter to match. Apply the profile overlay (app has none: the
template is the app profile), then run
`python C:/Users/flow/CursorProjects/app-factory/tools/sync.py . --apply`, which records the
files it dropped in `.claude/factory.json`. Private mode: set it up as in `/pm-upgrade` step 6.
The tier's rules govern everything after this.

**4. Draft the scope → `PLAN.md`, Purpose and scope:** purpose, in scope, **explicitly out of
scope** (be aggressive), design direction, domain assumptions to get from the user. Sign-off
gate at every tier.

**5. Design tokens** (app and site profiles): palette (4–6 values), type pairing, spacing scale,
justified against what the app is. Sign-off gate at every tier: design is always the user's.

**6. Fill in the rest of `PLAN.md`:** Roadmap (phases with goals and done-conditions; Phase A
ends at something runnable), Architecture (stack and system shape), Data model (initial schema;
no database, no section). Seed `FEATURES.md` from the scope. Open questions → `DECISIONS.md`;
things only the user can do (repo settings, accounts) → `CHORES.md`.

**7. Commit** once `python verify.py` is green: under `main`, on main, then push; under `pr`, on
`chore/new-app`, pushed with a PR. A repo with no remote yet: commit locally and add a chore for
the user to create it.

**8. Then:** at L1, announce Phase A in `CURRENT.md` and start building. At L2/L3, present the
Phase A plan and stop for sign-off.
