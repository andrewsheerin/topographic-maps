---
description: Add a feature — registered, scoped to the tier, then built on the right branch
argument-hint: <one-line description of the feature>
---

$ARGUMENTS is the feature (or take it from the conversation).

1. **Restate it in one sentence.** Features are always the user's call, at every tier; ask now
   if it's ambiguous.
2. **Register it:** the next `F-n` row in `FEATURES.md` (Item, Type, Phase, PLANNED).
   If it belongs later, Phase `later`, and ask whether to pull it forward.
3. **Blast radius:** which files change, and whether it touches schema, design tokens,
   architecture, a dependency, auth or domain logic. Route those per the tier (L1 mostly
   decide-and-log; L2 the reversibility test; L3 stop).
4. **Propose the smallest version that delivers the value**, and name what's left out. Nothing
   above the tier.
5. **Build** (L1 proceeds; L2/L3 wait if a gate applies). Set the row to WIP; one WIP at a time.
   Where to work, per `.claude/rules/git.md`:
   - `main` policy: on main, or a short `feat/F-n-slug` branch if risky or multi-session.
   - `pr` policy: a fresh `feat/F-n-slug` branch from an up-to-date main.
   - Something else already running in this repo: a worktree under `.claude/worktrees/`.
   Commit at logical points. A second problem found on the way → a new row; keep going.
6. **Ship it** per the policy: `main` → merge any branch, verify, push main. `pr` → push the
   branch, open the PR (body: what changed, F-n, verify summary), main folder back on main.

**Done means:** DONE with `Ref` filled (hash, or `#PR` under `pr`), PLAN.md's Data model updated
if the schema moved, no dead code, no off-token CSS, no dependency that skipped its gate, and
`python verify.py` green (or its reds registered).
