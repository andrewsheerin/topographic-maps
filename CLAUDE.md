# CLAUDE.md

> In context every turn. Keep it under ~120 lines. Detail lives in `.claude/rules/`.

## Project

- **Name:** TOPO2STL (topographic-maps)
- **One-line purpose:** Draw or upload an area, pull a DEM (+ optional OSM roads), and export a 3D-printable
  terrain STL. Runs locally for the author.
- **Tier:** see `.claude/TIER.md` — **read it at session start and load the matching
  `.claude/rules/tier-*.md`. The tier is the ruleset. It sets both your autonomy and the
  engineering rigor.** Switch tiers only via `/set-tier`.
- **Stack:** React (frontend) + FastAPI (backend). Deviations follow the tier's decision rules.

## Non-negotiables (every tier, no exceptions)

1. **Git.** Work on a **feature branch per register item** (`feat/F-12-short-slug`,
   `fix/F-31-…`), never directly on `main`. `git add`, `git commit`, and branch creation freely;
   **never** `git push`, `git merge`, `git rebase`, or force anything — the user does all pushing
   and merging. (Enforced by hook.) Commits: `<type>(F-n): summary` with type ∈ feat · fix ·
   refactor · docs · chore · test; drop the `(F-n)` only when no register item applies.
2. **Terraform.** Never `apply`, `destroy`, or `import` without explicit per-instance permission.
3. **Browser automation** only with explicit permission.
4. **No secrets in the repo.** Ever.
5. **Science.** Every scientific or domain assumption — units, factors, formulas, defaults,
   statistical treatment — belongs to the user (PhD, environmental engineering). Propose and
   recommend; never choose. See `.claude/rules/science-integrity.md`.
6. **Destructive data operations** (dropping tables, deleting user data, irreversible migrations)
   always need explicit permission.
7. If a permission you lack would make the work meaningfully faster or better, **say so and ask** —
   don't silently work around it.

## The decision router

Every non-trivial decision gets routed, and **the tier sets the threshold** for which lane it takes:

- **Decide, log, keep moving** → one line in the Decision Log at the bottom of
  `project-management/DECISIONS.md`.
- **Stop and ask** → full entry (options + recommendation) in `DECISIONS.md`, then continue on
  whatever doesn't depend on it.

Where the threshold sits per tier — L1: almost everything is decide-and-log; stop only for the
non-negotiables above, design direction, and adding/cutting features. L2: the reversibility test —
anything with real reversal cost (architecture, dependencies, schema) stops. L3: architecture,
dependencies, schema, auth, infra, and anything touching production data all stop. The tier file
is the authority.

**Phase boundaries follow the same pattern** — L1: announce the phase plan and proceed. L2/L3:
present it and wait.

## Session protocol

1. Read `project-management/CURRENT.md`. It says where things stand.
2. Work. Keep `CURRENT.md` updated as you go — scratchpad, not report.
3. Route decisions per the tier. Log chores to `CHORES.md` and feature ideas to
   `FEATURE_REGISTER.md` **the moment they come up**, then return to the task.
4. Before wrapping, run `python verify.py` and fix failures.
5. End with `/wrap-session` → session report + reset `CURRENT.md`.

## Code posture

Minimal, not clever. The enemy is sprawl: defensive layers, speculative abstractions,
workaround-on-workaround. Prefer deleting to adding. No dead code or commented-out blocks left
behind. If a fix requires a workaround, say so and log the underlying issue in the register.

## Where things live

| Need | File |
| --- | --- |
| Ruleset for this project | `.claude/TIER.md` → `.claude/rules/tier-*.md` |
| PM system rules | `.claude/rules/project-management.md` |
| Status right now | `project-management/CURRENT.md` |
| All features/fixes/bugs + status + commit | `project-management/FEATURE_REGISTER.md` |
| Decisions (user's + the log) | `project-management/DECISIONS.md` |
| User's manual chores | `project-management/CHORES.md` |
| Plan / roadmap / architecture / database | `project-management/plans/` |
| Frontend, backend, science conventions | `.claude/rules/*.md` |

Read a rules file when the work touches it. Don't preload all of them.
