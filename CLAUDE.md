# CLAUDE.md

> Always in context, so it stays under ~80 lines. Detail lives in `.claude/rules/` (each file
> loads only when you touch the files it covers) and in the slash commands.

## Project

- **Name:** TOPO2STL (topographic-maps)
- **Purpose:** Draw or upload an area, pull a DEM (+ optional OSM roads), and export a
  3D-printable terrain STL. Runs locally for the author.
- **Profile:** app · **Stack:** React (frontend) + FastAPI (backend)
- **Vault note:** C:/Users/flow/Vault/20 Projects/Topographic Maps.md
- Not a Tawa repo: Tawa workflow rules (issues, self-merge, releases) don't apply here.

@.claude/TIER.md

## Lean by default

- **Trivial prompts** (questions, lookups, explanations, one-line fixes): just answer or fix.
  Touch no PM file, run no verify, write no report.
- **During work**, capture only what would otherwise be lost: one register row (IDEA is enough)
  or a `D-n` stub (the question, status OPEN). Don't rewrite CURRENT.md or tidy docs mid-task.
- **`/wrap-session`** is the routine bookkeeping point, a couple of minutes. **`/cleanup`** is
  the deep pass, run when asked.
- Open a rules file or plan doc only when the task needs it.

## Git — the policy is in TIER.md; the full guide is `.claude/rules/git.md`

- **`main` policy:** commit on main; push main once `python verify.py` passes. Branch only for
  risky or multi-session work, merge it locally, then push.
- **`pr` policy:** one branch per register item (`feat/F-12-slug`, `fix/F-31-slug`). Push it and
  open a PR whose body carries the verify results. The user merges, because merging deploys.
  Never push to main, never merge a PR. Afterwards, put the main folder back on main.
- **`Deploys on push: yes`** (a live site): never push main; commit, then hand the user the push.
- **Always:** no force-push, no rewriting pushed history, no deleting unmerged work.
  Commits read `<type>(F-n): summary`, type ∈ feat · fix · refactor · docs · chore · test.
- Two things at once in this repo → a worktree under `.claude/worktrees/`, removed once merged.

`.claude/hooks/guard_commands.py` enforces the hard parts.

## Non-negotiables (every tier)

1. **Terraform:** never `apply`, `destroy` or `import` without explicit per-instance permission.
2. **Browser automation** only with explicit permission.
3. **No secrets in the repo.** Ever.
4. **Science.** Units, factors, formulas, defaults and statistical treatment belong to the user
   (PhD, environmental engineering). Propose and recommend; never choose.
   See `.claude/rules/science-integrity.md`.
5. **Destructive data operations** (dropping tables, deleting user data, irreversible
   migrations) need explicit permission.
6. If a permission you lack would make the work meaningfully faster or better, say so and ask.

## Decisions — the tier sets which lane

- **Decide and log:** one line in the Decision Log at the bottom of `DECISIONS.md`.
- **Stop and ask:** a `D-n` under Needs you, status OPEN, with options and a recommendation.
  Carry on with whatever doesn't depend on it.
- **Built on a provisional choice that's the user's to make:** a `D-n` with status PROPOSED.

## Code posture

Minimal, not clever. Prefer deleting to adding. No dead code, no speculative layers. A fix that
needs a workaround says so, and the real problem gets a register row.

## Where things live

| Need | File |
| --- | --- |
| Where things stand (printed at session start) | `project-management/CURRENT.md` |
| Every feature, fix, bug and idea (the register) | `project-management/FEATURES.md` |
| Decisions: the user's, resolved, and Claude's log | `project-management/DECISIONS.md` |
| Chores only the user can do | `project-management/CHORES.md` |
| Scope, roadmap, architecture, data model | `project-management/PLAN.md` |
| PM rules, git guide, code standards | `.claude/rules/` |
| Health check, code and PM | `python verify.py` |
