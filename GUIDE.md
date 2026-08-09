# App Factory — Owner's Guide (v2.1)

Your reference for what this thing is and how to run it. Written for you — Claude's instructions
live in `CLAUDE.md` and `.claude/rules/`.

---

## 1. What it is

A **template repo**: the standard skeleton for every new app. It bundles rules (how Claude
behaves), enforcement (hooks that make the important rules mechanical), and a project-management
system (living documents that make cold starts fast and autonomous work reviewable).

The core idea is **one dial**: the tier in `.claude/TIER.md`. In v2 the tier is the whole
ruleset — it sets **autonomy** (what Claude decides alone vs. brings to you) and **rigor**
(auth, tests, CI, infra, error handling) together:

| | Autonomy | Rigor | You use it for |
| --- | --- | --- | --- |
| **L1 — Build it** | Builds phase after phase without waiting; interviews you for design + features, decides the rest and logs it | Lean, anti-overbuild | Starting a passion project |
| **L2 — Get it ready** | Phase plans wait for you; real-reversal-cost decisions stop | Beta: auth, backups, CI, smoke tests | Refining an app that needs progress |
| **L3 — Don't break it** | Architecture, schema, deps, auth, infra, prod data all stop | Production everything | Deployed apps that matter |

What never changes with tier: you push and merge (Claude only commits), Terraform and browser
automation need per-instance permission, secrets never enter the repo, destructive data operations
always ask, and **science is always yours** — Claude proposes, never chooses.

Switching: run `/set-tier L2` — it walks the implications, and an upgrade proposes a hardening
phase for the roadmap. Under the hood it's one line in `.claude/TIER.md`, so it's trivially
reversible; the command exists so the implications get walked, not skipped.

## 2. Starting a new app

1. (Recommended) Ramble the idea to Claude in a regular chat → get a scope brief back.
2. Create the GitHub repo, clone into Cursor. Turn on **branch protection for `main`**
   (Settings → Branches: require a PR, no force-pushes) — that's the real wall behind the hooks.
3. Copy this template's full contents in — **including dotfiles** (`.claude/`, `.gitignore`).
   On Windows, copy from the terminal so nothing hidden gets missed:
   `xcopy path\to\app-factory path\to\new-repo /E /H` (or `cp -R app-factory/. new-repo/` in
   Git Bash). Nothing to chmod — the hooks are Python.
4. Commit — clean baseline.
5. In Claude Code: `/new-app` (+ paste the brief). It interviews, confirms tier, writes the four
   plans docs, seeds the register. Two sign-offs are yours at every tier: scope and design tokens.
6. At L1 it then starts building. At L2/L3 it presents Phase A and waits.

## 3. What's in the box

```
CLAUDE.md            Claude's constitution — short, always in context.
GUIDE.md             This file.
verify.py            One command: is the codebase healthy?  (python verify.py)
.claude/
  TIER.md            The dial: L1 / L2 / L3.
  settings.json      Permissions + hook wiring.
  hooks/             guard_commands.py (blocks push/merge/terraform incl. sneaky
                     variants — tested), format_on_write.py (auto ruff/prettier).
                     Pure Python: cross-platform, nothing to chmod.
  rules/             tier-L1/2/3 (the rulesets), project-management,
                     frontend-standards, backend-standards, science-integrity.
  commands/          /new-app /add-feature /log-decision /phase-report
                     /wrap-session /set-tier
  agents/ skills/    Still deliberately empty.
project-management/
  CURRENT.md         Live session snapshot (<20 lines).
  FEATURE_REGISTER.md  THE work table: every feature/fix/bug/idea, incl. "later";
                       checkoff = DONE + commit hash.
  DECISIONS.md       Top: decisions for YOU (options + recommendation).
                     Bottom: one-line log of decisions Claude made — the L1 audit trail.
  CHORES.md          Manual tasks only you can do, with exact steps.
  plans/             PLAN.md · ROADMAP.md · ARCHITECTURE.md (system + stack) · DATABASE.md
  reports/phases/    End-of-phase reports.
  reports/sessions/  End-of-session reports (incl. next-session prompt).
backend/ frontend/ data/ infra/
data/DATA_SOURCES.md One provenance entry per dataset (source, version, license, CRS…).
```

## 4. The commands

| Command | When | Notes |
| --- | --- | --- |
| `/new-app` | Once | Interview → tier → plans docs → seeded register |
| `/add-feature` | Every feature | Registers first, sizes to tier, then builds |
| `/log-decision` | Decision surfaces | Routes it: full entry for you, or one Decision Log line |
| `/phase-report` | Phase ends | Report + **codebase audit** + doc prune; L1 rolls on, L2/L3 stop |
| `/wrap-session` | **Every session end** | Register sync + session report + **next-session prompt** + CURRENT reset |
| `/set-tier` | Project changes life-stage | Walks implications; upgrades propose a hardening phase |

Two habits carry the system: **always `/wrap-session`**, and **answer your D-entries** — at L2/L3
they block work, and at L1 the Decision Log is your periodic read to keep autonomous choices honest.

## 5. verify.py — the health check

`python verify.py` runs every check that applies — backend formatting, lint, tests; frontend
lint, type-check, tests — and skips anything not set up yet, so it works from day one. Green exit
means healthy. Claude runs it before every session wrap and phase close; you can run it any time
you come back to a repo and want to know where it stands. It's also the seed of the v3 "factory
produces verified runnable apps" contract.

## 6. Branches and commits

Claude never works on `main`: every register item gets a branch (`feat/F-12-slug`,
`fix/F-31-slug`) and commits like `feat(F-12): add renewal calendar`. Wrap-session tells you which
branches are ready; **you merge** (PR or locally) and push. With branch protection on, this is
enforced by GitHub itself, not just by rules. Commit prefixes: feat · fix · refactor · docs ·
chore · test.

## 7. Reviewing autonomous work (mostly an L1 concern)

L1 trades approval-in-advance for review-after-the-fact. Your review surface, cheapest first:
the **Decision Log** (skim weekly), **session reports** (what happened while you weren't looking),
**phase reports** (the audit section), and `git log` when something smells off. If a logged
decision bothers you, say so — reversing it becomes a register row, and the pattern becomes a rule.

## 8. Session rhythm

**Start:** paste the prompt from the last session report; say today's goal in a sentence.
**During:** one register item WIP at a time; new ideas → "log it" (a row costs nothing, chasing it
mid-task costs the session); answer D-entries as they appear.
**End:** `/wrap-session`, review commits, push, clear your chores.

**Smells:**

| Smell | Meaning |
| --- | --- |
| CURRENT.md > 20 lines | Detail belongs in the register |
| Work happening with no register row | The "discussed = registered" rule is slipping |
| DONE rows with empty Commit column | Checkoff discipline slipping |
| DATABASE.md ≠ actual schema | The task that changed it wasn't finished |
| New docs (NOTES.md, TODO.md…) | System bypass — fold them in |
| Hex/px values outside tokens.css | Token system eroding |
| L1 build "needs" caching/RBAC/queues | Tier mismatch — `/set-tier` conversation |

## 9. Maintaining the factory

Copies diverge per project; port proven improvements back by hand. Keep the template
project-agnostic. `agents/` and `skills/` stay empty until a real recurring need names itself —
candidates remain frontend-ui, science-reviewer, janitor. If hand-syncing across many repos ever
genuinely hurts, the upgrade path is packaging `.claude/` as a Claude Code plugin.

## 10. Migrating an existing app into the factory (Beacon, then SWPT, then LCA Pave)

1. Copy in `.claude/`, `CLAUDE.md`, `GUIDE.md`, `verify.py`, and the `project-management/`
   skeleton. Don't touch app code.
2. Set the tier honestly (`.claude/TIER.md`) — Beacon L1; SWPT/LCA Pave likely L2 until proven L3.
3. First Claude Code session is a **migration session, not a build session**: have Claude read the
   codebase and backfill the four plans docs *as built* (ARCHITECTURE and DATABASE especially),
   convert any existing todo lists into `FEATURE_REGISTER.md` rows, log open questions to
   `DECISIONS.md`, run `python verify.py` and register what's red.
4. Turn on branch protection for `main`.
5. From then on it's a normal factory repo. For LCA Pave: collaborators make branch protection
   and PR review non-optional — do that migration only after SWPT has proven the flow.
