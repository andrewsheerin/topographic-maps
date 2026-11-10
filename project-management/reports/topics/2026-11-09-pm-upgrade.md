# v3 PM upgrade — topographic-maps

Run 2026-11-09 from 20:01, interactively with Andrew, following
`app-factory/template/.claude/commands/pm-upgrade.md` under the dispatch
`Vault/60 Reference/Dispatches/2026-11-09 topo-v3-pilot.md`. Steps 0–5 on the short branch
`chore/pm-upgrade` off main `4fcf6d2`; step 6 on main.

## Checklist

- [x] 0 preconditions   - [x] 3 settings confirmed   - [ ] 6 private mode (or n/a)
- [x] 1 layout found    - [x] 4 .claude installed    - [ ] 7 verified, reported, pushed
- [x] 2 code deps       - [x] 5 content mapped

## 0. Preconditions

- A git repository with a remote (`origin` = github.com/andrewsheerin/topographic-maps, public).
- Working tree clean; no other worktrees, no stashes; Andrew present, no other session.
- Before this, the dispatch's git step: `main` fast-forwarded `7e7fb3d → 4fcf6d2` (the 34-commit
  F-21…F-37 stack) and 29 merged branches deleted with `git branch -d`.
- Branch: `chore/pm-upgrade` (git policy `main`, multi-step work).

## 1. Layout found: v2.1

All four v2.1 files, a `Commit(s)` column, no CURRENT frontmatter.

| File | Lines | IDs | Status words | Notes |
| --- | --- | --- | --- | --- |
| `CURRENT.md` | 24 | — | — | No frontmatter; 4 over the limit |
| `FEATURE_REGISTER.md` | 63 | F-1…F-37 | IDEA, PLANNED, DONE | `Commit(s)` column; F-24…F-37 separated by blank lines, which breaks the table; Type `test` on F-5; empty `## Dropped` table |
| `DECISIONS.md` | 108 | D-1, D-2; two unnumbered (`—`) | OPEN | Resolved table `ID · Decision · Chose · Date`; Decision Log 41 rows |
| `CHORES.md` | 30 | C-1, C-2 | TODO | No Due column; `### C-n` detail blocks |
| `plans/PLAN.md` | 53 | — | — | |
| `plans/ROADMAP.md` | 29 | — | — | |
| `plans/ARCHITECTURE.md` | 50 | — | — | |
| `plans/DATABASE.md` | 31 | — | — | No database |
| `reports/sessions/2026-08-09-2148.md` | 78 | — | — | Ends `## Opening prompt for next session`, not `## Next session prompt` |
| `reports/phases/` | — | — | — | Empty |

Outside `project-management/`: a root `GUIDE.md` (the app-factory v2.1 owner's guide, 151 lines).

**pm_check before the upgrade:** 24 errors, 15 warnings.

| Code | Count | What |
| --- | --- | --- |
| `table` | 14 | register rows F-24…F-37 outside the table (blank lines between rows) |
| `tier` | 2 | no `Git policy:` / `PM mode:` line in TIER.md |
| `id-style` + `status` | 4 | the two unnumbered Resolved rows |
| `header` | 2 | register has no Ref column; chores have no Due column |
| `frontmatter`, `current-length` | 2 | CURRENT.md |
| warnings | 15 | no next-session prompt; main folder off main; 13 merged branches (all cleared by the git step before this) |

## 2. What code depends on: nothing frozen

Searched every non-PM file (excluding `.venv/`, `node_modules/`) for `project-management`, the PM
file names and `F-n`/`D-n`/`C-n`. No code reads or writes a PM file, so nothing is frozen and
TIER.md needs no `pm_check ignore:` line. Mentions only, in comments and docs:

- `backend/constants.py:6`, `backend/tests/test_carve_order.py:7` (cite `DECISIONS.md`, D-2, F-2),
  `frontend/src/lib/geometry.js:10` (cites DECISIONS). Under private mode these point at files a
  clone of the public repo won't have; left as they are (no app code changes).
- `F-n` IDs in code comments and test docstrings across `backend/` and `frontend/`: IDs are kept,
  so they stay valid.
- `README.md` (lines 80, 81, 84) and `CLAUDE.md`: handled in step 4.

## 3. Settings, confirmed by Andrew 2026-11-09 (one batched question)

| Setting | Value | Basis |
| --- | --- | --- |
| Profile | app | STANDARD.md repo table; the template is the app profile (no overlay file) |
| Git policy | `main` | STANDARD.md repo table |
| PM mode | `private` | STANDARD.md; the GitHub repo is public. Set up in step 6 — TIER.md says `committed` until then (see step 4) |
| Deploys on push | no | `infra/` holds only `.gitkeep`; no `.github/`, Dockerfile or deploy config in the repo |
| Vault note | `C:/Users/flow/Vault/20 Projects/Topographic Maps.md` | exists |
| Tier | L1, unchanged | |

## 4. The v3 `.claude`, installed

Every `.claude` file in the repo was an unmodified copy of the v2.1 template (diffed ignoring
line endings), so the v3 versions replace them; nothing repo-specific needed merging.

- **Copied from the template:** `hooks/` (guard, guard tests, formatter, `session_start.py`),
  all seven `commands/` (adds `pm-upgrade.md`), `skills/cleanup/` (SKILL.md, pm_check.py), the
  rules `git.md`, `project-management.md`, `testing.md`, `security.md`, and the template's
  `backend-standards.md`, `frontend-standards.md`, `science-integrity.md` (identical to the
  repo's apart from the new `paths:` frontmatter). `skills/.gitkeep` removed.
- **Tiers:** `rules/tier-L{1,2,3}.md` moved with `git mv` to `tiers/L{1,2,3}.md`, then given the
  v3 text.
- **`settings.json`, merged:** dropped the `git push:*` and `git merge:*` denies (the `main`
  policy contradicts them); kept `git rebase:*` and the `rm -rf` / `docker` asks; the repo's
  `terraform …:*` denies folded into the template's broader `terraform …*` ones; added the
  template's force-push, history-rewrite, `gh pr merge` and `.env`/tfstate read denies. Hooks
  switched from absolute paths (F-12's fix for a cwd-relative path that wedged a session) to
  `$CLAUDE_PROJECT_DIR`, which solves the same problem and also works from a worktree. The
  SessionStart hook no longer prints TIER.md: CLAUDE.md imports it now.
- **`verify.py`:** the template's (adds pm_check and the hook tests). One change kept beyond the
  template: ruff is taken from `.venv` like Python is. The template looks ruff up only on PATH,
  and ruff here exists only in `.venv/Scripts`, so it was silently skipped — which is how
  F-38's reds went unnoticed.
- **`TIER.md`:** v3 shape: L1, `Git policy: main`, `Deploys on push: no`, `@tiers/L1.md`, no
  exceptions (there were none). **`PM mode: committed` for now:** pm_check errors on `private`
  until the nested repo exists, so writing the final value here would keep verify red through
  step 5. Step 6 flips it.
- **`CLAUDE.md`:** the template's shape, 75 lines, Project block filled (name and purpose kept
  verbatim), vault-note and "Not a Tawa repo" lines. Nothing project-specific was left over, so
  no `.claude/rules/project.md`. Old rules replaced:
  - "Feature branch per register item, never directly on main" → `main` policy.
  - "Never git push, git merge, git rebase — the user does all pushing and merging (enforced by
    hook)" → Claude merges locally and pushes main after verify.
  - "Keep CURRENT.md updated as you go" → lean by default; CURRENT is rewritten at wrap.
  - The per-tier decision-router thresholds → the Decisions section plus `tiers/L1.md`.
- **`.gitignore`:** added `.claude/worktrees/` (the worktree test needs it; the template has it).
- **`.gitattributes`:** left alone. It exists (LFS rules for shapefiles, rasters, STL, ZIP; no
  LFS files are tracked), so "add if missing" doesn't apply. The index is LF throughout
  (`core.autocrlf=true`), so `eol=lf` would renormalize nothing.
- **Profile overlay:** none; the template is the app profile.
- **Checked:** guard tests 90/90; the guard allows `merge` and `push origin main` under this
  TIER.md and denies force-push.

## 5. Content mapped

**ID map** (old → new, in old order):

| Old | New | Notes |
| --- | --- | --- |
| F-1…F-37 | unchanged | Rows byte-for-byte; `Commit(s)` renamed `Ref`; blank lines between rows removed |
| — | F-38 (PLANNED) | New: the ruff reds found in step 4 |
| D-1, D-2 (OPEN) | unchanged, OPEN | Status and Raised joined onto one line; text untouched |
| `—` Frontend framework (Resolved) | D-3, RESOLVED | Original row → `archive/DECISIONS-answered.md` |
| `—` Migration scope (Resolved) | D-4, RESOLVED | Same |
| CURRENT "Blocked on: SRTM/NASADEM labels" | D-5, PROPOSED | From the 2026-09-03 Decision Log line; the log line stays as history |
| CURRENT "Blocked on: email in Overpass UA?" | D-6, PROPOSED | Same |
| C-1, C-2 (TODO) | unchanged, TODO | Due column added, blank. Closed by /cleanup, per the dispatch |

**Moved to `archive/`:**

- `GUIDE-v2.1.md` ← root `GUIDE.md` (`git mv`, header added): the v2.1 factory guide. v3 ships no
  GUIDE and its rules (Andrew does all merging; branch protection as "the real wall") contradict
  the `main` policy. README's two pointers to it now point at CLAUDE.md.
- `DECISIONS-answered.md`: the two unnumbered Resolved rows, as their full text.

**Rewritten:** `CURRENT.md`, with frontmatter; body 15 lines. Its old body was a snapshot, not a
log, so no session report was needed for its history: the F-36 summary is in the register, the
branch-stack note is obsolete (merged), and every "Blocked on: yours" item survives (D-1, D-2,
D-5, D-6, the Killington print check; C-1 is in CHORES).

**Small fixes:** `plans/PLAN.md` pointed at `.claude/rules/tier-L1.md` → `.claude/tiers/L1.md`.
Empty v2.1 `## Dropped` table removed (v3 uses the DROPPED status in the one table).

**Left as found:**

- `reports/sessions/2026-08-09-2148.md` ends `## Opening prompt for next session`. Reports are
  history, so it isn't edited; the warning clears once a newer report exists.
- F-5 has Type `test`, outside v3's five types; pm_check doesn't check Type.
- Other "confirm"/"flag" asks inside Decision Log and register Notes (F-17's 2 mm base default,
  F-11's county subdivisions, the equirectangular square) were not promoted to D-n: only the two
  CURRENT.md carried as open asks were. Candidates if Andrew wants them tracked.
- `plans/` content (PLAN's in-scope list predates water, 3MF and F-18's buffer removal) — a
  content refresh, not a migration step.

**pm_check after step 5:** 0 errors, 3 warnings (the old report's prompt heading; main folder on
`chore/pm-upgrade`; main 34 ahead of origin).
