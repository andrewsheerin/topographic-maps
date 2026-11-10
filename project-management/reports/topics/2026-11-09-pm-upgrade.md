# v3 PM upgrade — topographic-maps

Run 2026-11-09 from 20:01, interactively with Andrew, following
`app-factory/template/.claude/commands/pm-upgrade.md` under the dispatch
`Vault/60 Reference/Dispatches/2026-11-09 topo-v3-pilot.md`. Steps 0–5 on the short branch
`chore/pm-upgrade` off main `4fcf6d2`; step 6 on main.

## Checklist

- [x] 0 preconditions   - [ ] 3 settings confirmed   - [ ] 6 private mode (or n/a)
- [x] 1 layout found    - [ ] 4 .claude installed    - [ ] 7 verified, reported, pushed
- [x] 2 code deps       - [ ] 5 content mapped

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
