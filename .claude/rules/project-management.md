---
paths:
  - "project-management/**"
---

# Project management (v3)

The PM files exist so any session can start cold in two minutes, so autonomous work can be
reviewed afterwards, and so the vault can build the user's daily plan. The vault reads
`CURRENT.md`, `DECISIONS.md`, `CHORES.md` and the newest session report, and never edits them:
keep those four small and in the shapes below. `pm_check.py` (run by `verify.py`) checks them.

## The files

| File | Holds | Limit |
| --- | --- | --- |
| `CURRENT.md` | Frontmatter + where things stand | ≤ 20 lines below the frontmatter |
| `FEATURES.md` | The register: every feature, fix, bug, refactor, idea | One table, one line per row |
| `DECISIONS.md` | Needs you · Resolved · Decision Log | Needs you holds only OPEN and PROPOSED |
| `CHORES.md` | Tasks only the user can do | One table; DONE and DROPPED rows move to the archive at /cleanup |
| `PLAN.md` | Purpose and scope · Roadmap · Architecture · Data model | About 500 lines, a soft target; the data model changes in the same commit as the schema |
| `reports/sessions/` | `YYYY-MM-DD-HHMM[-slug].md`, clock time from `date`; one per session | Ends with `## Next session prompt`; optional `## Factory feedback` before it |
| `reports/phases/` | `phase-<X>[-slug].md` | One per phase |
| `reports/topics/` | `YYYY-MM-DD-topic.md`: research, reviews, audits, migrations | — |
| `archive/` | Answered decisions, done chores, anything moved aside | Never deleted |

No other documents: no NOTES.md, no TODO.md, no parallel trackers.

## CURRENT.md

```
---
pm: v3
profile: app            # app · paper · engagement · partner · site
git_policy: main        # main · pr — mirrors .claude/TIER.md
pm_mode: committed      # committed · private — mirrors .claude/TIER.md
updated: 2026-11-09     # date of the last rewrite
branch: main            # branch the work is on
phase: A
wip: F-12               # the one WIP item, or —
blocked_on: D-3, C-2    # or —
---
```

Body: right now, blocked on, next. A snapshot, never a log. Rewritten at /wrap-session, not
mid-task. A session in a linked worktree leaves CURRENT.md to the main-folder session.

## IDs and statuses

- `F-n`, `D-n`, `C-n`, numbered per repo, never reused, never zero-padded. Subtasks: `F-12a`.
  Cite another repo's item as `swpt-app D-39`. An item migrated from an old scheme keeps its old
  ID in Notes as `was A.1`.
- Register: `IDEA → PLANNED → WIP → DONE`, plus `BLOCKED` and `DROPPED`. One WIP at a time.
  Deferred work is PLANNED with Phase `later`.
- Decisions: `OPEN` (the user decides; work waits), `PROPOSED` (Claude built on a provisional
  choice; the user confirms or corrects), `RESOLVED`, `SUPERSEDED`.
- Chores: `TODO`, `DONE`, `DROPPED` (no longer needed; the reason in Notes).

## FEATURES.md, the register

`| ID | Item | Type | Phase | Status | Ref | Notes |`. Type ∈ feature · fix · bug · refactor ·
test · chore · idea. `Ref` is filled at DONE: commit hash(es), a PR (`#42`) or a file. If
losing a thought would cost something, it gets a row the moment it comes up. Phase names a
phase in PLAN.md's Roadmap, or `later`.

## PLAN.md

Four sections: Purpose and scope (with what's explicitly out of scope), Roadmap (phases with a
goal, a done-when and their F-n), Architecture (as built), Data model (engine, schema, schema
changes; it changes in the same commit as the schema). Profiles drop the sections they don't
need. About 500 lines is the target, not a cap: detail that can live with the code goes there,
linked, and a plan for a single feature is a topic report.

## DECISIONS.md

- `## Needs you`: one `### D-n: question` entry each, with a `**Status:**` line, what it blocks,
  2–3 sentences of context, an options table with trade-offs and reversibility, and a
  recommendation. PROPOSED entries say what was chosen and the cost of reversing it. An entry
  the user can't act on in 30 seconds isn't finished; a mid-task stub gets finished at wrap.
- `## Resolved`: `| ID | Decision | Status | Outcome | Date |`, one line per decision. When one
  is answered, its full entry is appended to `archive/DECISIONS-answered.md`.
- `## Decision Log (made by Claude)`: `| Date | Decision | Why |`, newest first.

## CHORES.md

`| ID | Chore | Blocks | Due | Status | Notes |`. Due is `YYYY-MM-DD` or blank. Each chore is
doable without further thought: exact steps and values in Notes, or a `### C-n` block below the
table for multi-step ones.

## Table hygiene

Every row stays on one line. Write a literal pipe as `\|`. A cell that needs paragraphs belongs
in a report or a `### C-n` block, linked from the row.

## Private PM mode

`project-management/` is its own git repo, excluded from the host through `.git/info/exclude`.
Commit PM changes inside it (`git -C project-management commit …`); its own main may be pushed to
its private remote under either git policy. Worktrees of the host don't contain it: write to the
main folder's `project-management/`, which the session-start print names (`PM files: <path>`).
