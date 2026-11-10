---
description: Migrate this repo's project management and .claude setup to the app-factory v3 standard — resumable, never deletes content
argument-hint: "[profile=app|paper|engagement|partner|site] [git=main|pr] [pm=committed|private]"
---

Moves this repo onto v3 as `C:/Users/flow/CursorProjects/app-factory/STANDARD.md` defines it,
copying from `C:/Users/flow/CursorProjects/app-factory/template/`. No application code changes.
To run it in a repo that doesn't have it yet, open Claude Code there and say: "Read
C:/Users/flow/CursorProjects/app-factory/template/.claude/commands/pm-upgrade.md and follow it."

**Ground rules**
- **Never delete content.** What can't be placed goes to `project-management/archive/` with a
  first line saying where it came from and when. Original files that were mapped go there too.
- Move with `git mv`, so history follows the file.
- Every migrated item keeps its old ID in Notes: `was A.1`, `was M-3`.
- **Resumable:** the migration report `project-management/reports/topics/YYYY-MM-DD-pm-upgrade.md`
  is created in step 1 and carries the checklist below. If one exists with open boxes, read it
  and resume at the first open box. Commit after every step
  (`chore(pm): v3 upgrade, step N — …`) and tick its box in the same commit.

```
- [ ] 0 preconditions   - [ ] 3 settings confirmed   - [ ] 6 private mode (or n/a)
- [ ] 1 layout found    - [ ] 4 .claude installed    - [ ] 7 verified, reported, pushed
- [ ] 2 code deps       - [ ] 5 content mapped
```

## 0. Preconditions — stop and ask if any fails

- Not a git repository: stop and ask. For kalshi-bot it's decided (app-factory D-9): `git init`,
  local only with no remote; write a `.gitignore` (covering `.env`, the key folder, `data/db`,
  `logs/`) and stop for the user to review it before the first commit; update its CLAUDE.md
  "don't run git" line in the same step. The current state is the first commit, so the
  migration is reviewable as a diff.
- Uncommitted changes this session didn't make, or another session active in the repo: ask.
- Branch: under `pr`, `chore/pm-upgrade-YYYY-MM-DD`; under `main`, a short branch
  `chore/pm-upgrade` (this is multi-step and risky), merged when verify is green.

## 1. Detect the layout (read only)

| Layout | Signals |
| --- | --- |
| v2.1 | `project-management/{CURRENT,FEATURE_REGISTER,DECISIONS,CHORES}.md`, a `Commit(s)` column, no CURRENT frontmatter |
| v1 | `tasks/tasks.md`, `chores/decisions.md`, `chores/manual-tasks.md`, `plans/roadmap.md` |
| stripped v1 | some v1 files, e.g. a flat `tasks.md` |
| hybrid | v2.1 file names with their own ID schemes (`F-A01`, `D-001`, `CH-12`) |
| custom | PM in `docs/`, a root `ROADMAP.md`, or a single notes file |

Create the migration report. Record every PM file found with its line count, ID scheme and the
status words in use, plus `pm_check.py` output from before the upgrade.

## 2. Find what code depends on (read only)

Search the non-PM files for `project-management`, the PM file names and the ID patterns. A path
or line that code reads or writes is **frozen**; list each one in the report. For example:
kalshi-bot's `research/registry.py` needs a hypothesis ID, its spec hash and "pre-registered" on
one line of `DECISIONS.md`, and its `cli.py` writes `reports/YYYY-MM-DD_data-quality.md`.
- A frozen file whose name or place breaks a v3 rule stays put. Add
  `- pm_check ignore: <glob> (<reason>)` to the Exceptions in `.claude/TIER.md`.
- A frozen line keeps its exact text, in the file the code reads. If v3 would move it (to the
  archive, into a heading, across lines), stop and ask.
- Commit messages that cite old IDs (beacon-finance) are why old IDs stay in Notes.

## 3. Settle the settings — one batch of questions

Defaults come from the repo table in STANDARD.md; confirm them: profile, git policy, PM mode,
`Deploys on push`, the vault note path, and the tier (unchanged unless the user says so).

## 4. Install the v3 `.claude`

From the template: `hooks/` (all), `commands/`, `skills/cleanup/`, `tiers/`, and the rules
`git.md`, `project-management.md`, `testing.md`, `security.md`.
- **Existing files are merged, not overwritten.** `settings.json`: keep the repo's permissions
  and hooks, add the guard, formatter and session-start hook, drop deny rules the new policy
  contradicts (e.g. a blanket `git push` deny under `main`). `verify.py`: keep its checks, add
  the PM and hook checks. `.gitattributes`: add if missing, and renormalize line endings in a
  commit of its own.
- **Repo-specific rules stay.** Give each a `paths:` frontmatter so it loads only when needed;
  one that truly applies to every task stays unscoped, noted in the report.
- **TIER.md** in v3 shape: the same tier, `Git policy:`, `PM mode:`, `Deploys on push:`, the
  `@tiers/Ln.md` import, and every existing Exceptions line.
- **CLAUDE.md** in the template's shape, under ~80 lines, with the vault-note line and the
  "Not a Tawa repo" line (except in wsa). Project notes that don't fit move to a path-scoped
  `.claude/rules/project.md`; moved, never dropped. Old git rules that contradict the chosen
  policy are replaced; list them in the report.
- Apply the profile overlay: `C:/Users/flow/CursorProjects/app-factory/profiles/<profile>.md`.
- **Partner and engagement repos** (wsa, Eagle Eye): commit nothing of ours to the host. Our
  `CLAUDE.md`, `.claude/`, `verify.py` and `project-management/` go in `.git/info/exclude`
  (app-factory D-8). No `.gitattributes` or other host-file changes either. If the host already
  tracks a `CLAUDE.md` or `.claude/`, stop and ask.

## 5. Map the content

| Found | v3 |
| --- | --- |
| v1 `tasks/tasks.md` IDs `A.1`, `B.2.1`; flat `tasks.md`; kalshi `F-A01`; wsa ROADMAP rows 0.1–3.41 | `FEATURE_REGISTER.md` rows `F-n`, Phase = the old letter, `was <old id>` in Notes |
| v1 statuses TODO, WIP, BLOCKED, DONE; kalshi Planned, In progress, Blocked, Deferred | PLANNED, WIP, BLOCKED, DONE; Deferred → PLANNED with Phase `later` |
| v1 `chores/decisions.md`; kalshi `D-001` | `DECISIONS.md` `D-n`; kalshi keeps its numbers, unpadded (`D-1`) |
| beacon's "Provisional" table; kalshi Proposed | PROPOSED entries in Needs you |
| answered entries in Needs you; ANSWERED, APPROVED, DECIDED and similar | RESOLVED: full text to `archive/DECISIONS-answered.md`, one line under Resolved |
| v1 `chores/manual-tasks.md` `M-n`; kalshi `CH-nn` | `CHORES.md` `C-n`, `was M-3` in Notes, Due blank unless known |
| `plans/roadmap.md` with Out of scope, Approach and Deferred sections | `ROADMAP.md`, PLAN's out-of-scope, `ARCHITECTURE.md`, and register rows with Phase `later` |
| `reports/session-YYYY-MM-DD.md`, `*-session-handoff.md`, slug-named sessions | `reports/sessions/YYYY-MM-DD-HHMM[-slug].md`; when the real time is unknown, the time of the commit that added it (`git log --diff-filter=A --format=%ci -- <file>`) |
| research, review, audit and change-log reports; `reports/reviews/`; kalshi `YYYY-MM-DD_topic.md` | `reports/topics/YYYY-MM-DD-topic.md` (not the frozen ones) |
| an oversized `CURRENT.md` (e.g. a 237-line log) | its history → one session report dated by its last commit; CURRENT rewritten under 20 lines |

- New IDs are assigned in the old order, so the map reads top to bottom. Put the full
  old → new map in the migration report.
- Every table row stays on one line; long text becomes a short Notes summary plus a link to the
  archived original.
- Whatever fits nowhere (prompts, scratch notes, drafts) → `archive/`, header first.

## 6. Private PM mode (only if chosen)

1. If the host repo tracks `project-management/`, untracking it is the one step that removes
   files from the host's index. Confirm with the user first, and say plainly that the files
   stay in the host's history; on a public repo, anyone can still read them there.
2. `git -C project-management init -b main`, add everything, commit.
3. In the host: `git rm -r --cached project-management`, commit
   `chore: move project management to a private nested repo`.
4. Append `project-management/` to the host's `.git/info/exclude`.
5. A private remote for the nested repo is the user's to create: log it as a chore.

## 7. Verify, report, push

- `python verify.py` until `pm_check` shows no errors. Register code reds; don't fix app code
  here.
- Finish the migration report: layout found, settings chosen, the ID map, what went to the
  archive and why, frozen items and ignore lines, questions raised as `D-n` entries, and a
  `## Next session prompt` for the first v3 session.
- Push per the policy: `main` → merge the branch, push main once verify passes (ask first if
  `Deploys on push: yes`); `pr` → push the branch and open a PR whose body summarizes the report.
- Tell the user what moved, what's frozen, and what needs their answer.
