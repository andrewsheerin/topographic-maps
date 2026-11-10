---
description: Migrate this repo's project management and .claude setup to the app-factory v3 standard — resumable, never deletes content
argument-hint: "[profile=app|paper|engagement|partner|site] [git=main|pr] [pm=committed|private]"
---

Moves this repo onto v3 as `C:/Users/flow/CursorProjects/app-factory/STANDARD.md` defines it,
copying from `C:/Users/flow/CursorProjects/app-factory/template/`. No application code changes.
To run it in a repo that doesn't have it yet, open Claude Code there and say: "Read
C:/Users/flow/CursorProjects/app-factory/template/.claude/commands/pm-upgrade.md and follow it."

## Already on v3: update, don't migrate

If CURRENT.md's frontmatter has `pm: v3`, skip the steps below and do this instead
(`<factory>` is `C:/Users/flow/CursorProjects/app-factory` throughout this file):
1. `python <factory>/tools/sync.py .` (read only). It sorts every template-owned file into
   current, stale (untouched since install), changed (edited here) and missing, and prints the
   repo's factory version and the current one. Show the user the lists.
2. Where to work: under `main`, on main; under `pr`, a `chore/factory-<version>` branch.
3. With the user's OK, `python <factory>/tools/sync.py . --apply`: it replaces stale files, adds
   missing ones and records the version in `.claude/factory.json`. It never touches PM
   content, CLAUDE.md, TIER.md or the repo's own rules.
4. Each changed file: diff it against `<factory>/template/<path>` and merge the factory's
   changes by hand, keeping the local ones. Say which you merged and which you left.
5. Run the "Upgrade from" steps in `<factory>/CHANGELOG.md` for every version after the
   repo's, oldest first.
6. `python verify.py`, commit (`chore: app-factory <version>`), push per the policy (`pr`: push
   the branch and open a PR). Note it in the session report.

**Ground rules**
- **Never delete content.** What can't be placed goes to `project-management/archive/` with a
  first line saying where it came from and when. When a differently named file replaces an
  original (v1, custom layouts), the original goes there too; v2.1 files keep their names and
  are edited in place, so git already has their history.
- Move with `git mv`, so history follows the file.
- Every migrated item keeps its old ID in Notes: `was A.1`, `was M-3`.
- **Resumable:** the migration report `project-management/reports/topics/YYYY-MM-DD-pm-upgrade.md`
  is created in step 1 and carries the checklist below. If one exists with open boxes, read it
  and resume at the first open box. Commit after every step
  (`chore(pm): v3 upgrade, step N — …`) and tick its box in the same commit; steps 0 and 1 share
  a commit, since step 0 has nothing to commit.

```
- [ ] 0 preconditions   - [ ] 3 settings confirmed   - [ ] 6 private mode (or n/a)
- [ ] 1 layout found    - [ ] 4 .claude installed    - [ ] 7 tidied, verified, reported, pushed
- [ ] 2 code deps       - [ ] 5 content mapped (and merged, under main)
```

## 0. Preconditions — stop and ask if any fails

- Not a git repository: stop and ask. For kalshi-bot it's decided (app-factory D-9): `git init`,
  local only with no remote; write a `.gitignore` (covering `.env`, the key folder, `data/db`,
  `logs/`) and stop for the user to review it before the first commit; update its CLAUDE.md
  "don't run git" line in the same step. The current state is the first commit, so the
  migration is reviewable as a diff.
- Uncommitted changes this session didn't make, or another session active in the repo: ask.
- **Inventory git, don't tidy it yet:** list unmerged branches, worktrees and stashes in the
  report, but leave fast-forwards, branch deletions and worktree removals to step 7. The repo's
  old guard and settings may block them; by step 7 the v3 guard is in force.
- Where the work happens:
  - `main` policy: a short branch `chore/pm-upgrade` in the main folder (this is multi-step and
    risky). It merges into main after step 5; steps 6 and 7 run on main.
  - `pr` policy: a worktree, so the main folder stays on main throughout:
    `git fetch`, then `git worktree add .claude/worktrees/pm-upgrade -b
    chore/pm-upgrade-YYYY-MM-DD origin/main`. If `.claude/worktrees/` isn't ignored yet, add it
    to `.git/info/exclude` too.

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
- Also list mentions that aren't dependencies (README lines, code comments citing a PM file):
  step 6 needs them under private mode.

## 3. Settle the settings — one batch of questions

Defaults come from the repo table in STANDARD.md; confirm them: profile, git policy, PM mode,
`Deploys on push`, the vault note path, and the tier (unchanged unless the user says so). Ask
too where this repo's domain defaults live (units, factors, model or print defaults), for the
science rule's `paths:` in step 4.

## 4. Install the v3 `.claude`

From the template: `hooks/` (all), `commands/`, `skills/cleanup/`, `tiers/`, and the rules
`git.md`, `project-management.md`, `testing.md`, `security.md`, plus `backend-standards.md`,
`frontend-standards.md` and `science-integrity.md` unless the profile drops them.
- **Existing files are merged, not overwritten.** `settings.json`: keep the repo's permissions
  and hooks, add the guard, formatter and session-start hook, drop deny rules the new policy
  contradicts (e.g. a blanket `git push` or `git merge` deny under `main`). The template's
  push-main lines go under `allow` only for `main` without a live site; under `pr` or with
  `Deploys on push: yes` they go under `deny`. `verify.py`: keep its checks, add the PM and
  hook checks.
- **`.gitignore`:** add the template's lines it lacks; `.claude/worktrees/` at least.
- **`.gitattributes`:** missing → the template's, then renormalize line endings in a commit of
  its own. Present without `eol=lf` → leave it, unless `git ls-files --eol` shows `i/crlf`
  (CRLF in the index); then add `* text=auto eol=lf` and renormalize in a commit of its own.
- **Repo-specific rules stay.** Give each a `paths:` frontmatter so it loads only when needed;
  one that truly applies to every task stays unscoped, noted in the report.
- **`science-integrity.md`:** add the folders from step 3 to its `paths:`. The template's globs
  (`backend/core/**`, `backend/constants.py`, `data/**`) miss defaults that live in frontend
  code, models or API schemas.
- **TIER.md** in v3 shape: the same tier, `Git policy:`, `PM mode: committed`,
  `Deploys on push:`, the `@tiers/Ln.md` import, and every existing Exceptions line. Write
  `committed` even when private was chosen: pm_check errors on `private` until the nested repo
  exists, so step 6 flips it.
- **CLAUDE.md** in the template's shape, under ~80 lines, with the vault-note line and the
  "Not a Tawa repo" line (except in wsa). Project notes that don't fit move to a path-scoped
  `.claude/rules/project.md`; moved, never dropped. Old git rules that contradict the chosen
  policy are replaced; list them in the report.
- **A root `GUIDE.md` from app-factory v2.1** (the owner's guide; it says "you push and merge"
  and calls branch protection "the real wall"): `git mv` it to
  `project-management/archive/GUIDE-v2.1.md`, header first, and point README's references to it
  at CLAUDE.md. v3 ships no per-repo guide.
- **The profile overlay:** `C:/Users/flow/CursorProjects/app-factory/profiles/<profile>.md`.
  The app profile has none: the template is the app profile.
- **Partner and engagement repos** (wsa, Eagle Eye): commit nothing of ours to the host. Our
  `CLAUDE.md`, `.claude/`, `verify.py` and `project-management/` go in `.git/info/exclude`
  (app-factory D-8). No `.gitattributes` or other host-file changes either. If the host already
  tracks a `CLAUDE.md` or `.claude/`, stop and ask.

## 5. Map the content

| Found | v3 |
| --- | --- |
| v1 `tasks/tasks.md` IDs `A.1`, `B.2.1`; flat `tasks.md`; kalshi `F-A01`; wsa ROADMAP rows 0.1–3.41 | `FEATURES.md` rows `F-n`, Phase = the old letter, `was <old id>` in Notes |
| v2.1 `FEATURE_REGISTER.md` | `git mv` to `FEATURES.md`; rows keep their IDs |
| v2.1 `plans/PLAN.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `DATABASE.md` | the four sections of `PLAN.md`: `git mv plans/PLAN.md PLAN.md` so history follows, fold the other three in whole, then `git rm` them. Past ~500 lines, list in the report what could move next to the code; the migration itself moves nothing into the host |
| any other file in `plans/` (one feature's plan, e.g. swpt-app's `F-336-publication-plan.md`) | `reports/topics/YYYY-MM-DD-<slug>.md`, dated by its last commit (`git mv`) |
| v1 statuses TODO, WIP, BLOCKED, DONE; kalshi Planned, In progress, Blocked, Deferred | PLANNED, WIP, BLOCKED, DONE; Deferred → PLANNED with Phase `later` |
| v1 `chores/decisions.md`; kalshi `D-001` | `DECISIONS.md` `D-n`; kalshi keeps its numbers, unpadded (`D-1`) |
| beacon's "Provisional" table; kalshi Proposed | PROPOSED entries in Needs you |
| answered entries in Needs you; ANSWERED, APPROVED, DECIDED and similar | RESOLVED: full text to `archive/DECISIONS-answered.md`, one line under Resolved |
| v2.1 Resolved rows with no ID (`—`) | the next free `D-n`s in file order, RESOLVED; each original row's full text to `archive/DECISIONS-answered.md` |
| a Decision Log line or a CURRENT "Blocked on" item asking the user to confirm something | a PROPOSED `D-n` in Needs you; the log line stays, as history |
| v1 `chores/manual-tasks.md` `M-n`; kalshi `CH-nn` | `CHORES.md` `C-n`, `was M-3` in Notes, Due blank unless known |
| v1 `plans/roadmap.md` with Out of scope, Approach and Deferred sections | `PLAN.md`'s Roadmap, its out-of-scope list and Architecture, and register rows with Phase `later` |
| `reports/session-YYYY-MM-DD.md`, `*-session-handoff.md`, slug-named sessions | `reports/sessions/YYYY-MM-DD-HHMM[-slug].md`; when the real time is unknown, the time of the commit that added it (`git log --diff-filter=A --format=%ci -- <file>`) |
| research, review, audit and change-log reports; `reports/reviews/`; kalshi `YYYY-MM-DD_topic.md` | `reports/topics/YYYY-MM-DD-topic.md` (not the frozen ones) |
| an oversized `CURRENT.md` (e.g. a 237-line log) | its history → one session report dated by its last commit; CURRENT rewritten under 20 lines |

- New IDs are assigned in the old order, so the map reads top to bottom. Put the full
  old → new map in the migration report.
- Every table row stays on one line; long text becomes a short Notes summary plus a link to the
  archived original.
- Whatever fits nowhere (prompts, scratch notes, drafts) → `archive/`, header first.

**Then record the factory version:** `python <factory>/tools/sync.py . --apply` writes
`.claude/factory.json` (step 4 already copied the current files, so nothing else should
change; files the profile dropped go under `skip`). **Then `python verify.py`** until pm_check
shows no errors; register code reds, don't fix app code here. Under `main`, merge now, so steps
6 and 7 run on main with the v3 guard in force:
`git switch main`, `git merge --no-ff chore/pm-upgrade`, `git branch -d chore/pm-upgrade`. If
the old guard or settings still block the merge, hand the user the command
(`! git switch main && git merge --no-ff chore/pm-upgrade`). Under `pr`, stay in the worktree.

## 6. Private PM mode (only if chosen)

Before anything moves:
- **Confirm the untracking with the user.** It is the one step that removes files from the
  host's index. Say plainly that the files stay in the host's history; on a public repo, anyone
  can still read them there.
- **List what will dangle** in a clone of the host (step 2's mentions): README lines go in the
  untracking commit; for code comments that cite a PM file, suggest citing the ID alone
  (`D-2`), and change code only if the user agrees.
- **Never switch the host's branch** from the untracking until the nested repo is in place in
  the main folder. Git treats ignored files as expendable: checking out a commit that still
  tracks `project-management/` overwrites the nested repo's working files without a word, and
  switching back deletes them. Committed nested content can be restored; uncommitted edits are
  lost.
- **The nested repo lives only in the main folder**, never in a worktree: `git worktree remove`
  deletes ignored files, a nested repo included, without asking.

### Under `main`: on main, in this order

1. Edit CURRENT.md's frontmatter: `pm_mode: private`, `branch: main`.
2. Write `project-management/.gitattributes` with `* text=auto eol=lf`: the host's
   `.gitattributes` doesn't reach into the nested repo.
3. `git -C project-management init -b main`, then add everything and commit
   (`docs(pm): project management moves to its own repo`).
4. Append `project-management/` to the host's `.git/info/exclude`.
5. One host commit, `chore: move project management to a private nested repo`:
   `git rm -r --cached project-management`, `.claude/TIER.md` → `PM mode: private`, and the
   README lines.
6. Check: `git ls-files project-management` prints nothing, `git check-ignore -q
   project-management` succeeds, `git -C project-management status` is clean, and pm_check has
   no errors.

### Under `pr`: two commits on the branch, then the main folder after the merge

Main only moves by the PR, so the untracking reaches main when the user merges, and the main
folder rebuilds the PM files from the branch. In the worktree, with verify green from step 5:

1. Edit CURRENT.md's frontmatter (`pm_mode: private`, `branch: main`) and write
   `project-management/.gitattributes` (`* text=auto eol=lf`). In the migration report, mark
   step 6 "content committed; finish in the main folder after the merge". Commit:
   `chore(pm): v3 upgrade, step 6 — content for the nested repo`. This is the **content
   commit**: the nested repo is built from it, so every PM change goes in before it.
2. One last commit: `git rm -r --cached project-management`, `.claude/TIER.md` →
   `PM mode: private`, the README lines; `chore: move project management to a private nested
   repo`. Nothing is committed on the branch after it.
3. Leave the worktree's leftover `project-management/` alone; it duplicates the content commit,
   and no nested repo goes here.
4. Push the branch and open the PR (step 7's body). Add: the content commit's hash; verify was
   green at the end of step 5, and pm_check's private-mode errors on the last commit are
   expected until the main-folder steps run. Ask the user to merge soon, and until the
   main-folder steps are done, to make no PM edits and switch no branches in the main folder.

After the user merges, in the main folder (on main, where it has been all along):

5. `git status --porcelain -- project-management` must print nothing. Anything it shows was
   written during the wait: stop and ask; it gets carried into the nested repo by hand.
6. Append `project-management/` to `.git/info/exclude`.
7. `git pull --ff-only`. The merge removes the tracked PM files here (git keeps their history).
8. `git restore --source=chore/pm-upgrade-YYYY-MM-DD~1 --worktree -- project-management`: the
   content commit's files, untracked and ignored. The local branch's parent is the content
   commit however the PR was merged (merge commit, squash or rebase), so restore before
   deleting the branch.
9. `git -C project-management init -b main`, add everything, commit (`docs(pm): project
   management moves to its own repo`). Tick step 6 in the migration report, with the content
   commit's hash, and commit that inside the nested repo.
10. Check as under `main` (step 6 above). Then `git worktree remove
    .claude/worktrees/pm-upgrade` (it deletes the leftover copy) and delete the branch:
    `git branch -d`, or after a squash or rebase merge, which git can't see as merged, confirm
    the PR is merged and use `git branch -D` (the guard asks).

`tools/tests/test_private_mode.py` in app-factory runs both sequences, and the old-branch case,
on throwaway repos.

### The private remote

Once the nested repo exists in the main folder, it is the only copy of the PM history. Say so,
and offer to give it a private remote now. With the user's yes (the guard asks too):
`gh repo create <owner>/<repo>-pm --private --source=project-management --remote=origin --push`.
If the user declines or `gh` isn't set up, add a chore, "Create a private remote for the PM
repo", with Blocks `off-disk copy of the PM files`; pm_check warns (`pm-remote`) until it's done.

## 7. Tidy, verify, report, push

Under `pr` the PR comes first (step 6, or now if PM stays committed) and the tidy follows the
user's merge.

- **Git tidy**, once main carries the v3 guard: delete branches merged into main
  (`git branch -d`), remove clean worktrees whose branch is merged, and ask about unmerged
  branches, stashes and orphaned copies, as /cleanup does. Private mode: pm_check names any
  branch that still tracks `project-management/`; handle it as `.claude/rules/git.md` says.
- `python verify.py` until `pm_check` shows no errors. Register code reds; don't fix app code
  here.
- Finish the migration report: layout found, settings chosen, the ID map, what went to the
  archive and why, frozen items and ignore lines, questions raised as `D-n` entries, and a
  `## Next session prompt` for the first v3 session.
- Push per the policy: `main` → push main once verify passes (with `Deploys on push: yes`,
  the user pushes: hand over the command); `pr` → push the branch and open a PR whose body
  summarizes the report and carries the verify results.
- Tell the user what moved, what's frozen, and what needs their answer.
