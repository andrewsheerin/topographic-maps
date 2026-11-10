---
name: cleanup
description: Deep clean of this repo's git state and project-management docs — inventory, reconcile against git, merge and push per the git policy, archive answered decisions, write a catch-up report. Run only when the user asks (/cleanup); never as part of routine work.
---

# /cleanup — the deep pass

Run when asked, not routinely. Git steps follow `.claude/rules/git.md` for this repo's policy
(`.claude/TIER.md`). `pm_check.py` sits next to this file.

**Never:** force-push, rewrite history, delete unmerged work, or discard uncommitted changes.

## 1. Inventory (read only)

Collect, then show the user one short table before changing anything:

- `git status`; current branch; whether this is the main folder or a linked worktree.
- Branches: merged into main or not, ahead/behind their remote
  (`git branch -vv`, `git branch --merged main`, `git fetch --prune` first if there's a remote).
- Worktrees (`git worktree list`), folders under `.claude/worktrees/` that aren't registered,
  and sibling copies of the repo next to it (e.g. `<repo>-old`, `<repo> (2)`): orphaned copies.
- Stashes (`git stash list`; the stack is shared by every worktree).
- `pr` policy: open PRs (`gh pr list`).
- `python .claude/skills/cleanup/pm_check.py . --json`.

## 2. Stop and ask about

- uncommitted changes this session didn't make,
- unmerged branches whose fate is unclear (finish, park, or abandon is the user's call),
- deleting anything other than merged branches and clean, merged worktrees,
- creating a remote,
- stashes (apply, keep, or drop — the user's call).

Carry on with everything that doesn't depend on the answers.

## 3. Reconcile the PM docs

- **Register against git:** a WIP or PLANNED row whose work is merged → DONE with `Ref`. A WIP
  row with no commits in two weeks → BLOCKED or PLANNED, with the reason in Notes. Exactly one
  WIP at most.
- **Decisions:** every answered entry in Needs you → full text appended to
  `archive/DECISIONS-answered.md`, one line under Resolved. Statuses outside OPEN, PROPOSED,
  RESOLVED, SUPERSEDED get mapped.
- **Chores:** close the ones with evidence they're done (say what the evidence is); one that
  no longer needs doing is DROPPED, with the reason in Notes. Move DONE and DROPPED rows to
  `archive/CHORES-done.md`.
- **Vault intake:** if CLAUDE.md names a vault note, read that one note (read only). Items the
  PM files lack go in, Notes `from vault YYYY-MM-DD`: work items → register rows; tasks only
  the user can do → CHORES rows.
- **Catch-up session report** covering the commits since the newest session report
  (`git log --since=<its date>`): what landed, by register item.
- **CURRENT.md** rewritten: frontmatter current, body ≤ 20 lines.
- Fix whatever else `pm_check.py` reports: report names, broken table rows, Ref gaps.

## 4. Git, per policy

- **`main`:** commit the PM changes; merge finished branches (`git merge --no-ff`), run
  `python verify.py`, and push main if it passes (`Deploys on push: yes` → the user pushes:
  hand over `! git push origin main`).
- **`pr`:** put the changes on `chore/cleanup-YYYY-MM-DD`, push it, and open a PR with the
  inventory and verify summary in its body. The user merges.
- **Either way:** delete branches merged into main (`git branch -d`), remove clean worktrees
  whose branch is merged (`git worktree remove`, no `--force`), and leave the main folder on
  main.
- **Private PM mode:** the PM changes are committed inside `project-management/`
  (`git -C project-management commit`), then `git -C project-management push` if it has a
  remote, under either policy. No remote (`pm-remote`): say the PM files have no off-disk
  copy and offer to create one, as `/pm-upgrade` step 6 does.

## 5. Report

Write `project-management/reports/sessions/YYYY-MM-DD-HHMM-cleanup.md` (clock time from `date`):
the inventory table, what changed, what is waiting on the user, `## Factory feedback` if the
template caused trouble (one line each), and `## Next session prompt` last. When /cleanup is
the session's last act, this is the session report: it also covers whatever else the session
did, and no separate /wrap-session report is written. Re-run `pm_check.py` — no errors — and
commit the report. Then tell the user, in a few lines, what was cleaned and what needs their
answer.
