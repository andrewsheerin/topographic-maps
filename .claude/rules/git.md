---
paths:
  - ".gitignore"
  - ".gitattributes"
  - ".github/**"
---

# Git: main, a branch, or a worktree

Read this before starting a branch or a worktree; /add-feature, /wrap-session and /cleanup
follow it. The repo's policy is the `Git policy:` line in `.claude/TIER.md`, and
`.claude/hooks/guard_commands.py` enforces the hard limits whatever this file says.

## The two policies

| | `main` | `pr` |
| --- | --- | --- |
| Used by | most repos | repos where merging deploys (swpt-app, lca-pave-app, wsa) |
| Day-to-day work | commit on main | one branch per register item |
| Push | main, after `python verify.py` passes | the branch only |
| Merge | Claude, locally, then push main | the user merges the PR |
| Claude never | force-pushes, rewrites pushed history | pushes to main, commits or merges on main, merges a PR |

Under `main` Claude pushes main itself once verify passes; `settings.json` allows
`git push origin main` so it doesn't stop to ask. `Deploys on push: yes` (a live site, for
example fathomsolutions) is the exception: pushing main deploys, so the user pushes. Claude
commits, then hands over `! git push origin main`; the guard denies Claude's own push. In
those repos and under `pr`, `settings.json` lists the same push commands under `deny`, not
`allow`: a second lock that holds even if the guard doesn't run. Pushes are allowed per repo,
never in user-level settings.

## Which one to use

| Situation | `main` policy | `pr` policy |
| --- | --- | --- |
| Small change, done in one session | on main | branch `fix/F-31-slug` |
| Risky change (refactor, migration, dependency swap) | short branch, merge when green | branch |
| Work that spans sessions | short branch, pushed as a backup | branch |
| Two things at once in the same repo | worktree | worktree |

Branch names: `feat/F-12-short-slug`, `fix/F-31-…`, `refactor/…`, `chore/…`, `docs/…`.
Commits: `<type>(F-n): summary`, type ∈ feat · fix · refactor · docs · chore · test; drop
`(F-n)` only when no register item applies. Prefer new commits to amending.

## The `main` flow

1. Work and commit on main (or on a short branch for risky or multi-session work).
2. A short branch, once green: `git switch main`, `git merge --no-ff <branch>`,
   `git branch -d <branch>`.
3. `python verify.py` passes → `git push origin main`.

## The `pr` flow

1. From an up-to-date main (`git switch main`, `git pull --ff-only`):
   `git switch -c feat/F-12-short-slug`.
2. Commit as you go. Before pushing, run `python verify.py`.
3. `git push -u origin feat/F-12-short-slug`, then
   `gh pr create --base main --title "feat(F-12): …" --body-file <file>`, where the body lists
   what changed, the register items, and the verify summary (PASS/FAIL per check).
4. `git switch main`. The main folder goes back to main after every branch, so the next task
   starts clean and the vault reads main.
5. After the user merges: `git pull --ff-only`, `git branch -d feat/F-12-short-slug`.

Branches don't track each other. Before assuming a branch has a fix "because it was built
around the same time", check `git log <other>..<this> --oneline`.

## Worktrees, for when two things happen at once

A worktree is a second folder checked out from the same repo, on its own branch. Both folders
share one history, so a commit in either is visible to both, but each has its own files on disk.
Use one when two things must run in the same repo at the same time:

- a background Claude session working while you keep working in the main folder,
- a long render, build or model run you don't want to disturb,
- comparing two versions side by side.

Don't use one for a small sequential task: a branch is enough.

```
git worktree add .claude/worktrees/F-12-short-slug -b feat/F-12-short-slug   # create
git worktree list                                                           # see them all
git worktree remove .claude/worktrees/F-12-short-slug                       # once merged
git branch -d feat/F-12-short-slug
```

Rules:

- Worktrees always live under `.claude/worktrees/` (gitignored), never as sibling copies of the
  repo, and each is removed once its branch is merged ("Landing a worktree" below). `/cleanup`
  finds the ones left behind.
- Git refuses to check out one branch in two worktrees; give each its own.
- A worktree session doesn't rewrite `CURRENT.md`; the main-folder session owns it. It writes its
  own session report and register rows.
- Landing a worktree (merging it, pushing main, removing it) runs from the main folder, below.
  Under `pr`, the branch may be pushed and its PR opened from the worktree.
- The stash is shared by every worktree, so park work in a WIP commit instead.

## Private PM mode: each worktree gets a PM checkout

In private mode `project-management/` is its own repo in the main folder, ignored by the host.
A worktree of the host gets its own **PM checkout**: a linked worktree of that nested repo at
`<worktree>/project-management`, on a **PM branch** with the host branch's name, made from the
PM repo's main. The SessionStart hook creates it when a session starts in the worktree
(`claude -w`, `claude --bg -w`), and the session-start print says so (`PM files:
project-management/ is this worktree's own checkout…`). A session that moved in later
(EnterWorktree) runs `python .claude/hooks/session_start.py .` from the worktree. Commits there
go straight into the main folder's nested repo, on the PM branch, so a background session can
write its report, register rows and chores without touching the main folder.

- **Commit PM work as you go** (`git -C project-management commit …`), and never stop with it
  uncommitted. Git deletes the ignored checkout along with its host worktree, and Claude Code's
  own worktree cleanup goes by the host's git status, which doesn't show an ignored folder.
  Committed work is safe on the PM branch.
- **Don't push the PM branch.** It lands in the PM repo's main, and that is what gets pushed.
- **Before taking a new `F-n`, `D-n` or `C-n`**, run `git -C project-management merge
  --no-edit main`. It brings in what the main folder has committed, so the next free number
  really is free. (`main` is the PM repo's main; the checkout shares its refs.)
- Never `git init` a nested repo in a worktree, and never copy `project-management/` into one
  (nor list it in `.worktreeinclude`). A copy isn't connected to the nested repo, and its
  commits die with the folder.
- No checkout means the worktree reads and writes the main folder's `project-management/`
  (`PM files: <path>` at session start), which a background session can't edit. That happens
  with a detached HEAD, a branch that still tracks `project-management/`, or a worktree made
  before 3.4.

## Landing a worktree

From the main folder, on main. `<name>` is the worktree's folder, `<branch>` its branch
(`worktree-<name>` when Claude Code made it). Skip the `project-management` lines when the
worktree has no PM checkout (committed mode, or a worktree from before 3.4).

1. Both checkouts are clean, or stop; the session isn't finished. Both commands print nothing:
   ```
   git -C .claude/worktrees/<name> status --porcelain
   git -C .claude/worktrees/<name>/project-management status --porcelain
   ```
2. A background session's worktree is locked by its job (`git worktree list` shows `locked`).
   Run `git worktree unlock .claude/worktrees/<name>` before removing it. Delete the job
   (`claude rm`) only after the PM checkout is gone: deleting it can remove the host worktree,
   and the PM checkout with it.

Under `main`:
```
git merge --no-ff --no-edit <branch>
git -C project-management merge --no-ff --no-edit <branch>
python verify.py
git push origin main                     # Deploys on push: yes → the user pushes
git -C project-management push           # if it has a remote
git -C project-management worktree remove ../.claude/worktrees/<name>/project-management
git worktree remove .claude/worktrees/<name>
git -C project-management branch -d <branch>
git branch -d <branch>
```

Under `pr` (verify checks the worktree, and reads its PM checkout there):
```
python .claude/worktrees/<name>/verify.py
git -C .claude/worktrees/<name> push -u origin <branch>
gh pr create --base main --head <branch> --title "<type>(F-n): …" --body-file <file>
git -C project-management merge --no-ff --no-edit <branch>
git -C project-management push           # if it has a remote
git -C project-management worktree remove ../.claude/worktrees/<name>/project-management
git worktree remove .claude/worktrees/<name>
git -C project-management branch -d <branch>
```
After the user merges the PR: `git pull --ff-only`, then `git branch -d <branch>`. After a squash
or rebase merge, which git can't see as merged, confirm the PR is merged and use
`git branch -D <branch>` (the guard asks). The PM branch lands without waiting for the PR,
because the PM repo moves on its own. The exception is a PM branch that changes the PM layout
(a factory update): it waits for the merge, so main's verify stays green. The worktree may
also stay until the merge if follow-ups are likely; the PM checkout still comes off first.

**When the PM merge stops on a conflict:**
- `CURRENT.md`: take the main folder's side (`git -C project-management checkout --ours
  CURRENT.md`).
- A table (FEATURES, CHORES, the tables in DECISIONS): keep every row from both sides, one line
  each, in ID order. A row both sides changed gets combined: the later status, and both sides'
  Notes. If both sides took the same ID for different items, the branch's item takes the next
  free number, with `was F-12 on <branch>` in Notes, and its session report uses the new
  number.
- Prose (a `D-n` entry, PLAN.md): merge by hand, keeping both sides' content.
- Then `git -C project-management add <files>`, `git -C project-management commit --no-edit`,
  and check that pm_check shows no errors.
- Unsure? Run `git -C project-management merge --abort`. Nothing is lost: the PM branch and its
  checkout stay as they were. Land the host branch if it's ready, leave the worktree and its PM
  checkout in place, and hand the PM merge to the user as a chore. `branch -d` refuses an
  unmerged PM branch anyway.

**Removed in the wrong order?** pm_check in the main folder warns `pm-checkout`: the checkout is
gone, but its branch keeps every commit. Land the PM branch as above, then run
`git -C project-management worktree prune`. Uncommitted edits in the checkout are gone, which is
why it comes off first.

## Private PM mode: branches from before the conversion

In private mode the main folder holds `project-management/` as its own repo, and git treats that
ignored folder as expendable. A branch made before the conversion still tracks the folder:
checking it out in the main folder overwrites the PM files without a word, and switching back
deletes them. Committed PM content can be restored from the nested repo; uncommitted edits are
gone. pm_check names these branches (`pm-old-branch`) and the guard asks before such a switch.

Open such a branch only in a worktree, and merge main into it there:

```
git worktree add .claude/worktrees/F-9-slug feat/F-9-slug
cd .claude/worktrees/F-9-slug
git diff $(git merge-base HEAD main) HEAD -- project-management > <patch outside the repo>
git merge main
```

- No PM edits on the branch: the merge is clean and removes the worktree's PM copy.
- PM edits: git reports a modify/delete conflict for each. Apply the patch to the main folder's
  nested repo (`git -C <main folder>/project-management apply -p2 <patch>`) and commit it there,
  then take main's side in the worktree (`git rm -r project-management`; it removes only the
  worktree's copy) and `git commit`.

After that the branch no longer tracks the folder and is safe anywhere. Under `pr`, push it;
the PR picks up the merge. The next session that starts in the worktree gives it a PM checkout.

## What the guard enforces

Denied under both policies: force-push, `push --all/--mirror`, history rewrites, `gh pr merge`,
`terraform apply/destroy/import`, publishing and deploy commands, and `git worktree remove` of a
worktree that still holds its PM checkout. Denied under `pr`: pushing
to main, committing or merging on main (`merge --ff-only` is fine). Denied under `main` with
`Deploys on push: yes`: pushing main. The user is asked first for: `reset --hard`,
`clean -f`, `branch -D`, `worktree remove --force`, deleting a remote branch, adding or
changing a remote, `gh repo create`, `stash clear`, and, in private mode, checking out a
branch that still tracks
`project-management/` in the main folder. Text in single quotes, heredocs and PowerShell
here-strings (a commit message) is data, unless it is fed to a shell or expands a `$( )`. If a
blocked action is genuinely needed, prepare it and hand the command to the user; never route
around the guard.
