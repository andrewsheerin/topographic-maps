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

`Deploys on push: yes` (for example fathomsolutions) adds one rule under `main`: say that the
push deploys, and ask before every push to main. The guard asks too.

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
  repo, and each is removed once its branch is merged. `/cleanup` finds the ones left behind.
- Git refuses to check out one branch in two worktrees; give each its own.
- A worktree session doesn't rewrite `CURRENT.md`; the main-folder session owns it. It writes its
  own session report and register rows.
- In private PM mode the worktree has no `project-management/`: write to the main folder's.
- The stash is shared by every worktree, so park work in a WIP commit instead.

## What the guard enforces

Denied under both policies: force-push, `push --all/--mirror`, history rewrites, `gh pr merge`,
`terraform apply/destroy/import`, publishing and deploy commands. Denied under `pr`: pushing
to main, committing or merging on main (`merge --ff-only` is fine). The user is asked first
for: `reset --hard`, `clean -f`, `branch -D`, `worktree remove --force`, deleting a remote
branch, adding or changing a remote, `gh repo create`, `stash clear`, and pushing main when
`Deploys on push: yes`. If a blocked action is genuinely needed, prepare it and hand the
command to the user; never route around the guard.
