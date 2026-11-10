---
description: End the session — verify, commit, push per policy, sync the PM files, short report with a next-session prompt
---

Routine bookkeeping, aimed at a couple of minutes. Skip any step with nothing to do. Git steps
follow `.claude/rules/git.md` for this repo's policy (`.claude/TIER.md`).

1. **Verify:** `python verify.py`. A red code check gets fixed, or registered as a row and named
   to the user; never wrap over red silently. A `current-stale` warning is expected (step 4).
2. **Commit** the session's own outstanding work, `<type>(F-n): summary`, staging explicit
   paths. Changes this session didn't make: ask, don't sweep them in.
3. **Sync the PM files**, only the rows this session touched:
   - Register: statuses moved; `Ref` on every DONE row (hash, `#PR` or file). Ideas raised in
     conversation but never registered become IDEA rows.
   - Decisions: finish any `D-n` stub (options + recommendation); log the calls Claude made.
   - Chores: anything only the user can do, with exact steps.
   - **Vault intake:** if CLAUDE.md names a vault note, read that one note, read only. Each item
     in it that the PM files don't have yet goes in, Notes `from vault YYYY-MM-DD`: work items
     → register rows; tasks only the user can do → CHORES rows. Never edit the note.
4. **Reset CURRENT.md** (skip in a linked worktree: the main-folder session owns it).
   Frontmatter `updated` = today, plus `branch`, `phase`, `wip`, `blocked_on`; body ≤ 20 lines:
   right now, blocked on, next.
5. **Report** to `project-management/reports/sessions/YYYY-MM-DD-HHMM.md`, the time taken from
   the system clock (`date +%Y-%m-%d-%H%M`, or `Get-Date -Format yyyy-MM-dd-HHmm`). If /cleanup
   already wrote this session's report, add today's other work to that one instead:
   - `## What happened`: 3–5 sentences.
   - `## Touched`: F-n → status, D-n raised or resolved, C-n added, the verify result.
   - `## Plan ahead`: next tasks, each pointing at a register row.
   - `## Factory feedback` (only if there is some): problems with the app-factory template
     found today (a command step that didn't fit, a checker false alarm, a missing rule), one
     line each. The factory collects them with `tools/portfolio.py --feedback`.
   - `## Next session prompt`, last: ready to paste, for a Claude with no memory of today —
     what we're doing, the next register item, what to read first, any gotcha. The vault reads
     this.
6. **Commit the PM files** (`docs(pm): wrap session`) once
   `python .claude/skills/cleanup/pm_check.py .` shows no errors. Private PM mode: commit inside
   `project-management/` with `git -C project-management`, then, if it has a remote,
   `git -C project-management push`, under either git policy (no verify needed: it's the PM
   repo). In a worktree, that folder is the main folder's (`PM files:` at session start).
7. **Push per the policy:**
   - `main`: verify passed → merge any finished short branch, then `git push origin main`. With
     `Deploys on push: yes` (a live site) the user pushes: hand over `! git push origin main`.
     Red verify: don't push; say why.
   - `pr`: push the branch. No PR yet → `gh pr create` with a body listing what changed, the
     F-n items and the verify summary; otherwise mention the new commits. Then put the main
     folder back on main.
8. **Worktree:** if this session ran in a worktree whose branch is now merged, remove it from
   the main folder (`git worktree remove <path>`, `git branch -d <branch>`). Never `--force`:
   with uncommitted changes it stays, and you say so.
9. **Tell the user**, five lines at most: what happened, what waits on them (D-n, C-n, a PR to
   merge), what's next, and any permission that would have made today faster.
