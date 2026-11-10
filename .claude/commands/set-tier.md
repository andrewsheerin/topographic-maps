---
description: Change the tier (L1/L2/L3), the git policy or Deploys on push, with a proper walkthrough
argument-hint: "L1|L2|L3 · git=main|pr · deploys=yes|no"
---

Target: $ARGUMENTS (ask if not given). The PM mode is not changed here: committed ↔ private
moves files, so it goes through `/pm-upgrade` step 6.

**Tier**
1. Read `.claude/TIER.md` and the current and target files in `.claude/tiers/`. Summarize what
   changes for THIS project in two short lists: autonomy, rigor.
2. **Upgrading** (L1→L2, L2→L3): audit the gap against the target's rigor table (auth, secrets,
   tests, CI, migrations, docs) and propose a **hardening phase** for PLAN.md's Roadmap, its
   items as register rows. The upgrade is done when that phase closes, not when the file says so.
3. **Downgrading:** confirm intent explicitly. For a deployed app with real users moving off
   L3, push back: deployed users don't care what this file says.

**Git policy / Deploys on push**
- `main → pr`: from now on the user merges and main only moves by PRs. Check `gh auth status`
  and that the remote's main is protected (a chore for the user if not).
- `pr → main`: Claude will commit on and push main. Confirm that pushing main doesn't deploy, or
  set `Deploys on push: yes`.
- Whenever the result is `pr` or `Deploys on push: yes`, the push-main lines in
  `.claude/settings.json` sit under `deny`; for `main` with no live site, under `allow`. Settings
  files are the user's to edit: hand over the change.

**On the user's confirmation:** edit `.claude/TIER.md` (the `TIER:` line, the `@tiers/Ln.md`
import, `Git policy:`, `Deploys on push:`), mirror `git_policy` in `CURRENT.md`'s frontmatter,
log the switch and its reason in the Decision Log, run
`python .claude/skills/cleanup/pm_check.py .`, and commit.
