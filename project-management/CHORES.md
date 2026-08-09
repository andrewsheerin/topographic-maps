# Chores

Manual tasks only the user can do: accounts, credentials, DNS, payments, console clicking,
pushing/merging, `terraform apply`, anything needing a human in a browser.

Each chore must be executable without further thought — exact steps and values.

| ID | Chore | Blocks | Status | Notes |
| --- | --- | --- | --- | --- |
| C-1 | Rotate the OpenTopography API key (it is exposed in git history) | security | TODO | See details. Do this even though the app still works with the old key. |
| C-2 | Enable branch protection on `main` when the repo goes to GitHub | factory workflow | TODO | Settings → Branches → require a PR, block force-pushes. The real wall behind the git hooks. |

## Details

### C-1
The current key `0939b30…` appears in commits `4f6458f` and `1bebe89`, so it must be treated as
compromised (anyone with repo history has it).

1. Sign in at https://portal.opentopography.org/ → **My Account → API keys**.
2. Revoke/regenerate the key.
3. Put the new value in `.env` at the repo root:  `OPEN_TOPO_API_KEY=<new key>`.
4. (Optional) The old value stays in git history. Rotating makes it useless, so history scrubbing
   (BFG / `git filter-repo`) is only needed if you specifically want it gone before pushing.

**Done when:** the old key is revoked and `.env` holds a working new key.

### C-2
Only actionable once `main` is on GitHub. Until then the git hooks (`.claude/hooks`) are the guard.

**Done when:** GitHub requires a PR to merge into `main` and blocks force-pushes.
