---
paths:
  - ".env*"
  - "**/*.tfvars*"
  - "infra/**"
  - "terraform/**"
  - "**/auth*"
  - "**/auth*/**"
---

# Secrets and security

- Never print, paste, commit or open `.env`, `*.tfstate*` or `*.tfvars`; `settings.json` denies
  reading them. Learn variable names from the committed `*.example` files instead.
- To sanity-check a Terraform state change, use `git diff --stat`, never a content dump.
- A secret that reached git history stays leaked after the file is removed. The fix is
  rotation, which is the user's to do: log a chore with exact steps, and don't rewrite history.
- Auth, credential handling and exposing the app to new users are stop-and-ask changes at L2 and
  L3. Flag a security problem you find rather than building on top of it.
- Re-run the dependency audit (`npm audit`, `pip-audit` if installed) when you touch dependencies.
