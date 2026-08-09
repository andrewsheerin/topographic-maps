# Tier L3 — Don't break it

Deployed, real users, real consequences. **Tread carefully: bias to the smallest safe change,
prove it, then move.**

## Autonomy

- **Phases wait**, and the phase report is mandatory before the next one is proposed.
- **Stop and ask for:** any architecture change, any new dependency, any schema change or
  migration, anything touching auth, infra, or **production data** — regardless of how small the
  diff looks. Decide-and-log is reserved for genuinely local implementation detail.
- Every change traces to a register ID. No drive-by edits outside the current item.
- Bug fixes ship with a regression test, no exceptions.
- Never run anything against production (data, migrations, deploys) yourself — prepare it,
  document it, and hand it to the user as a chore with exact steps.

## Rigor — everything in L2, plus

| Axis | Posture |
| --- | --- |
| Auth | Full authn + authz, per-resource checks, expiring sessions. |
| Secrets | Manager with rotation; humans don't read prod secrets in normal work. |
| Persistence | Managed Postgres. Migrations reviewed before merge and reversible. Restores actually tested. |
| Input | Every external input validated and bounded: size, type, rate. |
| Errors | Nothing internal reaches the client; errors reported to a tracker with context. |
| Tests | Unit + integration + E2E on critical paths. |
| CI/CD | + type-check + dependency audit + staging; deploys reversible. |
| Infra | Fully Terraform-managed; `plan` reviewed by the user before any `apply`. |
| Observability | Structured logs, metrics on paths that matter, alerts that reach a human. |
| Data | Retention/deletion policy decided before collecting anything personal. |
