# Tier L2 — Get it ready

An app with real momentum that still needs real progress — refinement, hardening, pre-deployment.
**Your job is to improve it without breaking what works.**

## Autonomy

- **Phases wait.** Present the phase plan and get sign-off before starting.
- **Decisions route by the reversibility test:** undoable in under an hour with no downstream
  cleanup → decide and log. Real reversal cost — architecture, new dependencies, schema changes,
  auth approach, anything baked into existing behavior — → stop, full entry in `DECISIONS.md`,
  continue on what doesn't depend on it.
- **Refactors** of working code get a one-paragraph plan in `CURRENT.md` before you start, so a
  session never ends mid-rearrangement without a record of the intent.

## Rigor — everything in L1, plus

| Axis | Posture |
| --- | --- |
| Auth | Real authentication (managed provider preferred). Coarse authz (user vs admin). |
| Secrets | Platform env vars or a secrets manager. Never in repo or image. |
| Persistence | Postgres. Alembic migrations. **Backups configured before launch.** |
| Errors | Handled at the API boundary; clean 500s, never a traceback to the client. |
| Tests | + happy-path API tests per endpoint + a boot smoke test. Regression test with every bug fix. |
| CI | Tests + lint on every PR; red blocks merge. |
| Infra | As code in `infra/`. Terraform still needs per-instance permission. |
| Logging | Structured, with levels and request IDs. |
| Docs | README + runbook: deploy, roll back, where logs live. |

**Still not warranted:** multi-region, autoscaling, caching layers, microservices, event sourcing,
comprehensive E2E. Ask first.
