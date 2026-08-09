# Tier L1 — Build it

Greenfield passion project. Local, one user (the author). **Your job is momentum.** The user wants
to describe an app and watch it exist — not approve every step of it.

## Autonomy

- **Interview first, then run.** Design direction, look and feel, and the feature set are the
  user's — get them at bootstrap (`/new-app`) and whenever a new feature is being shaped. After
  that, build without waiting.
- **Phases:** write the plan, announce it in `CURRENT.md`, and proceed. Do not wait for sign-off.
  The phase report is where the user reviews what happened.
- **Decisions:** architecture, libraries, schema shape, file layout — decide and log (one line in
  the Decision Log, `DECISIONS.md`). Stop only for: the non-negotiables in `CLAUDE.md` (science,
  git push, destructive data ops), adding or cutting a *feature* (that's scope, which is theirs),
  and genuine forks in design direction.
- Still true at L1: log chores and feature ideas the moment they surface, keep `CURRENT.md` live,
  wrap sessions properly. Autonomy is not exemption from the PM system — the PM system is what
  makes autonomy reviewable after the fact.

## Rigor — lean by design; the risk at L1 is over-building

| Axis | Posture |
| --- | --- |
| Auth | None. |
| Secrets | `.env`, gitignored. |
| Persistence | SQLite or flat files. No migrations framework unless the schema genuinely churns. |
| Errors | Let it crash with a readable traceback. |
| Validation | Pydantic at the API boundary. That's it. |
| Tests | Calculation / data-transformation logic only (science always gets tests). |
| CI / Infra | None. `infra/` stays empty. |
| Logging | `print()` is fine. |
| Perf | Ignore until visibly slow. |

**Actively refuse to build:** rate limiting, RBAC, audit logs, caching layers, job queues,
feature flags, retry wrappers, health checks, repository patterns, DI containers, error-tracker
integrations. Wanting one is a signal the project might be L2 — raise it, don't build it.
