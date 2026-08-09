# Backend standards

FastAPI + Python. Same posture as everywhere else: minimal, explicit, no speculative structure.

## Layout

```
backend/
  main.py            app creation, router registration, nothing else
  api/               routers, one file per resource
  models/            Pydantic schemas (request/response)
  core/              business logic and calculations — no FastAPI imports here
  db/                persistence
  constants.py       all domain constants and factors, with sources
```

The rule that matters: **`core/` never imports FastAPI.** Calculation and business logic must be
callable from a test or a notebook without an HTTP layer. This is what makes the science testable.

## Rules

1. **Type hints everywhere.** Not optional. They're the cheapest correctness tool available.
2. **Pydantic at the boundary.** Every request and response body is a model. No bare dicts crossing
   the API line.
3. **Thin routers.** A route handler validates, calls into `core/`, and shapes the response. If a
   handler has business logic in it, that logic is in the wrong place.
4. **No premature layers.** No repository pattern over SQLite, no service-class wrapper around a
   single function, no dependency-injection container. Add structure when the second caller exists.
5. **Sync until proven otherwise.** Use `async def` only where something is genuinely I/O-bound and
   concurrent. `async` on everything is cargo cult and complicates testing.
6. **Errors say what to do.** `HTTPException` with a message a human can act on. At L1 an unhandled
   exception with a real traceback is fine and often better than a swallowed one.
7. **Constants come from `constants.py`.** See `science-integrity.md` — no magic numbers in formulas.

## Data

- `data/` holds inputs and reference datasets. Treat it as read-only from the app's perspective
  unless the app's whole job is editing it.
- Large or generated files are gitignored. If a dataset can be regenerated, script the regeneration
  and commit the script, not the output.
- Any transformation applied to source data is code in the repo, not a manual step someone did once.
  If a manual step is unavoidable, it goes in `project-management/CHORES.md` with the exact procedure.
