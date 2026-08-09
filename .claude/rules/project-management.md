# Project management rules

The PM system exists so (a) any session can start cold and be productive in two minutes, and
(b) at high-autonomy tiers, everything Claude did on its own is reviewable after the fact.
It is not paperwork. If a document isn't earning that, it's too heavy.

## The files

| File | What it is | Cadence |
| --- | --- | --- |
| `CURRENT.md` | Where we are right now. Scratchpad, <20 lines. | Continuously |
| `FEATURE_REGISTER.md` | Every feature, fix, and bug — including ones planned for later | The moment one is discussed |
| `DECISIONS.md` | Top: decisions the USER must make. Bottom: log of decisions Claude made | The moment one surfaces / is made |
| `CHORES.md` | Manual tasks only the user can do | The moment one is discovered |
| `plans/PLAN.md` | What the app is, scope, out-of-scope | Rarely — scope changes only |
| `plans/ROADMAP.md` | Phases A/B/C with goals and done-conditions | Phase boundaries |
| `plans/ARCHITECTURE.md` | System architecture + tech stack, as built | When either actually changes |
| `plans/DATABASE.md` | Schema + how the DB is managed | **Same commit as any schema change** |
| `reports/phases/` | End-of-phase reports | Phase close (`/phase-report`) |
| `reports/sessions/` | End-of-session reports, `YYYY-MM-DD-HHMM.md` so same-day sessions never collide | Every session (`/wrap-session`) |

Everything except `reports/` is a **living document** — edited in place. History lives in git and
in the reports.

## FEATURE_REGISTER.md

One table, one row per item:

`| ID | Item | Type | Phase | Status | Commit(s) | Notes |`

- **ID:** `F-1`, `F-2`, … sequential forever, never reused.
- **Type:** feature · fix · bug · refactor · idea
- **Phase:** the phase it's slotted into, or `later`.
- **Status:** `IDEA → PLANNED → WIP → DONE` (or `DROPPED`). One item `WIP` at a time.
- **Commit(s):** filled in at DONE — the short hash(es) that delivered it. This is the checkoff.
- **Rule — the worth-remembering test:** if losing the thought would cost something, it gets a row
  the moment it comes up (IDEA is enough), even for "later." Pure speculation that neither of us
  would miss doesn't need a row. When unsure, register — a row costs one line.

## DECISIONS.md

Two sections:

- **Needs you** — full entries: `D-n`, status, what it blocks, context, options with trade-offs,
  and a recommendation. An entry the user can't act on in 30 seconds isn't finished.
- **Decision Log** — one line each for decisions Claude made autonomously (tier permitting):
  date, what, why in a clause. This is what makes L1 autonomy auditable.

## CHORES.md

One table: `| ID | Chore | Blocks | Status | Notes |`, IDs `C-n`. Each chore must be executable
without further thought — exact steps and values in the Notes (or a short block under the table
for multi-step ones). Statuses `TODO / DONE`.

## Rules for the agent

1. **One PRIMARY item WIP at a time.** It may carry bounded subtasks (F-12a, F-12b) when the work
   genuinely splits — bounded means enumerable up front, not discovered forever. A second
   *problem* found mid-task → new row, keep going.
2. **Log immediately, not at wrap-up.** Chores, decisions, and feature ideas are worthless if they
   surface at the end of the session.
3. **Don't create documents.** The files above are the whole system — no NOTES.md, no TODO.md, no
   parallel trackers. New doc types get proposed to the user first.
4. **Reports are for handoff.** Write what the next session (or the user reviewing a phase) needs.
   Git is the changelog; don't duplicate it.
5. `DATABASE.md` lies to nobody: if the schema changed and the doc didn't, the task isn't done.

6. **`python verify.py` before every wrap and every phase close.** Failures get fixed or
   registered — never silently wrapped over.
