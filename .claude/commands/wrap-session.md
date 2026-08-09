---
description: End the session — report with next-session prompt, reset CURRENT
---

Wrap up.

**1. Run `python verify.py`.** Fix failures or register them with the user's knowledge — never wrap over a red verify silently.

**2. Commit** anything uncommitted (`<type>(F-n): summary`). No push. Name the branch(es) ready for me to merge.

**3. Sync the register:** statuses moved, commit hashes filled for anything DONE. Ideas that came
up in conversation but never got rows → add them now as IDEA.

**4. Write `reports/sessions/<YYYY-MM-DD-HHMM>.md`:**
- What happened this session, in 3–5 sentences
- Register items touched (F-n: status), decisions made or raised, chores added
- **Plan-ahead:** tasks for future sessions and things to think about later — concrete enough to
  act on, each either pointing at a register row or getting one
- **A ready-to-paste opening prompt for next session** — written for a Claude with no memory of
  today: what we're doing, which register item is next, what to read first, any gotcha. This is
  the most important section.

**5. Reset `CURRENT.md`** to phase / next item / blockers. Under 20 lines.

**6. Tell me** in a few lines: what happened, what's blocked on me (D-n / C-n), what's next — and
any permission I could have granted that would have made today faster.
