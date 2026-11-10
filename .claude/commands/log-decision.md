---
description: Route a decision — OPEN or PROPOSED entry for the user, or one line in the Decision Log; or record the user's answer
---

Route the decision at hand per the tier (`.claude/TIER.md`):

- **The user's call, and work waits on it** → `### D-n: question` under **Needs you** in
  `project-management/DECISIONS.md`: `**Status:** OPEN · **Raised:** date · **Blocks:** F-n`,
  2–3 sentences of context, an options table (upside, downside, reversibility) and your
  recommendation. Do the analysis first; a bare question isn't an entry. Then say what you'll
  do meanwhile.
- **The user's call, but you can keep moving on a provisional choice** → the same entry with
  `**Status:** PROPOSED` and `**Chose (provisional):** … · **Why:** … · **Reverse cost:** …`.
  Build on it; the user confirms or corrects.
- **Yours to make under this tier** → decide, add one line at the top of the **Decision Log**
  (date, decision, why), keep moving.

When in doubt about the lane, it's the user's.

**Recording an answer** (the user decides, or confirms a PROPOSED entry): append the full entry
plus the answer and date to `archive/DECISIONS-answered.md`; add
`| D-n | short title | RESOLVED | what was chosen | date |` under **Resolved**; delete the entry
from Needs you. A decision replaced by a later one is SUPERSEDED, with `by D-m` in Outcome.
