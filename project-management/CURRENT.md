---
pm: v3
profile: app
git_policy: main
pm_mode: committed
updated: 2026-11-09
branch: chore/pm-upgrade
phase: C
wip: —
blocked_on: D-1, D-2
---
# Current

## Right now
Moving to app-factory v3 (/pm-upgrade, 2026-11-09). The F-21…F-37 stack is merged: main is
`4fcf6d2`, 34 commits ahead of origin, not pushed yet. Phase C, feature growth; F-37 was the
last item. Verify is red on ruff (F-38), hidden until now because verify skipped ruff.

## Blocked on
Yours: D-1/D-2 (Phase B road widths, carve overlap); confirm D-5 (SRTM/NASADEM bounds) and D-6
(email in the Overpass UA); print-check F-36/F-37 — Killington, Straight edges → Square → 100,
expect exactly 100 × 100 mm.

## Next
F-38 ruff reds; F-29 Strahler taper if stream prints warrant; Phase B correctness (D-1/D-2 →
F-2, F-4, F-5); F-6 3D preview.
