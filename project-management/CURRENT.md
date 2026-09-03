# Current

**Phase:** C — feature growth
**Register item:** F-21 — squares (GPS-centered square form + square-constrained draw tool)
**Status:** WIP on `feat/F-21-square-point` (off `main`; user merged the F-11..F-20 stack).

## Right now
Building F-21: (a) Draw-mode form — paste "lat, lon" + side length km → square centered on the
point; (b) `L.Draw.Square` toolbar tool (rectangle subclass, pixel-space constraint).
F-7 pulled forward (user-approved): global DEM datasets in the dropdown — the F-21 test point
50.86047, -115.55197 is in Alberta and USGS datasets don't cover it.

## Blocked on
Yours: C-1 (rotate API key), D-1/D-2. Re-slice the RI STL to confirm F-16/F-19/F-20 end-to-end.
Confirm 2 mm base default (F-17).

## Next
F-21 → F-7 → user click-through with the Alberta point → Phase B correctness (D-1/D-2 → F-2,
F-4, F-5) or F-6 3D preview.

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
