# Current

**Phase:** C — feature growth
**Register item:** F-11 — Area step redesign (3 modes: upload shp/GeoJSON · draw · TIGER place picker)
**Status:** WIP on `feat/F-11-area-modes` (stacked on `feat/F3-shapefile-upload`).

## Right now
F-11..F-16 + F-10 all DONE. Area UI (3 modes, true boundaries, state outlines, full lists);
STLs now watertight sealed solids (F-16 — fixes RI half-missing-at-slice); mesh build
vectorized. Dataset: subdivisions.gpkg (35,400 towns + 51 states). Tests 28/28, verify green.

## Blocked on
Yours: C-1 (rotate API key), D-1/D-2, merging the branch stack to `main` (tip:
`fix/F-18-remove-buffer`; each F-n branch stacks on the previous). Re-slice the RI STL to
confirm F-16 end-to-end. Also F-17/F-18: base now print-mm (default 2 mm — confirm), buffer
param removed, backend auto-reloads.

## Next
Merge the stack, click through the 3 modes, then Phase B correctness (D-1/D-2 → F-2, F-4, F-5)
or more features (F-7 DEM datasets, F-6 3D preview).

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
