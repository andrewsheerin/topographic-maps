# Current

**Phase:** C — feature growth
**Register item:** F-11 — Area step redesign (3 modes: upload shp/GeoJSON · draw · TIGER place picker)
**Status:** WIP on `feat/F-11-area-modes` (stacked on `feat/F3-shapefile-upload`).

## Right now
F-11 DONE (3-mode Area UI + TIGER picker; tests 17/17, verify green, RI+MA data fetched).
F-12 DONE (hook paths absolute). startup.bat reworked @ 9ea0440. Full TIGER dataset fetched:
35,400 subdivisions, 51 states, 0 failures (data/tiger/subdivisions.gpkg, 68 MB).

## Blocked on
Yours: C-1 (rotate API key), D-1/D-2, merging stacked branches
(feat/F3-shapefile-upload → feat/F-11-area-modes) to `main`. Browser click-through of the
new Area UI still worth a quick look.

## Next
Merge the stack, click through the 3 modes, then Phase B correctness (D-1/D-2 → F-2, F-4, F-5)
or more features (F-7 DEM datasets, F-6 3D preview).

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
