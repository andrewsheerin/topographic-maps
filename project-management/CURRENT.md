# Current

**Phase:** C — feature growth
**Register item:** F-11 — Area step redesign (3 modes: upload shp/GeoJSON · draw · TIGER place picker)
**Status:** WIP on `feat/F-11-area-modes` (stacked on `feat/F3-shapefile-upload`).

## Right now
F-11/12/13/14/15 all DONE. Area UI: draw / upload (shp+GeoJSON) / city-town picker with true
land-clipped boundaries, state outlines, full un-paginated lists. Dataset: subdivisions.gpkg
(35,400 towns + 51 state outlines). Tests 24/24, verify green.

## Blocked on
Yours: C-1 (rotate API key), D-1/D-2, merging the branch stack to `main`
(F3-shapefile-upload → F-11-area-modes → F-13-area-boundaries → F-14-state-outline →
F-15-picker-full-list). Browser click-through of the new Area UI still worth a look.

## Next
Merge the stack, click through the 3 modes, then Phase B correctness (D-1/D-2 → F-2, F-4, F-5)
or more features (F-7 DEM datasets, F-6 3D preview).

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
