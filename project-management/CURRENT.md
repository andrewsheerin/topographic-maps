# Current

**Phase:** C — feature growth
**Register items:** F-28, F-31, F-32, F-33, F-34 (auto resolution) — DONE.
**Status:** branch stack, tip `feat/F-34-auto-resolution` ← `feat/F-33-ocean` ←
`feat/F-32-color-bands` ← … ← `main` (main still at F-20; the whole stack is unmerged).

## Right now
F-28 DONE (f4289fa backend, 758bcd5 UI): dem_upsample 1–4 (bilinear, GDAL, NaN-safe),
anti-aliased fractional-depth carve (4× fine grid), V-profile rings, per-class widths
river/stream/brook 1.6/1.0/0.6 mm (user-confirmed), width floor = max(cell, 0.42 mm) clamped
backend + warned in UI, cell-size readout on the detail multiplier. 51/51 tests, verify green.
F-29 (Strahler) and F-30 (downsample nodata smear) registered for later.

## Blocked on
Yours: merge the stack (tip `feat/F-34-auto-resolution`); hard-refresh, then regenerate
Aquidneck with Ocean checked + Auto resolution on (expect ~180k triangles, not 15.7M, and a
blue bay); D-1/D-2; C-1; SRTM/NASADEM labels; email in Overpass UA?

## Next
F-29 Strahler taper if F-28 prints well; Phase B correctness (D-1/D-2 → F-2, F-4, F-5); F-6.

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
