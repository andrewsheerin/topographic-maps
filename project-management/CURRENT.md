# Current

**Phase:** C — feature growth
**Register items:** F-22..F-26 all DONE this session.
**Status:** branch stack, tip `feat/F-25-ui-toggles` ← `fix/F-26-overpass-rate` ←
`fix/F-24-overpass-endpoints` ← `feat/F-23-color-3mf` ← `feat/F-22-water` ← … ← `main`.

## Right now
F-25 DONE (05fe84d): Add roads/water/color toggles (all default off → no surprise Overpass
calls), Color section, Output last with STL / 3MF / ZIP. F-26 DONE (44b24b1): one water query,
Retry-After on 429. Earlier: F-22 water, F-23 colored 3MF, F-24 UA/endpoint fix.
45/45 tests, verify green.

## Blocked on
Yours: merge the stack (tip `feat/F-25-ui-toggles`); reload the app and retry the Alberta square
(50.86047, -115.55197 + COP30): water-only first, then color 3MF → Bambu import; D-1/D-2;
C-1 (rotate API key); confirm SRTM/NASADEM coverage labels; email in Overpass UA? (decision log).

## Next
Revisit stream/base coloring in 3MF; Phase B correctness (D-1/D-2 → F-2, F-4, F-5); F-6 3D preview.

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
