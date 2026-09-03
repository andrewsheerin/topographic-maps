# Current

**Phase:** C — feature growth
**Register items:** F-22 (water) + F-23 (colored 3MF) + F-24 (Overpass UA/endpoint fix) — DONE.
**Status:** branch stack, tip `fix/F-24-overpass-endpoints` ← `feat/F-23-color-3mf` ←
`feat/F-22-water` ← `feat/F-7-dem-datasets` ← `feat/F-21-square-point` ← `main`.

## Right now
F-22 DONE (6fe8c96): per-lake flatten + stream carve, Water panel, all exports. F-23 DONE
(983f8a2): colored 3MF (Materials ext, no new dep), /api/generate-3mf, required science fields.
40/40 tests, verify green. Water constants user-confirmed; F-7 was already done last session.

## Blocked on
Yours: merge the stack (tip `fix/F-24-overpass-endpoints`); re-click water + 3MF on the Alberta
square (50.86047, -115.55197 + COP30 — Overpass fetch now live-verified: 6 lakes, 110 streams)
and import the 3MF in Bambu Studio; D-1/D-2 (untouched); C-1 (rotate API key); confirm
SRTM/NASADEM coverage labels (F-7).

## Next
Revisit stream/base coloring in 3MF if AMS behaviour warrants; Phase B correctness
(D-1/D-2 → F-2, F-4, F-5); F-6 3D preview.

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
