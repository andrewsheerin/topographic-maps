# Current

**Phase:** C — feature growth
**Register items:** F-28, F-31..F-37 (scale off-by-one fix) — DONE.
**Status:** branch stack, tip `fix/F-37-scale-off-by-one` ← `feat/F-36-clip-dims` ←
`feat/F-35-frame` ← … ← `main` (main still at F-20; the whole stack is unmerged).

## Right now
F-36 DONE (4a24efd): Straight edges gained shape control — Largest fit / Square / Custom W×H
with exact mm (dims govern scale; Target max ignored + noted). Square is cell-exact; clip
centers on the drawn area. 65/65 tests, verify green. Earlier this arc: F-35 clean-edge clip
(reworked from frame), F-34 auto resolution, F-33 ocean, F-32 color bands, F-31 named streams.

## Blocked on
Yours: merge the stack (tip `fix/F-37-scale-off-by-one`); hard-refresh; regenerate Killington
with Straight edges → Square → 100 (exact 100×100 for the wooden frame);
D-1/D-2; C-1; SRTM/NASADEM labels; email in Overpass UA?

## Next
F-29 Strahler taper if stream prints warrant; Phase B correctness (D-1/D-2 → F-2, F-4, F-5);
F-6 3D preview.

---
*Under 20 lines. Detail belongs in the register. Reset at the end of each session.*
