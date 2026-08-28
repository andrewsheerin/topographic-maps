# Feature Register

Every feature, fix, bug, refactor, and idea — including ones planned for later. If it was
discussed, it has a row. Checkoff = Status DONE + the commit hash(es) that delivered it.

Statuses: `IDEA → PLANNED → WIP → DONE` · `DROPPED`. One WIP at a time. IDs never reused.

| ID | Item | Type | Phase | Status | Commit(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| F-1 | Migrate to app-factory (L1): restructure into backend/ + frontend/, React rewrite, PM system | refactor | A | DONE | c4dda13 | Behaviour-preserving. verify.py green; server serves the built SPA. |
| F-2 | Road carve overlaps accumulate depth instead of deepest-wins | bug | B | PLANNED | | `core/mesh.carve_roads` uses `-=`; contradicts its own docstring + `test_carve_order`. Blocked by D-2. |
| F-3 | Shapefile upload — UI calls `/api/upload-shapefile` | bug | C | DONE | e3efa84 | Endpoint added: zip → WGS84 polygon; missing-CRS rejected. Tests + live-verified. |
| F-4 | Reconcile per-class road-width defaults (constants vs UI/RoadEtchParams) | fix | B | PLANNED | | Blocked by D-1. |
| F-5 | Tests for mesh scale + carve against a user-verified worked example | test | B | PLANNED | | Science-integrity requires calc tests; extend the two existing carve tests. |
| F-6 | In-browser 3D preview of the generated mesh | idea | later | IDEA | | Would remove the download-to-inspect loop. |
| F-7 | Surface more DEM datasets in the UI (COP30/SRTM/NASADEM/AW3D30) | idea | later | IDEA | | Already wired in `DATASET_ALIASES`; just not in the dropdown. |
| F-10 | Vectorize `dem_to_mesh` face construction | refactor | C | DONE | 53f1e20 | Became visibly slow with state outlines (F-14); pulled forward. 1.3M-cell grid meshes+seals in ~18s. |
| F-8 | Polygon drawing aborts at the 3rd vertex | bug | C | DONE | acb749d | Leaflet.draw 1.0.4 `readableArea` assigns an undeclared `type` → ESM strict-mode throw. Patched in `lib/leafletDrawFix.js`. |
| F-9 | Add rectangle + circle draw tools (circle → polygon for DEM) | feature | C | DONE | acb749d | Squares via the rectangle tool; circle → 64-gon polygon. |
| F-11 | Area step redesign: 3 modes — upload (shp zip + GeoJSON), draw, TIGER city/town picker | feature | C | DONE | 3666b87, 3456f64 | TIGER county subdivisions, mirrored from swpt-app F-79. Tests 17/17; smoke-tested on real RI/MA data; browser click-through pending user run. |
| F-12 | Hooks in `.claude/settings.json` used cwd-relative paths — one `cd backend` wedged the whole session | bug | C | DONE | 1639a57 | Both hook commands now absolute paths (user-approved). Takes full effect next session. |
| F-13 | Land-clipped area polygons: keep true (Multi)Polygon boundaries for places + uploads, no convex hull | feature | C | DONE | 5467b90 | Verified: Gosnold 5 parts, Nantucket 3. Roads query hull ring + clips back to true boundary. |
| F-14 | State outline as selectable area (Census CB 2023 state boundaries, `states` layer in gpkg) | feature | C | DONE | d120055 | "Use the whole state outline" button. Large states may exceed OpenTopography area limits — surfaced as API error. |
| F-15 | Place picker shows the full result list — remove limit/offset pagination and "Show more" | fix | C | DONE | 49e14d3 | WI = 1913 rows in one scrollable list. |
| F-16 | STL not watertight: `add_base` concatenated two open sheets (no walls, no inverted bottom) → slicers drop regions | bug | C | DONE | 161b31a | Symptom: half of RI missing at slice time. Now a sealed manifold (Euler 2, outward normals, exact slab volume verified); non-watertight export refused. |
| F-17 | Base thickness in print millimetres (was real-world metres) — consistent with the page's other units | fix | C | DONE | 9d4f298 | User-directed unit change. Old 50 m default gave paper-thin bases at state scale (0.13 mm for RI). UI default 2 mm — user to confirm. |
| F-18 | Remove the buffer-distance parameter | fix | C | DONE | 01fe0f3 | User approved. Field, API param, and pipeline buffer dropped; buffer(0) kept solely as validity normalization after reprojection. |
| F-19 | Crop the mesh to the area boundary (state-shaped prints); STL always sealed — "Add base" checkbox removed | feature | C | DONE | d3d24a5 | Slicer can't slice the no-base open sheet; flat water plate wastes the bed. Walls along shoreline, islands = separate solids, DEM nodata holes stay visible. |
| F-20 | RI state outline hits the watertight guard: diagonal (corner-touching) cells make the seal non-manifold | bug | C | WIP | | Pinch vertex → vertical wall edge shared by 4 faces. Fix: erode one cell of each diagonal pair before meshing. |

## Dropped

| ID | Item | Why |
| --- | --- | --- |
