# Feature Register

Every feature, fix, bug, refactor, and idea — including ones planned for later. If it was
discussed, it has a row. Checkoff = Status DONE + the commit hash(es) that delivered it.

Statuses: `IDEA → PLANNED → WIP → DONE` · `DROPPED`. One WIP at a time. IDs never reused.

| ID | Item | Type | Phase | Status | Commit(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| F-1 | Migrate to app-factory (L1): restructure into backend/ + frontend/, React rewrite, PM system | refactor | A | DONE | c4dda13 | Behaviour-preserving. verify.py green; server serves the built SPA. |
| F-2 | Road carve overlaps accumulate depth instead of deepest-wins | bug | B | PLANNED | | `core/mesh.carve_roads` uses `-=`; contradicts its own docstring + `test_carve_order`. Blocked by D-2. |
| F-3 | Shapefile upload is broken — UI calls `/api/upload-shapefile`, no such endpoint | bug | B | PLANNED | | Needs a backend multipart endpoint returning `{polygon_geojson}` (python-multipart already pinned). |
| F-4 | Reconcile per-class road-width defaults (constants vs UI/RoadEtchParams) | fix | B | PLANNED | | Blocked by D-1. |
| F-5 | Tests for mesh scale + carve against a user-verified worked example | test | B | PLANNED | | Science-integrity requires calc tests; extend the two existing carve tests. |
| F-6 | In-browser 3D preview of the generated mesh | idea | later | IDEA | | Would remove the download-to-inspect loop. |
| F-7 | Surface more DEM datasets in the UI (COP30/SRTM/NASADEM/AW3D30) | idea | later | IDEA | | Already wired in `DATASET_ALIASES`; just not in the dropdown. |
| F-8 | Polygon drawing aborts at the 3rd vertex | bug | C | DONE | acb749d | Leaflet.draw 1.0.4 `readableArea` assigns an undeclared `type` → ESM strict-mode throw. Patched in `lib/leafletDrawFix.js`. |
| F-9 | Add rectangle + circle draw tools (circle → polygon for DEM) | feature | C | DONE | acb749d | Squares via the rectangle tool; circle → 64-gon polygon. |

## Dropped

| ID | Item | Why |
| --- | --- | --- |
