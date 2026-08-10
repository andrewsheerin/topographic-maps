# Architecture & Tech Stack

*As built — not aspirational. Update when reality changes, in the same commit where it changes.*

## System overview

Single-user local app, two processes in development, one in "just run it" mode.

- **Frontend** (React + Vite) renders a Leaflet map + control panel. It calls the backend over
  relative `/api/*` paths. In dev, the Vite server (`:5173`) proxies `/api` to the backend
  (`:8020`); for local use, `npm run build` emits `frontend/dist`, which the backend serves at `/`.
- **Backend** (FastAPI, `:8020`) exposes three endpoints and orchestrates the pipeline. It holds no
  state; each request writes to a fresh temp directory.
- **External services:** OpenTopography (DEM GeoTIFFs) and OpenStreetMap via Overpass (road
  centerlines). The OpenTopography API key is the only secret, loaded from a gitignored `.env`.

Pipeline (per request): polygon (WGS84) → download DEM for bounds → clip to polygon → reproject to
the polygon's UTM zone → buffer + re-clip in metres → read to array → mesh to millimetres (relief-
only Z) → optional base → STL. For the bundle, roads are fetched (WGS84), reprojected+clipped to
the same UTM footprint, and recessed into a second mesh; raw STL + carved STL + roads GeoJSON are
zipped.

## Tech stack

| Layer | Choice | Why / notes |
| --- | --- | --- |
| Frontend | React 18 + Vite (plain JS/JSX) | SPA; CSS Modules + `tokens.css`. Raw Leaflet + leaflet-draw (not react-leaflet) for a faithful port. |
| Backend | FastAPI + uvicorn | Thin routers; all logic in `core/` (no FastAPI imports there). |
| Database | None | Stateless; per-request temp dirs. See DATABASE.md. |
| Geospatial | rasterio, geopandas, shapely, pyproj, numpy, trimesh | DEM I/O, reprojection, geometry, meshing. |
| External services | OpenTopography (DEM), Overpass/OSM (roads) | Fetched live per request; see DATA_SOURCES.md. |
| Styling | CSS Modules + `frontend/src/styles/tokens.css` | Dark, one acid-green accent. |
| Secrets | `python-dotenv` → `.env` (`OPEN_TOPO_API_KEY`) | Never committed. |

## Key structural decisions

- `core/` never imports FastAPI — the science is callable from tests/notebooks (Decision Log 2026-08-09).
- `dem_stl.py` split into `core/{geometry,dem,roads,mesh,pipeline}.py` + `constants.py` + `config.py` (Log 2026-08-09).
- Domain constants centralized in `constants.py` with sourced comments (`science-integrity.md`).
- Frontend built and served by FastAPI at `/`; Vite proxy for dev (Log 2026-08-09).
- Dead DXF export path and `ezdxf`/`osm2geojson` deps dropped (Log 2026-08-09).

## Repo layout notes

- `backend/` is the app root: run with `uvicorn main:app --app-dir backend`. `conftest.py` puts it
  on `sys.path` so tests import `core.*` directly. `ruff.toml` pins the lint rule set.
- `backend/core/pipeline.py` is orchestration only; the per-step functions live in the sibling core
  modules.
- `frontend/` is a standard Vite app; `node_modules/` and `dist/` are gitignored.
- `data/` and `infra/` are empty by design at L1 (no committed datasets, no infra).
