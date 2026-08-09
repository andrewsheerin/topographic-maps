# TOPO2STL — Topographic Maps → Terrain STL

Draw or upload an area, pull a DEM (with optional OpenStreetMap roads), and export a
3D-printable terrain STL. Runs locally.

- Draw a polygon on a Leaflet map, or upload a zipped shapefile
- Download terrain-only **STL**
- Download a **ZIP bundle**: raw terrain STL + carved terrain STL (roads recessed) + roads GeoJSON
- Road classes `motorway · trunk · primary · secondary · tertiary · residential`, each with a
  per-class carve width + depth (mm)

## Stack

- **Backend:** FastAPI (Python) — `backend/`
- **Frontend:** React + Vite — `frontend/`
- **Data source:** OpenTopography DEMs + OSM roads (Overpass)

## Requirements

- Python 3.10+
- Node.js 18+ (for the React frontend)
- A free **OpenTopography API key** → https://portal.opentopography.org/

## Configure your API key

Copy `.env.example` to `.env` (gitignored) and set your key:

```
OPEN_TOPO_API_KEY=your_key_here
```

## Quick start (Windows)

```
startup.bat
```

It creates `.venv`, installs backend deps, builds the frontend, and serves the app at
http://127.0.0.1:8000.

## Manual setup

**Backend**

```bash
python -m venv .venv
. .venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r backend/requirements.txt
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

**Frontend — development** (hot reload, proxies `/api` to the backend on :8000)

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

**Frontend — build** (FastAPI then serves it at `/`)

```bash
cd frontend
npm run build          # outputs frontend/dist, served by the backend at http://127.0.0.1:8000
```

## Health check

```bash
python verify.py       # backend format/lint/tests + frontend build/lint
```

## Repo layout

```
backend/     FastAPI app — main.py, api/ (routers), core/ (DEM/roads/mesh/pipeline),
             models/ (pydantic), constants.py, config.py, tests/
frontend/    React + Vite app (Leaflet map, controls)
data/        input/reference datasets (DATA_SOURCES.md logs provenance)
project-management/   living plan, feature register, decisions, chores, reports
.claude/     tier + rules + hooks + commands (see GUIDE.md)
```

See `GUIDE.md` for how the project is run day to day.

## Notes / troubleshooting

- Installing `geopandas` / `rasterio` on Windows can be the hardest part. If `pip install`
  fails, use **conda-forge** for the geospatial stack.
- Road fetches use the public Overpass API and may occasionally be slow or rate-limited.

## License

Add a license file (MIT, Apache-2.0, …) if you plan to share publicly.
