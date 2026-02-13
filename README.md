# Topographic Maps → Terrain STL (with roads)

Generate printable terrain STL files from a drawn polygon or uploaded shapefile.
Optionally fetch OpenStreetMap road centerlines and carve them into the terrain.

## Features

- Draw a polygon on a Leaflet map, or upload a zipped shapefile
- Download terrain-only **STL**
- Download a **ZIP bundle** containing:
  - raw terrain STL
  - carved terrain STL (roads recessed)
  - roads centerlines GeoJSON
- Road classes: `motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `residential`
- Per-road-class carve width + depth controls

## Requirements

- Python 3.10+ (Windows, macOS, or Linux)
- An **OpenTopography API key**

## Setup

### 1) Configure your OpenTopography API key

Set an environment variable:

- `OPEN_TOPO_API_KEY=<your_key>`

Or create a local file at the repo root:

- `API_KEY.txt`

> Note: `API_KEY.txt` is ignored by git and won’t be committed.

### 2) Install Python dependencies

Create and activate a virtualenv, then install requirements:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r WEBAPP/requirements.txt
```

### 3) Run the server

```bash
python -m uvicorn WEBAPP.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- http://127.0.0.1:8000

## Windows quick start

You can also use:

- `startup.bat`

It creates `.venv`, installs requirements, loads `API_KEY.txt` into `OPEN_TOPO_API_KEY` (if set), and starts the server.

## Notes / troubleshooting

- Installing `geopandas` / `rasterio` on Windows can be the hardest part.
  If `pip install` fails, consider using **conda-forge** for geospatial deps.
- Road fetches use the public Overpass API and may occasionally be slow or rate-limited.

## Repo layout

- `WEBAPP/` – FastAPI backend + static frontend
- `WEBAPP/static/` – UI assets (Leaflet app)
- `openapi.json` – API schema snapshot

## License

Add a license file (MIT, Apache-2.0, etc.) if you plan to share publicly.

