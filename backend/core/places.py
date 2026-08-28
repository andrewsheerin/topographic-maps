"""Named-area lookup for the Area picker (F-11; ported from swpt-app F-79).

Reads the TIGER County Subdivisions GeoPackage produced by
`scripts/fetch_tiger_subdivisions.py` and returns selectable towns/cities with
their bbox, plus the full polygon geometry on selection. Provenance in
`data/DATA_SOURCES.md`.

Dataset schema: columns `geoid`, `name`, `state` (2-letter abbr), `county`
(county name, for disambiguation — subdivision names repeat heavily),
`geometry` (polygons, EPSG:4326).
"""

import math
from functools import lru_cache
from pathlib import Path

import geopandas as gpd

import config
from core.shapefile import to_area_geometry

SUBDIVISIONS_GPKG = config.REPO_ROOT / "data" / "tiger" / "subdivisions.gpkg"

MISSING_DATASET_MSG = (
    "Place dataset not found. Fetch it once with: "
    ".venv/Scripts/python.exe backend/scripts/fetch_tiger_subdivisions.py"
)

# Approximate km per degree at the equator (WGS84); display-only bbox area for
# the picker list. Ported verbatim from swpt-app `boundaries/places.py`.
_KM_PER_DEG_LON_EQUATOR = 111.320
_KM_PER_DEG_LAT = 110.574


def _bbox_area_km2(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> float:
    mean_lat = math.radians((min_lat + max_lat) / 2.0)
    return (
        abs(max_lon - min_lon)
        * _KM_PER_DEG_LON_EQUATOR
        * math.cos(mean_lat)
        * abs(max_lat - min_lat)
        * _KM_PER_DEG_LAT
    )


@lru_cache(maxsize=4)
def _load_layer(path_str: str, layer: str) -> gpd.GeoDataFrame:
    """Load + cache a GeoPackage layer in WGS84. Cache keyed by (path, layer)
    (restart the server after re-fetching the dataset)."""
    gdf = gpd.read_file(path_str, layer=layer)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _summary(row) -> dict:
    geom = row.geometry
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in geom.bounds)
    county = str(row["county"]).strip()
    return {
        "geoid": str(row["geoid"]),
        "name": str(row["name"]),
        "state": str(row["state"]),
        "county": county or None,
        "bbox": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        },
        "area_km2": round(_bbox_area_km2(min_lon, min_lat, max_lon, max_lat), 1),
    }


def query_places(
    state: str | None = None,
    q: str | None = None,
    limit: int = 100,
    offset: int = 0,
    gpkg_path: Path = SUBDIVISIONS_GPKG,
) -> list[dict]:
    """Subdivisions filtered by state abbr and/or case-insensitive name search,
    sorted by name then county, windowed by offset/limit. Raises FileNotFoundError
    (with an actionable message) when the dataset hasn't been fetched."""
    if not gpkg_path.exists():
        raise FileNotFoundError(MISSING_DATASET_MSG)

    gdf = _load_layer(str(gpkg_path), "subdivisions")
    if state:
        gdf = gdf[gdf["state"].astype(str).str.upper() == state.upper()]
    if q:
        gdf = gdf[
            gdf["name"].astype(str).str.contains(q, case=False, na=False, regex=False)
        ]
    gdf = gdf.sort_values(["name", "county"]).iloc[offset : offset + int(limit)]

    return [
        _summary(row)
        for _, row in gdf.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    ]


def get_place(geoid: str, gpkg_path: Path = SUBDIVISIONS_GPKG) -> dict | None:
    """One subdivision by GEOID, with its boundary geometry (GeoJSON) for map
    display and DEM extraction. None when the GEOID is unknown.

    The true shoreline-clipped boundary is kept (F-13): coastal towns with
    islands come back as MultiPolygon — the STL covers land only, and the map
    shows exactly the area that gets meshed."""
    if not gpkg_path.exists():
        raise FileNotFoundError(MISSING_DATASET_MSG)

    gdf = _load_layer(str(gpkg_path), "subdivisions")
    match = gdf[gdf["geoid"].astype(str) == str(geoid)]
    if match.empty:
        return None
    row = match.iloc[0]
    if row.geometry is None or row.geometry.is_empty:
        return None
    return {**_summary(row), "geometry": to_area_geometry([row.geometry])}


MISSING_STATES_MSG = (
    "State outlines not in the dataset. Add them with: "
    ".venv/Scripts/python.exe backend/scripts/fetch_tiger_subdivisions.py --only states"
)


def get_state(abbr: str, gpkg_path: Path = SUBDIVISIONS_GPKG) -> dict | None:
    """One state outline by 2-letter abbreviation, with its true (Multi)Polygon
    boundary (F-14). None when the abbreviation is unknown."""
    if not gpkg_path.exists():
        raise FileNotFoundError(MISSING_DATASET_MSG)
    try:
        gdf = _load_layer(str(gpkg_path), "states")
    except Exception:
        raise FileNotFoundError(MISSING_STATES_MSG)

    match = gdf[gdf["state"].astype(str).str.upper() == str(abbr).upper()]
    if match.empty:
        return None
    row = match.iloc[0]
    if row.geometry is None or row.geometry.is_empty:
        return None
    return {**_summary(row), "geometry": to_area_geometry([row.geometry])}
