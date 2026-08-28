"""Fetch US Census TIGER County Subdivisions into a GeoPackage for the Area
place picker (F-11; ported from swpt-app F-79/D-28).

County subdivisions (COUSUB — towns/townships; MCDs in the ~20 strong-MCD
states, statistical CCDs elsewhere) are used instead of TIGER Places because
they tile each county with no unincorporated holes. Downloads the Census
Cartographic Boundary cousub files (generalized 1:500k) per state, joins county
NAMEs from the national county CB file (subdivision names repeat heavily),
normalizes to `[geoid, name, state, county, geometry]` in EPSG:4326, and writes
`data/tiger/subdivisions.gpkg` (gitignored; regenerate with this script).
Public domain (US Census). Provenance: `data/DATA_SOURCES.md`.

Run from the repo root with the repo venv (default = all 50 states + DC):
    .venv/Scripts/python.exe backend/scripts/fetch_tiger_subdivisions.py
    .venv/Scripts/python.exe backend/scripts/fetch_tiger_subdivisions.py --states MA RI
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from core.places import SUBDIVISIONS_GPKG  # noqa: E402

# Census Cartographic Boundary files, 2023 vintage, 1:500k generalization.
COUSUB_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_{fips}_cousub_500k.zip"
)
COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
STATE_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"
_HEADERS = {"User-Agent": "topo2stl-tiger-subdivisions/1.0 (F-11)"}

STATE_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}

# Census fills coastal/water gaps with "County subdivisions not defined" records;
# they aren't selectable areas and are dropped from the picker dataset.
_PSEUDO_NAME_MARKERS = ("not defined",)


def _download(url: str, attempts: int = 3) -> bytes:
    for i in range(attempts):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=180)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException:
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def _read_zip_shapefile(data: bytes) -> gpd.GeoDataFrame:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as fh:
        fh.write(data)
        zip_path = fh.name
    return gpd.read_file(f"zip://{zip_path}")  # GDAL reads the shapefile inside the zip


def _county_names() -> dict[str, str]:
    """(STATEFP+COUNTYFP 5-digit) -> county NAME, from the national county CB file."""
    gdf = _read_zip_shapefile(_download(COUNTY_URL))
    return {
        f"{row['STATEFP']}{row['COUNTYFP']}": str(row["NAME"])
        for _, row in gdf.iterrows()
    }


def _fetch_state(abbr: str, county_names: dict[str, str]) -> gpd.GeoDataFrame:
    fips = STATE_FIPS[abbr]
    gdf = _read_zip_shapefile(_download(COUSUB_URL.format(fips=fips)))
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Drop water-only pseudo-subdivisions (no land area).
    if "ALAND" in gdf.columns:
        gdf = gdf[gdf["ALAND"].fillna(0).astype("int64") > 0]

    county = (gdf["STATEFP"].astype(str) + gdf["COUNTYFP"].astype(str)).map(
        county_names
    )
    out = gpd.GeoDataFrame(
        {
            "geoid": gdf["GEOID"].astype(str),
            "name": gdf["NAME"].astype(str),
            "state": abbr,
            "county": county.fillna("").astype(str),
        },
        geometry=gdf.geometry,
        crs="EPSG:4326",
    )
    mask = ~out["name"].str.contains(
        "|".join(_PSEUDO_NAME_MARKERS), case=False, na=False
    )
    return out[mask]


def _fetch_state_outlines() -> gpd.GeoDataFrame:
    """State boundaries (F-14) from the national state CB file, normalized to
    the same schema as subdivisions (county empty)."""
    gdf = _read_zip_shapefile(_download(STATE_URL))
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf["STUSPS"].isin(STATE_FIPS.keys())]  # 50 states + DC only
    return gpd.GeoDataFrame(
        {
            "geoid": gdf["GEOID"].astype(str),
            "name": gdf["NAME"].astype(str),
            "state": gdf["STUSPS"].astype(str),
            "county": "",
        },
        geometry=gdf.geometry,
        crs="EPSG:4326",
    )


def _fetch_subdivisions(states: list[str]) -> gpd.GeoDataFrame:
    print("Fetching national county names for disambiguation ...")
    county_names = _county_names()
    print(f"  {len(county_names)} counties")

    frames = []
    for abbr in states:
        try:
            gdf = _fetch_state(abbr, county_names)
            frames.append(gdf)
            print(f"  {abbr}: {len(gdf)} subdivisions")
        except Exception as e:  # noqa: BLE001 — report + continue so one bad state doesn't abort all
            print(f"  {abbr}: FAILED ({e})")

    if not frames:
        raise SystemExit("No subdivisions fetched.")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch TIGER County Subdivisions + state outlines into a GeoPackage"
    )
    ap.add_argument(
        "--states", nargs="+", help="State abbreviations (default: all 50 + DC)"
    )
    ap.add_argument(
        "--only",
        choices=["all", "subdivisions", "states"],
        default="all",
        help="Which layer(s) to (re)fetch; the other layer in the gpkg is kept",
    )
    args = ap.parse_args()

    states = (
        [s.upper() for s in args.states] if args.states else list(STATE_FIPS.keys())
    )
    unknown = [s for s in states if s not in STATE_FIPS]
    if unknown:
        raise SystemExit(f"Unknown state abbreviations: {unknown}")

    SUBDIVISIONS_GPKG.parent.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "subdivisions"):
        subs = _fetch_subdivisions(states)
        subs.to_file(SUBDIVISIONS_GPKG, layer="subdivisions", driver="GPKG")
        print(f"wrote layer 'subdivisions'  ({len(subs)} subdivisions)")

    if args.only in ("all", "states"):
        outlines = _fetch_state_outlines()
        outlines.to_file(SUBDIVISIONS_GPKG, layer="states", driver="GPKG")
        print(f"wrote layer 'states'  ({len(outlines)} state outlines)")

    print(f"\ndone: {SUBDIVISIONS_GPKG}")


if __name__ == "__main__":
    main()
