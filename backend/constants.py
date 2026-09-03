"""Domain constants and factors for terrain/road processing.

Every value here is a domain assumption. Per `.claude/rules/science-integrity.md`
these belong to the author (PhD environmental engineer) — they are carried over
verbatim from the original app and must not be changed without a decision logged
in `project-management/DECISIONS.md`. Sources are named inline.
"""

# --- Units ---------------------------------------------------------------

# Millimetres per metre. Exact by definition.
METERS_TO_MM = 1000.0

# --- Road classes (carve order + fallback widths) ------------------------

# The canonical set of OSM `highway` classes the app carves, and the fallback
# print-width (mm) used only when the request omits a per-class width. The UI
# normally supplies widths/depths per request (see models.schemas.RoadEtch), so
# these act as defaults, not the operating values.
#
# NOTE (open decision D-1): these fallback widths differ from the UI defaults in
# the frontend / RoadEtchParams. Carried over from the original app as-is.
# Source: original app author defaults.
ROAD_WIDTHS_MM = {
    "motorway": 4.0,
    "trunk": 3.0,
    "primary": 2.0,
    "secondary": 0.5,
    "tertiary": 0.35,
    "residential": 0.25,
}

# Fallback recess depth (mm) when a request omits a per-class depth.
# Source: original app author default.
RECESS_DEPTH_MM = 2.0

# Sanity caps applied to per-request carve params to reject absurd input.
# Source: original app author defaults.
CARVE_WIDTH_CAP_MM = 50.0
CARVE_DEPTH_CAP_MM = 20.0

# --- Water: lakes + streams (F-22) ---------------------------------------

# Lake-surface recess below each lake's flattened plane, print mm.
# Source: user-provided 2026-09-03.
WATER_RECESS_MM = 0.6

# Stream carve width/depth, print mm — same convention as ROAD_WIDTHS_MM: real
# stream widths are far below one extrusion width at print scale, so print-mm
# values are used instead. Source: user-provided 2026-09-03.
STREAM_WIDTH_MM = 1.0
STREAM_DEPTH_MM = 0.4

# Percentile of the in-polygon DEM samples used as a lake's plane elevation.
# COP30 is radar-derived and noisy/void over open water; a low percentile
# rejects shoreline and noise highs. Source: user spec 2026-09-03.
WATER_LEVEL_PERCENTILE = 10

# OSM tag selections for water features. Source: user spec 2026-09-03.
WATER_POLYGON_VALUES = ("lake", "reservoir", "pond")  # natural=water + water=…
WATERWAY_VALUES = ("stream", "river")  # waterway=…

# --- Mesh / print guards -------------------------------------------------

# Maximum vertical exaggeration accepted; higher values are clamped.
# Source: original app author default.
Z_EXAG_MAX = 20.0

# If computed relief exceeds this, treat it as a unit/CRS error and fail loudly
# rather than emit a broken mesh. Metres. Source: original app author default.
RELIEF_SANITY_MAX_M = 20000.0

# --- DEM datasets --------------------------------------------------------

# Default DEM product when none is specified. Source: original app.
DEFAULT_DATASET = "COP30"

# UI label -> OpenTopography dataset id. USGS products are served by the
# /API/usgsdem endpoint (datasetName=...); the rest by /API/globaldem
# (demtype=...). See core.dem for the endpoint split.
DATASET_ALIASES = {
    "USGS10m": "USGS10m",
    "USGS30m": "USGS30m",
    "SRTMGL1": "SRTMGL1",
    "COP30": "COP30",
    "COP90": "COP90",
    "NASADEM": "NASADEM",
    "AW3D30": "AW3D30",
}

# Datasets that must use the USGS DEM endpoint rather than the global one.
USGSDEM_DATASETS = {"USGS10m", "USGS30m"}

# --- Overpass (OSM roads) ------------------------------------------------

# Public Overpass endpoints, tried in order with retries (see core.roads).
# overpass.nchc.org.tw removed 2026-09-03 (F-24): its DNS no longer resolves.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Identifying User-Agent per OSM usage policy. overpass-api.de rejects the
# default python-requests agent with HTTP 406 (verified 2026-09-03, F-24).
OVERPASS_USER_AGENT = "TOPO2STL/1.0 (personal terrain-print app)"
