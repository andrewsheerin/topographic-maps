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
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]
