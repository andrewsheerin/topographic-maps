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

# Stream carve widths per OSM waterway class and shared depth, print mm —
# same convention as ROAD_WIDTHS_MM: real stream widths are far below one
# extrusion width at print scale, so print-mm values are used instead.
# Wider classes = downstream, so streams taper toward the headwaters (F-28
# W-4). Source: user-provided 2026-09-03.
STREAM_WIDTHS_MM = {
    "river": 1.6,
    "stream": 1.0,
    "brook": 0.6,
}
STREAM_DEPTH_MM = 0.4

# V-profile cross-section for stream carves (F-28 W-3): (width_fraction,
# depth_fraction) rings applied shallow-to-deep, same ordering contract as
# build_carve_plan. Source: user-provided 2026-09-03.
STREAM_PROFILE = ((1.00, 0.35), (0.60, 0.70), (0.30, 1.00))

# Anti-aliasing factor for the stream carve (F-28 W-2): the buffer is
# rasterized on a grid this many times finer than the working DEM, then
# area-averaged back to a fractional depth. Source: user spec 2026-09-03.
STREAM_AA_FACTOR = 4

# Minimum renderable feature width, print mm: one extrusion line of a 0.4 mm
# nozzle. Stream widths below max(cell size, this) are clamped (F-28 W-5).
# Source: user-provided 2026-09-03.
EXTRUSION_LINE_WIDTH_MM = 0.42

# Ocean threshold, metres (F-33): cells at or below this flatten to a
# recessed sea plane and take the water color. Headroom over COP30's radar
# noise at ~0 m without swallowing coastal lowland.
# Source: user-confirmed 2026-09-04.
SEA_LEVEL_MAX_M = 0.5

# Percentile of the in-polygon DEM samples used as a lake's plane elevation.
# COP30 is radar-derived and noisy/void over open water; a low percentile
# rejects shoreline and noise highs. Source: user spec 2026-09-03.
WATER_LEVEL_PERCENTILE = 10

# OSM tag selections for water features. Source: user spec 2026-09-03
# (brook added with F-28 W-4; note it is rare in OSM tagging).
WATER_POLYGON_VALUES = ("lake", "reservoir", "pond")  # natural=water + water=…
WATERWAY_VALUES = ("river", "stream", "brook")  # waterway=…

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
