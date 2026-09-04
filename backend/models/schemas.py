"""Pydantic request models — the API boundary. All values are print-mm or metres
as named; see `.claude/rules/science-integrity.md` (units are explicit)."""

from typing import Any

from pydantic import BaseModel, Field

import constants


class RoadEtch(BaseModel):
    width_mm: float = 0.0
    depth_mm: float = 0.0


class StreamWidths(BaseModel):
    """Per-class stream carve widths, print mm (F-28 W-4): wider classes are
    downstream, so streams taper toward the headwaters. Defaults from
    constants (user-provided 2026-09-03)."""

    river: float = constants.STREAM_WIDTHS_MM["river"]
    stream: float = constants.STREAM_WIDTHS_MM["stream"]
    brook: float = constants.STREAM_WIDTHS_MM["brook"]


class WaterParams(BaseModel):
    """Lake flattening + stream carving (F-22). Depth/width are print mm;
    defaults come from constants (user-provided 2026-09-03)."""

    lakes: bool = False
    streams: bool = False
    # F-31: keep only waterways with a `name` tag (unnamed = mostly traced
    # gullies). Defaults on — the dense-network complaint that motivated it.
    named_only: bool = True
    recess_mm: float = constants.WATER_RECESS_MM
    stream_widths_mm: StreamWidths = StreamWidths()
    stream_depth_mm: float = constants.STREAM_DEPTH_MM


class RoadEtchParams(BaseModel):
    # NOTE: these per-class defaults differ from the frontend's default inputs
    # (see open decision D-1). Preserved from the original app. The UI normally
    # supplies explicit values, so these defaults are rarely exercised.
    motorway: RoadEtch = RoadEtch(width_mm=2.5, depth_mm=1.2)
    trunk: RoadEtch = RoadEtch(width_mm=2.0, depth_mm=1.1)
    primary: RoadEtch = RoadEtch(width_mm=1.6, depth_mm=1.0)
    secondary: RoadEtch = RoadEtch(width_mm=1.1, depth_mm=0.8)
    tertiary: RoadEtch = RoadEtch(width_mm=0.8, depth_mm=0.6)
    residential: RoadEtch = RoadEtch(width_mm=0.6, depth_mm=0.5)


_HEX = r"^#[0-9A-Fa-f]{6}$"


class ColorParams(BaseModel):
    """The 3MF palette in class order (F-23). All required — no default
    colors are chosen here."""

    water: str = Field(pattern=_HEX)
    forest: str = Field(pattern=_HEX)
    rock: str = Field(pattern=_HEX)
    snow: str = Field(pattern=_HEX)


class ColorBands(BaseModel):
    """Color-band parameters for the 3MF (F-23/F-25). treeline_m and
    snowline_m are scientific per-request values (they vary by mountain
    range) — required, no defaults (science-integrity)."""

    treeline_m: float
    snowline_m: float
    colors: ColorParams


class GenerateRequest(BaseModel):
    polygon_geojson: dict[str, Any]
    dem_dataset: str
    downsample: int
    # Bilinear DEM upsample (F-28 W-1): buys placement resolution for water
    # features; invents no terrain information. Cost is quadratic in triangles.
    dem_upsample: int = Field(default=1, ge=1, le=4)
    z_scale: float
    target_max_mm: float
    base_thickness_mm: float  # print millimetres (F-17); STL is always sealed (F-19)
    road_levels: list[str] = []
    road_etch: RoadEtchParams = RoadEtchParams()
    water: WaterParams = WaterParams()
    # Present when the "Add color" toggle is on (F-25): the ZIP bundle then
    # also contains the colored 3MF.
    color: ColorBands | None = None


class Generate3MFRequest(GenerateRequest):
    """Colored 3MF export: color is mandatory here."""

    color: ColorBands


class RoadsRequest(BaseModel):
    polygon_geojson: dict[str, Any]
    road_levels: list[str] = []


class BBox(BaseModel):
    """Geographic extent in WGS84 decimal degrees."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class PlaceSummary(BaseModel):
    """One TIGER county subdivision (town/city) in the picker list.
    `area_km2` is the approximate bbox area, display-only."""

    geoid: str
    name: str
    state: str
    county: str | None = None
    bbox: BBox
    area_km2: float


class PlaceDetail(PlaceSummary):
    geometry: dict[str, Any]  # GeoJSON polygon, WGS84
