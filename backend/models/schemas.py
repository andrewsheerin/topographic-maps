"""Pydantic request models — the API boundary. All values are print-mm or metres
as named; see `.claude/rules/science-integrity.md` (units are explicit)."""

from typing import Any

from pydantic import BaseModel, Field

import constants


class RoadEtch(BaseModel):
    width_mm: float = 0.0
    depth_mm: float = 0.0


class WaterParams(BaseModel):
    """Lake flattening + stream carving (F-22). Depth/width are print mm;
    defaults come from constants (user-provided 2026-09-03)."""

    lakes: bool = False
    streams: bool = False
    recess_mm: float = constants.WATER_RECESS_MM
    stream_width_mm: float = constants.STREAM_WIDTH_MM
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


class GenerateRequest(BaseModel):
    polygon_geojson: dict[str, Any]
    dem_dataset: str
    downsample: int
    z_scale: float
    target_max_mm: float
    base_thickness_mm: float  # print millimetres (F-17); STL is always sealed (F-19)
    road_levels: list[str] = []
    road_etch: RoadEtchParams = RoadEtchParams()
    water: WaterParams = WaterParams()


_HEX = r"^#[0-9A-Fa-f]{6}$"


class ColorParams(BaseModel):
    """The 3MF palette in class order (F-23). All required — no default
    colors are chosen here."""

    water: str = Field(pattern=_HEX)
    forest: str = Field(pattern=_HEX)
    rock: str = Field(pattern=_HEX)
    snow: str = Field(pattern=_HEX)


class Generate3MFRequest(GenerateRequest):
    """Colored 3MF export (F-23). treeline_m and snowline_m are scientific
    per-request values (they vary by mountain range) — required, no defaults
    (science-integrity)."""

    treeline_m: float
    snowline_m: float
    colors: ColorParams


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
