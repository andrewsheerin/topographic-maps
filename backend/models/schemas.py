"""Pydantic request models — the API boundary. All values are print-mm or metres
as named; see `.claude/rules/science-integrity.md` (units are explicit)."""

from typing import Any

from pydantic import BaseModel


class RoadEtch(BaseModel):
    width_mm: float = 0.0
    depth_mm: float = 0.0


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
    buffer_m: float
    target_max_mm: float
    add_base: bool
    base_thickness_m: float
    road_levels: list[str] = []
    road_etch: RoadEtchParams = RoadEtchParams()


class RoadsRequest(BaseModel):
    polygon_geojson: dict[str, Any]
    road_levels: list[str] = []
