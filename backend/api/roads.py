"""Roads endpoint: fetch OSM road centerlines within a polygon."""

from fastapi import APIRouter, HTTPException

from core import roads as roads_mod
from core.geometry import polygon_from_geojson
from models import RoadsRequest

router = APIRouter(prefix="/api", tags=["roads"])


@router.post("/roads")
def get_roads(req: RoadsRequest):
    try:
        polygon = polygon_from_geojson(req.polygon_geojson)
        roads = roads_mod.fetch_roads_geojson_overpass(
            polygon_wgs84=polygon,
            highway_levels=req.road_levels,
        )
        return {"roads_geojson": roads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
