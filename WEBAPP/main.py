import os
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import dem_stl


app = FastAPI(title="TOPO2STL")


# -----------------------------
# Models
# -----------------------------

class RoadEtch(BaseModel):
    width_mm: float = 0.0
    depth_mm: float = 0.0


class RoadEtchParams(BaseModel):
    motorway: RoadEtch = RoadEtch(width_mm=2.5, depth_mm=1.2)
    trunk: RoadEtch = RoadEtch(width_mm=2.0, depth_mm=1.1)
    primary: RoadEtch = RoadEtch(width_mm=1.6, depth_mm=1.0)
    secondary: RoadEtch = RoadEtch(width_mm=1.1, depth_mm=0.8)
    tertiary: RoadEtch = RoadEtch(width_mm=0.8, depth_mm=0.6)
    residential: RoadEtch = RoadEtch(width_mm=0.6, depth_mm=0.5)


class GenerateRequest(BaseModel):
    polygon_geojson: Dict[str, Any]
    dem_dataset: str
    downsample: int
    z_scale: float
    buffer_m: float
    target_max_mm: float
    add_base: bool
    base_thickness_m: float
    road_levels: List[str] = []
    road_etch: RoadEtchParams = RoadEtchParams()


class RoadsRequest(BaseModel):
    polygon_geojson: Dict[str, Any]
    road_levels: List[str] = []


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.post("/api/roads")
def get_roads(req: RoadsRequest):
    try:
        polygon = dem_stl.polygon_from_geojson(req.polygon_geojson)
        roads = dem_stl.fetch_roads_geojson_overpass(
            polygon_wgs84=polygon,
            highway_levels=req.road_levels
        )
        return {"roads_geojson": roads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-stl")
def generate_stl(req: GenerateRequest):
    try:
        polygon = dem_stl.polygon_from_geojson(req.polygon_geojson)
        path = dem_stl.generate_stl_from_polygon(
            polygon_wgs84=polygon,
            dem_dataset=req.dem_dataset,
            downsample=req.downsample,
            z_scale=req.z_scale,
            buffer_m=req.buffer_m,
            target_max_mm=req.target_max_mm,
            add_base_flag=req.add_base,
            base_thickness_m=req.base_thickness_m,
        )
        return FileResponse(path, filename="terrain.stl", media_type="application/sla")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-bundle")
def generate_bundle(req: GenerateRequest):
    try:
        polygon = dem_stl.polygon_from_geojson(req.polygon_geojson)
        zip_path = dem_stl.generate_bundle_from_polygon(
            polygon_wgs84=polygon,
            dem_dataset=req.dem_dataset,
            downsample=req.downsample,
            z_scale=req.z_scale,
            buffer_m=req.buffer_m,
            target_max_mm=req.target_max_mm,
            add_base_flag=req.add_base,
            base_thickness_m=req.base_thickness_m,
            road_levels=req.road_levels,
            road_etch=req.road_etch.model_dump(),
        )
        return FileResponse(zip_path, filename="terrain_bundle.zip", media_type="application/zip")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Static
# -----------------------------

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
