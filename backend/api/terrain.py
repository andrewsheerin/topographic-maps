"""Terrain endpoints: generate a terrain-only STL, or a ZIP bundle with carved
roads."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from core import pipeline
from core.geometry import polygon_from_geojson
from models import GenerateRequest

router = APIRouter(prefix="/api", tags=["terrain"])


@router.post("/generate-stl")
def generate_stl(req: GenerateRequest):
    try:
        polygon = polygon_from_geojson(req.polygon_geojson)
        path = pipeline.generate_stl_from_polygon(
            polygon_wgs84=polygon,
            dem_dataset=req.dem_dataset,
            downsample=req.downsample,
            z_scale=req.z_scale,
            target_max_mm=req.target_max_mm,
            add_base_flag=req.add_base,
            base_thickness_mm=req.base_thickness_mm,
        )
        return FileResponse(path, filename="terrain.stl", media_type="application/sla")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-bundle")
def generate_bundle(req: GenerateRequest):
    try:
        polygon = polygon_from_geojson(req.polygon_geojson)
        zip_path = pipeline.generate_bundle_from_polygon(
            polygon_wgs84=polygon,
            dem_dataset=req.dem_dataset,
            downsample=req.downsample,
            z_scale=req.z_scale,
            target_max_mm=req.target_max_mm,
            add_base_flag=req.add_base,
            base_thickness_mm=req.base_thickness_mm,
            road_levels=req.road_levels,
            road_etch=req.road_etch.model_dump(),
        )
        return FileResponse(
            zip_path, filename="terrain_bundle.zip", media_type="application/zip"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
