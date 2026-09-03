"""Terrain endpoints: generate a terrain-only STL, or a ZIP bundle with carved
roads."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from core import pipeline
from core.geometry import polygon_from_geojson
from models import Generate3MFRequest, GenerateRequest

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
            base_thickness_mm=req.base_thickness_mm,
            water=req.water.model_dump(),
        )
        return FileResponse(path, filename="terrain.stl", media_type="application/sla")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-3mf")
def generate_3mf(req: Generate3MFRequest):
    try:
        polygon = polygon_from_geojson(req.polygon_geojson)
        path = pipeline.generate_3mf_from_polygon(
            polygon_wgs84=polygon,
            dem_dataset=req.dem_dataset,
            downsample=req.downsample,
            z_scale=req.z_scale,
            target_max_mm=req.target_max_mm,
            base_thickness_mm=req.base_thickness_mm,
            treeline_m=req.treeline_m,
            snowline_m=req.snowline_m,
            # Palette in class order: 0 water, 1 forest, 2 rock, 3 snow.
            colors_hex=[
                req.colors.water,
                req.colors.forest,
                req.colors.rock,
                req.colors.snow,
            ],
            road_levels=req.road_levels,
            road_etch=req.road_etch.model_dump(),
            water=req.water.model_dump(),
        )
        return FileResponse(path, filename="terrain.3mf", media_type="model/3mf")
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
            base_thickness_mm=req.base_thickness_mm,
            road_levels=req.road_levels,
            road_etch=req.road_etch.model_dump(),
            water=req.water.model_dump(),
        )
        return FileResponse(
            zip_path, filename="terrain_bundle.zip", media_type="application/zip"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
