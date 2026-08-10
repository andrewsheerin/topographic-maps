"""Shapefile upload: a zipped shapefile -> a polygon GeoJSON geometry the map and
DEM pipeline can use."""

import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from core import shapefile as shp

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload-shapefile")
async def upload_shapefile(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")

    tmp = tempfile.mkdtemp()
    zip_path = os.path.join(tmp, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(contents)

    try:
        geometry = shp.polygon_geojson_from_zip(zip_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read shapefile: {e}")

    return {"polygon_geojson": geometry}
