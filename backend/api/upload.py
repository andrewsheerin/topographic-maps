"""Boundary upload: a zipped shapefile or a GeoJSON file -> a polygon GeoJSON
geometry the map and DEM pipeline can use."""

import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from core import shapefile as shp

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload-boundary")
async def upload_boundary(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")

    name = (file.filename or "").lower()
    try:
        if name.endswith(".zip"):
            tmp = tempfile.mkdtemp()
            zip_path = os.path.join(tmp, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(contents)
            geometry = shp.polygon_geojson_from_zip(zip_path)
        elif name.endswith((".geojson", ".json")):
            geometry = shp.polygon_geojson_from_geojson(
                contents.decode("utf-8", errors="replace")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload a zipped shapefile (.zip) or GeoJSON (.geojson/.json).",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read boundary file: {e}"
        )

    return {"polygon_geojson": geometry}
