"""Read an uploaded zipped shapefile into a single WGS84 polygon geometry.

The DEM pipeline works in WGS84 (EPSG:4326) and needs one polygon area. Per
science-integrity, CRS is never assumed: a shapefile without a .prj is rejected,
not guessed."""

import glob
import os
import tempfile
import zipfile

import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union

WGS84 = 4326


def polygon_geojson_from_zip(zip_path: str) -> dict:
    """Extract a zipped shapefile and return one WGS84 polygon as a GeoJSON
    geometry dict. Raises ValueError (with a user-actionable message) on bad
    input; the caller maps that to HTTP 400."""
    extract_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise ValueError("The upload is not a valid .zip file.")

    shps = glob.glob(os.path.join(extract_dir, "**", "*.shp"), recursive=True)
    if not shps:
        raise ValueError("No .shp file found in the uploaded zip.")

    gdf = gpd.read_file(shps[0])
    if gdf.empty:
        raise ValueError("The shapefile contains no features.")
    if gdf.crs is None:
        raise ValueError(
            "The shapefile has no CRS (.prj missing). Add a .prj or reproject to "
            "EPSG:4326 before uploading."
        )

    gdf = gdf.to_crs(WGS84)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError("The shapefile has no usable geometry.")

    union = unary_union(geoms)
    if union.geom_type == "Polygon":
        poly = union
    elif union.geom_type in ("MultiPolygon", "GeometryCollection"):
        # A multi-part upload is reduced to its convex hull so the pipeline has a
        # single contiguous area to extract.
        poly = union.convex_hull
    else:
        raise ValueError(
            f"The shapefile geometry is {union.geom_type}; a polygon area is required."
        )

    if poly.is_empty or poly.geom_type != "Polygon":
        raise ValueError("Could not derive a polygon area from the shapefile.")

    return mapping(poly)
