"""Read an uploaded boundary file (zipped shapefile or GeoJSON) into a single
WGS84 areal geometry (Polygon or MultiPolygon — true boundaries are kept, F-13).

The DEM pipeline works in WGS84 (EPSG:4326) and needs one areal geometry. Per
science-integrity, CRS is never assumed: a shapefile without a .prj is rejected,
not guessed. GeoJSON is WGS84 by spec (RFC 7946); a legacy `crs` member naming
anything else is rejected."""

import glob
import json
import os
import tempfile
import zipfile

import geopandas as gpd
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

WGS84 = 4326

# Legacy (pre-RFC 7946) GeoJSON `crs` names that still mean WGS84 lon/lat.
_WGS84_CRS_NAMES = {
    "urn:ogc:def:crs:OGC:1.3:CRS84",
    "urn:ogc:def:crs:OGC::CRS84",
    "urn:ogc:def:crs:EPSG::4326",
    "EPSG:4326",
}


def to_area_geometry(geoms: list) -> dict:
    """Union geometries into one areal GeoJSON geometry dict (Polygon or
    MultiPolygon). True boundaries are preserved — multi-part coastal areas stay
    multi-part; nothing is hulled over water (F-13). Raises ValueError when no
    polygonal area can be derived."""
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError("The file has no usable geometry.")

    union = unary_union(geoms)
    if union.geom_type == "GeometryCollection":
        polygonal = [
            g for g in union.geoms if g.geom_type in ("Polygon", "MultiPolygon")
        ]
        if not polygonal:
            raise ValueError("The file contains no polygon area.")
        union = unary_union(polygonal)

    if union.is_empty or union.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"The geometry is {union.geom_type}; a polygon area is required."
        )

    return mapping(union)


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
    return to_area_geometry(list(gdf.geometry))


def polygon_geojson_from_geojson(text: str) -> dict:
    """Parse GeoJSON (FeatureCollection, Feature, or bare geometry) and return
    one WGS84 polygon as a GeoJSON geometry dict. Raises ValueError on bad
    input; the caller maps that to HTTP 400."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("The upload is not valid JSON.")
    if not isinstance(data, dict) or "type" not in data:
        raise ValueError("The upload is not valid GeoJSON (no `type` member).")

    # RFC 7946 GeoJSON is always WGS84. A legacy `crs` member naming another
    # system means the coordinates aren't lon/lat — reject rather than assume.
    crs = data.get("crs")
    if crs is not None:
        name = (
            str((crs.get("properties") or {}).get("name", ""))
            if isinstance(crs, dict)
            else str(crs)
        )
        if name not in _WGS84_CRS_NAMES:
            raise ValueError(
                f"The GeoJSON declares CRS {name or crs!r}; reproject to WGS84 "
                "(EPSG:4326) before uploading."
            )

    if data["type"] == "FeatureCollection":
        geometries = [f.get("geometry") for f in data.get("features", [])]
    elif data["type"] == "Feature":
        geometries = [data.get("geometry")]
    else:
        geometries = [data]

    geoms = []
    for g in geometries:
        if not g:
            continue
        try:
            geoms.append(shape(g))
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ValueError("The GeoJSON contains an invalid geometry.")

    return to_area_geometry(geoms)
