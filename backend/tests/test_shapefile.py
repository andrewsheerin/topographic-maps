"""Tests for the boundary-file -> WGS84 polygon transformations. Self-contained:
builds shapefiles/GeoJSON on the fly; no network."""

import json
import os
import tempfile
import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, shape

from core.shapefile import polygon_geojson_from_geojson, polygon_geojson_from_zip


def _write_zip(gdf, tmp, drop_prj=False):
    shp_dir = os.path.join(tmp, "shp")
    os.makedirs(shp_dir, exist_ok=True)
    gdf.to_file(os.path.join(shp_dir, "area.shp"))
    zip_path = os.path.join(tmp, "area.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in os.listdir(shp_dir):
            if drop_prj and name.endswith(".prj"):
                continue  # simulate a shapefile uploaded without CRS info
            z.write(os.path.join(shp_dir, name), name)
    return zip_path


def test_reprojects_utm_polygon_to_wgs84():
    tmp = tempfile.mkdtemp()
    # ~1 km box in UTM 19N (EPSG:32619), near Rhode Island.
    box = Polygon(
        [(300000, 4600000), (301000, 4600000), (301000, 4601000), (300000, 4601000)]
    )
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box], crs="EPSG:32619")
    zip_path = _write_zip(gdf, tmp)

    geom = polygon_geojson_from_zip(zip_path)

    assert geom["type"] == "Polygon"
    poly = shape(geom)
    lon, lat = poly.centroid.x, poly.centroid.y
    # Reprojected to WGS84 lon/lat, should land in Rhode Island's neighbourhood.
    assert -72 < lon < -70
    assert 41 < lat < 42


def test_missing_crs_raises():
    tmp = tempfile.mkdtemp()
    box = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box], crs="EPSG:4326")
    zip_path = _write_zip(gdf, tmp, drop_prj=True)

    with pytest.raises(ValueError):
        polygon_geojson_from_zip(zip_path)


SQUARE = {
    "type": "Polygon",
    "coordinates": [
        [[-71.5, 41.5], [-71.4, 41.5], [-71.4, 41.6], [-71.5, 41.6], [-71.5, 41.5]]
    ],
}


def test_geojson_feature_collection():
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": SQUARE}],
    }
    geom = polygon_geojson_from_geojson(json.dumps(fc))
    assert geom["type"] == "Polygon"
    assert shape(geom).equals(shape(SQUARE))


def test_geojson_bare_geometry():
    geom = polygon_geojson_from_geojson(json.dumps(SQUARE))
    assert shape(geom).equals(shape(SQUARE))


def test_geojson_multipolygon_reduces_to_convex_hull():
    multi = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]],
        ],
    }
    geom = polygon_geojson_from_geojson(json.dumps(multi))
    assert geom["type"] == "Polygon"
    assert shape(geom).contains(shape(multi))


def test_geojson_foreign_crs_rejected():
    data = {
        **SQUARE,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32619"}},
    }
    with pytest.raises(ValueError, match="WGS84"):
        polygon_geojson_from_geojson(json.dumps(data))


def test_geojson_invalid_json_rejected():
    with pytest.raises(ValueError, match="JSON"):
        polygon_geojson_from_geojson("not json {")
