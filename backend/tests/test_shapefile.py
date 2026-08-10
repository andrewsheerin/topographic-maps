"""Tests for the shapefile -> WGS84 polygon transformation. Self-contained:
builds a shapefile on the fly and zips it; no network."""

import os
import tempfile
import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, shape

from core.shapefile import polygon_geojson_from_zip


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
