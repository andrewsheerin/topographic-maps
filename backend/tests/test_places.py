"""Tests for the TIGER place lookup. Self-contained: builds a small GeoPackage
on the fly; no network."""

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, shape

from core.places import get_place, query_places


def _box(lon: float, lat: float, d: float = 0.1) -> Polygon:
    return Polygon([(lon, lat), (lon + d, lat), (lon + d, lat + d), (lon, lat + d)])


@pytest.fixture(scope="module")
def gpkg(tmp_path_factory):
    gdf = gpd.GeoDataFrame(
        {
            "geoid": ["4400001", "4400002", "2500001", "4400003"],
            "name": ["Barrington", "Warren", "Barnstable", "New Shoreham"],
            "state": ["RI", "RI", "MA", "RI"],
            "county": ["Bristol", "Bristol", "Barnstable", "Washington"],
        },
        geometry=[
            _box(-71.4, 41.7),
            _box(-71.3, 41.7),
            _box(-70.3, 41.7),
            # Island town: multi-part boundary.
            MultiPolygon([_box(-71.6, 41.1, 0.05), _box(-71.5, 41.2, 0.02)]),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path_factory.mktemp("tiger") / "subdivisions.gpkg"
    gdf.to_file(path, layer="subdivisions", driver="GPKG")
    return path


def test_query_filters_by_state(gpkg):
    out = query_places(state="ri", gpkg_path=gpkg)
    assert [p["name"] for p in out] == ["Barrington", "New Shoreham", "Warren"]
    assert all(p["state"] == "RI" for p in out)


def test_query_search_is_case_insensitive_substring(gpkg):
    out = query_places(q="barn", gpkg_path=gpkg)
    assert [p["name"] for p in out] == ["Barnstable"]


def test_query_pagination(gpkg):
    first = query_places(limit=1, offset=0, gpkg_path=gpkg)
    second = query_places(limit=1, offset=1, gpkg_path=gpkg)
    assert len(first) == len(second) == 1
    assert first[0]["geoid"] != second[0]["geoid"]


def test_summary_has_bbox_and_area(gpkg):
    (p,) = query_places(q="Warren", gpkg_path=gpkg)
    assert p["bbox"]["min_lon"] == pytest.approx(-71.3)
    assert p["bbox"]["max_lat"] == pytest.approx(41.8)
    assert p["area_km2"] > 0


def test_get_place_returns_geometry(gpkg):
    place = get_place("4400001", gpkg_path=gpkg)
    assert place is not None
    geom = shape(place["geometry"])
    assert geom.geom_type == "Polygon"
    assert geom.contains(shape({"type": "Point", "coordinates": [-71.35, 41.75]}))


def test_get_place_reduces_multipolygon_to_single_polygon(gpkg):
    place = get_place("4400003", gpkg_path=gpkg)
    geom = shape(place["geometry"])
    # Multi-part island boundary -> one contiguous polygon (convex hull), same
    # policy as file uploads; the roads query needs a single exterior ring.
    assert geom.geom_type == "Polygon"
    assert geom.contains(shape({"type": "Point", "coordinates": [-71.575, 41.125]}))


def test_get_place_unknown_geoid(gpkg):
    assert get_place("0000000", gpkg_path=gpkg) is None


def test_missing_dataset_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_tiger_subdivisions"):
        query_places(gpkg_path=tmp_path / "nope.gpkg")
