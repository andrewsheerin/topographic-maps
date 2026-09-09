"""Tests for lake flattening + stream carving (F-22). No network calls —
Overpass element parsing is tested on canned dicts.

The scientific contract under test: each lake is flattened to ITS OWN plane
(10th percentile of the DEM inside its polygon) minus the print-mm recess,
delta = mm / (scale_xy * z_scale * METERS_TO_MM); streams carve everywhere
except inside a flattened lake; the sealed-solid guard still holds afterwards.
"""

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from core.mesh import (
    add_base,
    carve_streams,
    dem_to_mesh,
    flatten_sea,
    flatten_water,
)
from core.water import _relation_polygon, _way_polygon


def stream_gdf(records, epsg=32611):
    """(waterway_class, LineString) tuples -> the GeoDataFrame carve expects."""
    return gpd.GeoDataFrame(
        {"waterway": [c for c, _ in records]},
        geometry=[g for _, g in records],
        crs=epsg,
    )

PX_M = 10.0
SCALE_XY = 0.001  # mm per metre / METERS_TO_MM
Z_SCALE = 2.0
# Grid: 20x20 pixels, 10 m each, origin top-left (0, 200).
TRANSFORM = from_origin(0.0, 200.0, PX_M, PX_M)

# Pixel windows rows/cols 2..6 and 12..16, inset 1 unit so all_touched
# rasterization doesn't bleed into neighbouring pixels.
LAKE1_BOX = box(21.0, 131.0, 69.0, 179.0)  # rows 2:7, cols 2:7
LAKE2_BOX = box(121.0, 31.0, 169.0, 79.0)  # rows 12:17, cols 12:17


def _dem_with_two_lakes():
    rng = np.random.default_rng(7)
    dem = np.full((20, 20), 100.0, dtype=np.float32)
    # Noisy, non-planar "radar over open water" surfaces at two elevations.
    dem[2:7, 2:7] = 150.0 + rng.uniform(-5.0, 5.0, (5, 5))
    dem[12:17, 12:17] = 300.0 + rng.uniform(-5.0, 5.0, (5, 5))
    return dem


def test_each_lake_flattens_to_its_own_plane():
    dem = _dem_with_two_lakes()
    recess_mm = 0.6
    # delta = mm / (scale_xy * z_scale * METERS_TO_MM) = 0.6 / 2.0 = 0.3 m
    expected_delta_m = 0.3

    dem2, water_mask = flatten_water(
        dem, TRANSFORM, [LAKE1_BOX, LAKE2_BOX], SCALE_XY, Z_SCALE, recess_mm
    )

    lake1 = dem2[2:7, 2:7]
    lake2 = dem2[12:17, 12:17]

    # Planar: every cell of a lake sits at one Z...
    assert np.all(lake1 == lake1[0, 0])
    assert np.all(lake2 == lake2[0, 0])
    # ...which is that lake's own 10th percentile minus the recess...
    assert lake1[0, 0] == np.float32(
        np.percentile(dem[2:7, 2:7], 10) - expected_delta_m
    )
    assert lake2[0, 0] == np.float32(
        np.percentile(dem[12:17, 12:17], 10) - expected_delta_m
    )
    # ...and the two planes differ (no global water elevation).
    assert lake1[0, 0] != lake2[0, 0]

    # Mask covers exactly the two lakes; everything else untouched.
    assert water_mask[2:7, 2:7].all() and water_mask[12:17, 12:17].all()
    assert water_mask.sum() == 50
    assert np.all(dem2[~water_mask] == dem[~water_mask])


def test_stream_carves_outside_but_not_through_a_lake():
    dem = _dem_with_two_lakes()
    dem2, water_mask = flatten_water(
        dem, TRANSFORM, [LAKE1_BOX], SCALE_XY, Z_SCALE, 0.6
    )
    lake_z = dem2[4, 4]

    # A stream straight across row 4 (y=155), through the middle of lake 1.
    streams = stream_gdf([("stream", LineString([(0.0, 155.0), (200.0, 155.0)]))])
    dem3, color_mask = carve_streams(
        dem2, TRANSFORM, streams, SCALE_XY, Z_SCALE,
        {"stream": 1.0}, 0.4, protect_mask=water_mask,
    )

    assert np.all(dem3[4, :2] < 100.0)  # carved before the lake
    assert np.all(dem3[4, 7:] < 100.0)  # carved after the lake
    assert np.all(dem3[4, 2:7] == lake_z)  # lake plane untouched
    assert np.all(dem3[:4, :] == dem2[:4, :])  # other rows untouched
    # The color mask covers the carved stream but never the protected lake.
    assert color_mask[4, :2].all() and color_mask[4, 7:].all()
    assert not color_mask[4, 2:7].any()


def test_flatten_and_carve_still_seal_watertight():
    rng = np.random.default_rng(11)
    dem = rng.uniform(0.0, 400.0, size=(30, 30)).astype(np.float32)
    dem[:6, 20:] = np.nan  # nodata region, as after boundary cropping
    transform = from_origin(0.0, 300.0, PX_M, PX_M)

    dem2, water_mask = flatten_water(
        dem, transform, [box(51.0, 51.0, 129.0, 129.0)], SCALE_XY, Z_SCALE, 0.6
    )
    dem3, _ = carve_streams(
        dem2, transform,
        stream_gdf([("river", LineString([(0.0, 90.0), (300.0, 210.0)]))]),
        SCALE_XY, Z_SCALE, {"river": 1.6}, 0.4, protect_mask=water_mask,
    )

    solid = add_base(dem_to_mesh(dem3, PX_M, SCALE_XY, Z_SCALE), 2.0)
    assert solid.is_watertight
    assert solid.is_winding_consistent
    assert solid.volume > 0


def test_sea_flattens_to_recessed_plane_and_protects_from_streams():
    # Coastal DEM (F-33): noisy ocean around 0 m, land at 50 m, one NaN hole.
    rng = np.random.default_rng(21)
    dem = np.full((20, 20), 50.0, dtype=np.float32)
    dem[:, :8] = rng.uniform(-0.4, 0.4, (20, 8))  # the bay
    dem[0, 0] = np.nan

    # delta = 0.6 / (0.001 * 2.0 * 1000) = 0.3 m; plane = 0.5 - 0.3 = 0.2
    dem2, sea_mask = flatten_sea(dem, 0.5, SCALE_XY, Z_SCALE, 0.6)

    assert sea_mask[1:, :8].all() and not sea_mask[:, 8:].any()
    assert not sea_mask[0, 0]  # NaN is never sea
    assert np.all(dem2[sea_mask] == np.float32(0.5 - 0.3))
    assert np.all(dem2[:, 8:] == 50.0)  # land untouched
    assert np.isnan(dem2[0, 0])

    # A stream flowing into the bay carves land but never the sea plane.
    stream = LineString([(0.0, 155.0), (200.0, 155.0)])
    dem3, _ = carve_streams(
        dem2, TRANSFORM, stream_gdf([("stream", stream)]),
        SCALE_XY, Z_SCALE, {"stream": 1.0}, 0.4, protect_mask=sea_mask,
    )
    assert np.all(dem3[4, :8] == np.float32(0.2))  # bay plane intact
    assert np.all(dem3[4, 8:] < 50.0)  # land carved


def test_way_polygon_parses_closed_ways_only():
    closed = {
        "type": "way",
        "geometry": [
            {"lon": 0.0, "lat": 0.0},
            {"lon": 1.0, "lat": 0.0},
            {"lon": 1.0, "lat": 1.0},
            {"lon": 0.0, "lat": 0.0},
        ],
    }
    open_way = {"type": "way", "geometry": closed["geometry"][:-1]}
    assert _way_polygon(closed) is not None
    assert _way_polygon(open_way) is None


def test_relation_polygon_stitches_outers_and_subtracts_islands():
    # Outer square split across two member ways; one inner island.
    el = {
        "type": "relation",
        "members": [
            {
                "role": "outer",
                "geometry": [
                    {"lon": 0.0, "lat": 0.0},
                    {"lon": 4.0, "lat": 0.0},
                    {"lon": 4.0, "lat": 4.0},
                ],
            },
            {
                "role": "outer",
                "geometry": [
                    {"lon": 4.0, "lat": 4.0},
                    {"lon": 0.0, "lat": 4.0},
                    {"lon": 0.0, "lat": 0.0},
                ],
            },
            {
                "role": "inner",
                "geometry": [
                    {"lon": 1.0, "lat": 1.0},
                    {"lon": 2.0, "lat": 1.0},
                    {"lon": 2.0, "lat": 2.0},
                    {"lon": 1.0, "lat": 2.0},
                    {"lon": 1.0, "lat": 1.0},
                ],
            },
        ],
    }
    geom = _relation_polygon(el)
    assert geom is not None
    assert geom.area == 16.0 - 1.0  # outer minus island
