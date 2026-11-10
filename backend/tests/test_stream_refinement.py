"""Tests for the F-28 stream-rendering refinement, at the scale from the
motivating case: ~9.5 km box printed at 250 mm on COP30 (30 m cells), i.e.
scale_xy ≈ 2.63e-5 and one DEM cell ≈ 0.79 mm at print scale."""

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString

import constants
from core.dem import upsample_dem
from core.mesh import add_base, carve_streams, dem_to_mesh, stream_width_floor_mm
from tests.test_water import stream_gdf

PX_M = 30.0
SCALE_XY = 250.0 / (9500.0 * 1000.0)  # ~2.63e-5: 9.5 km box at 250 mm
Z_SCALE = 1.0
DELTA_FULL_M = 0.4 / (SCALE_XY * Z_SCALE * 1000.0)  # full-depth carve, metres
EPSG = 32611


def _flat_dem(n=40, value=1000.0):
    dem = np.full((n, n), value, dtype=np.float32)
    transform = from_origin(0.0, n * PX_M, PX_M, PX_M)
    return dem, transform


def test_fractional_coverage_produces_intermediate_depths():
    # A diagonal sub-cell-width stream: edge cells must get partial-depth
    # carves, not the on/off chain the old rasterize produced.
    dem, transform = _flat_dem()
    streams = stream_gdf([("brook", LineString([(0.0, 0.0), (1200.0, 1200.0)]))])
    dem2, _ = carve_streams(
        dem, transform, streams, SCALE_XY, Z_SCALE, {"brook": 0.6}, 0.4
    )
    carved = 1000.0 - dem2[dem2 < 1000.0]
    assert carved.size > 0
    # Intermediate depths exist (strictly between zero and the full delta).
    intermediate = carved[
        (carved > 0.01 * DELTA_FULL_M) & (carved < 0.95 * DELTA_FULL_M)
    ]
    assert intermediate.size > 0
    # And nothing exceeds the requested full depth.
    assert carved.max() <= DELTA_FULL_M * 1.0001


def test_v_profile_is_monotonically_deeper_toward_centerline():
    # A wide straight stream: carve depth per row must increase toward the
    # centerline row and be symmetric about it.
    dem, transform = _flat_dem()
    y_center = 600.0  # centre of row 19 (rows span 585-615 at 30 m)
    streams = stream_gdf([("river", LineString([(0.0, y_center), (1200.0, y_center)]))])
    # Width of ~6 cells so all three rings resolve: 4.7 mm print width.
    dem2, _ = carve_streams(
        dem, transform, streams, SCALE_XY, Z_SCALE, {"river": 4.7}, 0.4
    )
    depth_by_row = (1000.0 - dem2).mean(axis=1)
    center_row = int(np.argmax(depth_by_row))
    profile = depth_by_row[depth_by_row > 0]
    assert profile.size >= 5
    rows = np.where(depth_by_row > 0)[0]
    for r in rows:
        if r < center_row:
            assert depth_by_row[r] <= depth_by_row[r + 1] + 1e-6
        if r > center_row:
            assert depth_by_row[r] <= depth_by_row[r - 1] + 1e-6
    # The centerline reaches (close to) the full requested depth.
    assert depth_by_row[center_row] >= 0.9 * DELTA_FULL_M


def test_dem_upsample_shape_and_range():
    rng = np.random.default_rng(9)
    dem = rng.uniform(500.0, 1500.0, size=(30, 30)).astype(np.float32)
    dem[:6, 22:] = np.nan
    transform = from_origin(0.0, 900.0, PX_M, PX_M)

    up, px_m, t2 = upsample_dem(dem, transform, 2, EPSG)

    assert up.shape == (60, 60)
    assert px_m == PX_M / 2
    assert t2.a == transform.a / 2 and abs(t2.e) == abs(transform.e) / 2
    # Bilinear invents no new extremes and loses little at the peaks.
    assert np.nanmin(up) >= np.nanmin(dem) - 1e-3
    assert np.nanmax(up) <= np.nanmax(dem) + 1e-3
    assert abs(np.nanmin(up) - np.nanmin(dem)) < 100.0
    assert abs(np.nanmax(up) - np.nanmax(dem)) < 100.0
    # Nodata stays nodata: the interior of the NaN block is still NaN.
    assert np.isnan(up[2 * 2, 2 * 26])


def test_watertight_after_upsample_and_v_carve():
    rng = np.random.default_rng(13)
    dem = rng.uniform(800.0, 1400.0, size=(25, 25)).astype(np.float32)
    dem[:5, 18:] = np.nan
    transform = from_origin(0.0, 750.0, PX_M, PX_M)

    up, px_m, t2 = upsample_dem(dem, transform, 2, EPSG)
    streams = stream_gdf(
        [
            ("river", LineString([(0.0, 100.0), (750.0, 650.0)])),
            ("stream", LineString([(0.0, 400.0), (750.0, 200.0)])),
        ]
    )
    carved, _ = carve_streams(
        up, t2, streams, SCALE_XY, Z_SCALE, constants.STREAM_WIDTHS_MM, 0.4
    )
    solid = add_base(dem_to_mesh(carved, px_m, SCALE_XY, Z_SCALE), 2.0)
    assert solid.is_watertight
    assert solid.is_winding_consistent
    assert solid.volume > 0


def test_width_floor_clamps_sub_floor_requests():
    # floor = max(cell size at print scale, one extrusion line)
    assert stream_width_floor_mm(30.0, SCALE_XY) == 30.0 * SCALE_XY * 1000.0
    assert stream_width_floor_mm(10.0, SCALE_XY) == constants.EXTRUSION_LINE_WIDTH_MM

    # A request below the floor carves exactly as if it were at the floor —
    # never the old ragged one-cell chain from a degenerate buffer.
    dem, transform = _flat_dem()
    line = LineString([(0.0, 605.0), (1200.0, 605.0)])
    floor_mm = stream_width_floor_mm(PX_M, SCALE_XY)
    below, _ = carve_streams(
        dem,
        transform,
        stream_gdf([("stream", line)]),
        SCALE_XY,
        Z_SCALE,
        {"stream": 0.1},
        0.4,
    )
    at_floor, _ = carve_streams(
        dem,
        transform,
        stream_gdf([("stream", line)]),
        SCALE_XY,
        Z_SCALE,
        {"stream": floor_mm},
        0.4,
    )
    assert np.allclose(below, at_floor, equal_nan=True)
    assert (below < 1000.0).any()
