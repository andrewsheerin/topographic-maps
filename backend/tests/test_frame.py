"""Tests for the clean-edge clip (F-35): the inscribed-rectangle search and
the pre-mesh DEM clip that gives rectangular prints straight 90° edges
instead of the stair-stepped UTM-reprojection margin."""

import numpy as np
import pytest
from rasterio.transform import from_origin

from core.mesh import add_base, dem_to_mesh, largest_true_rect
from core.pipeline import _clip_clean_rect

PX_M = 30.0
SCALE_XY = 0.0001
Z_SCALE = 1.0


def test_largest_true_rect_on_skewed_footprint():
    # A rotated-rectangle footprint like a reprojected drawn box: build a
    # sheared band and check the found rect is all-True and substantial.
    h, w = 40, 60
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        shift = r // 4  # ~14 degree shear
        mask[r, shift : shift + 40] = True

    r0, r1, c0, c1 = largest_true_rect(mask)
    assert mask[r0 : r1 + 1, c0 : c1 + 1].all()
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert area >= 0.6 * mask.sum() * 0.5  # a substantial inscribed rect


def test_largest_true_rect_exact_on_simple_block():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:9, 3:8] = True
    assert largest_true_rect(mask) == (2, 8, 3, 7)


def test_largest_true_rect_raises_on_empty():
    with pytest.raises(RuntimeError, match="frame"):
        largest_true_rect(np.zeros((5, 5), dtype=bool))


def _skewed_dem():
    h, w = 30, 30
    dem = np.full((h, w), np.nan, dtype=np.float32)
    for r in range(h):
        shift = r // 5
        dem[r, shift : shift + 22] = 100.0 + r
    return dem


def test_clip_clean_rect_yields_fully_finite_rectangle():
    dem = _skewed_dem()
    transform = from_origin(0.0, 30 * PX_M, PX_M, PX_M)
    dem2, t2 = _clip_clean_rect(dem, transform)

    # The clipped DEM is a solid rectangle: every vertex finite.
    assert np.isfinite(dem2).all()
    assert dem2.shape[0] >= 2 and dem2.shape[1] >= 2
    # The transform shifted by the clip offset (origin moved east/south).
    assert t2.c >= transform.c and t2.f <= transform.f
    assert t2.a == transform.a and t2.e == transform.e


def test_clipped_mesh_has_straight_edges_and_seals():
    dem = _skewed_dem()
    transform = from_origin(0.0, 30 * PX_M, PX_M, PX_M)
    dem2, _ = _clip_clean_rect(dem, transform)

    solid = add_base(dem_to_mesh(dem2, PX_M, SCALE_XY, Z_SCALE), 2.0)
    assert solid.is_watertight
    assert solid.is_winding_consistent

    # Straight edges: every boundary vertex lies exactly on the bbox — the
    # XY outline is a perfect rectangle (no stair-steps).
    h2, w2 = dem2.shape
    cell = PX_M * SCALE_XY * 1000.0
    x, y = solid.vertices[:, 0], solid.vertices[:, 1]
    assert x.min() == pytest.approx(0.0)
    assert x.max() == pytest.approx((w2 - 1) * cell)
    assert y.min() == pytest.approx(0.0)
    assert y.max() == pytest.approx((h2 - 1) * cell)
    # Full-grid mesh: no cell was dropped, so vertex count is exactly the
    # grid twice (top + bottom) — the signature of a clean rectangle.
    assert len(solid.vertices) == 2 * h2 * w2


def test_unclipped_skewed_mesh_is_ragged_by_contrast():
    # Sanity check of the premise: without the clip, the same DEM meshes to a
    # cropped, stair-stepped outline (fewer vertices than the full grid).
    dem = _skewed_dem()
    surface = dem_to_mesh(dem, PX_M, SCALE_XY, Z_SCALE)
    assert len(surface.vertices) < 30 * 30
