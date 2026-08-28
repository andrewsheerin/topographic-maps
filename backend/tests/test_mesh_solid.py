"""Tests for watertight STL generation (F-16). The regression that motivated
this: `add_base` used to concatenate two open sheets, so slicers dropped whole
regions of the model (half of Rhode Island failed to slice)."""

import numpy as np
import pytest

from core.mesh import add_base, dem_to_mesh

PX_M = 10.0
SCALE_XY = 0.001  # mm per metre / METERS_TO_MM
Z_SCALE = 1.0


def _solid(dem, thickness_mm=2.0):
    surface = dem_to_mesh(dem, PX_M, SCALE_XY, Z_SCALE)
    return add_base(surface, thickness_mm)


def test_flat_dem_seals_to_watertight_slab():
    dem = np.full((5, 7), 100.0, dtype=np.float32)
    solid = _solid(dem, thickness_mm=2.0)

    assert solid.is_watertight
    assert solid.is_winding_consistent
    assert solid.euler_number == 2  # closed surface, sphere topology
    # Flat terrain -> pure slab: volume = footprint area x thickness.
    x_mm = (dem.shape[1] - 1) * PX_M * SCALE_XY * 1000.0
    y_mm = (dem.shape[0] - 1) * PX_M * SCALE_XY * 1000.0
    assert solid.volume == pytest.approx(x_mm * y_mm * 2.0, rel=1e-6)


def test_terrain_with_nodata_regions_seals_watertight():
    # Mixed terrain with NaN regions (outside a multi-part boundary), like a
    # state outline over water: NaN flattens to the base plane but the solid
    # must still close.
    rng = np.random.default_rng(42)
    dem = rng.uniform(0.0, 500.0, size=(20, 30)).astype(np.float32)
    dem[:8, 12:] = np.nan  # a bay
    dem[15:, :5] = np.nan  # offshore corner
    solid = _solid(dem, thickness_mm=3.0)

    assert solid.is_watertight
    assert solid.is_winding_consistent
    assert solid.euler_number == 2
    assert solid.volume > 0


def test_normals_point_outward():
    dem = np.full((4, 4), 10.0, dtype=np.float32)
    solid = _solid(dem, thickness_mm=1.0)
    # Positive volume under trimesh's convention == outward-facing normals.
    assert solid.volume > 0
    top_z = solid.vertices[:, 2].max()
    top_faces = solid.faces[
        np.all(solid.vertices[solid.faces][:, :, 2] == top_z, axis=1)
    ]
    top_normals = solid.face_normals[
        np.all(solid.vertices[solid.faces][:, :, 2] == top_z, axis=1)
    ]
    assert len(top_faces) > 0
    assert np.all(top_normals[:, 2] > 0)


def test_zero_thickness_is_rejected():
    dem = np.full((3, 3), 5.0, dtype=np.float32)
    surface = dem_to_mesh(dem, PX_M, SCALE_XY, Z_SCALE)
    with pytest.raises(ValueError, match="thickness"):
        add_base(surface, 0.0)
