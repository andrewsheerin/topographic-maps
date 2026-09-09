"""Tests for the 3MF frame (F-35): the inscribed-rectangle search, the ring
solid's watertightness, and the multi-object 3MF round-trip."""

import zipfile
from xml.etree import ElementTree

import numpy as np
import pytest
from rasterio.transform import from_origin

from core.mesh import (
    add_base,
    dem_to_mesh,
    frame_solid,
    largest_true_rect,
)
from core.pipeline import _build_frame
from core.threemf import CORE_NS, write_3mf

PX_M = 30.0
SCALE_XY = 0.0001
Z_SCALE = 1.0


def test_largest_true_rect_on_skewed_footprint():
    # A rotated-rectangle footprint like a reprojected drawn box: build a
    # parallelogram mask and check the found rect is all-True and non-trivial.
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


def test_frame_solid_watertight_with_exact_volume():
    solid = frame_solid((10, 10, 90, 80), (0, 0, 100, 100), -2.0, 15.0)
    assert solid.is_watertight
    assert solid.is_winding_consistent
    assert solid.euler_number == 0  # ring topology (genus 1)
    expected = (100 * 100 - 80 * 70) * 17.0
    assert solid.volume == pytest.approx(expected, rel=1e-9)


def test_frame_solid_rejects_non_nested_rects():
    with pytest.raises(ValueError, match="inside"):
        frame_solid((0, 0, 100, 100), (10, 10, 90, 90), 0.0, 5.0)


def _skewed_dem():
    h, w = 30, 30
    dem = np.full((h, w), np.nan, dtype=np.float32)
    for r in range(h):
        shift = r // 5
        dem[r, shift : shift + 22] = 100.0 + r
    return dem


def test_build_frame_covers_ragged_margin_and_spans_terrain_height():
    dem = _skewed_dem()
    solid = add_base(dem_to_mesh(dem, PX_M, SCALE_XY, Z_SCALE), 2.0)
    frame = _build_frame(dem, PX_M, SCALE_XY, solid, frame_width_mm=5.0)

    assert frame.is_watertight
    # Full vertical span: base bottom to terrain peak.
    assert frame.vertices[:, 2].min() == pytest.approx(
        solid.vertices[:, 2].min()
    )
    assert frame.vertices[:, 2].max() == pytest.approx(
        solid.vertices[:, 2].max()
    )
    # Outer rect = terrain bbox + width on every side.
    cell = PX_M * SCALE_XY * 1000.0
    assert frame.vertices[:, 0].min() == pytest.approx(-5.0)
    assert frame.vertices[:, 0].max() == pytest.approx((30 - 1) * cell + 5.0)
    # Inner rect sits strictly inside the terrain bbox (covers the wedges).
    inner_x = sorted(set(np.round(frame.vertices[:, 0], 6)))[1:-1]
    assert min(inner_x) > 0.0


def test_3mf_gains_second_uncolored_object(tmp_path):
    dem = _skewed_dem()
    solid = add_base(dem_to_mesh(dem, PX_M, SCALE_XY, Z_SCALE), 2.0)
    frame = _build_frame(dem, PX_M, SCALE_XY, solid, frame_width_mm=5.0)

    out = str(tmp_path / "framed.3mf")
    face_cls = np.zeros(len(solid.faces), dtype=int)
    write_3mf(
        out, solid.vertices, solid.faces, face_cls, ["#123456"], 0,
        extra_meshes=[(frame.vertices, frame.faces)],
    )

    with zipfile.ZipFile(out) as z:
        root = ElementTree.fromstring(z.read("3D/3dmodel.model"))
    objects = root.findall(f".//{{{CORE_NS}}}object")
    assert [o.get("id") for o in objects] == ["2", "3"]
    assert objects[1].get("pid") is None  # frame carries no color
    frame_tris = objects[1].findall(f".//{{{CORE_NS}}}triangle")
    assert len(frame_tris) == len(frame.faces) == 32
    items = root.findall(f".//{{{CORE_NS}}}item")
    assert [i.get("objectid") for i in items] == ["2", "3"]
