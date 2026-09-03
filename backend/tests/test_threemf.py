"""Round-trip tests for the colored 3MF export (F-23): the written package
must contain exactly the mesh's triangles, every face a valid palette
reference, and the requested palette verbatim."""

import zipfile
from xml.etree import ElementTree

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from core.mesh import (
    add_base,
    classify_terrain_cells,
    dem_to_mesh,
    face_classes_for_solid,
    flatten_water,
)
from core.threemf import CORE_NS, MATERIAL_NS, write_3mf

PX_M = 10.0
SCALE_XY = 0.001
Z_SCALE = 1.0
PALETTE = ["#1D4ED8", "#166534", "#78716C", "#F8FAFC"]  # test-only values


def _solid_with_classes():
    """A small terrain with all four classes: a lake, low ground, mid slopes,
    a summit; returns (solid, face_classes)."""
    rng = np.random.default_rng(5)
    dem = rng.uniform(900.0, 1000.0, size=(12, 12)).astype(np.float32)
    dem[:4, :4] += 1500.0  # summit above the test snowline
    dem[8:, 8:] += 600.0  # slopes between treeline and snowline
    transform = from_origin(0.0, 120.0, PX_M, PX_M)

    dem2, water_mask = flatten_water(
        dem, transform, [box(61.0, 11.0, 109.0, 49.0)], SCALE_XY, Z_SCALE, 0.6
    )
    solid = add_base(dem_to_mesh(dem2, PX_M, SCALE_XY, Z_SCALE), 2.0)
    cells = classify_terrain_cells(
        dem2, water_mask, treeline_m=1200.0, snowline_m=2000.0
    )
    face_cls = face_classes_for_solid(dem2, cells, len(solid.faces))
    return solid, face_cls


def _read_model_xml(path):
    with zipfile.ZipFile(path) as z:
        return ElementTree.fromstring(z.read("3D/3dmodel.model"))


def test_roundtrip_counts_colors_and_pids(tmp_path):
    solid, face_cls = _solid_with_classes()
    out = str(tmp_path / "terrain.3mf")
    write_3mf(out, solid.vertices, solid.faces, face_cls, PALETTE, default_index=2)

    root = _read_model_xml(out)
    colors = root.findall(f".//{{{MATERIAL_NS}}}colorgroup/{{{MATERIAL_NS}}}color")
    assert [c.get("color") for c in colors] == PALETTE

    obj = root.find(f".//{{{CORE_NS}}}object")
    assert obj.get("pid") == "1"
    assert obj.get("pindex") == "2"  # base/walls default to rock

    verts = root.findall(f".//{{{CORE_NS}}}vertex")
    tris = root.findall(f".//{{{CORE_NS}}}triangle")
    assert len(verts) == len(solid.vertices)
    assert len(tris) == len(solid.faces)

    n_verts = len(solid.vertices)
    with_p1 = 0
    for t in tris:
        v = [int(t.get(k)) for k in ("v1", "v2", "v3")]
        assert all(0 <= i < n_verts for i in v)
        p1 = t.get("p1")
        if p1 is not None:
            assert 0 <= int(p1) < len(PALETTE)
            with_p1 += 1
    # Exactly the terrain-surface faces carry a class; base/walls rely on the
    # object default.
    assert with_p1 == int(np.sum(face_cls >= 0))
    assert with_p1 > 0


def test_all_four_classes_present_and_water_wins():
    solid, face_cls = _solid_with_classes()
    surface_cls = face_cls[face_cls >= 0]
    assert set(np.unique(surface_cls)) == {0, 1, 2, 3}

    # Water is elevation-independent: the flattened lake sits near 900 m
    # (forest band), but still classifies as water.
    dem = np.full((3, 3), 900.0, dtype=np.float32)
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, :] = True
    cells = classify_terrain_cells(dem, mask, treeline_m=1200.0, snowline_m=2000.0)
    assert np.all(cells[0, :] == 0)  # touching the lake -> water
    assert np.all(cells[1, :] == 1)  # below treeline -> forest


def test_classification_bands():
    dem = np.array(
        [[1000.0, 1000.0], [1500.0, 1500.0], [2500.0, 2500.0]], dtype=np.float32
    )
    cells = classify_terrain_cells(dem, None, treeline_m=1200.0, snowline_m=2000.0)
    # Cell means: row0 1250 (rock band boundary? 1250 >= 1200 -> rock),
    # row1 2000 -> snow (not < snowline).
    assert cells.shape == (2, 1)
    assert cells[0, 0] == 2
    assert cells[1, 0] == 3


def test_writer_rejects_bad_input(tmp_path):
    solid, face_cls = _solid_with_classes()
    out = str(tmp_path / "bad.3mf")
    with pytest.raises(ValueError, match="hex"):
        write_3mf(out, solid.vertices, solid.faces, face_cls, ["blue"], 0)
    with pytest.raises(ValueError, match="length"):
        write_3mf(out, solid.vertices, solid.faces, face_cls[:-1], PALETTE, 2)
