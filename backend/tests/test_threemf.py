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
    # Bands equivalent to the old treeline/snowline model: water slot at 0,
    # then <1200, <2000, above.
    cells = classify_terrain_cells(
        dem2, water_mask, [1200.0, 2000.0], band_offset=1
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
    # (lowest band), but still classifies as water.
    dem = np.full((3, 3), 900.0, dtype=np.float32)
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, :] = True
    cells = classify_terrain_cells(dem, mask, [1200.0, 2000.0], band_offset=1)
    assert np.all(cells[0, :] == 0)  # touching the lake -> water slot
    assert np.all(cells[1, :] == 1)  # below the first cutoff -> first band


def test_carved_stream_cells_classify_as_water():
    # F-27: the water mask passed to classification is lakes | streams, so a
    # carved stream in the forest band prints in the water color.
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import LineString

    from core.mesh import carve_streams

    dem = np.full((6, 6), 900.0, dtype=np.float32)
    transform = from_origin(0.0, 60.0, 10.0, 10.0)
    streams = gpd.GeoDataFrame(
        {"waterway": ["stream"]},
        geometry=[LineString([(0.0, 35.0), (60.0, 35.0)])],
        crs=32611,
    )
    _, stream_mask = carve_streams(
        dem, transform, streams, 0.001, 1.0, {"stream": 1.0}, 0.4
    )
    assert stream_mask.any()
    cells = classify_terrain_cells(dem, stream_mask, [1200.0], band_offset=1)
    assert np.all(cells[stream_mask[:-1, :-1]] == 0)


def test_classification_bands():
    dem = np.array(
        [[1000.0, 1000.0], [1500.0, 1500.0], [2500.0, 2500.0]], dtype=np.float32
    )
    # No water slot: band_offset 0, cutoffs [1200, 2000].
    cells = classify_terrain_cells(dem, None, [1200.0, 2000.0])
    # Cell means: row0 1250 -> second band; row1 2000 (== cutoff, strict <)
    # -> third band. Boundary behavior matches the old treeline/snowline.
    assert cells.shape == (2, 1)
    assert cells[0, 0] == 1
    assert cells[1, 0] == 2


def test_classification_arbitrary_band_count():
    # Column of cells whose means step 150, 350, 550, 750, 950.
    col = np.array([100.0, 200.0, 500.0, 600.0, 900.0, 1000.0], dtype=np.float32)
    dem = np.column_stack([col, col])
    # Single unbounded band: everything is index 0.
    assert np.all(classify_terrain_cells(dem, None, []) == 0)
    # Five bands, no water slot: each cell lands in its own band.
    cells = classify_terrain_cells(dem, None, [200.0, 400.0, 600.0, 800.0])
    assert list(cells[:, 0]) == [0, 1, 2, 3, 4]
    # With a water slot, everything shifts up by one.
    shifted = classify_terrain_cells(dem, None, [200.0, 400.0, 600.0, 800.0],
                                     band_offset=1)
    assert list(shifted[:, 0]) == [1, 2, 3, 4, 5]


def test_write_colored_3mf_sorts_bands_and_builds_palette(tmp_path):
    # Pipeline-level (F-32): bands arrive unsorted, water slot present.
    from core.pipeline import _write_colored_3mf

    rng = np.random.default_rng(3)
    dem = rng.uniform(900.0, 1000.0, size=(8, 8)).astype(np.float32)
    transform = from_origin(0.0, 80.0, PX_M, PX_M)
    dem2, water_mask = flatten_water(
        dem, transform, [box(21.0, 21.0, 59.0, 59.0)], SCALE_XY, Z_SCALE, 0.6
    )
    solid = add_base(dem_to_mesh(dem2, PX_M, SCALE_XY, Z_SCALE), 2.0)

    color = {
        "water_color": "#001489",
        "bands": [
            {"upto_m": None, "color": "#FFFFFF"},
            {"upto_m": 950.0, "color": "#034638"},  # unsorted on purpose
            {"upto_m": 980.0, "color": "#7F7E83"},
        ],
    }
    out = str(tmp_path / "bands.3mf")
    _write_colored_3mf(out, dem2, water_mask, solid, color)

    root = _read_model_xml(out)
    colors = root.findall(f".//{{{MATERIAL_NS}}}colorgroup/{{{MATERIAL_NS}}}color")
    # Palette: water first, then bands sorted by cutoff, unbounded last.
    assert [c.get("color") for c in colors] == [
        "#001489", "#034638", "#7F7E83", "#FFFFFF",
    ]
    obj = root.find(f".//{{{CORE_NS}}}object")
    assert obj.get("pindex") == "1"  # base/walls take the lowest band

    # Two unbounded bands is a user error, said plainly.
    bad = {
        "water_color": None,
        "bands": [
            {"upto_m": None, "color": "#FFFFFF"},
            {"upto_m": None, "color": "#000000"},
        ],
    }
    with pytest.raises(RuntimeError, match="highest color band"):
        _write_colored_3mf(str(tmp_path / "bad.3mf"), dem2, None, solid, bad)


def test_writer_rejects_bad_input(tmp_path):
    solid, face_cls = _solid_with_classes()
    out = str(tmp_path / "bad.3mf")
    with pytest.raises(ValueError, match="hex"):
        write_3mf(out, solid.vertices, solid.faces, face_cls, ["blue"], 0)
    with pytest.raises(ValueError, match="length"):
        write_3mf(out, solid.vertices, solid.faces, face_cls[:-1], PALETTE, 2)
