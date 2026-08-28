"""Tests for the multi-part-area handling in the Overpass road fetch (F-13):
query ring construction and clipping lines back to the true boundary."""

from shapely.geometry import MultiPolygon, Polygon

from core.roads import clip_line_features_to_area, outer_ring_coords

BOX_A = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
BOX_B = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
MULTI = MultiPolygon([BOX_A, BOX_B])


def test_outer_ring_is_own_exterior_for_polygon():
    assert outer_ring_coords(BOX_A) == list(BOX_A.exterior.coords)


def test_outer_ring_is_hull_for_multipolygon():
    ring = Polygon(outer_ring_coords(MULTI))
    assert ring.equals(MULTI.convex_hull)
    # The hull covers the water gap, so the query region spans both parts.
    assert ring.contains(Polygon([(1.2, 0.2), (1.8, 0.2), (1.8, 0.8), (1.2, 0.8)]))


def test_clip_splits_line_crossing_water_gap():
    road = {
        "type": "Feature",
        "properties": {"highway": "primary"},
        "geometry": {
            "type": "LineString",
            "coordinates": [(0.5, 0.5), (2.5, 0.5)],
        },
    }
    clipped = clip_line_features_to_area([road], MULTI)

    assert len(clipped) == 2
    assert all(f["properties"] == {"highway": "primary"} for f in clipped)
    xs = sorted(x for f in clipped for x, _ in f["geometry"]["coordinates"])
    # Nothing survives inside the gap between x=1 and x=2.
    assert all(x <= 1.0 or x >= 2.0 for x in xs)


def test_clip_drops_lines_entirely_outside():
    road = {
        "type": "Feature",
        "properties": {"highway": "primary"},
        "geometry": {
            "type": "LineString",
            "coordinates": [(1.2, 0.5), (1.8, 0.5)],
        },
    }
    assert clip_line_features_to_area([road], MULTI) == []
