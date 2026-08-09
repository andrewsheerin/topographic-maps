"""Verify the carve plan orders roads least-depth-first (deepest carved last).

Exercises the real `core.mesh.build_carve_plan`; no rasterio/shapely/network."""

from core.mesh import build_carve_plan


def test_plan_sorted_least_depth_first_and_drops_zero_width():
    road_etch = {
        "motorway": {"width_mm": 3.0, "depth_mm": 1.5},
        "trunk": {"width_mm": 3.0, "depth_mm": 0.9},
        "primary": {"width_mm": 2.0, "depth_mm": 1.1},
        "secondary": {"width_mm": 2.0, "depth_mm": 0.7},
        "tertiary": {"width_mm": 1.0, "depth_mm": 1.2},
        "residential": {"width_mm": 0.0, "depth_mm": 0.5},  # zero width -> dropped
    }

    plan = build_carve_plan(road_etch)
    order = [(p["level"], p["depth_mm"]) for p in plan]

    assert order == [
        ("secondary", 0.7),
        ("trunk", 0.9),
        ("primary", 1.1),
        ("tertiary", 1.2),
        ("motorway", 1.5),
    ]


if __name__ == "__main__":
    test_plan_sorted_least_depth_first_and_drops_zero_width()
    print("OK")
