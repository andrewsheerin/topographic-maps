"""Smoke test: verify carve plan ordering is least depth first.

This intentionally doesn't call rasterio/shapely. It just checks the sorting logic
that carve_roads uses.

Run:
  python WEBAPP/test_carve_plan_sort.py
"""


def build_plan(road_widths_mm, road_etch, recess_depth_mm=2.0):
    def _finite_nonneg(v, *, cap):
        try:
            v = float(v)
        except Exception:
            return 0.0
        if v != v or v < 0:  # NaN check
            return 0.0
        return min(v, cap)

    plan = []
    for level in road_widths_mm.keys():
        cfg = road_etch.get(level) or {}
        width_mm = _finite_nonneg(cfg.get("width_mm", road_widths_mm.get(level, 0.0)), cap=50.0)
        depth_mm = _finite_nonneg(cfg.get("depth_mm", recess_depth_mm), cap=20.0)
        if width_mm <= 0 or depth_mm <= 0:
            continue
        plan.append({"level": level, "width_mm": width_mm, "depth_mm": depth_mm})

    plan.sort(key=lambda d: (d["depth_mm"], d["level"]))
    return plan


def main():
    road_widths = {
        "motorway": 4.0,
        "secondary": 1.0,
        "tertiary": 0.5,
    }

    road_etch = {
        "motorway": {"width_mm": 3, "depth_mm": 1.5},
        "secondary": {"width_mm": 2, "depth_mm": 0.7},
        "tertiary": {"width_mm": 1, "depth_mm": 1.2},
    }

    plan = build_plan(road_widths, road_etch)
    got = [(p["level"], p["depth_mm"]) for p in plan]
    expected = [("secondary", 0.7), ("tertiary", 1.2), ("motorway", 1.5)]

    assert got == expected, f"Expected {expected}, got {got}"
    print("OK", got)


if __name__ == "__main__":
    main()

