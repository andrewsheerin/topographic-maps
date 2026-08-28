"""DEM -> printable mesh, plus road carving.

Units: DEM elevations are metres; output vertices are millimetres. `scale_xy` is
the print scale as mm-per-metre / METERS_TO_MM (i.e. target_mm / extent_mm);
`z_scale` is vertical exaggeration."""

import numpy as np
import trimesh
from rasterio import features

import constants

METERS_TO_MM = constants.METERS_TO_MM


def _finite_nonneg(v, *, cap):
    """Coerce to a finite value in [0, cap]; non-numeric/negative/NaN -> 0."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v) or v < 0:
        return 0.0
    return min(v, cap)


def build_carve_plan(road_etch=None):
    """Build the ordered carve plan (least depth first, deepest last).

    Ordering is the contract: applying carves shallow->deep lets deeper roads
    overwrite shallower carves where buffers overlap. Ties broken by level name
    for determinism. Widths/depths are print-mm, clamped to the sanity caps.

    Returns a list of {"level", "width_mm", "depth_mm"}.
    """
    if road_etch is None:
        road_etch = {}

    plan = []
    for level in constants.ROAD_WIDTHS_MM.keys():
        cfg = road_etch.get(level) or {}
        width_mm = _finite_nonneg(
            cfg.get("width_mm", constants.ROAD_WIDTHS_MM.get(level, 0.0)),
            cap=constants.CARVE_WIDTH_CAP_MM,
        )
        depth_mm = _finite_nonneg(
            cfg.get("depth_mm", constants.RECESS_DEPTH_MM),
            cap=constants.CARVE_DEPTH_CAP_MM,
        )
        if width_mm <= 0 or depth_mm <= 0:
            continue
        plan.append({"level": level, "width_mm": width_mm, "depth_mm": depth_mm})

    plan.sort(key=lambda d: (d["depth_mm"], d["level"]))
    return plan


def carve_roads(dem, transform, gdf, scale_xy, z_scale, road_etch=None):
    """Recess road buffers into the DEM, applying carves least-depth-first so
    the deepest carve wins at overlaps. Returns a new array."""
    dem2 = dem.copy()

    plan = build_carve_plan(road_etch)
    if not plan:
        return dem2

    for item in plan:
        level = item["level"]
        width_mm = item["width_mm"]
        depth_mm = item["depth_mm"]

        subset = gdf[gdf["highway"] == level]
        if subset.empty:
            continue

        delta = depth_mm / (scale_xy * z_scale * METERS_TO_MM)
        if delta <= 0 or not np.isfinite(delta):
            continue

        half_m = (width_mm / 2) / (scale_xy * METERS_TO_MM)
        buffered = subset.geometry.buffer(half_m)
        shapes = [
            (geom, 1) for geom in buffered if geom is not None and not geom.is_empty
        ]
        if not shapes:
            continue

        mask = features.rasterize(
            shapes,
            out_shape=dem2.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        dem2[(mask == 1) & np.isfinite(dem2)] -= np.float32(delta)

    return dem2


def dem_to_mesh(dem, px_m, scale_xy, z_scale):
    """Convert a DEM (metres) to a triangular mesh in millimetres.

    X/Y are scaled to the requested print size; Z is relief-only (min elevation
    -> 0) using the same mm-per-metre factor as X/Y so aspect ratio is
    preserved, times `z_scale` vertical exaggeration."""
    h, w = dem.shape

    # Reverse X so east is to the right in the exported STL regardless of the
    # raster transform's sign conventions.
    xs = np.arange(w)[::-1] * px_m * scale_xy * METERS_TO_MM
    ys = np.arange(h) * px_m * scale_xy * METERS_TO_MM
    xv, yv = np.meshgrid(xs, ys)

    finite = np.isfinite(dem)
    if not finite.any():
        raise RuntimeError("DEM contains no finite elevation samples after clipping.")

    z0 = float(np.nanmin(dem[finite]))
    dem_rel = dem - z0
    dem_rel[~finite] = 0.0

    # Guard against unit/CRS mistakes producing absurd relief.
    relief_m = float(np.nanmax(dem_rel))
    if relief_m > constants.RELIEF_SANITY_MAX_M:
        raise RuntimeError(
            f"DEM relief looks unrealistic ({relief_m:.1f} m). "
            "This usually indicates a bad DEM or unit mismatch."
        )

    z_exag = float(z_scale)
    if z_exag <= 0:
        z_exag = 1.0
    z_exag = min(z_exag, constants.Z_EXAG_MAX)

    mm_per_meter = scale_xy * METERS_TO_MM
    z = dem_rel * z_exag * mm_per_meter

    vertices = np.column_stack([xv.ravel(), yv.ravel(), z.ravel()])

    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x
            faces.append([i, i + 1, i + w])
            faces.append([i + 1, i + w + 1, i + w])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)


def add_base(surface, thickness_mm):
    """Seal an open heightmap surface into a watertight solid (F-16): an
    inverted copy of the surface `thickness_mm` below its lowest point forms the
    bottom, and quad walls close the open boundary. Normals are oriented
    outward, and a non-watertight result is refused rather than exported —
    slicers silently drop regions of non-manifold meshes.
    """
    thickness_mm = float(thickness_mm)
    if thickness_mm <= 0:
        raise ValueError("Base thickness must be > 0 mm to form a printable solid.")

    n = len(surface.vertices)
    minz = surface.vertices[:, 2].min()

    bottom_vertices = surface.vertices.copy()
    bottom_vertices[:, 2] = minz - thickness_mm
    bottom_faces = surface.faces[:, ::-1] + n  # reversed winding -> faces down

    # Directed edges that belong to exactly one face are the open boundary.
    directed = surface.edges
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    boundary = directed[counts[inverse] == 1]
    if len(boundary) == 0:
        raise RuntimeError("Surface has no open boundary; cannot seal a base onto it.")

    # Wall quads: for a boundary edge a->b on the top, the wall presents b->a
    # (opposite direction), keeping the winding consistent with top and bottom.
    a, b = boundary[:, 0], boundary[:, 1]
    walls = np.concatenate(
        [
            np.column_stack([b, a, a + n]),
            np.column_stack([b, a + n, b + n]),
        ]
    )

    solid = trimesh.Trimesh(
        vertices=np.vstack([surface.vertices, bottom_vertices]),
        faces=np.vstack([surface.faces, bottom_faces, walls]),
        process=False,
    )
    if solid.volume < 0:
        solid.invert()

    if not solid.is_watertight or not solid.is_winding_consistent:
        raise RuntimeError(
            "Generated mesh is not watertight — refusing to export a broken STL. "
            "This is a bug; please report the polygon and settings used."
        )
    return solid
