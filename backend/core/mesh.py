"""DEM -> printable mesh, plus road carving.

Units: DEM elevations are metres; output vertices are millimetres. `scale_xy` is
the print scale as mm-per-metre / METERS_TO_MM (i.e. target_mm / extent_mm);
`z_scale` is vertical exaggeration."""

import numpy as np
import trimesh
from rasterio import features
from rasterio.transform import Affine

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


def flatten_water(dem, transform, water_polys, scale_xy, z_scale, recess_mm):
    """Flatten each water polygon to its own plane, recessed below shore (F-22).

    COP30 is radar-derived and its returns over open water are noise or voids,
    so lake surfaces are not planar in the source raster. Each polygon is
    rasterized SEPARATELY and set to the WATER_LEVEL_PERCENTILE of the original
    DEM inside it, minus the print-mm recess — lakes at different elevations
    each get their own plane; there is no global water elevation.

    Returns (new_dem, water_mask); the mask protects streams from carving
    through a lake plane and feeds the 3MF face classification (F-23)."""
    dem2 = dem.copy()
    water_mask = np.zeros(dem.shape, dtype=bool)

    recess_mm = _finite_nonneg(recess_mm, cap=constants.CARVE_DEPTH_CAP_MM)
    delta = recess_mm / (scale_xy * z_scale * METERS_TO_MM)

    for geom in water_polys:
        if geom is None or geom.is_empty:
            continue
        mask = (
            features.rasterize(
                [(geom, 1)],
                out_shape=dem.shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True,
            )
            == 1
        ) & np.isfinite(dem)
        if not mask.any():
            continue
        plane_m = float(
            np.percentile(dem[mask], constants.WATER_LEVEL_PERCENTILE)
        )
        dem2[mask] = np.float32(plane_m - delta)
        water_mask |= mask

    return dem2, water_mask


def flatten_sea(dem, sea_level_m, scale_xy, z_scale, recess_mm):
    """Flatten every cell at or below sea level to one recessed plane (F-33).

    Ocean needs no fetch — COP30 sits at ~0 m (noisy) over open water, so an
    elevation threshold is the mask. The plane is sea_level_m minus the same
    print-mm recess lakes use. Returns (new_dem, sea_mask); the mask joins
    the water color class and protects the bay from stream carves. Known
    limitation: below-sea-level inland basins would also match."""
    dem2 = dem.copy()
    sea_mask = np.isfinite(dem) & (dem <= float(sea_level_m))
    if sea_mask.any():
        recess_mm = _finite_nonneg(recess_mm, cap=constants.CARVE_DEPTH_CAP_MM)
        delta = recess_mm / (scale_xy * z_scale * METERS_TO_MM)
        dem2[sea_mask] = np.float32(float(sea_level_m) - delta)
    return dem2, sea_mask


def stream_width_floor_mm(px_m, scale_xy):
    """The narrowest renderable stream width, print mm (F-28 W-5): one DEM
    cell at print scale, or one extrusion line, whichever is larger."""
    cell_mm = px_m * scale_xy * METERS_TO_MM
    return max(cell_mm, constants.EXTRUSION_LINE_WIDTH_MM)


def _fractional_coverage(geoms_buffered, shape, transform, aa):
    """Area fraction of each working cell covered by the buffers (F-28 W-2):
    rasterize on an `aa`-times finer grid (cell-center test, unbiased), then
    block-average back down to [0, 1] per cell."""
    shapes = [(g, 1) for g in geoms_buffered if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=np.float32)
    h, w = shape
    fine = features.rasterize(
        shapes,
        out_shape=(h * aa, w * aa),
        transform=transform * Affine.scale(1.0 / aa),
        fill=0,
        dtype=np.uint8,
    )
    return fine.reshape(h, aa, w, aa).mean(axis=(1, 3), dtype=np.float32)


def carve_streams(
    dem, transform, gdf_streams, scale_xy, z_scale, widths_mm, depth_mm,
    protect_mask=None, aa=constants.STREAM_AA_FACTOR,
):
    """Recess waterway centerlines into the DEM (F-22, reworked in F-28).

    Widths/depth are print millimetres (ROAD_WIDTHS_MM convention) — real
    stream widths are far below one extrusion width at print scale. Per
    waterway class (`gdf_streams.waterway`, widths from `widths_mm`), the
    carve is a V-profile: STREAM_PROFILE's concentric rings applied
    shallow-to-deep — the ordering contract build_carve_plan implements for
    roads — with deeper rings overwriting shallower ones (per-cell max).
    Each ring's depth is scaled by its fractional cell coverage
    (anti-aliasing, W-2), so sub-cell widths carve shallow instead of
    ragged. Requested widths are clamped up to stream_width_floor_mm (W-5).

    Cells in `protect_mask` (flattened lakes) are left untouched. Returns
    (new_dem, color_mask): color_mask is the cells with ≥50% total coverage,
    which join the water color class in the 3MF (F-27)."""
    dem2 = dem.copy()
    color_mask = np.zeros(dem.shape, dtype=bool)

    depth_mm = _finite_nonneg(depth_mm, cap=constants.CARVE_DEPTH_CAP_MM)
    if depth_mm <= 0 or len(gdf_streams) == 0:
        return dem2, color_mask

    px_m = abs(transform.a)
    floor_mm = stream_width_floor_mm(px_m, scale_xy)
    delta_full = depth_mm / (scale_xy * z_scale * METERS_TO_MM)

    depth_field = np.zeros(dem.shape, dtype=np.float32)
    coverage_total = np.zeros(dem.shape, dtype=np.float32)

    for wclass in constants.STREAM_WIDTHS_MM:
        subset = gdf_streams[gdf_streams["waterway"] == wclass]
        if subset.empty:
            continue
        width_mm = _finite_nonneg(
            (widths_mm or {}).get(wclass, constants.STREAM_WIDTHS_MM[wclass]),
            cap=constants.CARVE_WIDTH_CAP_MM,
        )
        if width_mm <= 0:
            continue
        width_mm = max(width_mm, floor_mm)  # W-5: clamp, never a ragged chain

        # Shallow-to-deep ring order (build_carve_plan's contract); np.maximum
        # makes deeper rings win where they overlap the shallower ones.
        for width_frac, depth_frac in sorted(
            constants.STREAM_PROFILE, key=lambda p: p[1]
        ):
            half_m = (width_mm * width_frac / 2) / (scale_xy * METERS_TO_MM)
            coverage = _fractional_coverage(
                subset.geometry.buffer(half_m), dem.shape, transform, aa
            )
            np.maximum(
                depth_field,
                np.float32(delta_full * depth_frac) * coverage,
                out=depth_field,
            )
            if width_frac == 1.0:
                np.maximum(coverage_total, coverage, out=coverage_total)

    apply_mask = np.isfinite(dem2) & (depth_field > 0)
    if protect_mask is not None:
        apply_mask &= ~protect_mask
    dem2[apply_mask] -= depth_field[apply_mask]

    color_mask = (coverage_total >= 0.5) & apply_mask
    return dem2, color_mask


def _remove_pinch_cells(cell_ok):
    """Drop one cell of every diagonally-touching pair (F-20).

    Two cells sharing only a corner put two boundary loops through the same
    vertex; sealing then gives that corner's vertical wall edge four incident
    faces — non-manifold, so the watertight guard rejects it. Eroding one cell
    (one DEM pixel, invisible at print scale) restores manifoldness. Iterates
    because a removal can expose a new diagonal pair; strictly monotone, so it
    terminates."""
    cell_ok = cell_ok.copy()
    while True:
        nw_se = (
            cell_ok[:-1, :-1] & cell_ok[1:, 1:] & ~cell_ok[:-1, 1:] & ~cell_ok[1:, :-1]
        )
        ne_sw = (
            cell_ok[:-1, 1:] & cell_ok[1:, :-1] & ~cell_ok[:-1, :-1] & ~cell_ok[1:, 1:]
        )
        if not (nw_se.any() or ne_sw.any()):
            return cell_ok
        drop = np.zeros_like(cell_ok)
        drop[1:, 1:] |= nw_se  # drop the south-east cell of a NW/SE pair
        drop[1:, :-1] |= ne_sw  # drop the south-west cell of a NE/SW pair
        cell_ok &= ~drop


def cell_ok_mask(dem):
    """The (h-1, w-1) mask of grid cells that become faces: all four corners
    finite (F-19), diagonal pinches eroded (F-20). Shared by dem_to_mesh and
    the 3MF face classification (F-23) so face count and order always agree."""
    finite = np.isfinite(dem)
    cell_ok = finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, :-1] & finite[1:, 1:]
    return _remove_pinch_cells(cell_ok)


def classify_terrain_cells(dem, water_mask, cutoffs_m, band_offset=0):
    """Per-cell palette index for the 3MF export (F-32: dynamic bands).

    `cutoffs_m` is the ascending list of band upper bounds in metres —
    n_bands - 1 entries, the highest band is unbounded. A cell's 4-corner
    mean elevation picks the first band whose cutoff exceeds it (strict `<`,
    exactly as the old treeline/snowline behaved). Band i maps to palette
    index band_offset + i. Water wins first: cells with any corner pixel in
    `water_mask` override to palette index 0 (the mask-driven water slot) —
    pass water_mask=None when the palette has no water slot. The cutoffs are
    per-request user values (science-integrity: never defaulted here)."""
    mean_m = (dem[:-1, :-1] + dem[:-1, 1:] + dem[1:, :-1] + dem[1:, 1:]) / 4.0
    idx = np.searchsorted(
        np.asarray(cutoffs_m, dtype=np.float64), mean_m, side="right"
    )
    cls = (idx + int(band_offset)).astype(np.int16)
    if water_mask is not None:
        water_cell = (
            water_mask[:-1, :-1]
            | water_mask[:-1, 1:]
            | water_mask[1:, :-1]
            | water_mask[1:, 1:]
        )
        cls[water_cell] = 0
    return cls


def face_classes_for_solid(dem, cell_classes, n_solid_faces):
    """Per-face classes aligned with dem_to_mesh + add_base face order:
    [cell triangles A, cell triangles B, bottom, walls]. Base and wall faces
    get -1 — the 3MF object-level default color."""
    ok = cell_ok_mask(dem).ravel()
    c = cell_classes.ravel()[ok]
    surface = np.concatenate([c, c])
    base = np.full(n_solid_faces - len(surface), -1, dtype=surface.dtype)
    return np.concatenate([surface, base])


def dem_to_mesh(dem, px_m, scale_xy, z_scale):
    """Convert a DEM (metres) to a triangular surface mesh in millimetres,
    cropped to the finite (in-boundary) cells.

    X/Y are scaled to the requested print size; Z is relief-only (min elevation
    -> 0) using the same mm-per-metre factor as X/Y so aspect ratio is
    preserved, times `z_scale` vertical exaggeration.

    Cells with any nodata corner are dropped (F-19): the mesh takes the shape
    of the selected area — a state print is state-shaped, islands become
    separate patches, and nodata gaps inside the DEM stay visible as holes
    (never silently filled). `add_base` later seals every open boundary."""
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
    dem_rel[~finite] = 0.0  # placeholder; these vertices are dropped below

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

    # Two triangles per grid cell, vectorized (F-10), keeping only cells whose
    # four corners are all finite (F-19).
    idx = np.arange(h * w).reshape(h, w)
    cell_ok = cell_ok_mask(dem).ravel()
    if not cell_ok.any():
        raise RuntimeError(
            "No complete grid cells inside the area — the area is too small for "
            "this DEM resolution; reduce downsample or pick a larger area."
        )
    tl = idx[:-1, :-1].ravel()[cell_ok]
    tr = idx[:-1, 1:].ravel()[cell_ok]
    bl = idx[1:, :-1].ravel()[cell_ok]
    br = idx[1:, 1:].ravel()[cell_ok]
    faces = np.concatenate(
        [
            np.column_stack([tl, tr, bl]),
            np.column_stack([tr, br, bl]),
        ]
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


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
