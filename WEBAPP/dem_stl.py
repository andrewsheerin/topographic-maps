import os
import json
import tempfile
import zipfile
import requests
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio import features
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import trimesh
import ezdxf
from pyproj import CRS, Transformer

# -----------------------------
# Constants
# -----------------------------

METERS_TO_MM = 1000.0

ROAD_WIDTHS_MM = {
    "motorway": 4.0,
    "trunk": 3.0,
    "primary": 2.0,
    "secondary": 0.5,
    "tertiary": 0.35,
    "residential": 0.25,
}

RECESS_DEPTH_MM = 2.0

# -----------------------------
# API key handling
# -----------------------------

def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (f.read() or "").strip()
    except OSError:
        return ""


def get_opentopo_api_key() -> str:
    """Return the OpenTopography API key if configured, else empty string.

    Supported sources (first non-empty wins):
    - env: OPEN_TOPO_API_KEY (preferred)
    - env: OPENTOPOGRAPHY_API_KEY (common alternative)
    - file: <repo-root>/API_KEY.txt
    - file: <WEBAPP>/API_KEY.txt
    """
    key = (os.environ.get("OPEN_TOPO_API_KEY") or "").strip()
    if key:
        return key

    key = (os.environ.get("OPENTOPOGRAPHY_API_KEY") or "").strip()
    if key:
        return key

    repo_root = os.path.dirname(os.path.dirname(__file__))
    key = _read_text_file(os.path.join(repo_root, "API_KEY.txt"))
    if key:
        return key

    key = _read_text_file(os.path.join(os.path.dirname(__file__), "API_KEY.txt"))
    return key


def require_opentopo_api_key() -> str:
    key = get_opentopo_api_key()
    if not key:
        raise RuntimeError(
            "Missing OpenTopography API key. Set OPEN_TOPO_API_KEY (or OPENTOPOGRAPHY_API_KEY) "
            "or create API_KEY.txt at the project root."
        )
    return key

# -----------------------------
# Geometry helpers
# -----------------------------

def polygon_from_geojson(geojson_obj):
    if geojson_obj.get("type") == "Feature":
        return shape(geojson_obj["geometry"])
    return shape(geojson_obj)


def utm_epsg_from_lon_lat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


# -----------------------------
# DEM acquisition + prep
# -----------------------------

# UI -> OpenTopography product mapping.
# Some products are served by /API/globaldem (demtype=...), while USGS DEMs
# are served by /API/usgsdem (datasetName=...).
DATASET_ALIASES = {
    # UI label -> OpenTopography datasetName (usgsdem)
    "USGS10m": "USGS10m",
    "USGS30m": "USGS30m",

    # Allow selecting these explicitly from the UI later.
    "SRTMGL1": "SRTMGL1",
    "COP30": "COP30",
    "COP90": "COP90",
    "NASADEM": "NASADEM",
    "AW3D30": "AW3D30",
}

USGSDEM_DATASETS = {"USGS10m", "USGS30m"}


def _normalize_dataset(name: str) -> str:
    if not name:
        return "COP30"
    name = str(name).strip()
    return DATASET_ALIASES.get(name, name)


def download_dem(w, s, e, n, out_path, dataset="USGS10m"):
    api_key = require_opentopo_api_key()
    dataset = _normalize_dataset(dataset)

    # USGS DEMs are NOT provided by the GlobalDEM endpoint.
    if dataset in USGSDEM_DATASETS:
        url = "https://portal.opentopography.org/API/usgsdem"
        params = dict(
            datasetName=dataset,
            west=w,
            south=s,
            east=e,
            north=n,
            outputFormat="GTiff",
            API_Key=api_key,
        )
        r = requests.get(url, params=params, timeout=180)
        if not r.ok:
            raise RuntimeError(
                "OpenTopography USGSDEM request failed "
                f"(HTTP {r.status_code}) for datasetName='{dataset}'. Response: {r.text.strip()}"
            )
    else:
        # Global products
        url = "https://portal.opentopography.org/API/globaldem"
        params = dict(
            demtype=dataset,
            west=w,
            south=s,
            east=e,
            north=n,
            outputFormat="GTiff",
            API_Key=api_key,
        )
        r = requests.get(url, params=params, timeout=180)
        if not r.ok:
            raise RuntimeError(
                "OpenTopography GlobalDEM request failed "
                f"(HTTP {r.status_code}) for demtype='{dataset}'. Response: {r.text.strip()}"
            )

    with open(out_path, "wb") as f:
        f.write(r.content)


def clip_dem_by_polygon(dem_path, polygon, crs, out_path):
    # `rasterio.mask` is a separate submodule; use `from rasterio.mask import mask`.
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
    with rasterio.open(dem_path) as src:
        gdf = gdf.to_crs(src.crs)
        out, transform = rio_mask(src, gdf.geometry, crop=True)
        meta = src.meta.copy()
        meta.update({
            "height": out.shape[1],
            "width": out.shape[2],
            "transform": transform
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(out)


def reproject_dem(src_path, dst_path, epsg):
    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(epsg)
        transform, w, h = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()

        # Preserve nodata through reprojection to avoid nodata becoming huge "valid" values.
        src_nodata = src.nodata
        dst_nodata = src_nodata if src_nodata is not None else -9999.0

        meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": w,
            "height": h,
            "nodata": dst_nodata,
        })

        with rasterio.open(dst_path, "w", **meta) as dst:
            rasterio.warp.reproject(
                rasterio.band(src, 1),
                rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=dst_nodata,
                resampling=Resampling.bilinear,
            )


def read_dem(path, downsample):
    with rasterio.open(path) as src:
        transform = src.transform
        nodata = src.nodata

        if downsample > 1:
            data = src.read(
                1,
                out_shape=(max(1, src.height // downsample), max(1, src.width // downsample)),
                resampling=Resampling.average,
            )
            transform = transform * Affine.scale(downsample)
        else:
            data = src.read(1)

        px_m = abs(transform.a)

    data = data.astype(np.float32)

    # Mask nodata explicitly (many DEMs use a large finite nodata value).
    if nodata is not None and np.isfinite(nodata):
        data[data == np.float32(nodata)] = np.nan

    data[~np.isfinite(data)] = np.nan
    return data, px_m, transform


# -----------------------------
# Roads (Overpass)
# -----------------------------

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]


def _post_overpass(query: str, *, timeout_s: int = 60, max_retries: int = 3) -> dict:
    """POST an Overpass query with retries and endpoint fallback.

    Overpass frequently returns 429/504 under load; this helper retries with
    exponential backoff and tries alternate public endpoints.

    Returns parsed JSON dict on success.
    """
    last_err = None

    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(max_retries):
            try:
                r = requests.post(endpoint, data=query.encode("utf-8"), timeout=timeout_s)
                r.raise_for_status()
                try:
                    return r.json()
                except ValueError as je:
                    # Overpass sometimes returns HTML even on 200
                    raise RuntimeError(
                        f"Overpass returned non-JSON response from {endpoint}. "
                        f"First 200 chars: {(r.text or '')[:200]}"
                    ) from je
            except Exception as e:
                last_err = e
                # backoff: 1s, 2s, 4s...
                try:
                    import time
                    time.sleep(2 ** attempt)
                except Exception:
                    pass
        # next endpoint

    raise RuntimeError(
        "Overpass request failed after retries. This is usually temporary (server load). "
        "Try again, reduce road detail, or use a smaller polygon. "
        f"Last error: {last_err}"
    )


def fetch_roads_geojson_overpass(polygon_wgs84, highway_levels):
    if not highway_levels:
        return {"type": "FeatureCollection", "features": []}

    coords = list(polygon_wgs84.exterior.coords)

    # Overpass polygons must be fairly small/short, otherwise it times out.
    # If the polygon has lots of vertices, downsample the ring to reduce query cost.
    if len(coords) > 300:
        step = max(1, len(coords) // 300)
        coords = coords[::step]
        # ensure closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

    poly_str = " ".join(f"{lat} {lon}" for lon, lat in coords)

    hw = "|".join(highway_levels)

    # 'out geom' is heavy but required for linework. Add a generous timeout.
    query = f"""
    [out:json][timeout:60];
    way["highway"~"{hw}"](poly:"{poly_str}");
    out geom;
    """

    data = _post_overpass(query, timeout_s=70, max_retries=3)

    features = []
    for el in data.get("elements", []):
        if "geometry" not in el or not el.get("tags"):
            continue
        highway = el["tags"].get("highway")
        if not highway:
            continue
        features.append({
            "type": "Feature",
            "properties": {"highway": highway},
            "geometry": {
                "type": "LineString",
                "coordinates": [(p["lon"], p["lat"]) for p in el["geometry"]],
            },
        })

    return {"type": "FeatureCollection", "features": features}


# -----------------------------
# Roads exports
# -----------------------------

def write_roads_geojson(roads_fc: dict, out_path: str) -> str:
    """Write a GeoJSON FeatureCollection (assumed EPSG:4326) to disk."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(roads_fc, f)
    return out_path


def roads_featurecollection_to_utm(roads_fc: dict, epsg: int) -> dict:
    """Reproject a roads FeatureCollection from EPSG:4326 into the given EPSG (UTM).

    Output is still GeoJSON coordinates, but in projected meters.
    We also set a `crs` member for convenience (note: deprecated in strict GeoJSON).
    """
    if not roads_fc.get("features"):
        return {
            "type": "FeatureCollection",
            "features": [],
            "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        }

    gdf = gpd.GeoDataFrame.from_features(roads_fc["features"], crs="EPSG:4326").to_crs(epsg)
    fc = {
        "type": "FeatureCollection",
        "features": json.loads(gdf.to_json()).get("features", []),
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
    }
    return fc


def clip_roads_gdf_to_polygon(gdf_roads_utm: gpd.GeoDataFrame, poly_utm) -> gpd.GeoDataFrame:
    """Clip roads to the terrain polygon/buffer in the same UTM CRS."""
    if gdf_roads_utm.empty:
        return gdf_roads_utm

    # Only keep geometries that intersect, then clip to the polygon boundary.
    gdf2 = gdf_roads_utm[gdf_roads_utm.geometry.intersects(poly_utm)].copy()
    if gdf2.empty:
        return gdf2

    gdf2["geometry"] = gdf2.geometry.intersection(poly_utm)
    gdf2 = gdf2[~gdf2.is_empty & gdf2.geometry.is_valid]
    return gdf2


# -----------------------------
# (Deprecated) DXF writer
# -----------------------------

# NOTE: Keeping this for now to avoid breaking older code paths, but the bundle
# now exports GeoJSON instead.

def write_roads_dxf(gdf, bounds, scale_xy, out_path):
    left, bottom, _, _ = bounds
    doc = ezdxf.new()
    msp = doc.modelspace()

    for level, width in ROAD_WIDTHS_MM.items():
        doc.layers.new(name=f"ROADS_{level.upper()}")

    for _, row in gdf.iterrows():
        geom = row.geometry
        level = row["highway"]
        pts = []
        for x, y in geom.coords:
            xm = (x - left) * scale_xy * METERS_TO_MM
            ym = (y - bottom) * scale_xy * METERS_TO_MM
            pts.append((xm, ym))
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": f"ROADS_{level.upper()}"})

    doc.saveas(out_path)


# -----------------------------
# DEM carving
# -----------------------------

def carve_roads(dem, transform, gdf, scale_xy, z_scale, road_etch=None):
    """Carve (recess) road buffers into the DEM.

    Contract / rule:
    - Carving MUST be applied from least depth -> most depth (deepest last).
      This allows deeper roads to overwrite shallower carves at overlaps.

    `road_etch` is expected to be a dict like:
      {
        "motorway": {"width_mm": 2.5, "depth_mm": 1.2},
        ...
      }

    Width/depth are in *print mm*.
    """
    dem2 = dem.copy()

    if road_etch is None:
        road_etch = {}

    def _finite_nonneg(v, *, cap):
        try:
            v = float(v)
        except Exception:
            return 0.0
        if not np.isfinite(v) or v < 0:
            return 0.0
        return min(v, cap)

    # 1) Build a stable carve plan (least depth first).
    plan = []
    for level in ROAD_WIDTHS_MM.keys():
        cfg = road_etch.get(level) or {}
        width_mm = _finite_nonneg(cfg.get("width_mm", ROAD_WIDTHS_MM.get(level, 0.0)), cap=50.0)
        depth_mm = _finite_nonneg(cfg.get("depth_mm", RECESS_DEPTH_MM), cap=20.0)
        if width_mm <= 0 or depth_mm <= 0:
            continue
        plan.append({
            "level": level,
            "width_mm": width_mm,
            "depth_mm": depth_mm,
        })

    if not plan:
        return dem2

    # sort by requested depth (ascending). If equal depths, keep deterministic ordering by level name.
    plan.sort(key=lambda d: (d["depth_mm"], d["level"]))

    # 2) Apply sequential carves in that order.
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
        shapes = [(geom, 1) for geom in buffered if geom is not None and not geom.is_empty]
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


# -----------------------------
# Mesh
# -----------------------------

def dem_to_mesh(dem, px_m, scale_xy, z_scale):
    """Convert a DEM (meters) to a triangular mesh in millimeters.

    X/Y are scaled to fit the requested print size via `scale_xy`.

    Z handling:
    - Normalize elevations so min elevation becomes 0 (relief only).
    - Convert meters -> millimeters using a *consistent* mm-per-meter factor.
      We intentionally do NOT allow absolute elevations to inflate Z.

    Note: `scale_xy` is mm-per-meter in X/Y. Using that same factor for Z
    preserves aspect ratio; `z_scale` is vertical exaggeration.
    """
    h, w = dem.shape

    # X axis flip fix:
    # Raster grids are indexed left->right by column, but depending on the
    # raster's transform sign conventions and how downstream slicers interpret
    # coordinates, users can perceive a west/east mirror.
    # Using a reversed X coordinate ensures the STL matches the map orientation
    # (east to the right) consistently.
    xs = np.arange(w)[::-1] * px_m * scale_xy * METERS_TO_MM
    ys = np.arange(h) * px_m * scale_xy * METERS_TO_MM
    xv, yv = np.meshgrid(xs, ys)

    finite = np.isfinite(dem)
    if not finite.any():
        raise RuntimeError("DEM contains no finite elevation samples after clipping.")

    z0 = float(np.nanmin(dem[finite]))
    dem_rel = dem - z0
    dem_rel[~finite] = 0.0

    # Guard against unit mistakes: if the DEM is accidentally in millimeters or
    # contains a huge relief, cap it to something reasonable for printing.
    relief_m = float(np.nanmax(dem_rel))
    if relief_m > 20000:  # 20km relief is almost certainly a unit/CRS issue
        raise RuntimeError(
            f"DEM relief looks unrealistic ({relief_m:.1f} m). "
            "This usually indicates a bad DEM or unit mismatch."
        )

    z_exag = float(z_scale)
    if z_exag <= 0:
        z_exag = 1.0
    if z_exag > 20:
        z_exag = 20.0

    # scale_xy is currently (target_mm / (max_dim_m*1000)). Multiply by 1000 to get mm/m.
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


def add_base(mesh, thickness_mm):
    minz = mesh.vertices[:, 2].min()
    base = mesh.copy()
    base.vertices[:, 2] = minz - thickness_mm
    return trimesh.util.concatenate([mesh, base])


# -----------------------------
# Public entry points
# -----------------------------

def generate_stl_from_polygon(
    polygon_wgs84,
    dem_dataset,
    downsample,
    z_scale,
    buffer_m,
    target_max_mm,
    add_base_flag,
    base_thickness_m,
):
    lon, lat = polygon_wgs84.centroid.coords[0]
    epsg = utm_epsg_from_lon_lat(lon, lat)

    tmp = tempfile.mkdtemp()
    dem = os.path.join(tmp, "dem.tif")
    dem2 = os.path.join(tmp, "dem_clip_wgs84.tif")
    dem3 = os.path.join(tmp, "dem_utm.tif")
    dem4 = os.path.join(tmp, "dem_utm_clip_buffer.tif")

    download_dem(*polygon_wgs84.bounds, dem, dem_dataset)
    clip_dem_by_polygon(dem, polygon_wgs84, "EPSG:4326", dem2)
    reproject_dem(dem2, dem3, epsg)

    poly_utm = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326").to_crs(epsg).iloc[0]
    poly_utm = poly_utm.buffer(float(buffer_m) if buffer_m else 0.0)

    # Clip again in UTM after buffering so the output includes the requested margin.
    clip_dem_by_polygon(dem3, poly_utm, epsg, dem4)

    dem_arr, px_m, transform = read_dem(dem4, downsample)

    # Compute scale based on real-world extent in meters (UTM), not pixel count.
    left = transform.c
    top = transform.f
    right = left + dem_arr.shape[1] * transform.a
    bottom = top + dem_arr.shape[0] * transform.e
    width_m = abs(right - left)
    height_m = abs(top - bottom)
    max_dim_m = max(width_m, height_m)
    if max_dim_m <= 0 or not np.isfinite(max_dim_m):
        raise RuntimeError("Clipped DEM has invalid extent; try reducing downsample or adjusting the polygon.")

    scale_xy = float(target_max_mm) / (max_dim_m * METERS_TO_MM)

    mesh = dem_to_mesh(dem_arr, px_m, scale_xy, z_scale)
    if add_base_flag:
        # base_thickness_m is a real-world thickness in meters; convert to mm after scaling.
        mesh = add_base(mesh, float(base_thickness_m) * METERS_TO_MM * scale_xy)

    out = os.path.join(tmp, "terrain.stl")
    mesh.export(out)
    return out


def generate_bundle_from_polygon(
    polygon_wgs84,
    dem_dataset,
    downsample,
    z_scale,
    buffer_m,
    target_max_mm,
    add_base_flag,
    base_thickness_m,
    road_levels,
    road_etch=None,
):
    lon, lat = polygon_wgs84.centroid.coords[0]
    epsg = utm_epsg_from_lon_lat(lon, lat)

    tmp = tempfile.mkdtemp()

    dem = os.path.join(tmp, "dem.tif")
    dem2 = os.path.join(tmp, "dem_clip_wgs84.tif")
    dem3 = os.path.join(tmp, "dem_utm.tif")
    dem4 = os.path.join(tmp, "dem_utm_clip_buffer.tif")

    download_dem(*polygon_wgs84.bounds, dem, dem_dataset)
    clip_dem_by_polygon(dem, polygon_wgs84, "EPSG:4326", dem2)
    reproject_dem(dem2, dem3, epsg)

    poly_utm = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326").to_crs(epsg).iloc[0]
    poly_utm = poly_utm.buffer(float(buffer_m) if buffer_m else 0.0)
    clip_dem_by_polygon(dem3, poly_utm, epsg, dem4)

    dem_arr, px_m, transform = read_dem(dem4, downsample)

    # Scale based on real-world UTM extent.
    left = transform.c
    top = transform.f
    right = left + dem_arr.shape[1] * transform.a
    bottom = top + dem_arr.shape[0] * transform.e
    width_m = abs(right - left)
    height_m = abs(top - bottom)
    max_dim_m = max(width_m, height_m)
    if max_dim_m <= 0 or not np.isfinite(max_dim_m):
        raise RuntimeError("Clipped DEM has invalid extent; try reducing downsample or adjusting the polygon.")

    scale_xy = float(target_max_mm) / (max_dim_m * METERS_TO_MM)

    # Roads (fetch in WGS84, then reproject + clip in UTM to match terrain footprint)
    roads_fc_wgs84 = fetch_roads_geojson_overpass(polygon_wgs84, road_levels)

    gdf_roads_utm = gpd.GeoDataFrame.from_features(
        roads_fc_wgs84.get("features", []),
        crs="EPSG:4326",
    ).to_crs(epsg)

    gdf_roads_utm = clip_roads_gdf_to_polygon(gdf_roads_utm, poly_utm)

    roads_fc_utm = {
        "type": "FeatureCollection",
        "features": json.loads(gdf_roads_utm.to_json()).get("features", []),
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
    }

    roads_geojson_utm_path = os.path.join(tmp, "roads_centerlines_utm.geojson")
    write_roads_geojson(roads_fc_utm, roads_geojson_utm_path)

    # Use the clipped UTM roads for carving
    gdf = gdf_roads_utm

    # Raw mesh
    raw_mesh = dem_to_mesh(dem_arr, px_m, scale_xy, z_scale)
    if add_base_flag:
        raw_mesh = add_base(raw_mesh, float(base_thickness_m) * METERS_TO_MM * scale_xy)

    raw_stl = os.path.join(tmp, "terrain_raw.stl")
    raw_mesh.export(raw_stl)

    # Carved
    carved_dem = carve_roads(dem_arr, transform, gdf, scale_xy, z_scale, road_etch=road_etch)
    carved_mesh = dem_to_mesh(carved_dem, px_m, scale_xy, z_scale)
    if add_base_flag:
        carved_mesh = add_base(carved_mesh, float(base_thickness_m) * METERS_TO_MM * scale_xy)

    carved_stl = os.path.join(tmp, "terrain_with_roads_recess.stl")
    carved_mesh.export(carved_stl)

    zip_path = os.path.join(tmp, "terrain_bundle.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(raw_stl, "terrain_raw.stl")
        z.write(roads_geojson_utm_path, "roads_centerlines_utm.geojson")
        z.write(carved_stl, "terrain_with_roads_recess.stl")

    return zip_path
