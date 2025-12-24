import os
import requests
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from affine import Affine
import trimesh


# -----------------------------
# CONFIG
# -----------------------------

API_KEY = "YOUR_API_KEY"

SHAPEFILE_PATH = "DATA/JacksonMountain.shp"
OUTPUT_DIR = "outputs"

DEM_DATASET = "USGS10m"
DOWNSAMPLE = 10  # integer downsample factor (1 = none)
Z_SCALE = 2.0  # vertical exaggeration factor

ADD_BASE = True
METERS_TO_MM = 1000.0
BASE_THICKNESS_MM = 50.0 * 1000.0  # 50 m base thickness expressed in mm
BUFFER_METERS = 50.0               # buffer around polygon in meters (UTM meters)
TARGET_MAX_MM = 200.0              # max X/Y size of the print in millimeters


os.makedirs(OUTPUT_DIR, exist_ok=True)

DEM_PATH = os.path.join(OUTPUT_DIR, "dem.tif")
CLIPPED_DEM_PATH = os.path.join(OUTPUT_DIR, "dem_clipped.tif")
PROJECTED_DEM_PATH = os.path.join(OUTPUT_DIR, "dem_utm.tif")
BUFFERED_CLIPPED_UTM_PATH = os.path.join(OUTPUT_DIR, "dem_utm_buffered_clip.tif")
STL_PATH = os.path.join(OUTPUT_DIR, "terrain.stl")


# -----------------------------
# IO
# -----------------------------

def utm_epsg_from_lon_lat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def download_dem(west, south, east, north, out_path):
    url = "https://portal.opentopography.org/API/usgsdem"
    params = {
        "datasetName": DEM_DATASET,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "outputFormat": "GTiff",
        "API_Key": API_KEY,
    }

    req = requests.Request("GET", url, params=params).prepare()
    print("Request URL:")
    print(req.url)

    r = requests.get(req.url, timeout=180)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


def load_polygon_wgs84(shapefile):
    gdf = gpd.read_file(shapefile)
    geom = gdf.geometry.union_all()
    if gdf.crs is None:
        raise RuntimeError("Shapefile has no CRS")
    return gpd.GeoSeries([geom], crs=gdf.crs).to_crs(epsg=4326).iloc[0]


def clip_dem_by_polygon(dem_path, polygon, polygon_crs, out_path):
    """
    Masks/crops a raster by a polygon. polygon_crs must be the CRS of `polygon`.
    """
    with rasterio.open(dem_path) as src:
        poly_in_src = gpd.GeoSeries([polygon], crs=polygon_crs).to_crs(src.crs).iloc[0]
        data, transform = mask(src, [poly_in_src], crop=True)
        meta = src.meta.copy()
        meta.update({"height": data.shape[1], "width": data.shape[2], "transform": transform})

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)

    return out_path


def reproject_dem_to_epsg(src_path, dst_path, epsg):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, f"EPSG:{epsg}", src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({"crs": f"EPSG:{epsg}", "transform": transform, "width": width, "height": height})

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=f"EPSG:{epsg}",
                resampling=Resampling.bilinear,
            )

    return dst_path


# -----------------------------
# PROCESSING
# -----------------------------

def read_dem(path, downsample):
    with rasterio.open(path) as src:
        nodata = src.nodata
        transform = src.transform

        if downsample > 1:
            data = src.read(
                1,
                out_shape=(src.height // downsample, src.width // downsample),
                resampling=Resampling.average,
            )
            transform = transform * Affine.scale(downsample)
        else:
            data = src.read(1)

        px = abs(transform.a)  # meters per pixel (UTM)
        crs = src.crs

    data = data.astype(np.float32)

    # Nodata cleanup (this is critical)
    data[~np.isfinite(data)] = np.nan
    if nodata is not None:
        data[data == nodata] = np.nan

    # Kill negatives
    data[data < 0] = np.nan

    # Fill to sea level
    # data[np.isnan(data)] = 0.0 # leave as NaN for mesh generation

    width_m = data.shape[1] * px
    height_m = data.shape[0] * px
    z_min, z_max = float(data.min()), float(data.max())

    print(f"DEM CRS: {crs}")
    print(f"DEM size: {width_m:.1f} m × {height_m:.1f} m")
    print(f"DEM elevation (clean): {z_min:.1f} m → {z_max:.1f} m")

    return data, px


def dem_to_mesh(dem, px_m, scale):
    h, w = dem.shape

    xs, ys = np.meshgrid(
        np.arange(w) * px_m * scale,
        np.arange(h)[::-1] * px_m * scale
    )
    zs = dem * Z_SCALE * scale

    # Build vertices (NaNs allowed for now)
    verts = np.column_stack([
        xs.ravel() * METERS_TO_MM,
        ys.ravel() * METERS_TO_MM,
        zs.ravel() * METERS_TO_MM,
    ])

    faces = []

    def valid(i):
        return not np.isnan(verts[i, 2])

    for y in range(h - 1):
        for x in range(w - 1):
            i0 = y * w + x
            i1 = i0 + 1
            i2 = i0 + w
            i3 = i2 + 1

            # triangle 1
            if valid(i0) and valid(i1) and valid(i2):
                faces.append([i0, i1, i2])

            # triangle 2
            if valid(i1) and valid(i3) and valid(i2):
                faces.append([i1, i3, i2])

    mesh = trimesh.Trimesh(
        verts,
        np.asarray(faces),
        process=True
    )

    # Now it’s safe to flatten remaining NaNs (if any slipped through)
    mesh.vertices[np.isnan(mesh.vertices[:, 2]), 2] = 0.0

    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    print(f"STL (surface) size: {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm")

    return mesh



def add_base(mesh, base_thickness_mm):
    """
    Extrudes a base downward using the *mesh boundary*.
    Because we clipped by polygon+buffer, this base matches that shape.
    """
    v_top = mesh.vertices
    f_top = mesh.faces

    z_bottom = v_top[:, 2].min() - base_thickness_mm

    v_bottom = v_top.copy()
    v_bottom[:, 2] = z_bottom

    vertices = np.vstack([v_top, v_bottom])
    offset = len(v_top)

    f_bottom = f_top[:, ::-1] + offset

    edges = mesh.edges_sorted
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]

    side_faces = []
    for a, b in boundary_edges:
        side_faces.append([a, b, b + offset])
        side_faces.append([a, b + offset, a + offset])

    faces = np.vstack([f_top, f_bottom, np.asarray(side_faces)])

    solid = trimesh.Trimesh(vertices, faces, process=True)

    bounds = solid.bounds
    size = bounds[1] - bounds[0]
    print(f"STL (solid) size:   {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm")
    print("Watertight:", solid.is_watertight)

    return solid


# -----------------------------
# MAIN
# -----------------------------

def main():
    polygon_wgs84 = load_polygon_wgs84(SHAPEFILE_PATH)
    west, south, east, north = polygon_wgs84.bounds

    lon, lat = polygon_wgs84.centroid.coords[0]
    TARGET_EPSG = utm_epsg_from_lon_lat(lon, lat)

    download_dem(west, south, east, north, DEM_PATH)

    # Clip once in whatever CRS the downloaded DEM uses (reduces size before reprojection)
    clip_dem_by_polygon(DEM_PATH, polygon_wgs84, "EPSG:4326", CLIPPED_DEM_PATH)

    # Reproject to UTM (meters)
    reproject_dem_to_epsg(CLIPPED_DEM_PATH, PROJECTED_DEM_PATH, TARGET_EPSG)

    # Build buffered polygon in UTM meters and clip again (THIS enforces polygon+buffer footprint)
    poly_utm = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326").to_crs(epsg=TARGET_EPSG).iloc[0]
    poly_utm_buffered = poly_utm.buffer(BUFFER_METERS) if BUFFER_METERS > 0 else poly_utm

    clip_dem_by_polygon(PROJECTED_DEM_PATH, poly_utm_buffered, f"EPSG:{TARGET_EPSG}", BUFFERED_CLIPPED_UTM_PATH)

    dem, px = read_dem(BUFFERED_CLIPPED_UTM_PATH, DOWNSAMPLE)

    # Compute print scale BEFORE meshing
    width_m = dem.shape[1] * px
    height_m = dem.shape[0] * px

    scale_factor = TARGET_MAX_MM / (max(width_m, height_m) * METERS_TO_MM)

    print(f"Print scale factor: {scale_factor:.6f}")

    mesh = dem_to_mesh(dem, px, scale_factor)

    if ADD_BASE:
        mesh = add_base(mesh, BASE_THICKNESS_MM * scale_factor)

    mesh.export(STL_PATH)
    print("STL written:", STL_PATH)


if __name__ == "__main__":
    main()
