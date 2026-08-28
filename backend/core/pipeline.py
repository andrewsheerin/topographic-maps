"""End-to-end pipelines: polygon -> DEM -> (roads) -> STL / ZIP bundle.

Each step's CRS is explicit: DEM downloaded in WGS84, clipped, reprojected to the
polygon's UTM zone, buffered and re-clipped in metres, then meshed."""

import json
import os
import tempfile
import zipfile

import geopandas as gpd

import constants
from core import dem as dem_mod
from core import mesh as mesh_mod
from core import roads as roads_mod
from core.geometry import utm_epsg_from_lon_lat

METERS_TO_MM = constants.METERS_TO_MM


def _prepare_dem(polygon_wgs84, dem_dataset, downsample, tmp):
    """Shared DEM prep: download -> clip -> reproject to UTM -> re-clip -> read.
    Returns (dem_arr, px_m, transform, epsg, poly_utm) so callers compute scale
    from real-world extent."""
    lon, lat = polygon_wgs84.centroid.coords[0]
    epsg = utm_epsg_from_lon_lat(lon, lat)

    dem = os.path.join(tmp, "dem.tif")
    dem2 = os.path.join(tmp, "dem_clip_wgs84.tif")
    dem3 = os.path.join(tmp, "dem_utm.tif")
    dem4 = os.path.join(tmp, "dem_utm_clip.tif")

    dem_mod.download_dem(*polygon_wgs84.bounds, dem, dem_dataset)
    dem_mod.clip_dem_by_polygon(dem, polygon_wgs84, "EPSG:4326", dem2)
    dem_mod.reproject_dem(dem2, dem3, epsg)

    poly_utm = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326").to_crs(epsg).iloc[0]
    # buffer(0) adds no margin (F-18 removed the user-facing buffer) — it only
    # normalizes any invalidity introduced by reprojection.
    poly_utm = poly_utm.buffer(0.0)

    dem_mod.clip_dem_by_polygon(dem3, poly_utm, epsg, dem4)

    dem_arr, px_m, transform = dem_mod.read_dem(dem4, downsample)
    return dem_arr, px_m, transform, epsg, poly_utm


def _scale_xy_from_extent(dem_arr, transform, target_max_mm):
    """Print scale (mm-per-metre / METERS_TO_MM) from the DEM's real-world UTM
    extent, not its pixel count."""
    left = transform.c
    top = transform.f
    right = left + dem_arr.shape[1] * transform.a
    bottom = top + dem_arr.shape[0] * transform.e
    width_m = abs(right - left)
    height_m = abs(top - bottom)
    max_dim_m = max(width_m, height_m)
    if max_dim_m <= 0 or not (max_dim_m == max_dim_m):  # NaN-safe
        raise RuntimeError(
            "Clipped DEM has invalid extent; try reducing downsample or adjusting the polygon."
        )
    return float(target_max_mm) / (max_dim_m * METERS_TO_MM)


def generate_stl_from_polygon(
    polygon_wgs84,
    dem_dataset,
    downsample,
    z_scale,
    target_max_mm,
    add_base_flag,
    base_thickness_mm,
):
    tmp = tempfile.mkdtemp()
    dem_arr, px_m, transform, epsg, poly_utm = _prepare_dem(
        polygon_wgs84, dem_dataset, downsample, tmp
    )
    scale_xy = _scale_xy_from_extent(dem_arr, transform, target_max_mm)

    mesh = mesh_mod.dem_to_mesh(dem_arr, px_m, scale_xy, z_scale)
    if add_base_flag:
        # base_thickness_mm is a print dimension (F-17), like road widths/depths.
        mesh = mesh_mod.add_base(mesh, float(base_thickness_mm))

    out = os.path.join(tmp, "terrain.stl")
    mesh.export(out)
    return out


def generate_bundle_from_polygon(
    polygon_wgs84,
    dem_dataset,
    downsample,
    z_scale,
    target_max_mm,
    add_base_flag,
    base_thickness_mm,
    road_levels,
    road_etch=None,
):
    tmp = tempfile.mkdtemp()
    dem_arr, px_m, transform, epsg, poly_utm = _prepare_dem(
        polygon_wgs84, dem_dataset, downsample, tmp
    )
    scale_xy = _scale_xy_from_extent(dem_arr, transform, target_max_mm)

    # Roads: fetch in WGS84, reproject + clip to the terrain footprint in UTM.
    roads_fc_wgs84 = roads_mod.fetch_roads_geojson_overpass(polygon_wgs84, road_levels)
    gdf_roads_utm = gpd.GeoDataFrame.from_features(
        roads_fc_wgs84.get("features", []),
        crs="EPSG:4326",
    ).to_crs(epsg)
    gdf_roads_utm = roads_mod.clip_roads_gdf_to_polygon(gdf_roads_utm, poly_utm)

    roads_fc_utm = {
        "type": "FeatureCollection",
        "features": json.loads(gdf_roads_utm.to_json()).get("features", []),
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
    }
    roads_geojson_utm_path = os.path.join(tmp, "roads_centerlines_utm.geojson")
    roads_mod.write_roads_geojson(roads_fc_utm, roads_geojson_utm_path)

    # Raw mesh.
    raw_mesh = mesh_mod.dem_to_mesh(dem_arr, px_m, scale_xy, z_scale)
    if add_base_flag:
        raw_mesh = mesh_mod.add_base(raw_mesh, float(base_thickness_mm))
    raw_stl = os.path.join(tmp, "terrain_raw.stl")
    raw_mesh.export(raw_stl)

    # Carved mesh.
    carved_dem = mesh_mod.carve_roads(
        dem_arr, transform, gdf_roads_utm, scale_xy, z_scale, road_etch=road_etch
    )
    carved_mesh = mesh_mod.dem_to_mesh(carved_dem, px_m, scale_xy, z_scale)
    if add_base_flag:
        carved_mesh = mesh_mod.add_base(carved_mesh, float(base_thickness_mm))
    carved_stl = os.path.join(tmp, "terrain_with_roads_recess.stl")
    carved_mesh.export(carved_stl)

    zip_path = os.path.join(tmp, "terrain_bundle.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(raw_stl, "terrain_raw.stl")
        z.write(roads_geojson_utm_path, "roads_centerlines_utm.geojson")
        z.write(carved_stl, "terrain_with_roads_recess.stl")

    return zip_path
