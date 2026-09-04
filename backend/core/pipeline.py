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
from core import threemf as threemf_mod
from core import water as water_mod
from core.geometry import utm_epsg_from_lon_lat

METERS_TO_MM = constants.METERS_TO_MM


def _prepare_dem(polygon_wgs84, dem_dataset, downsample, tmp, dem_upsample=1):
    """Shared DEM prep: download -> clip -> reproject to UTM -> re-clip ->
    read -> optional bilinear upsample (F-28 W-1). Returns (dem_arr, px_m,
    transform, epsg, poly_utm) so callers compute scale from real-world
    extent."""
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
    if int(dem_upsample) > 1:
        dem_arr, px_m, transform = dem_mod.upsample_dem(
            dem_arr, transform, dem_upsample, epsg
        )
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


def _apply_water(
    dem_arr, transform, polygon_wgs84, epsg, poly_utm, scale_xy, z_scale, water
):
    """Fetch water and apply it to the DEM (F-22): flatten each lake to its own
    plane first, then carve streams (lakes protected). Returns
    (dem, water_mask, lakes_utm, streams_utm); the last three are None when
    water is disabled."""
    if not water or not (water.get("lakes") or water.get("streams")):
        return dem_arr, None, None, None

    lake_polys, stream_lines = water_mod.fetch_water_overpass(
        polygon_wgs84,
        lakes=bool(water.get("lakes")),
        streams=bool(water.get("streams")),
        named_streams_only=bool(water.get("named_only")),
    )
    lakes_utm = water_mod.water_to_utm_clipped(lake_polys, epsg, poly_utm)
    streams_utm = water_mod.streams_to_utm_clipped(stream_lines, epsg, poly_utm)

    dem_arr, lake_mask = mesh_mod.flatten_water(
        dem_arr,
        transform,
        list(lakes_utm.geometry),
        scale_xy,
        z_scale,
        water.get("recess_mm", constants.WATER_RECESS_MM),
    )
    dem_arr, stream_mask = mesh_mod.carve_streams(
        dem_arr,
        transform,
        streams_utm,
        scale_xy,
        z_scale,
        water.get("stream_widths_mm", constants.STREAM_WIDTHS_MM),
        water.get("stream_depth_mm", constants.STREAM_DEPTH_MM),
        protect_mask=lake_mask,
    )
    # Streams join the water color class (F-27) — carving still protects
    # only the lake planes.
    return dem_arr, lake_mask | stream_mask, lakes_utm, streams_utm


def _write_colored_3mf(out_path, dem_arr, water_mask, solid, color):
    """Write the colored 3MF for a sealed solid (F-23). `color` is the
    ColorBands dict: treeline_m, snowline_m, colors{water,forest,rock,snow}."""
    cell_cls = mesh_mod.classify_terrain_cells(
        dem_arr, water_mask, color["treeline_m"], color["snowline_m"]
    )
    face_cls = mesh_mod.face_classes_for_solid(dem_arr, cell_cls, len(solid.faces))
    c = color["colors"]
    threemf_mod.write_3mf(
        out_path,
        solid.vertices,
        solid.faces,
        face_cls,
        # Palette in class order: 0 water, 1 forest, 2 rock, 3 snow.
        [c["water"], c["forest"], c["rock"], c["snow"]],
        default_index=2,  # base/walls print in the rock color
    )
    return out_path


def _water_geojson_utm(lakes_utm, streams_utm, epsg):
    """Fetched water as one UTM FeatureCollection: `kind` = lake | stream,
    streams also carry their `waterway` class."""
    feats = []
    if lakes_utm is not None:
        for geom in lakes_utm:
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"kind": "lake"},
                    "geometry": geom.__geo_interface__,
                }
            )
    if streams_utm is not None:
        for wclass, geom in zip(streams_utm["waterway"], streams_utm.geometry):
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"kind": "stream", "waterway": wclass},
                    "geometry": geom.__geo_interface__,
                }
            )
    return {
        "type": "FeatureCollection",
        "features": feats,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
    }


def generate_stl_from_polygon(
    polygon_wgs84,
    dem_dataset,
    downsample,
    dem_upsample,
    z_scale,
    target_max_mm,
    base_thickness_mm,
    water=None,
):
    tmp = tempfile.mkdtemp()
    dem_arr, px_m, transform, epsg, poly_utm = _prepare_dem(
        polygon_wgs84, dem_dataset, downsample, tmp, dem_upsample
    )
    scale_xy = _scale_xy_from_extent(dem_arr, transform, target_max_mm)

    dem_arr, _, _, _ = _apply_water(
        dem_arr, transform, polygon_wgs84, epsg, poly_utm, scale_xy, z_scale, water
    )

    # The STL is always a sealed solid (F-19) — an open surface can't be
    # sliced. base_thickness_mm is a print dimension (F-17), like road etch.
    mesh = mesh_mod.dem_to_mesh(dem_arr, px_m, scale_xy, z_scale)
    mesh = mesh_mod.add_base(mesh, float(base_thickness_mm))

    out = os.path.join(tmp, "terrain.stl")
    mesh.export(out)
    return out


def _fetch_and_carve_roads(
    dem_arr, transform, polygon_wgs84, epsg, poly_utm, scale_xy, z_scale,
    road_levels, road_etch,
):
    """Fetch roads and carve them into the DEM; returns (dem, roads_gdf_utm)."""
    roads_fc_wgs84 = roads_mod.fetch_roads_geojson_overpass(polygon_wgs84, road_levels)
    gdf_roads_utm = gpd.GeoDataFrame.from_features(
        roads_fc_wgs84.get("features", []),
        crs="EPSG:4326",
    ).to_crs(epsg)
    gdf_roads_utm = roads_mod.clip_roads_gdf_to_polygon(gdf_roads_utm, poly_utm)
    dem_arr = mesh_mod.carve_roads(
        dem_arr, transform, gdf_roads_utm, scale_xy, z_scale, road_etch=road_etch
    )
    return dem_arr, gdf_roads_utm


def generate_3mf_from_polygon(
    polygon_wgs84,
    dem_dataset,
    downsample,
    dem_upsample,
    z_scale,
    target_max_mm,
    base_thickness_mm,
    color,
    road_levels=None,
    road_etch=None,
    water=None,
):
    """Colored 3MF export (F-23): the same terrain pipeline as the STL, plus a
    per-face class raster written via the 3MF Materials extension.

    `color` is the ColorBands dict (per-request user values). Streams and
    roads carve geometry but carry no color in this version (register F-23
    notes the tradeoff)."""
    tmp = tempfile.mkdtemp()
    dem_arr, px_m, transform, epsg, poly_utm = _prepare_dem(
        polygon_wgs84, dem_dataset, downsample, tmp, dem_upsample
    )
    scale_xy = _scale_xy_from_extent(dem_arr, transform, target_max_mm)

    dem_arr, water_mask, _, _ = _apply_water(
        dem_arr, transform, polygon_wgs84, epsg, poly_utm, scale_xy, z_scale, water
    )
    if road_levels:
        dem_arr, _ = _fetch_and_carve_roads(
            dem_arr, transform, polygon_wgs84, epsg, poly_utm, scale_xy, z_scale,
            road_levels, road_etch,
        )

    surface = mesh_mod.dem_to_mesh(dem_arr, px_m, scale_xy, z_scale)
    solid = mesh_mod.add_base(surface, float(base_thickness_mm))

    out = os.path.join(tmp, "terrain.3mf")
    return _write_colored_3mf(out, dem_arr, water_mask, solid, color)


def generate_bundle_from_polygon(
    polygon_wgs84,
    dem_dataset,
    downsample,
    dem_upsample,
    z_scale,
    target_max_mm,
    base_thickness_mm,
    road_levels=None,
    road_etch=None,
    water=None,
    color=None,
):
    """The "everything" ZIP (F-25): raw STL + carved STL always; plus the
    colored 3MF when `color` is set, and UTM GeoJSON of whichever features
    (roads, water) were fetched. Roads/water/color are all opt-in — with
    nothing toggled, no Overpass call is made."""
    tmp = tempfile.mkdtemp()
    dem_arr, px_m, transform, epsg, poly_utm = _prepare_dem(
        polygon_wgs84, dem_dataset, downsample, tmp, dem_upsample
    )
    scale_xy = _scale_xy_from_extent(dem_arr, transform, target_max_mm)

    # Raw mesh — the base terrain, no water, no roads.
    raw_mesh = mesh_mod.dem_to_mesh(dem_arr, px_m, scale_xy, z_scale)
    raw_mesh = mesh_mod.add_base(raw_mesh, float(base_thickness_mm))
    raw_stl = os.path.join(tmp, "terrain_raw.stl")
    raw_mesh.export(raw_stl)

    # Carved mesh: lakes flatten first, streams carve around them (F-22),
    # then roads carve last.
    carved_dem, water_mask, lakes_utm, streams_utm = _apply_water(
        dem_arr, transform, polygon_wgs84, epsg, poly_utm, scale_xy, z_scale, water
    )
    gdf_roads_utm = None
    if road_levels:
        carved_dem, gdf_roads_utm = _fetch_and_carve_roads(
            carved_dem, transform, polygon_wgs84, epsg, poly_utm, scale_xy, z_scale,
            road_levels, road_etch,
        )

    carved_mesh = mesh_mod.dem_to_mesh(carved_dem, px_m, scale_xy, z_scale)
    carved_solid = mesh_mod.add_base(carved_mesh, float(base_thickness_mm))
    carved_stl = os.path.join(tmp, "terrain_carved.stl")
    carved_solid.export(carved_stl)

    zip_path = os.path.join(tmp, "terrain_bundle.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(raw_stl, "terrain_raw.stl")
        z.write(carved_stl, "terrain_carved.stl")

        if color:
            threemf_path = _write_colored_3mf(
                os.path.join(tmp, "terrain.3mf"),
                carved_dem,
                water_mask,
                carved_solid,
                color,
            )
            z.write(threemf_path, "terrain.3mf")

        if gdf_roads_utm is not None:
            roads_fc_utm = {
                "type": "FeatureCollection",
                "features": json.loads(gdf_roads_utm.to_json()).get("features", []),
                "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
            }
            roads_path = os.path.join(tmp, "roads_utm.geojson")
            roads_mod.write_roads_geojson(roads_fc_utm, roads_path)
            z.write(roads_path, "roads_utm.geojson")

        if lakes_utm is not None or streams_utm is not None:
            water_path = os.path.join(tmp, "water_utm.geojson")
            roads_mod.write_roads_geojson(
                _water_geojson_utm(lakes_utm, streams_utm, epsg), water_path
            )
            z.write(water_path, "water_utm.geojson")

    return zip_path
