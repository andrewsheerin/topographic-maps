"""OSM road centerlines via Overpass: fetch, reproject to UTM, clip, and export.

Roads are fetched in WGS84 (EPSG:4326) and reprojected explicitly to the
terrain's UTM CRS before any metric operation (buffering, clipping, carving)."""

import json
import time

import geopandas as gpd
import requests

import constants


def _post_overpass(query: str, *, timeout_s: int = 60, max_retries: int = 3) -> dict:
    """POST an Overpass query with retries and endpoint fallback.

    Overpass frequently returns 429/504 under load; retry with exponential
    backoff and try alternate public endpoints. Returns parsed JSON on success.
    """
    last_err = None

    for endpoint in constants.OVERPASS_ENDPOINTS:
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    endpoint, data=query.encode("utf-8"), timeout=timeout_s
                )
                r.raise_for_status()
                try:
                    return r.json()
                except ValueError as je:
                    # Overpass sometimes returns HTML even on 200.
                    raise RuntimeError(
                        f"Overpass returned non-JSON response from {endpoint}. "
                        f"First 200 chars: {(r.text or '')[:200]}"
                    ) from je
            except Exception as e:
                last_err = e
                time.sleep(2**attempt)

    raise RuntimeError(
        "Overpass request failed after retries. This is usually temporary (server load). "
        "Try again, reduce road detail, or use a smaller polygon. "
        f"Last error: {last_err}"
    )


def fetch_roads_geojson_overpass(polygon_wgs84, highway_levels):
    """Return an EPSG:4326 FeatureCollection of road centerlines within the
    polygon for the requested highway classes."""
    if not highway_levels:
        return {"type": "FeatureCollection", "features": []}

    coords = list(polygon_wgs84.exterior.coords)

    # Overpass polygons must be fairly short or the query times out; downsample
    # a dense ring to keep the query cheap.
    if len(coords) > 300:
        step = max(1, len(coords) // 300)
        coords = coords[::step]
        if coords[0] != coords[-1]:
            coords.append(coords[0])

    poly_str = " ".join(f"{lat} {lon}" for lon, lat in coords)
    hw = "|".join(highway_levels)

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
        features.append(
            {
                "type": "Feature",
                "properties": {"highway": highway},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [(p["lon"], p["lat"]) for p in el["geometry"]],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def write_roads_geojson(roads_fc: dict, out_path: str) -> str:
    """Write a GeoJSON FeatureCollection to disk."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(roads_fc, f)
    return out_path


def roads_featurecollection_to_utm(roads_fc: dict, epsg: int) -> dict:
    """Reproject a roads FeatureCollection from EPSG:4326 into the given UTM EPSG.

    Output stays GeoJSON-shaped but in projected metres. A `crs` member is set
    for convenience (note: deprecated in strict GeoJSON)."""
    if not roads_fc.get("features"):
        return {
            "type": "FeatureCollection",
            "features": [],
            "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        }

    gdf = gpd.GeoDataFrame.from_features(roads_fc["features"], crs="EPSG:4326").to_crs(
        epsg
    )
    return {
        "type": "FeatureCollection",
        "features": json.loads(gdf.to_json()).get("features", []),
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
    }


def clip_roads_gdf_to_polygon(
    gdf_roads_utm: gpd.GeoDataFrame, poly_utm
) -> gpd.GeoDataFrame:
    """Clip roads to the terrain polygon/buffer in the same UTM CRS."""
    if gdf_roads_utm.empty:
        return gdf_roads_utm

    gdf2 = gdf_roads_utm[gdf_roads_utm.geometry.intersects(poly_utm)].copy()
    if gdf2.empty:
        return gdf2

    gdf2["geometry"] = gdf2.geometry.intersection(poly_utm)
    gdf2 = gdf2[~gdf2.is_empty & gdf2.geometry.is_valid]
    return gdf2
