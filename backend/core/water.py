"""OSM water via Overpass: lake polygons and waterway centerlines (F-22).

Reuses the roads module's endpoint list, retry logic, and poly filter — no
parallel Overpass system. Fetched in WGS84 (EPSG:4326); reprojection to UTM is
explicit in the pipeline, like roads.

Tag selections (source: user spec 2026-09-03, see constants):
  polygons: natural=water + water=lake|reservoir|pond (ways and multipolygon
            relations — mountain reservoirs are commonly mapped as relations)
  lines:    waterway=stream|river
"""

import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

import constants
from core.roads import _post_overpass, overpass_poly_str


def _way_polygon(el) -> Polygon | None:
    """A closed Overpass way with inline geometry -> Polygon (else None)."""
    pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
    if len(pts) < 4 or pts[0] != pts[-1]:
        return None
    poly = Polygon(pts)
    return poly if not poly.is_empty and poly.is_valid else poly.buffer(0)


def _relation_polygon(el):
    """A multipolygon relation -> (Multi)Polygon: polygonize the outer member
    ways and subtract the inner ones (islands stay land)."""
    outers, inners = [], []
    for m in el.get("members", []):
        pts = [(p["lon"], p["lat"]) for p in m.get("geometry", [])]
        if len(pts) < 2:
            continue
        (inners if m.get("role") == "inner" else outers).append(LineString(pts))
    if not outers:
        return None
    outer = unary_union(list(polygonize(outers)))
    if outer.is_empty:
        return None
    if inners:
        holes = unary_union(list(polygonize(inners)))
        outer = outer.difference(holes)
    return None if outer.is_empty else outer


def fetch_water_overpass(polygon_wgs84, *, lakes=True, streams=True):
    """Fetch water bodies and waterways inside the polygon.

    Returns (lake_polys, stream_lines): two lists of shapely geometries in
    EPSG:4326. Either list is empty when its fetch is disabled."""
    poly_str = overpass_poly_str(polygon_wgs84)
    water_re = "|".join(constants.WATER_POLYGON_VALUES)
    waterway_re = "|".join(constants.WATERWAY_VALUES)

    lake_polys = []
    if lakes:
        query = f"""
        [out:json][timeout:60];
        (
          way["natural"="water"]["water"~"^({water_re})$"](poly:"{poly_str}");
          relation["natural"="water"]["water"~"^({water_re})$"](poly:"{poly_str}");
        );
        out geom;
        """
        data = _post_overpass(query, timeout_s=70, max_retries=3)
        for el in data.get("elements", []):
            geom = (
                _relation_polygon(el)
                if el.get("type") == "relation"
                else _way_polygon(el)
            )
            if geom is not None and not geom.is_empty:
                lake_polys.append(geom)

    stream_lines = []
    if streams:
        query = f"""
        [out:json][timeout:60];
        way["waterway"~"^({waterway_re})$"](poly:"{poly_str}");
        out geom;
        """
        data = _post_overpass(query, timeout_s=70, max_retries=3)
        for el in data.get("elements", []):
            pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(pts) >= 2:
                stream_lines.append(LineString(pts))

    return lake_polys, stream_lines


def water_to_utm_clipped(geoms_wgs84, epsg, poly_utm) -> gpd.GeoSeries:
    """Reproject WGS84 water geometries to the terrain's UTM CRS and clip to
    the terrain footprint. Explicit CRS on both ends; returns a GeoSeries."""
    if not geoms_wgs84:
        return gpd.GeoSeries([], crs=epsg)
    gs = gpd.GeoSeries(geoms_wgs84, crs="EPSG:4326").to_crs(epsg)
    gs = gs.intersection(poly_utm)
    gs = gs[~gs.is_empty & gs.is_valid]
    return gs
