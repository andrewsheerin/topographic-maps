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


def fetch_water_overpass(
    polygon_wgs84, *, lakes=True, streams=True, named_streams_only=False
):
    """Fetch water bodies and waterways inside the polygon in ONE Overpass
    request (F-26: separate lake/stream queries fired back-to-back tripped
    rate limits). Elements are split by their tags afterwards.

    named_streams_only (F-31) keeps only waterways carrying a `name` tag —
    named ways are the streams a paper topo would show, unnamed ones are
    mostly imagery-traced gullies. Applied in the query itself so the
    payload shrinks too. Lakes are never name-filtered.

    Returns (lake_polys, stream_lines): lake_polys is a list of shapely
    geometries, stream_lines a list of (waterway_class, LineString) tuples —
    the class drives per-class carve widths (F-28 W-4). Both in EPSG:4326;
    either list is empty when its fetch is disabled."""
    lake_polys, stream_lines = [], []
    if not (lakes or streams):
        return lake_polys, stream_lines

    poly_str = overpass_poly_str(polygon_wgs84)
    water_re = "|".join(constants.WATER_POLYGON_VALUES)
    waterway_re = "|".join(constants.WATERWAY_VALUES)

    blocks = []
    if lakes:
        blocks.append(
            f'way["natural"="water"]["water"~"^({water_re})$"](poly:"{poly_str}");'
        )
        blocks.append(
            f'relation["natural"="water"]["water"~"^({water_re})$"](poly:"{poly_str}");'
        )
    if streams:
        name_filter = '["name"]' if named_streams_only else ""
        blocks.append(
            f'way["waterway"~"^({waterway_re})$"]{name_filter}(poly:"{poly_str}");'
        )

    query = "[out:json][timeout:60];\n(\n  " + "\n  ".join(blocks) + "\n);\nout geom;\n"
    data = _post_overpass(query, timeout_s=70, max_retries=3)

    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        if tags.get("waterway") in constants.WATERWAY_VALUES:
            pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(pts) >= 2:
                stream_lines.append((tags["waterway"], LineString(pts)))
        else:
            geom = (
                _relation_polygon(el)
                if el.get("type") == "relation"
                else _way_polygon(el)
            )
            if geom is not None and not geom.is_empty:
                lake_polys.append(geom)

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


def streams_to_utm_clipped(stream_lines, epsg, poly_utm) -> gpd.GeoDataFrame:
    """(waterway_class, LineString) tuples -> UTM GeoDataFrame with a
    `waterway` column, clipped to the terrain footprint (F-28 W-4)."""
    if not stream_lines:
        return gpd.GeoDataFrame({"waterway": []}, geometry=[], crs=epsg)
    classes = [c for c, _ in stream_lines]
    geoms = [g for _, g in stream_lines]
    gdf = gpd.GeoDataFrame(
        {"waterway": classes}, geometry=geoms, crs="EPSG:4326"
    ).to_crs(epsg)
    gdf["geometry"] = gdf.geometry.intersection(poly_utm)
    gdf = gdf[~gdf.is_empty & gdf.geometry.is_valid]
    return gdf
