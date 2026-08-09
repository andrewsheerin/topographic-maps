"""Pure geometry helpers: GeoJSON parsing and UTM zone selection."""

from shapely.geometry import shape


def polygon_from_geojson(geojson_obj):
    """Return a shapely geometry from a GeoJSON Feature or bare geometry dict."""
    if geojson_obj.get("type") == "Feature":
        return shape(geojson_obj["geometry"])
    return shape(geojson_obj)


def utm_epsg_from_lon_lat(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat). Northern zones 326xx,
    southern 327xx."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone
