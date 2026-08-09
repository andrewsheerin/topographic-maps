# Data Sources

*One entry per dataset, added in the same commit that adds the data. See
`.claude/rules/science-integrity.md` → Dataset provenance.*

No datasets are committed to `data/` — all inputs are fetched live per request. These entries
document the runtime sources so their CRS, units, and licenses are on the record.

## OpenTopography DEMs

- **File(s):** none stored; fetched per request as GeoTIFF to a temp dir (`core/dem.py`).
- **Source / citation:** OpenTopography API — `/API/usgsdem` (USGS products) and `/API/globaldem`
  (global products). https://portal.opentopography.org/
- **Version / retrieved:** per request, at run time. Requires `OPEN_TOPO_API_KEY`.
- **License / terms:** varies by product — USGS 3DEP: public domain; Copernicus GLO-30/90 (COP30/
  COP90): ESA/Copernicus terms; SRTM/NASADEM: NASA; AW3D30: JAXA. Confirm the specific product's
  terms before redistributing outputs.
- **CRS / datum:** delivered in EPSG:4326 (WGS84); reprojected explicitly to the polygon's UTM zone
  in `core/dem.py` before any metric operation.
- **Temporal coverage:** product-dependent (single-epoch DEMs).
- **Units:** elevation in metres.
- **Nodata convention:** preserved through reprojection; masked to NaN on read (never silently
  filled). See `read_dem`.
- **Derived from / lineage:** primary (upstream provider via OpenTopography).
- **Known limitations / uncertainty:** vertical accuracy and resolution vary by product;
  `dem_to_mesh` rejects relief > `RELIEF_SANITY_MAX_M` as a likely unit/CRS error.

## OpenStreetMap roads (Overpass)

- **File(s):** none stored; fetched per request, exported as GeoJSON inside the ZIP bundle.
- **Source / citation:** OpenStreetMap contributors, via the Overpass API
  (`core/roads.py`, endpoints in `constants.OVERPASS_ENDPOINTS`).
- **Version / retrieved:** live at run time (current OSM data).
- **License / terms:** ODbL 1.0 — attribution "© OpenStreetMap contributors" required; share-alike
  applies to derived databases.
- **CRS / datum:** returned in EPSG:4326 (WGS84); reprojected to the terrain's UTM zone before
  buffering/clipping/carving.
- **Temporal coverage:** current snapshot at fetch time.
- **Units:** geographic degrees (source); metres after UTM reprojection. Carve widths/depths are
  print millimetres.
- **Nodata convention:** n/a (vector).
- **Derived from / lineage:** primary (OSM `highway` ways for the requested classes).
- **Known limitations / uncertainty:** OSM completeness/accuracy varies by area; dense polygon rings
  are downsampled to keep Overpass queries within timeout.
