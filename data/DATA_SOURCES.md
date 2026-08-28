# Data Sources

*One entry per dataset, added in the same commit that adds the data. See
`.claude/rules/science-integrity.md` → Dataset provenance.*

No datasets are committed to `data/` — inputs are fetched live per request, or (TIGER) fetched
once by a script into a gitignored file. These entries document the sources so their CRS, units,
and licenses are on the record.

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

## US Census TIGER county subdivisions (place picker)

- **File(s):** `data/tiger/subdivisions.gpkg`, layers `subdivisions` + `states` (gitignored;
  regenerate with `backend/scripts/fetch_tiger_subdivisions.py`).
- **Source / citation:** US Census Bureau, Cartographic Boundary Files 2023 — County Subdivisions
  (1:500,000), per-state `cb_2023_<FIPS>_cousub_500k.zip`; county names joined from
  `cb_2023_us_county_500k.zip`; state outlines (F-14) from `cb_2023_us_state_500k.zip`
  (all under `www2.census.gov/geo/tiger/GENZ2023/shp/`).
- **Version / retrieved:** 2023 vintage; retrieval date printed by the fetch script run.
- **License / terms:** public domain (US federal government work).
- **CRS / datum:** reprojected to EPSG:4326 (WGS84) by the fetch script; served as-is.
- **Temporal coverage:** 2023 boundary vintage.
- **Units:** geographic degrees; the picker's `area_km2` is an approximate bbox area
  (display-only, equirectangular km-per-degree factors in `core/places.py`).
- **Nodata convention:** water-only pseudo-subdivisions (`ALAND == 0`) and "County subdivisions
  not defined" filler records are dropped by the fetch script — they are not selectable areas.
- **Derived from / lineage:** `fetch_tiger_subdivisions.py` downloads, filters, joins county
  names, reprojects, and writes the GeoPackage; fully regenerable from that script.
- **Known limitations / uncertainty:** 1:500k generalized boundaries (not parcel-accurate);
  county subdivisions are real town/township governments in ~20 strong-MCD states but
  statistical Census County Divisions elsewhere. Approach ported from swpt-app (F-79/D-28),
  which chose subdivisions over TIGER Places to avoid unincorporated gaps.

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
