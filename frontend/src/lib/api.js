/* ---------------------------------------------------------
   fetch wrappers for the FastAPI backend.
   All calls use RELATIVE paths so the same build works in dev
   (via the Vite proxy) and in production (FastAPI serving dist/).
--------------------------------------------------------- */

const JSON_HEADERS = { 'Content-Type': 'application/json' };

/**
 * POST /api/roads -> GeoJSON FeatureCollection of road centerlines.
 */
export async function fetchRoads(polygonGeojson, roadLevels) {
  const res = await fetch('/api/roads', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({
      polygon_geojson: polygonGeojson,
      road_levels: roadLevels,
    }),
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(detail || `Road fetch failed (HTTP ${res.status}).`);
  }

  const fc = data?.roads_geojson;
  if (!fc || fc.type !== 'FeatureCollection') {
    throw new Error("Roads response wasn't a GeoJSON FeatureCollection.");
  }
  return fc;
}

/**
 * POST /api/generate-stl -> binary STL blob.
 */
export async function generateStl(body) {
  const res = await fetch('/api/generate-stl', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'STL generation failed.');
  }
  return res.blob();
}

/**
 * POST /api/generate-bundle -> binary ZIP blob.
 */
export async function generateBundle(body) {
  const res = await fetch('/api/generate-bundle', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Bundle generation failed.');
  }
  return res.blob();
}

/**
 * POST /api/upload-boundary (multipart) -> polygon GeoJSON geometry.
 * Accepts a zipped shapefile or a GeoJSON file; the backend reduces it to a
 * single WGS84 polygon. On failure the caller surfaces `detail` in the status
 * line.
 */
export async function uploadBoundary(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/upload-boundary', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Boundary upload failed.');
  }

  const data = await res.json();
  return data.polygon_geojson;
}

/**
 * GET /api/places -> list of place summaries (TIGER county subdivisions).
 */
export async function fetchPlaces({ state, q, limit = 100, offset = 0 }) {
  const params = new URLSearchParams();
  if (state) params.set('state', state);
  if (q) params.set('q', q);
  params.set('limit', String(limit));
  params.set('offset', String(offset));

  const res = await fetch(`/api/places?${params}`);
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(detail || `Place search failed (HTTP ${res.status}).`);
  }
  return data;
}

/**
 * GET /api/places/{geoid} -> place with its polygon geometry.
 */
export async function fetchPlaceDetail(geoid) {
  const res = await fetch(`/api/places/${encodeURIComponent(geoid)}`);
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(detail || `Place lookup failed (HTTP ${res.status}).`);
  }
  return data;
}

/**
 * GET /api/states/{abbr} -> state outline with its polygon geometry.
 */
export async function fetchStateOutline(abbr) {
  const res = await fetch(`/api/states/${encodeURIComponent(abbr)}`);
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(detail || `State lookup failed (HTTP ${res.status}).`);
  }
  return data;
}
