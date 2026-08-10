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
 * POST /api/upload-shapefile (multipart) -> polygon GeoJSON geometry.
 * Backend extracts the zipped shapefile, reprojects to WGS84, and returns a
 * single polygon. On failure the caller surfaces `detail` in the status line.
 */
export async function uploadShapefile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/upload-shapefile', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Shapefile upload failed.');
  }

  const data = await res.json();
  return data.polygon_geojson;
}
