/* Small geographic helpers shared by the map and the area panel.

   Both conversions use the equirectangular approximation
   (1° latitude ≈ 111 320 m; longitude scaled by cos(lat)) — the same
   approximation the circle draw tool has used since F-9. Fine at the
   map-area scales this app works at; the DEM pipeline only needs a
   WGS84 polygon. */

// Metres per degree of latitude (mean Earth). Source: same constant the
// circle tool shipped with (F-9); user-visible precedent, logged in DECISIONS.
const METERS_PER_DEG_LAT = 111320;

// A drawn circle (centre + radius in metres) as a polygon Feature. The DEM
// pipeline needs a polygon; a circle's toGeoJSON() is only a point with a
// radius property, which shapely can't use as an area.
export function circleToPolygonFeature(circle, steps = 64) {
  const center = circle.getLatLng();
  const radiusM = circle.getRadius();
  const latRad = (center.lat * Math.PI) / 180;
  const dLat = radiusM / METERS_PER_DEG_LAT;
  const dLng = radiusM / (METERS_PER_DEG_LAT * Math.cos(latRad));
  const ring = [];
  for (let i = 0; i <= steps; i++) {
    const theta = (i / steps) * 2 * Math.PI;
    ring.push([
      center.lng + dLng * Math.cos(theta),
      center.lat + dLat * Math.sin(theta),
    ]);
  }
  return {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [ring] },
  };
}

// Approximate max bbox extent of a WGS84 Feature in metres. Display-only —
// drives the cell-size readout and width-floor hint (F-28 W-5); the backend
// computes the exact value from the projected DEM.
export function maxExtentMeters(feature) {
  let minLon = Infinity;
  let maxLon = -Infinity;
  let minLat = Infinity;
  let maxLat = -Infinity;
  const walk = (coords) => {
    if (typeof coords[0] === 'number') {
      minLon = Math.min(minLon, coords[0]);
      maxLon = Math.max(maxLon, coords[0]);
      minLat = Math.min(minLat, coords[1]);
      maxLat = Math.max(maxLat, coords[1]);
    } else {
      coords.forEach(walk);
    }
  };
  walk(feature.geometry.coordinates);
  if (!Number.isFinite(minLon)) return null;
  const midLatRad = (((minLat + maxLat) / 2) * Math.PI) / 180;
  const widthM = (maxLon - minLon) * METERS_PER_DEG_LAT * Math.cos(midLatRad);
  const heightM = (maxLat - minLat) * METERS_PER_DEG_LAT;
  return Math.max(widthM, heightM);
}

// A square of side `sideKm` centred on (lat, lng), as a polygon Feature.
export function squareAroundPoint(lat, lng, sideKm) {
  const halfM = (sideKm * 1000) / 2;
  const latRad = (lat * Math.PI) / 180;
  const dLat = halfM / METERS_PER_DEG_LAT;
  const dLng = halfM / (METERS_PER_DEG_LAT * Math.cos(latRad));
  const ring = [
    [lng - dLng, lat - dLat],
    [lng + dLng, lat - dLat],
    [lng + dLng, lat + dLat],
    [lng - dLng, lat + dLat],
    [lng - dLng, lat - dLat],
  ];
  return {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [ring] },
  };
}
