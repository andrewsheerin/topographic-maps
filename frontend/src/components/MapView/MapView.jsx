import { useCallback, useEffect, useMemo, useRef } from 'react';
import L from 'leaflet';
import 'leaflet-draw';
import '../../lib/leafletDrawFix.js';

import styles from './MapView.module.css';

function clamp(num, min, max) {
  return Math.max(min, Math.min(max, num));
}

// Approximate a drawn circle (centre + radius in metres) as a polygon Feature.
// The DEM pipeline needs a polygon; a circle's toGeoJSON() is only a point with
// a radius property, which shapely can't use as an area.
function circleToPolygonFeature(circle, steps = 64) {
  const center = circle.getLatLng();
  const radiusM = circle.getRadius();
  const latRad = (center.lat * Math.PI) / 180;
  const dLat = radiusM / 111320; // metres -> degrees latitude
  const dLng = radiusM / (111320 * Math.cos(latRad)); // metres -> degrees longitude
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

// A drawn shape -> the polygon GeoJSON Feature the backend expects.
function shapeToPolygonFeature(layer) {
  return layer instanceof L.Circle
    ? circleToPolygonFeature(layer)
    : layer.toGeoJSON();
}

/**
 * Imperative Leaflet map. Driven by props:
 *  - polygon: current polygon GeoJSON Feature (or null)
 *  - roadsGeojson: roads FeatureCollection (or null)
 *  - roadWidths: { <class>: number } used to drive preview line thickness
 * Callbacks report user edits back up so React stays the source of truth.
 */
export default function MapView({
  polygon,
  roadsGeojson,
  roadWidths,
  onPolygonCreated,
  onPolygonEdited,
  onPolygonDeleted,
}) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const drawnItemsRef = useRef(null);
  const roadsLayerRef = useRef(null);

  // Tracks the polygon reference currently displayed on the map so that
  // polygons created/edited via the map don't get re-rendered (or re-fit) by
  // the sync effect below.
  const lastSyncedRef = useRef(null);

  // Keep the latest widths reachable from the (stable) style callback.
  const roadWidthsRef = useRef(roadWidths);
  roadWidthsRef.current = roadWidths;

  // Keep the latest callbacks reachable from the (mount-once) init effect.
  const cbRef = useRef({});
  cbRef.current = { onPolygonCreated, onPolygonEdited, onPolygonDeleted };

  // Read the accent colour from the design tokens rather than hardcoding it.
  const accent = useMemo(
    () =>
      getComputedStyle(document.documentElement)
        .getPropertyValue('--color-accent')
        .trim(),
    [],
  );

  const roadStyle = useCallback(
    (feature) => {
      const highway = feature?.properties?.highway || '';
      const mm = Number(roadWidthsRef.current?.[highway] || 0);
      // Map mm -> px with a small scaling, bounded so it stays usable.
      const px = mm > 0 ? mm * 1.4 : 1.0;
      return { color: accent, weight: clamp(px, 1, 10), opacity: 0.9 };
    },
    [accent],
  );

  /* ---------------- init map (once) ---------------- */
  useEffect(() => {
    const map = L.map(containerRef.current, {
      center: [41.6, -71.4],
      zoom: 9,
    });
    mapRef.current = map;

    const openTopo = L.tileLayer(
      'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
      {
        maxZoom: 17,
        attribution:
          'Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)',
      },
    );

    const esriTopo = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 19,
        attribution: 'Tiles © Esri — Source: Esri, USGS, NOAA',
      },
    );

    const osmStreets = L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors',
      },
    );

    const esriHillshade = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 19,
        opacity: 0.7,
        attribution: 'Hillshade © Esri — Source: Esri',
      },
    );

    openTopo.addTo(map);
    esriHillshade.addTo(map);

    L.control
      .layers(
        {
          'Topo (OpenTopoMap)': openTopo,
          'Topo (Esri)': esriTopo,
          'Streets (OSM)': osmStreets,
        },
        { Hillshade: esriHillshade },
        { collapsed: true },
      )
      .addTo(map);

    const drawnItems = new L.FeatureGroup();
    drawnItemsRef.current = drawnItems;
    map.addLayer(drawnItems);

    const drawControl = new L.Control.Draw({
      draw: {
        // Squares are drawn with the rectangle tool.
        polygon: {
          allowIntersection: false,
          showArea: true,
        },
        rectangle: {
          showArea: true,
          metric: true,
        },
        circle: {
          showRadius: true,
          metric: true,
        },
        polyline: false,
        marker: false,
        circlemarker: false,
      },
      edit: {
        featureGroup: drawnItems,
      },
    });
    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, (event) => {
      drawnItems.clearLayers();
      drawnItems.addLayer(event.layer);
      const feature = shapeToPolygonFeature(event.layer);
      lastSyncedRef.current = feature;
      cbRef.current.onPolygonCreated(feature);
    });

    map.on(L.Draw.Event.EDITED, () => {
      const layers = drawnItems.getLayers();
      if (layers.length > 0) {
        const feature = shapeToPolygonFeature(layers[0]);
        lastSyncedRef.current = feature;
        cbRef.current.onPolygonEdited(feature);
      }
    });

    map.on(L.Draw.Event.DELETED, () => {
      lastSyncedRef.current = null;
      cbRef.current.onPolygonDeleted();
    });

    // Ensure correct sizing once the grid layout has settled.
    const sizeTimer = setTimeout(() => map.invalidateSize(), 0);

    return () => {
      clearTimeout(sizeTimer);
      map.remove();
      mapRef.current = null;
      drawnItemsRef.current = null;
      roadsLayerRef.current = null;
    };
  }, []);

  /* ---------------- sync polygon prop -> map ---------------- */
  useEffect(() => {
    const map = mapRef.current;
    const drawnItems = drawnItemsRef.current;
    if (!map || !drawnItems) return;

    // Already represented on the map (drawn/edited by the user): nothing to do.
    if (lastSyncedRef.current === polygon) return;

    drawnItems.clearLayers();
    if (!polygon) return;

    // Externally-set polygon (e.g. shapefile upload): render and fit bounds.
    const layer = L.geoJSON(polygon).getLayers()[0];
    if (!layer) return;
    drawnItems.addLayer(layer);
    lastSyncedRef.current = polygon;

    const bounds = layer.getBounds && layer.getBounds();
    if (bounds && bounds.isValid()) {
      map.fitBounds(bounds.pad(0.2));
    }
  }, [polygon]);

  /* ---------------- sync roads prop -> map ---------------- */
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (roadsLayerRef.current) {
      map.removeLayer(roadsLayerRef.current);
      roadsLayerRef.current = null;
    }
    if (!roadsGeojson) return;

    roadsLayerRef.current = L.geoJSON(roadsGeojson, { style: roadStyle }).addTo(
      map,
    );
  }, [roadsGeojson, roadStyle]);

  /* ---------------- live-update road preview thickness ---------------- */
  useEffect(() => {
    if (roadsLayerRef.current) {
      roadsLayerRef.current.setStyle(roadStyle);
    }
  }, [roadWidths, roadStyle]);

  return <div ref={containerRef} className={styles.map} />;
}
