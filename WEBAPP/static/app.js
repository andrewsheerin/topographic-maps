/* ---------------------------------------------------------
   Leaflet map + polygon selection + API calls
--------------------------------------------------------- */

let map;
let drawnItems;
let roadsLayer = null;
let currentPolygonGeoJSON = null;

const el = (id) => document.getElementById(id);

function requiredEl(id) {
  const node = el(id);
  if (!node) {
    throw new Error(
      `Missing UI element #${id}. Hard refresh the page (Ctrl+Shift+R) to clear cached HTML/JS.`
    );
  }
  return node;
}

const statusEl = requiredEl("status");

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

/* ---------------------------------------------------------
   Road category selection (checkboxes)
--------------------------------------------------------- */

function getRoadLevels() {
  const checks = Array.from(document.querySelectorAll(".roadCheck"));
  return checks.filter((c) => c.checked).map((c) => c.value);
}

function updateRoadUIState() {
  const levels = getRoadLevels();
  const bundleBtn = el("bundleBtn");
  if (!bundleBtn) return;

  const enabled = levels.length > 0;
  bundleBtn.disabled = !enabled;
  bundleBtn.title = enabled ? "" : "Select at least one road class to enable this.";
}

/* ---------------------------------------------------------
   Parameter collection
--------------------------------------------------------- */

function getParams() {
  // If HTML and JS are out of sync (cached), this prevents a silent null crash.
  return {
    dem_dataset: requiredEl("demDataset").value,
    downsample: Number(requiredEl("downsample").value || 1),
    z_scale: Number(requiredEl("zScale").value || 1.0),
    buffer_m: Number(requiredEl("bufferM").value || 0),
    target_max_mm: Number(requiredEl("targetMaxMm").value || 200),
    add_base: Boolean(requiredEl("addBase").checked),
    base_thickness_m: Number(requiredEl("baseThicknessM").value || 0),
    road_levels: getRoadLevels(),
    road_etch: {
      motorway: {
        width_mm: Number(requiredEl("roadWidthMotorway").value || 0),
        depth_mm: Number(requiredEl("roadDepthMotorway").value || 0)
      },
      trunk: {
        width_mm: Number(requiredEl("roadWidthTrunk").value || 0),
        depth_mm: Number(requiredEl("roadDepthTrunk").value || 0)
      },
      primary: {
        width_mm: Number(requiredEl("roadWidthPrimary").value || 0),
        depth_mm: Number(requiredEl("roadDepthPrimary").value || 0)
      },
      secondary: {
        width_mm: Number(requiredEl("roadWidthSecondary").value || 0),
        depth_mm: Number(requiredEl("roadDepthSecondary").value || 0)
      },
      tertiary: {
        width_mm: Number(requiredEl("roadWidthTertiary").value || 0),
        depth_mm: Number(requiredEl("roadDepthTertiary").value || 0)
      },
      residential: {
        width_mm: Number(requiredEl("roadWidthResidential").value || 0),
        depth_mm: Number(requiredEl("roadDepthResidential").value || 0)
      }
    }
  };
}

/* ---------------------------------------------------------
   Map initialization
--------------------------------------------------------- */

function initMap() {
  map = L.map("map", {
    center: [41.6, -71.4],
    zoom: 9
  });

  // Terrain-friendly basemap options
  const osmStreets = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors"
  });

  const openTopo = L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
    maxZoom: 17,
    attribution:
      "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)"
  });

  const esriTopo = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 19,
      attribution: "Tiles © Esri — Source: Esri, USGS, NOAA"
    }
  );

  // Optional hillshade overlay to make relief pop
  const esriHillshade = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 19,
      opacity: 0.7,
      attribution: "Hillshade © Esri — Source: Esri"
    }
  );

  // Default layers (pick what you want users to see first)
  openTopo.addTo(map);
  esriHillshade.addTo(map);

  L.control
    .layers(
      {
        "Topo (OpenTopoMap)": openTopo,
        "Topo (Esri)": esriTopo,
        "Streets (OSM)": osmStreets
      },
      {
        Hillshade: esriHillshade
      },
      { collapsed: true }
    )
    .addTo(map);

  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  const drawControl = new L.Control.Draw({
    draw: {
      polygon: {
        allowIntersection: false,
        showArea: true
      },
      polyline: false,
      rectangle: false,
      circle: false,
      marker: false,
      circlemarker: false
    },
    edit: {
      featureGroup: drawnItems
    }
  });
  map.addControl(drawControl);

  map.on(L.Draw.Event.CREATED, (event) => {
    drawnItems.clearLayers();
    drawnItems.addLayer(event.layer);
    currentPolygonGeoJSON = event.layer.toGeoJSON();
    clearRoads();
    setStatus("Polygon set.");
  });

  map.on(L.Draw.Event.EDITED, () => {
    const layers = drawnItems.getLayers();
    if (layers.length > 0) {
      currentPolygonGeoJSON = layers[0].toGeoJSON();
      clearRoads();
      setStatus("Polygon updated.");
    }
  });

  map.on(L.Draw.Event.DELETED, () => {
    currentPolygonGeoJSON = null;
    clearRoads();
    setStatus("Polygon cleared.");
  });
}

/* ---------------------------------------------------------
   Shapefile upload
--------------------------------------------------------- */

async function uploadShapefileZip(file) {
  setStatus("Uploading shapefile...");

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/api/upload-shapefile", {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Shapefile upload failed.");
  }

  const data = await res.json();

  const feature = {
    type: "Feature",
    properties: {},
    geometry: data.polygon_geojson
  };

  drawnItems.clearLayers();
  const layer = L.geoJSON(feature).getLayers()[0];
  drawnItems.addLayer(layer);

  map.fitBounds(layer.getBounds().pad(0.2));
  currentPolygonGeoJSON = feature;

  clearRoads();
  setStatus("Polygon loaded from shapefile.");
}

/* ---------------------------------------------------------
   Roads preview
--------------------------------------------------------- */

function clamp(num, min, max) {
  return Math.max(min, Math.min(max, num));
}

function getRoadPreviewWeight(highway) {
  // Use the UI width (mm) to drive on-map line thickness.
  // Leaflet weight is in pixels; we map mm -> px with a small scaling.
  const idMap = {
    motorway: "roadWidthMotorway",
    trunk: "roadWidthTrunk",
    primary: "roadWidthPrimary",
    secondary: "roadWidthSecondary",
    tertiary: "roadWidthTertiary",
    residential: "roadWidthResidential"
  };

  const id = idMap[highway];
  const mm = id ? Number(el(id)?.value || 0) : 0;

  // Tunable mapping: 1.0 mm ≈ 1.4 px, with limits so it stays usable.
  const px = mm > 0 ? mm * 1.4 : 1.0;
  return clamp(px, 1, 10);
}

function roadStyle(feature) {
  const hw = feature?.properties?.highway || "";
  return {
    color: "#0bf5b2",
    weight: getRoadPreviewWeight(hw),
    opacity: 0.9
  };
}

function refreshRoadPreviewStyle() {
  if (!roadsLayer) return;
  roadsLayer.setStyle(roadStyle);
}

async function loadRoads() {
  if (!currentPolygonGeoJSON) {
    setStatus("Draw a polygon or upload a shapefile first.");
    return;
  }

  const levels = getRoadLevels();
  if (levels.length === 0) {
    clearRoads();
    setStatus("Select at least one road class.");
    return;
  }

  setStatus("Fetching roads from OpenStreetMap...");

  const body = {
    polygon_geojson: currentPolygonGeoJSON,
    road_levels: levels
  };

  const res = await fetch("/api/roads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(detail || `Road fetch failed (HTTP ${res.status}).`);
  }

  const fc = data?.roads_geojson;
  if (!fc || fc.type !== "FeatureCollection") {
    throw new Error("Roads response wasn't a GeoJSON FeatureCollection.");
  }

  clearRoads();

  roadsLayer = L.geoJSON(fc, {
    style: roadStyle
  }).addTo(map);

  refreshRoadPreviewStyle();

  const n = Array.isArray(fc.features) ? fc.features.length : 0;
  setStatus(`Roads loaded (${n}).`);
}

function clearRoads() {
  if (roadsLayer) {
    map.removeLayer(roadsLayer);
    roadsLayer = null;
  }
}

/* ---------------------------------------------------------
   STL + bundle generation
--------------------------------------------------------- */

async function generateSTL() {
  if (!currentPolygonGeoJSON) {
    setStatus("Draw a polygon or upload a shapefile first.");
    return;
  }

  setStatus("Generating STL...");

  const body = {
    polygon_geojson: currentPolygonGeoJSON,
    ...getParams()
  };

  const res = await fetch("/api/generate-stl", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "STL generation failed.");
  }

  const blob = await res.blob();
  downloadBlob(blob, "terrain.stl");
  setStatus("STL downloaded.");
}

async function generateBundle() {
  if (!currentPolygonGeoJSON) {
    setStatus("Draw a polygon or upload a shapefile first.");
    return;
  }

  setStatus("Generating bundle (this may take a while)...");

  const body = {
    polygon_geojson: currentPolygonGeoJSON,
    ...getParams()
  };

  const res = await fetch("/api/generate-bundle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Bundle generation failed.");
  }

  const blob = await res.blob();
  downloadBlob(blob, "terrain_bundle.zip");
  setStatus("Bundle downloaded.");
}

/* ---------------------------------------------------------
   Utilities
--------------------------------------------------------- */

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* ---------------------------------------------------------
   UI wiring
--------------------------------------------------------- */

function wireUI() {
  requiredEl("shpInput").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      await uploadShapefileZip(file);
    } catch (err) {
      setStatus(err.message);
    } finally {
      e.target.value = "";
    }
  });

  requiredEl("clearBtn").addEventListener("click", () => {
    drawnItems.clearLayers();
    currentPolygonGeoJSON = null;
    clearRoads();
    setStatus("Polygon cleared.");
  });

  requiredEl("roadsFetchBtn").addEventListener("click", async () => {
    try {
      await loadRoads();
    } catch (err) {
      setStatus(err.message);
    }
  });

  requiredEl("roadsClearBtn").addEventListener("click", () => {
    clearRoads();
    setStatus("Roads cleared.");
  });

  requiredEl("generateBtn").addEventListener("click", async () => {
    try {
      await generateSTL();
    } catch (err) {
      setStatus(err.message);
    }
  });

  requiredEl("bundleBtn").addEventListener("click", async () => {
    try {
      await generateBundle();
    } catch (err) {
      setStatus(err.message);
    }
  });

  // Enable/disable bundle button based on road checkbox selection.
  document.querySelectorAll(".roadCheck").forEach((cb) => {
    cb.addEventListener("change", updateRoadUIState);
  });
  updateRoadUIState();

  // Live-update road preview thickness when width params change.
  const previewWidthIds = [
    "roadWidthMotorway",
    "roadWidthTrunk",
    "roadWidthPrimary",
    "roadWidthSecondary",
    "roadWidthTertiary",
    "roadWidthResidential"
  ];
  previewWidthIds.forEach((id) => {
    const node = el(id);
    if (!node) return;
    node.addEventListener("input", () => {
      refreshRoadPreviewStyle();
    });
  });
}

/* ---------------------------------------------------------
   Init
--------------------------------------------------------- */

initMap();
wireUI();
