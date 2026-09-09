import { useCallback, useEffect, useMemo, useState } from 'react';

import MapView from '../../components/MapView/MapView.jsx';
import AreaSection from '../../components/panel/AreaSection.jsx';
import TerrainParams from '../../components/panel/TerrainParams.jsx';
import RoadsSection from '../../components/panel/RoadsSection.jsx';
import WaterSection from '../../components/panel/WaterSection.jsx';
import ColorSection, {
  colorComplete,
} from '../../components/panel/ColorSection.jsx';
import OutputSection from '../../components/panel/OutputSection.jsx';
import * as api from '../../lib/api.js';
import { extentMeters, squareAroundPoint } from '../../lib/geometry.js';
import styles from './Generator.module.css';

// Nominal dataset resolutions in metres — drives the cell-size readout only;
// the backend uses the actual raster. Mirrors backend DATASET_ALIASES.
const DATASET_RES_M = {
  USGS10m: 10,
  USGS30m: 30,
  SRTMGL1: 30,
  COP30: 30,
  COP90: 90,
  NASADEM: 30,
  AW3D30: 30,
};

// One extrusion line of a 0.4 mm nozzle — mirrors the backend constant
// (user-provided 2026-09-03, F-28 W-5).
const EXTRUSION_LINE_WIDTH_MM = 0.42;

const ROAD_CLASSES = [
  { key: 'motorway', label: 'Motorway', width: '2.0', depth: '0.5' },
  { key: 'trunk', label: 'Trunk', width: '1.5', depth: '0.5' },
  { key: 'primary', label: 'Primary', width: '1.0', depth: '0.5' },
  { key: 'secondary', label: 'Secondary', width: '0.7', depth: '0.5' },
  { key: 'tertiary', label: 'Tertiary', width: '0.5', depth: '0.5' },
  { key: 'residential', label: 'Residential', width: '0.3', depth: '0.5' },
];

const DEFAULT_TERRAIN = {
  demDataset: 'USGS10m',
  downsample: '1',
  demUpsample: '1',
  // F-34: when on, downsample/demUpsample track the auto-computed values
  // that land cells near the extrusion width; uncheck to edit by hand.
  autoRes: true,
  zScale: '2.0',
  targetMaxMm: '200',
  baseThicknessMm: '2',
};

const DEFAULT_ROADS = ROAD_CLASSES.reduce((acc, c) => {
  acc[c.key] = { checked: true, width: c.width, depth: c.depth };
  return acc;
}, {});

// Water defaults mirror backend constants (user-provided 2026-09-03). Lakes
// and streams start checked — the "Add water" toggle is the on/off switch.
// Per-class stream widths (F-28 W-4): wider = downstream.
const DEFAULT_WATER = {
  lakes: true,
  streams: true,
  sea: false,
  seaLevel: '0.5',
  namedOnly: true,
  recess: '0.6',
  riverWidth: '1.6',
  streamWidth: '1.0',
  brookWidth: '0.6',
  streamDepth: '0.4',
};

// 3MF color fields are deliberately empty — bands and colors are the user's
// per print, with no built-in classes (F-32). One unbounded band to start.
const DEFAULT_COLOR_3MF = {
  water: '',
  bands: [{ upto: '', color: '' }],
};

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function Generator() {
  const [polygon, setPolygon] = useState(null);
  // Where the current polygon came from: { label, geoid? } — label is shown in
  // the Area section, geoid highlights the picked place in the picker list.
  const [areaSource, setAreaSource] = useState(null);
  const [roadsGeojson, setRoadsGeojson] = useState(null);
  const [status, setStatus] = useState('');
  const [terrain, setTerrain] = useState(DEFAULT_TERRAIN);
  const [roads, setRoads] = useState(DEFAULT_ROADS);
  const [water, setWater] = useState(DEFAULT_WATER);
  const [color3mf, setColor3mf] = useState(DEFAULT_COLOR_3MF);
  // Opt-in feature toggles (F-25): off means no inputs shown and, above all,
  // no Overpass fetch for that feature.
  const [features, setFeatures] = useState({
    roads: false,
    water: false,
    color: false,
  });

  const toggleFeature = useCallback((key, value) => {
    setFeatures((f) => ({ ...f, [key]: value }));
    if (key === 'roads' && !value) setRoadsGeojson(null);
  }, []);

  const updateTerrain = useCallback((key, value) => {
    setTerrain((t) => ({ ...t, [key]: value }));
  }, []);

  const updateWater = useCallback((key, value) => {
    setWater((w) => ({ ...w, [key]: value }));
  }, []);

  const updateColor3mf = useCallback((key, value) => {
    setColor3mf((c) => ({ ...c, [key]: value }));
  }, []);

  const updateColorBand = useCallback((i, field, value) => {
    setColor3mf((c) => ({
      ...c,
      bands: c.bands.map((b, j) => (j === i ? { ...b, [field]: value } : b)),
    }));
  }, []);

  // New bands slot in just below the "Above" row.
  const addColorBand = useCallback(() => {
    setColor3mf((c) => ({
      ...c,
      bands: [
        ...c.bands.slice(0, -1),
        { upto: '', color: '' },
        c.bands[c.bands.length - 1],
      ],
    }));
  }, []);

  const removeColorBand = useCallback((i) => {
    setColor3mf((c) =>
      c.bands.length <= 1
        ? c
        : { ...c, bands: c.bands.filter((_, j) => j !== i) },
    );
  }, []);

  const updateRoad = useCallback((key, field, value) => {
    setRoads((r) => ({ ...r, [key]: { ...r[key], [field]: value } }));
  }, []);

  const selectedRoadLevels = useMemo(
    () => ROAD_CLASSES.filter((c) => roads[c.key].checked).map((c) => c.key),
    [roads],
  );

  const extents = useMemo(
    () => (polygon ? extentMeters(polygon) : null),
    [polygon],
  );

  // F-34: the downsample/upsample pair that lands one cell nearest the
  // extrusion width (never coarser) for the current area, dataset, and size.
  const autoFactors = useMemo(() => {
    const resM = DATASET_RES_M[terrain.demDataset];
    const targetMm = Number(terrain.targetMaxMm) || 0;
    if (!extents || !resM || targetMm <= 0) return null;
    const cellAt1x = resM * (targetMm / extents.maxM);
    if (cellAt1x < EXTRUSION_LINE_WIDTH_MM) {
      // Oversampled: coarsen, staying at or below one extrusion line.
      const ds = Math.floor(EXTRUSION_LINE_WIDTH_MM / cellAt1x);
      return { downsample: Math.min(20, Math.max(1, ds)), upsample: 1 };
    }
    // Undersampled: refine, staying at or below one extrusion line.
    const us = Math.ceil(cellAt1x / EXTRUSION_LINE_WIDTH_MM);
    return { downsample: 1, upsample: Math.min(4, Math.max(1, us)) };
  }, [extents, terrain.demDataset, terrain.targetMaxMm]);

  // Auto mode keeps the (still-visible) fields in sync with the computation.
  useEffect(() => {
    if (!terrain.autoRes || !autoFactors) return;
    const ds = String(autoFactors.downsample);
    const us = String(autoFactors.upsample);
    if (terrain.downsample !== ds || terrain.demUpsample !== us) {
      setTerrain((t) => ({ ...t, downsample: ds, demUpsample: us }));
    }
  }, [terrain.autoRes, terrain.downsample, terrain.demUpsample, autoFactors]);

  // Approximate print-mm size of one DEM cell for the current area/settings
  // (F-28): drives the detail readout and the stream width floor. Null until
  // an area is set.
  const cellMm = useMemo(() => {
    const resM = DATASET_RES_M[terrain.demDataset];
    const targetMm = Number(terrain.targetMaxMm) || 0;
    if (!extents || !resM || targetMm <= 0) return null;
    const downsample = Math.max(1, Number(terrain.downsample) || 1);
    const upsample = Math.max(1, Number(terrain.demUpsample) || 1);
    return ((resM * downsample) / upsample) * (targetMm / extents.maxM);
  }, [extents, terrain]);

  // Rough face count for the sealed solid: surface + mirrored bottom, walls
  // negligible. Warns before a 15M-triangle export, not after (F-34).
  const triEstimate = useMemo(() => {
    const resM = DATASET_RES_M[terrain.demDataset];
    if (!extents || !resM) return null;
    const downsample = Math.max(1, Number(terrain.downsample) || 1);
    const upsample = Math.max(1, Number(terrain.demUpsample) || 1);
    const effResM = (resM * downsample) / upsample;
    const cells = (extents.widthM / effResM) * (extents.heightM / effResM);
    return 4 * cells;
  }, [extents, terrain]);

  const widthFloorMm = useMemo(
    () => (cellMm == null ? null : Math.max(cellMm, EXTRUSION_LINE_WIDTH_MM)),
    [cellMm],
  );

  // Numeric widths keyed by class, used to drive the map road preview.
  const roadWidths = useMemo(() => {
    const w = {};
    ROAD_CLASSES.forEach((c) => {
      w[c.key] = Number(roads[c.key].width || 0);
    });
    return w;
  }, [roads]);

  const buildParams = useCallback(() => {
    const road_etch = {};
    ROAD_CLASSES.forEach((c) => {
      road_etch[c.key] = {
        width_mm: Number(roads[c.key].width || 0),
        depth_mm: Number(roads[c.key].depth || 0),
      };
    });
    // W-5: clamp sub-floor stream widths (the UI also warns inline).
    const clampWidth = (v) => {
      const w = Number(v || 0);
      return widthFloorMm != null && w > 0 && w < widthFloorMm
        ? widthFloorMm
        : w;
    };
    return {
      dem_dataset: terrain.demDataset,
      downsample: Number(terrain.downsample || 1),
      dem_upsample: Number(terrain.demUpsample || 1),
      z_scale: Number(terrain.zScale || 1.0),
      target_max_mm: Number(terrain.targetMaxMm || 200),
      base_thickness_mm: Number(terrain.baseThicknessMm || 2),
      road_levels: features.roads ? selectedRoadLevels : [],
      road_etch,
      water: {
        lakes: features.water && water.lakes,
        streams: features.water && water.streams,
        sea: features.water && water.sea,
        sea_level_m: Number(water.seaLevel || 0),
        named_only: water.namedOnly,
        recess_mm: Number(water.recess || 0),
        stream_widths_mm: {
          river: clampWidth(water.riverWidth),
          stream: clampWidth(water.streamWidth),
          brook: clampWidth(water.brookWidth),
        },
        stream_depth_mm: Number(water.streamDepth || 0),
      },
      color:
        features.color && colorComplete(color3mf)
          ? {
              water_color: color3mf.water,
              bands: color3mf.bands.map((b, i) => ({
                upto_m:
                  i === color3mf.bands.length - 1 ? null : Number(b.upto),
                color: b.color,
              })),
            }
          : null,
    };
  }, [terrain, roads, selectedRoadLevels, water, features, color3mf, widthFloorMm]);

  /* ---------------- polygon ---------------- */

  const handlePolygonCreated = useCallback((feature) => {
    setPolygon(feature);
    setAreaSource({ label: 'Drawn shape' });
    setRoadsGeojson(null);
    setStatus('Area set.');
  }, []);

  const handlePolygonEdited = useCallback((feature) => {
    setPolygon(feature);
    setAreaSource((s) => s ?? { label: 'Drawn shape' });
    setRoadsGeojson(null);
    setStatus('Area updated.');
  }, []);

  const handlePolygonDeleted = useCallback(() => {
    setPolygon(null);
    setAreaSource(null);
    setRoadsGeojson(null);
    setStatus('Area cleared.');
  }, []);

  const handleClearArea = useCallback(() => {
    setPolygon(null);
    setAreaSource(null);
    setRoadsGeojson(null);
    setStatus('Area cleared.');
  }, []);

  const handlePlaceSquare = useCallback((lat, lng, sideKm) => {
    setPolygon(squareAroundPoint(lat, lng, sideKm));
    setAreaSource({
      label: `${sideKm} km square at ${lat.toFixed(5)}, ${lng.toFixed(5)}`,
    });
    setRoadsGeojson(null);
    setStatus('Area set.');
  }, []);

  const handleUpload = useCallback(async (file) => {
    setStatus('Reading boundary file...');
    try {
      const geometry = await api.uploadBoundary(file);
      setPolygon({ type: 'Feature', properties: {}, geometry });
      setAreaSource({ label: file.name });
      setRoadsGeojson(null);
      setStatus('Area loaded from file.');
    } catch (err) {
      setStatus(err.message);
    }
  }, []);

  const handlePickState = useCallback(async (abbr) => {
    setStatus(`Loading state outline for ${abbr}...`);
    try {
      const detail = await api.fetchStateOutline(abbr);
      setPolygon({ type: 'Feature', properties: {}, geometry: detail.geometry });
      setAreaSource({ label: `${detail.name} (state outline)` });
      setRoadsGeojson(null);
      setStatus(`Area set to ${detail.name}.`);
    } catch (err) {
      setStatus(err.message);
    }
  }, []);

  const handlePickPlace = useCallback(async (place) => {
    setStatus(`Loading boundary for ${place.name}...`);
    try {
      const detail = await api.fetchPlaceDetail(place.geoid);
      setPolygon({ type: 'Feature', properties: {}, geometry: detail.geometry });
      setAreaSource({
        label: `${detail.name}, ${detail.state}`,
        geoid: detail.geoid,
      });
      setRoadsGeojson(null);
      setStatus(`Area set to ${detail.name}, ${detail.state}.`);
    } catch (err) {
      setStatus(err.message);
    }
  }, []);

  /* ---------------- roads ---------------- */

  const handleLoadRoads = useCallback(async () => {
    if (!polygon) {
      setStatus('Set an area first — draw, upload, or pick a city/town.');
      return;
    }
    if (selectedRoadLevels.length === 0) {
      setRoadsGeojson(null);
      setStatus('Select at least one road class.');
      return;
    }

    setStatus('Fetching roads from OpenStreetMap...');
    try {
      const fc = await api.fetchRoads(polygon, selectedRoadLevels);
      setRoadsGeojson(fc);
      const n = Array.isArray(fc.features) ? fc.features.length : 0;
      setStatus(`Roads loaded (${n}).`);
    } catch (err) {
      setStatus(err.message);
    }
  }, [polygon, selectedRoadLevels]);

  const handleClearRoads = useCallback(() => {
    setRoadsGeojson(null);
    setStatus('Roads cleared.');
  }, []);

  /* ---------------- output ---------------- */

  const handleGenerateStl = useCallback(async () => {
    if (!polygon) {
      setStatus('Set an area first — draw, upload, or pick a city/town.');
      return;
    }
    setStatus('Generating STL...');
    try {
      const blob = await api.generateStl({
        polygon_geojson: polygon,
        ...buildParams(),
      });
      downloadBlob(blob, 'terrain.stl');
      setStatus('STL downloaded.');
    } catch (err) {
      setStatus(err.message);
    }
  }, [polygon, buildParams]);

  const handleGenerate3mf = useCallback(async () => {
    if (!polygon) {
      setStatus('Set an area first — draw, upload, or pick a city/town.');
      return;
    }
    setStatus('Generating colored 3MF...');
    try {
      const blob = await api.generate3mf({
        polygon_geojson: polygon,
        ...buildParams(),
      });
      downloadBlob(blob, 'terrain.3mf');
      setStatus('3MF downloaded.');
    } catch (err) {
      setStatus(err.message);
    }
  }, [polygon, buildParams]);

  const handleGenerateBundle = useCallback(async () => {
    if (!polygon) {
      setStatus('Set an area first — draw, upload, or pick a city/town.');
      return;
    }
    setStatus('Generating bundle (this may take a while)...');
    try {
      const blob = await api.generateBundle({
        polygon_geojson: polygon,
        ...buildParams(),
      });
      downloadBlob(blob, 'terrain_bundle.zip');
      setStatus('Bundle downloaded.');
    } catch (err) {
      setStatus(err.message);
    }
  }, [polygon, buildParams]);

  return (
    <div className={styles.layout}>
      <aside className={styles.panel}>
        <h1>Terrain STL</h1>

        <AreaSection
          onUpload={handleUpload}
          onPickPlace={handlePickPlace}
          onPickState={handlePickState}
          onPlaceSquare={handlePlaceSquare}
          selectedGeoid={areaSource?.geoid ?? null}
          areaLabel={areaSource?.label ?? null}
          onClearArea={handleClearArea}
        />

        <TerrainParams
          terrain={terrain}
          onChange={updateTerrain}
          cellMm={cellMm}
          triEstimate={triEstimate}
        />

        <RoadsSection
          enabled={features.roads}
          onToggle={(v) => toggleFeature('roads', v)}
          classes={ROAD_CLASSES}
          roads={roads}
          onChange={updateRoad}
          onLoadRoads={handleLoadRoads}
          onClearRoads={handleClearRoads}
        />

        <WaterSection
          enabled={features.water}
          onToggle={(v) => toggleFeature('water', v)}
          water={water}
          onChange={updateWater}
          widthFloorMm={widthFloorMm}
        />

        <ColorSection
          enabled={features.color}
          onToggle={(v) => toggleFeature('color', v)}
          color={color3mf}
          onChange={updateColor3mf}
          onBandChange={updateColorBand}
          onAddBand={addColorBand}
          onRemoveBand={removeColorBand}
        />

        <OutputSection
          onGenerateStl={handleGenerateStl}
          onGenerate3mf={handleGenerate3mf}
          onGenerateBundle={handleGenerateBundle}
          threemfDisabled={
            !(features.color && colorComplete(color3mf))
          }
          zipDisabled={
            features.color && !colorComplete(color3mf)
          }
          status={status}
        />
      </aside>

      <main className={styles.mapWrap}>
        <MapView
          polygon={polygon}
          roadsGeojson={roadsGeojson}
          roadWidths={roadWidths}
          onPolygonCreated={handlePolygonCreated}
          onPolygonEdited={handlePolygonEdited}
          onPolygonDeleted={handlePolygonDeleted}
        />
      </main>
    </div>
  );
}
