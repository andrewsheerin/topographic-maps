import { useCallback, useMemo, useState } from 'react';

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
import { squareAroundPoint } from '../../lib/geometry.js';
import styles from './Generator.module.css';

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
const DEFAULT_WATER = {
  lakes: true,
  streams: true,
  recess: '0.6',
  streamWidth: '1.0',
  streamDepth: '0.4',
};

// 3MF science fields are deliberately empty: treeline/snowline and the palette
// vary by mountain range, so the user supplies them per print (F-23).
const DEFAULT_COLOR_3MF = {
  treeline: '',
  snowline: '',
  water: '',
  forest: '',
  rock: '',
  snow: '',
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

  const updateRoad = useCallback((key, field, value) => {
    setRoads((r) => ({ ...r, [key]: { ...r[key], [field]: value } }));
  }, []);

  const selectedRoadLevels = useMemo(
    () => ROAD_CLASSES.filter((c) => roads[c.key].checked).map((c) => c.key),
    [roads],
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
    return {
      dem_dataset: terrain.demDataset,
      downsample: Number(terrain.downsample || 1),
      z_scale: Number(terrain.zScale || 1.0),
      target_max_mm: Number(terrain.targetMaxMm || 200),
      base_thickness_mm: Number(terrain.baseThicknessMm || 2),
      road_levels: features.roads ? selectedRoadLevels : [],
      road_etch,
      water: {
        lakes: features.water && water.lakes,
        streams: features.water && water.streams,
        recess_mm: Number(water.recess || 0),
        stream_width_mm: Number(water.streamWidth || 0),
        stream_depth_mm: Number(water.streamDepth || 0),
      },
      color:
        features.color && colorComplete(color3mf)
          ? {
              treeline_m: Number(color3mf.treeline),
              snowline_m: Number(color3mf.snowline),
              colors: {
                water: color3mf.water,
                forest: color3mf.forest,
                rock: color3mf.rock,
                snow: color3mf.snow,
              },
            }
          : null,
    };
  }, [terrain, roads, selectedRoadLevels, water, features, color3mf]);

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

        <TerrainParams terrain={terrain} onChange={updateTerrain} />

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
        />

        <ColorSection
          enabled={features.color}
          onToggle={(v) => toggleFeature('color', v)}
          color={color3mf}
          onChange={updateColor3mf}
        />

        <OutputSection
          onGenerateStl={handleGenerateStl}
          onGenerate3mf={handleGenerate3mf}
          onGenerateBundle={handleGenerateBundle}
          threemfDisabled={!(features.color && colorComplete(color3mf))}
          zipDisabled={features.color && !colorComplete(color3mf)}
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
