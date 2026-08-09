import { useCallback, useMemo, useState } from 'react';

import MapView from '../../components/MapView/MapView.jsx';
import AreaSection from '../../components/panel/AreaSection.jsx';
import TerrainParams from '../../components/panel/TerrainParams.jsx';
import RoadsSection from '../../components/panel/RoadsSection.jsx';
import OutputSection from '../../components/panel/OutputSection.jsx';
import * as api from '../../lib/api.js';
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
  bufferM: '50',
  targetMaxMm: '200',
  addBase: true,
  baseThicknessM: '50',
};

const DEFAULT_ROADS = ROAD_CLASSES.reduce((acc, c) => {
  acc[c.key] = { checked: true, width: c.width, depth: c.depth };
  return acc;
}, {});

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
  const [roadsGeojson, setRoadsGeojson] = useState(null);
  const [status, setStatus] = useState('');
  const [terrain, setTerrain] = useState(DEFAULT_TERRAIN);
  const [roads, setRoads] = useState(DEFAULT_ROADS);

  const updateTerrain = useCallback((key, value) => {
    setTerrain((t) => ({ ...t, [key]: value }));
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
      buffer_m: Number(terrain.bufferM || 0),
      target_max_mm: Number(terrain.targetMaxMm || 200),
      add_base: Boolean(terrain.addBase),
      base_thickness_m: Number(terrain.baseThicknessM || 0),
      road_levels: selectedRoadLevels,
      road_etch,
    };
  }, [terrain, roads, selectedRoadLevels]);

  /* ---------------- polygon ---------------- */

  const handlePolygonCreated = useCallback((feature) => {
    setPolygon(feature);
    setRoadsGeojson(null);
    setStatus('Polygon set.');
  }, []);

  const handlePolygonEdited = useCallback((feature) => {
    setPolygon(feature);
    setRoadsGeojson(null);
    setStatus('Polygon updated.');
  }, []);

  const handlePolygonDeleted = useCallback(() => {
    setPolygon(null);
    setRoadsGeojson(null);
    setStatus('Polygon cleared.');
  }, []);

  const handleClearPolygon = useCallback(() => {
    setPolygon(null);
    setRoadsGeojson(null);
    setStatus('Polygon cleared.');
  }, []);

  const handleUpload = useCallback(async (file) => {
    setStatus('Uploading shapefile...');
    try {
      const geometry = await api.uploadShapefile(file);
      setPolygon({ type: 'Feature', properties: {}, geometry });
      setRoadsGeojson(null);
      setStatus('Polygon loaded from shapefile.');
    } catch (err) {
      setStatus(err.message);
    }
  }, []);

  /* ---------------- roads ---------------- */

  const handleLoadRoads = useCallback(async () => {
    if (!polygon) {
      setStatus('Draw a polygon or upload a shapefile first.');
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
      setStatus('Draw a polygon or upload a shapefile first.');
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

  const handleGenerateBundle = useCallback(async () => {
    if (!polygon) {
      setStatus('Draw a polygon or upload a shapefile first.');
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
          onClearPolygon={handleClearPolygon}
        />

        <TerrainParams terrain={terrain} onChange={updateTerrain} />

        <RoadsSection
          classes={ROAD_CLASSES}
          roads={roads}
          onChange={updateRoad}
          onLoadRoads={handleLoadRoads}
          onClearRoads={handleClearRoads}
        />

        <OutputSection
          onGenerateStl={handleGenerateStl}
          onGenerateBundle={handleGenerateBundle}
          bundleDisabled={selectedRoadLevels.length === 0}
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
