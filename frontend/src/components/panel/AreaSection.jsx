import { useState } from 'react';

import Button from '../ui/Button.jsx';
import PlacePicker from './PlacePicker.jsx';
import styles from './AreaSection.module.css';

const MODES = [
  { key: 'draw', label: 'Draw' },
  { key: 'upload', label: 'Upload' },
  { key: 'place', label: 'City / town' },
];

export default function AreaSection({
  onUpload,
  onPickPlace,
  onPickState,
  selectedGeoid,
  areaLabel,
  onClearArea,
}) {
  const [mode, setMode] = useState('draw');

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    // Reset so selecting the same file again re-triggers change.
    e.target.value = '';
  };

  return (
    <section>
      <h2>1. Area</h2>

      <div className={styles.modes} role="tablist" aria-label="Area source">
        {MODES.map((m) => (
          <button
            key={m.key}
            type="button"
            role="tab"
            aria-selected={mode === m.key}
            className={
              mode === m.key
                ? `${styles.modeBtn} ${styles.modeActive}`
                : styles.modeBtn
            }
            onClick={() => setMode(m.key)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {mode === 'draw' && (
        <p className={styles.hint}>
          Use the tools in the top-left corner of the map to draw a polygon,
          rectangle, or circle. Rectangles can be dragged square; shapes stay
          editable.
        </p>
      )}

      {mode === 'upload' && (
        <>
          <label className={styles.file}>
            <input
              type="file"
              accept=".zip,.geojson,.json"
              onChange={handleFile}
            />
            <span className={styles.fileBtn}>Choose a boundary file…</span>
          </label>
          <p className={styles.hint}>
            Zipped shapefile (.zip, must include .prj) or GeoJSON
            (.geojson / .json, WGS84).
          </p>
        </>
      )}

      {mode === 'place' && (
        <PlacePicker
          onPick={onPickPlace}
          onPickState={onPickState}
          selectedGeoid={selectedGeoid}
        />
      )}

      {areaLabel && (
        <div className={styles.current}>
          <span className={styles.currentLabel}>
            Area: <strong>{areaLabel}</strong>
          </span>
          <Button className={styles.clearBtn} onClick={onClearArea}>
            Clear area
          </Button>
        </div>
      )}
    </section>
  );
}
