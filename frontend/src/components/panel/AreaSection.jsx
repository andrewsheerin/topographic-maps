import { useState } from 'react';

import Button from '../ui/Button.jsx';
import Field from '../ui/Field.jsx';
import PlacePicker from './PlacePicker.jsx';
import styles from './AreaSection.module.css';

const MODES = [
  { key: 'draw', label: 'Draw' },
  { key: 'upload', label: 'Upload' },
  { key: 'place', label: 'City / town' },
];

// "lat, lon" text + side text -> {lat, lng, sideKm}, or null if not valid.
function parseSquareForm(centerText, sideText) {
  const parts = centerText.split(',');
  if (parts.length !== 2) return null;
  const lat = Number(parts[0].trim());
  const lng = Number(parts[1].trim());
  const sideKm = Number(sideText);
  if (![lat, lng, sideKm].every(Number.isFinite)) return null;
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180 || sideKm <= 0) {
    return null;
  }
  return { lat, lng, sideKm };
}

export default function AreaSection({
  onUpload,
  onPickPlace,
  onPickState,
  onPlaceSquare,
  selectedGeoid,
  areaLabel,
  onClearArea,
}) {
  const [mode, setMode] = useState('draw');
  const [squareCenter, setSquareCenter] = useState('');
  const [squareSide, setSquareSide] = useState('');
  const square = parseSquareForm(squareCenter, squareSide);

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
        <>
          <p className={styles.hint}>
            Use the tools in the top-left corner of the map to draw a polygon,
            rectangle, square, or circle — or place an exact square around a
            point below. Shapes stay editable.
          </p>
          <Field label="Center point (lat, lon)" htmlFor="squareCenter">
            <input
              id="squareCenter"
              type="text"
              placeholder="e.g. 50.86047, -115.55197"
              value={squareCenter}
              onChange={(e) => setSquareCenter(e.target.value)}
            />
          </Field>
          <Field label="Side length (km)" htmlFor="squareSide">
            <input
              id="squareSide"
              type="number"
              min="0"
              step="any"
              value={squareSide}
              onChange={(e) => setSquareSide(e.target.value)}
            />
          </Field>
          <Button
            disabled={!square}
            onClick={() => onPlaceSquare(square.lat, square.lng, square.sideKm)}
          >
            Place square
          </Button>
        </>
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
