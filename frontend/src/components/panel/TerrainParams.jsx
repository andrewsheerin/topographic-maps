import Field from '../ui/Field.jsx';
import styles from './TerrainParams.module.css';

export default function TerrainParams({ terrain, onChange }) {
  return (
    <section>
      <h2>2. Terrain parameters</h2>

      <Field label="DEM dataset" htmlFor="demDataset">
        <select
          id="demDataset"
          value={terrain.demDataset}
          onChange={(e) => onChange('demDataset', e.target.value)}
        >
          <option value="USGS10m">USGS 10 m</option>
          <option value="USGS30m">USGS 30 m</option>
        </select>
      </Field>

      <Field label="Downsample factor" htmlFor="downsample">
        <input
          id="downsample"
          type="number"
          min="1"
          max="20"
          value={terrain.downsample}
          onChange={(e) => onChange('downsample', e.target.value)}
        />
      </Field>

      <Field label="Vertical exaggeration (z-scale)" htmlFor="zScale">
        <input
          id="zScale"
          type="number"
          step="0.1"
          value={terrain.zScale}
          onChange={(e) => onChange('zScale', e.target.value)}
        />
      </Field>

      <Field label="Buffer distance (meters)" htmlFor="bufferM">
        <input
          id="bufferM"
          type="number"
          value={terrain.bufferM}
          onChange={(e) => onChange('bufferM', e.target.value)}
        />
      </Field>

      <Field label="Target max size (mm)" htmlFor="targetMaxMm">
        <input
          id="targetMaxMm"
          type="number"
          value={terrain.targetMaxMm}
          onChange={(e) => onChange('targetMaxMm', e.target.value)}
        />
      </Field>

      <Field label="Base thickness (meters)" htmlFor="baseThicknessM">
        <div className={styles.baseInline}>
          <div className={styles.baseCheck}>
            <input
              id="addBase"
              type="checkbox"
              className={styles.checkbox}
              checked={terrain.addBase}
              onChange={(e) => onChange('addBase', e.target.checked)}
            />
            <div className={styles.baseCaption}>Add base</div>
          </div>

          <div className={styles.baseInput}>
            <input
              id="baseThicknessM"
              type="number"
              value={terrain.baseThicknessM}
              onChange={(e) => onChange('baseThicknessM', e.target.value)}
            />
          </div>
        </div>
      </Field>
    </section>
  );
}
