import Field from '../ui/Field.jsx';

export default function TerrainParams({ terrain, onChange, cellMm }) {
  return (
    <section>
      <h2>2. Terrain parameters</h2>

      <Field label="DEM dataset" htmlFor="demDataset">
        <select
          id="demDataset"
          value={terrain.demDataset}
          onChange={(e) => onChange('demDataset', e.target.value)}
        >
          <option value="USGS10m">USGS 10 m (US only)</option>
          <option value="USGS30m">USGS 30 m (US only)</option>
          <option value="COP30">Copernicus 30 m (global)</option>
          <option value="COP90">Copernicus 90 m (global)</option>
          <option value="SRTMGL1">SRTM 30 m (global, 60°N–56°S)</option>
          <option value="NASADEM">NASADEM 30 m (global, 60°N–56°S)</option>
          <option value="AW3D30">ALOS AW3D 30 m (global)</option>
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

      <Field
        label={
          cellMm != null
            ? `Detail multiplier — ≈${cellMm.toFixed(2)} mm per DEM cell`
            : 'Detail multiplier'
        }
        htmlFor="demUpsample"
      >
        <input
          id="demUpsample"
          type="number"
          min="1"
          max="4"
          value={terrain.demUpsample}
          onChange={(e) => onChange('demUpsample', e.target.value)}
          title="Bilinear DEM upsample: sharpens water placement (no new terrain detail). Triangle count grows with the square of this."
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

      <Field label="Target max size (mm)" htmlFor="targetMaxMm">
        <input
          id="targetMaxMm"
          type="number"
          value={terrain.targetMaxMm}
          onChange={(e) => onChange('targetMaxMm', e.target.value)}
        />
      </Field>

      <Field label="Base thickness (mm)" htmlFor="baseThicknessMm">
        <input
          id="baseThicknessMm"
          type="number"
          step="0.5"
          min="0.5"
          value={terrain.baseThicknessMm}
          onChange={(e) => onChange('baseThicknessMm', e.target.value)}
        />
      </Field>
    </section>
  );
}
