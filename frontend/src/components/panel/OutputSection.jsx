import Button from '../ui/Button.jsx';
import styles from './OutputSection.module.css';

const HEX_RE = /^#[0-9A-Fa-f]{6}$/;

const COLOR_FIELDS = [
  { key: 'water', label: 'Water' },
  { key: 'forest', label: 'Forest' },
  { key: 'rock', label: 'Rock' },
  { key: 'snow', label: 'Snow' },
];

// Every science field is required — treeline/snowline and the palette vary by
// mountain range and have no defaults.
function color3mfIncomplete(c) {
  if (c.treeline === '' || !Number.isFinite(Number(c.treeline))) return true;
  if (c.snowline === '' || !Number.isFinite(Number(c.snowline))) return true;
  return COLOR_FIELDS.some((f) => !HEX_RE.test(c[f.key]));
}

export default function OutputSection({
  onGenerateStl,
  onGenerateBundle,
  bundleDisabled,
  color3mf,
  onColorChange,
  onGenerate3mf,
  status,
}) {
  const threemfDisabled = color3mfIncomplete(color3mf);

  return (
    <section>
      <h2>5. Output</h2>

      <div className={styles.row}>
        <Button variant="primary" onClick={onGenerateStl}>
          Terrain Only (STL)
        </Button>
        <Button
          variant="primary"
          onClick={onGenerateBundle}
          disabled={bundleDisabled}
          title={
            bundleDisabled
              ? 'Select at least one road class to enable this.'
              : ''
          }
        >
          Terrain w/ Roads (ZIP)
        </Button>
      </div>

      <div className={styles.threemf}>
        <div className={styles.threemfTitle}>
          Colored terrain (3MF) — per-face color by elevation band, for AMS
          multi-color printing
        </div>

        <div className={styles.bandGrid}>
          <label className={styles.fieldLabel} htmlFor="treelineM">
            Treeline (m)
            <input
              id="treelineM"
              type="number"
              value={color3mf.treeline}
              onChange={(e) => onColorChange('treeline', e.target.value)}
            />
          </label>
          <label className={styles.fieldLabel} htmlFor="snowlineM">
            Snowline (m)
            <input
              id="snowlineM"
              type="number"
              value={color3mf.snowline}
              onChange={(e) => onColorChange('snowline', e.target.value)}
            />
          </label>
        </div>

        <div className={styles.colorGrid}>
          {COLOR_FIELDS.map((f) => (
            <label className={styles.fieldLabel} key={f.key} htmlFor={`color-${f.key}`}>
              {f.label}
              <input
                id={`color-${f.key}`}
                type="text"
                placeholder="#RRGGBB"
                value={color3mf[f.key]}
                onChange={(e) => onColorChange(f.key, e.target.value)}
              />
            </label>
          ))}
        </div>

        <Button
          variant="primary"
          onClick={onGenerate3mf}
          disabled={threemfDisabled}
          title={
            threemfDisabled
              ? 'Fill treeline, snowline, and all four #RRGGBB colors to enable this.'
              : ''
          }
        >
          Colored terrain (3MF)
        </Button>
      </div>

      <div className={styles.status} aria-live="polite">
        {status}
      </div>

      <footer className={styles.footer}>
        <div className={styles.small}>Requires OpenTopography API key.</div>
      </footer>
    </section>
  );
}
