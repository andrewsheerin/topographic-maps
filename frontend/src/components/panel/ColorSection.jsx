import styles from './ColorSection.module.css';

export const COLOR_FIELDS = [
  { key: 'water', label: 'Water' },
  { key: 'forest', label: 'Forest' },
  { key: 'rock', label: 'Rock' },
  { key: 'snow', label: 'Snow' },
];

const HEX_RE = /^#[0-9A-Fa-f]{6}$/;

// Every science field is required — treeline/snowline and the palette vary by
// mountain range and have no defaults.
export function colorComplete(c) {
  if (c.treeline === '' || !Number.isFinite(Number(c.treeline))) return false;
  if (c.snowline === '' || !Number.isFinite(Number(c.snowline))) return false;
  return COLOR_FIELDS.every((f) => HEX_RE.test(c[f.key]));
}

export default function ColorSection({ enabled, onToggle, color, onChange }) {
  return (
    <section>
      <h2>5. Color</h2>

      <label className={styles.toggle}>
        <input
          type="checkbox"
          className={styles.checkbox}
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
        />
        <span>Add color</span>
      </label>

      {enabled && (
        <div className={styles.box}>
          <div className={styles.title}>
            Per-face color by elevation band, for AMS multi-color printing —
            exported in the 3MF
          </div>

          <div className={styles.grid}>
            <label className={styles.fieldLabel} htmlFor="treelineM">
              Treeline (m)
              <input
                id="treelineM"
                type="number"
                value={color.treeline}
                onChange={(e) => onChange('treeline', e.target.value)}
              />
            </label>
            <label className={styles.fieldLabel} htmlFor="snowlineM">
              Snowline (m)
              <input
                id="snowlineM"
                type="number"
                value={color.snowline}
                onChange={(e) => onChange('snowline', e.target.value)}
              />
            </label>
          </div>

          <div className={styles.grid}>
            {COLOR_FIELDS.map((f) => (
              <label
                className={styles.fieldLabel}
                key={f.key}
                htmlFor={`color-${f.key}`}
              >
                {f.label}
                <input
                  id={`color-${f.key}`}
                  type="text"
                  placeholder="#RRGGBB"
                  value={color[f.key]}
                  onChange={(e) => onChange(f.key, e.target.value)}
                />
              </label>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
