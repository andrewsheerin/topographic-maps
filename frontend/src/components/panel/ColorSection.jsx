import Button from '../ui/Button.jsx';
import styles from './ColorSection.module.css';

const HEX_RE = /^#[0-9A-Fa-f]{6}$/;

export const MAX_BANDS = 16;

// Every field is required — cutoffs and colors are the user's per print
// (F-32: no built-in classes, no defaults). The water slot is always part
// of the palette (stable slot order); it paints nothing when water is off.
export function colorComplete(c) {
  if (!HEX_RE.test(c.water)) return false;
  if (c.bands.length < 1) return false;
  return c.bands.every((b, i) => {
    if (!HEX_RE.test(b.color)) return false;
    const isLast = i === c.bands.length - 1;
    return isLast || (b.upto !== '' && Number.isFinite(Number(b.upto)));
  });
}

export default function ColorSection({
  enabled,
  onToggle,
  color,
  onChange,
  onBandChange,
  onAddBand,
  onRemoveBand,
}) {
  const lastIdx = color.bands.length - 1;

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
            Elevation bands, low to high — each face gets its band&apos;s
            color in the 3MF. Bands sort by elevation automatically; the base
            and walls use the lowest band.
          </div>

          {color.bands.map((b, i) => (
            <div className={styles.bandRow} key={i}>
              {i === lastIdx ? (
                <span className={styles.aboveLabel}>Above</span>
              ) : (
                <label className={styles.fieldLabel} htmlFor={`bandUpto-${i}`}>
                  Up to (m)
                  <input
                    id={`bandUpto-${i}`}
                    type="number"
                    value={b.upto}
                    onChange={(e) => onBandChange(i, 'upto', e.target.value)}
                  />
                </label>
              )}
              <label className={styles.fieldLabel} htmlFor={`bandColor-${i}`}>
                Color
                <input
                  id={`bandColor-${i}`}
                  type="text"
                  placeholder="#RRGGBB"
                  value={b.color}
                  onChange={(e) => onBandChange(i, 'color', e.target.value)}
                />
              </label>
              <Button
                className={styles.removeBtn}
                onClick={() => onRemoveBand(i)}
                disabled={color.bands.length <= 1}
                aria-label={`Remove band ${i + 1}`}
              >
                ✕
              </Button>
            </div>
          ))}

          <Button
            onClick={onAddBand}
            disabled={color.bands.length >= MAX_BANDS}
          >
            Add band
          </Button>

          <div className={styles.waterSlot}>
            <label className={styles.fieldLabel} htmlFor="waterColor">
              Lakes &amp; streams
              <input
                id="waterColor"
                type="text"
                placeholder="#RRGGBB"
                value={color.water}
                onChange={(e) => onChange('water', e.target.value)}
              />
            </label>
            <p className={styles.hint}>
              Painted on fetched water shapes, whatever their elevation —
              needs &quot;Add water&quot; to have shapes to paint.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
