import styles from './WaterSection.module.css';

const STREAM_CLASSES = [
  { key: 'riverWidth', label: 'River' },
  { key: 'streamWidth', label: 'Stream' },
  { key: 'brookWidth', label: 'Brook' },
];

export default function WaterSection({
  enabled,
  onToggle,
  water,
  onChange,
  widthFloorMm,
}) {
  const belowFloor =
    widthFloorMm != null &&
    STREAM_CLASSES.some((c) => {
      const w = Number(water[c.key] || 0);
      return w > 0 && w < widthFloorMm;
    });
  return (
    <section>
      <h2>4. Water</h2>

      <label className={styles.toggle}>
        <input
          type="checkbox"
          className={styles.checkbox}
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
        />
        <span>Add water</span>
      </label>

      {enabled && (
        <div className={styles.box}>
        <div className={styles.title}>
          Flatten lakes to their own level; carve streams like roads
        </div>

        <div className={styles.waterRow}>
          <label className={styles.check}>
            <input
              type="checkbox"
              className={styles.checkbox}
              checked={water.lakes}
              onChange={(e) => onChange('lakes', e.target.checked)}
            />
            <span className={styles.name}>Lakes</span>
          </label>

          <label className={styles.numLabel} htmlFor="waterRecess">
            Recess (mm)
          </label>
          <input
            id="waterRecess"
            className={styles.num}
            type="number"
            step="0.1"
            min="0"
            value={water.recess}
            onChange={(e) => onChange('recess', e.target.value)}
          />
        </div>

        <div className={styles.waterRow}>
          <label className={styles.check}>
            <input
              type="checkbox"
              className={styles.checkbox}
              checked={water.sea}
              onChange={(e) => onChange('sea', e.target.checked)}
            />
            <span className={styles.name}>Ocean</span>
          </label>

          <label className={styles.numLabel} htmlFor="seaLevel">
            Sea level (m)
          </label>
          <input
            id="seaLevel"
            className={styles.num}
            type="number"
            step="0.1"
            value={water.seaLevel}
            onChange={(e) => onChange('seaLevel', e.target.value)}
          />
        </div>

        <div className={styles.waterRow}>
          <label className={styles.check}>
            <input
              type="checkbox"
              className={styles.checkbox}
              checked={water.streams}
              onChange={(e) => onChange('streams', e.target.checked)}
            />
            <span className={styles.name}>Streams</span>
          </label>

          <label className={styles.numLabel} htmlFor="streamDepth">
            Depth (mm)
          </label>
          <input
            id="streamDepth"
            className={styles.num}
            type="number"
            step="0.1"
            min="0"
            value={water.streamDepth}
            onChange={(e) => onChange('streamDepth', e.target.value)}
          />
        </div>

        {water.streams && (
          <label className={`${styles.check} ${styles.subCheck}`}>
            <input
              type="checkbox"
              className={styles.checkbox}
              checked={water.namedOnly}
              onChange={(e) => onChange('namedOnly', e.target.checked)}
            />
            <span className={styles.name}>Named streams only</span>
          </label>
        )}

        {water.streams && (
          <div className={styles.widthRow}>
            {STREAM_CLASSES.map((c) => (
              <label className={styles.widthLabel} key={c.key} htmlFor={c.key}>
                {c.label} (mm)
                <input
                  id={c.key}
                  className={styles.num}
                  type="number"
                  step="0.1"
                  min="0"
                  value={water[c.key]}
                  onChange={(e) => onChange(c.key, e.target.value)}
                />
              </label>
            ))}
          </div>
        )}

        {water.streams && widthFloorMm != null && (
          <p className={belowFloor ? styles.floorWarn : styles.floorHint}>
            {belowFloor
              ? `Some widths are below the ≈${widthFloorMm.toFixed(2)} mm floor ` +
                '(DEM cell / extrusion line) and will print at the floor. ' +
                'Raise the detail multiplier for finer streams.'
              : `Narrowest printable stream ≈${widthFloorMm.toFixed(2)} mm ` +
                '(DEM cell / extrusion line).'}
          </p>
        )}
        </div>
      )}
    </section>
  );
}
