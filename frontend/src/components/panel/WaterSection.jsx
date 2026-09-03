import styles from './WaterSection.module.css';

export default function WaterSection({ enabled, onToggle, water, onChange }) {
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
              checked={water.streams}
              onChange={(e) => onChange('streams', e.target.checked)}
            />
            <span className={styles.name}>Streams</span>
          </label>

          <label className={styles.numLabel} htmlFor="streamWidth">
            Width (mm)
          </label>
          <input
            id="streamWidth"
            className={styles.num}
            type="number"
            step="0.1"
            min="0"
            value={water.streamWidth}
            onChange={(e) => onChange('streamWidth', e.target.value)}
          />

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
        </div>
      )}
    </section>
  );
}
