import Button from '../ui/Button.jsx';
import styles from './RoadsSection.module.css';

export default function RoadsSection({
  classes,
  roads,
  onChange,
  onLoadRoads,
  onClearRoads,
}) {
  return (
    <section>
      <h2>3. Roads</h2>

      <div className={styles.roadsPick}>
        <div className={styles.roadsTitle}>Show / carve these road classes</div>

        {classes.map((c) => (
          <div className={styles.roadRow} key={c.key}>
            <label className={styles.roadCheck}>
              <input
                type="checkbox"
                className={styles.checkbox}
                checked={roads[c.key].checked}
                onChange={(e) => onChange(c.key, 'checked', e.target.checked)}
              />
              <span className={styles.roadName}>{c.label}</span>
            </label>

            <input
              className={styles.roadNum}
              type="number"
              step="0.1"
              min="0"
              value={roads[c.key].width}
              title={`${c.label} width (mm)`}
              aria-label={`${c.label} width (mm)`}
              onChange={(e) => onChange(c.key, 'width', e.target.value)}
            />

            <input
              className={styles.roadNum}
              type="number"
              step="0.1"
              min="0"
              value={roads[c.key].depth}
              title={`${c.label} depth (mm)`}
              aria-label={`${c.label} depth (mm)`}
              onChange={(e) => onChange(c.key, 'depth', e.target.value)}
            />
          </div>
        ))}

        <div className={styles.roadHeader} aria-hidden="true">
          <div />
          <div className={styles.roadH}>Width (mm)</div>
          <div className={styles.roadH}>Depth (mm)</div>
        </div>
      </div>

      <div className={styles.row}>
        <Button onClick={onLoadRoads}>Load roads</Button>
        <Button onClick={onClearRoads}>Clear roads</Button>
      </div>
    </section>
  );
}
