import Button from '../ui/Button.jsx';
import styles from './OutputSection.module.css';

export default function OutputSection({
  onGenerateStl,
  onGenerate3mf,
  onGenerateBundle,
  threemfDisabled,
  zipDisabled,
  frame,
  onFrameChange,
  status,
}) {
  return (
    <section>
      <h2>6. Output</h2>

      <div className={styles.frameRow}>
        <label className={styles.frameCheck}>
          <input
            type="checkbox"
            className={styles.checkbox}
            checked={frame.enabled}
            onChange={(e) => onFrameChange('enabled', e.target.checked)}
          />
          <span>Add frame (3MF) — clean 90° outer edges</span>
        </label>
        {frame.enabled && (
          <>
            <label className={styles.frameLabel} htmlFor="frameWidth">
              Width (mm)
            </label>
            <input
              id="frameWidth"
              className={styles.frameNum}
              type="number"
              step="0.5"
              min="0.5"
              value={frame.width}
              onChange={(e) => onFrameChange('width', e.target.value)}
            />
          </>
        )}
      </div>

      <div className={styles.row}>
        <Button variant="primary" onClick={onGenerateStl}>
          Terrain (STL)
        </Button>
        <Button
          variant="primary"
          onClick={onGenerate3mf}
          disabled={threemfDisabled}
          title={
            threemfDisabled
              ? 'Check "Add color" and fill treeline, snowline, and all four #RRGGBB colors to enable this.'
              : ''
          }
        >
          Colored terrain (3MF)
        </Button>
        <Button
          variant="primary"
          onClick={onGenerateBundle}
          disabled={zipDisabled}
          title={
            zipDisabled
              ? 'Finish the color fields (or uncheck "Add color") to enable this.'
              : 'Everything that is enabled above: STLs, 3MF, and feature data.'
          }
        >
          Everything (ZIP)
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
