import Button from '../ui/Button.jsx';
import styles from './OutputSection.module.css';

export default function OutputSection({
  onGenerateStl,
  onGenerate3mf,
  onGenerateBundle,
  threemfDisabled,
  zipDisabled,
  cleanEdges,
  onCleanEdgesChange,
  clip,
  onClipChange,
  status,
}) {
  const dimsActive = cleanEdges && clip.shape !== 'largest';
  return (
    <section>
      <h2>6. Output</h2>

      <div className={styles.frameRow}>
        <label className={styles.frameCheck}>
          <input
            type="checkbox"
            className={styles.checkbox}
            checked={cleanEdges}
            onChange={(e) => onCleanEdgesChange(e.target.checked)}
          />
          <span>
            Straight edges — clip to a clean rectangle (trims the skewed
            margin)
          </span>
        </label>
      </div>

      {cleanEdges && (
        <div className={styles.clipRow}>
          <select
            className={styles.clipShape}
            value={clip.shape}
            aria-label="Clip shape"
            onChange={(e) => onClipChange('shape', e.target.value)}
          >
            <option value="largest">Largest fit</option>
            <option value="square">Square</option>
            <option value="custom">Custom W × H</option>
          </select>

          {clip.shape === 'square' && (
            <label className={styles.clipLabel} htmlFor="clipSide">
              Side (mm)
              <input
                id="clipSide"
                className={styles.clipNum}
                type="number"
                min="1"
                value={clip.side}
                onChange={(e) => onClipChange('side', e.target.value)}
              />
            </label>
          )}

          {clip.shape === 'custom' && (
            <>
              <label className={styles.clipLabel} htmlFor="clipW">
                Width (mm)
                <input
                  id="clipW"
                  className={styles.clipNum}
                  type="number"
                  min="1"
                  value={clip.width}
                  onChange={(e) => onClipChange('width', e.target.value)}
                />
              </label>
              <label className={styles.clipLabel} htmlFor="clipH">
                Height (mm)
                <input
                  id="clipH"
                  className={styles.clipNum}
                  type="number"
                  min="1"
                  value={clip.height}
                  onChange={(e) => onClipChange('height', e.target.value)}
                />
              </label>
            </>
          )}
        </div>
      )}

      {dimsActive && (
        <p className={styles.clipNote}>
          These dimensions set the printed size — Target max size is ignored.
        </p>
      )}

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
