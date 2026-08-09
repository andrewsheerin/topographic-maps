import Button from '../ui/Button.jsx';
import styles from './OutputSection.module.css';

export default function OutputSection({
  onGenerateStl,
  onGenerateBundle,
  bundleDisabled,
  status,
}) {
  return (
    <section>
      <h2>4. Output</h2>

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

      <div className={styles.status} aria-live="polite">
        {status}
      </div>

      <footer className={styles.footer}>
        <div className={styles.small}>Requires OpenTopography API key.</div>
      </footer>
    </section>
  );
}
