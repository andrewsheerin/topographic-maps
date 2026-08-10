import Button from '../ui/Button.jsx';
import styles from './AreaSection.module.css';

export default function AreaSection({ onUpload, onClearPolygon }) {
  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    // Reset so selecting the same file again re-triggers change.
    e.target.value = '';
  };

  return (
    <section>
      <h2>1. Area</h2>
      <p className={styles.hint}>
        Draw a polygon, rectangle, or circle on the map (use the tools top-left),
        or upload a zipped shapefile. Rectangles can be dragged square.
      </p>

      <label className={styles.file}>
        <input type="file" accept=".zip" onChange={handleFile} />
        <span>Upload shapefile (.zip)</span>
      </label>

      <Button onClick={onClearPolygon}>Clear polygon</Button>
    </section>
  );
}
