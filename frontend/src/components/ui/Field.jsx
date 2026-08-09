import styles from './Field.module.css';

/**
 * Label + control wrapper. The control itself is passed as children so the
 * same primitive works for <input> and <select>.
 */
export default function Field({ label, htmlFor, children }) {
  return (
    <div className={styles.field}>
      <label htmlFor={htmlFor}>{label}</label>
      {children}
    </div>
  );
}
