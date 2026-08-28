import { useEffect, useState } from 'react';

import Button from '../ui/Button.jsx';
import * as api from '../../lib/api.js';
import { US_STATES } from '../../lib/usStates.js';
import styles from './PlacePicker.module.css';

const PAGE = 50;

/**
 * Search TIGER county subdivisions (cities/towns) by state and name.
 * Owns its search/list state; reports the chosen place summary up via onPick —
 * the parent fetches the boundary and sets the area.
 */
export default function PlacePicker({ onPick, selectedGeoid }) {
  const [state, setState] = useState('');
  const [query, setQuery] = useState('');
  const [places, setPlaces] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');

  // Debounced search whenever the state filter or query changes.
  useEffect(() => {
    const q = query.trim();
    if (!state && !q) {
      setPlaces([]);
      setHasMore(false);
      setError('');
      return undefined;
    }
    setSearching(true);
    const timer = setTimeout(async () => {
      try {
        setError('');
        const rows = await api.fetchPlaces({ state, q, limit: PAGE, offset: 0 });
        setPlaces(rows);
        setHasMore(rows.length === PAGE);
      } catch (err) {
        setPlaces([]);
        setHasMore(false);
        setError(err.message);
      } finally {
        setSearching(false);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [state, query]);

  const loadMore = async () => {
    try {
      const rows = await api.fetchPlaces({
        state,
        q: query.trim(),
        limit: PAGE,
        offset: places.length,
      });
      setPlaces((prev) => [...prev, ...rows]);
      setHasMore(rows.length === PAGE);
    } catch (err) {
      setError(err.message);
    }
  };

  const showEmpty =
    !searching && !error && (state || query.trim()) && places.length === 0;

  return (
    <div>
      <div className={styles.filters}>
        <select
          value={state}
          aria-label="State"
          onChange={(e) => setState(e.target.value)}
        >
          <option value="">All states</option>
          {US_STATES.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </select>
        <input
          type="search"
          value={query}
          placeholder="Search cities and towns"
          aria-label="Search cities and towns"
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {!state && !query.trim() && (
        <p className={styles.note}>Pick a state or search by name.</p>
      )}
      {searching && <p className={styles.note}>Searching…</p>}
      {error && <p className={styles.error}>{error}</p>}
      {showEmpty && (
        <p className={styles.note}>
          No matches. Try another spelling or state.
        </p>
      )}

      {places.length > 0 && !searching && (
        <>
          <ul className={styles.results}>
            {places.map((p) => (
              <li key={p.geoid}>
                <button
                  type="button"
                  className={
                    p.geoid === selectedGeoid
                      ? `${styles.place} ${styles.placeActive}`
                      : styles.place
                  }
                  onClick={() => onPick(p)}
                >
                  <span className={styles.placeName}>
                    {p.name}
                    {p.county ? (
                      <span className={styles.placeMeta}> · {p.county} Co.</span>
                    ) : null}
                  </span>
                  <span className={styles.placeMeta}>
                    {p.state} · ~{p.area_km2} km²
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {hasMore && (
            <Button className={styles.more} onClick={loadMore}>
              Show more
            </Button>
          )}
        </>
      )}
    </div>
  );
}
