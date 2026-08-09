# Database

*Schema and how the database is managed. **This file changes in the same commit as any schema
change.** If they disagree, the code is wrong or this file is — either way the task isn't done.*

## Engine & management

**None.** The app is stateless. There is no persistent store, no ORM, and no migrations. Each
request:

- writes intermediate GeoTIFFs, meshes, and the STL/ZIP to a fresh `tempfile.mkdtemp()` directory,
- streams the result back as a file download,
- keeps nothing between requests.

Inputs (DEMs, OSM roads) are fetched live from external services per request (see DATA_SOURCES.md),
not stored. If persistence is ever needed (saved presets/projects), that's an L2-shaped decision —
raise it, don't add a store speculatively.

## Schema

n/a — no database.

## Conventions

n/a.

## Migration log

| Date | Change | Migration / commit |
| --- | --- | --- |
| — | No database in v1 | — |
