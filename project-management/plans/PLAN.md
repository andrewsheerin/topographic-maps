# Plan

*What the app is. Changes rarely — only when scope genuinely changes.*

## Purpose

TOPO2STL turns a real-world area into a 3D-printable terrain model. The user draws a polygon (or
uploads a shapefile), the app pulls a Digital Elevation Model for that footprint, optionally fetches
OpenStreetMap road centerlines and recesses them into the surface, and exports an STL sized for a
printer. Built for the author's own use.

## Tier

**L1 — Build it.** Greenfield, local, single user, no auth, no deployment. The risk here is
over-building; posture is lean. See `.claude/rules/tier-L1.md`. Graduate to L2 when it needs
hardening for anyone but the author.

## In scope (v1)

- Area selection: draw a polygon on a Leaflet map, or upload a zipped shapefile.
- DEM fetch from OpenTopography (USGS 10 m / 30 m today; other global products wired but not yet
  surfaced in the UI).
- Terrain mesh export as STL, with print controls: downsample, vertical exaggeration, buffer,
  target max size (mm), optional base.
- OSM roads (Overpass) per class, recessed into the terrain; ZIP bundle of raw STL + carved STL +
  roads GeoJSON.

## Explicitly out of scope

- Any hosting/deployment, multi-user, accounts, or auth.
- A database or saved projects — every run is stateless.
- In-browser 3D preview / slicing (idea for later, not v1).
- Non-DEM data products (land cover, imagery draping, contours).
- Automatic selection of scientific defaults — DEM choice, exaggeration, carve depths, and
  overlap rules are the user's (see Domain assumptions).

## Design direction

Dark, technical, single-screen tool: a control panel beside a full-height topo map. One acid-green
accent on near-black. It should read like an instrument, not a marketing page. Tokens live in
`frontend/src/styles/tokens.css`.

## Domain assumptions

Ground rules the user owns (anything not fixed here gets asked, per `science-integrity.md`):

- Elevation is relief-only (min elevation → 0); Z uses the same mm-per-metre scale as X/Y, times a
  vertical-exaggeration factor. Aspect ratio is preserved.
- Carve widths/depths are in print millimetres; DEM elevations and buffers are in metres.
- **Open (D-1):** the canonical per-class road-width defaults are unsettled (two different sets
  exist in the codebase).
- **Open (D-2):** how overlapping road carves should combine (accumulate vs. deepest-wins) is
  unsettled; current code accumulates.
