# Decisions

## Needs you

Decisions only the user makes. Each entry must be answerable in 30 seconds — options and a
recommendation are part of writing it, not a follow-up.

---

### D-1: Which per-class road-width defaults are canonical?
**Status:** OPEN
**Raised:** 2026-08-09 · **Blocks:** F-4

**Context:** Two disagreeing default sets exist. `backend/constants.py` `ROAD_WIDTHS_MM` has
motorway 4.0 / trunk 3.0 / primary 2.0 / secondary 0.5 / tertiary 0.35 / residential 0.25 (mm).
The frontend inputs and `RoadEtchParams` defaults differ (e.g. motorway 2.0–2.5). The UI always
sends explicit widths, so `constants` values only act as fallbacks — but the disagreement is a
correctness trap and the fallback path is untested. These are print dimensions, so the numbers are
yours.

**Options:**

| Option | Upside | Downside | Reversibility |
| --- | --- | --- | --- |
| Adopt the UI values as canonical; mirror them in `constants` | One source; matches what you print with | Must confirm each class's mm | Easy |
| Adopt the `constants` values; set the UI defaults to match | Keeps the wider motorway | UI defaults change under you | Easy |
| Keep both; document "UI wins, constants are fallback" | No number changes | The trap stays | Easy |

**Recommendation:** Pick one canonical set (I lean toward the UI values, since they're what you
tune when printing) and make `constants` mirror it — but the actual millimetres are your call.

---

### D-2: How should overlapping road carves combine — deepest-wins or accumulate?
**Status:** OPEN
**Raised:** 2026-08-09 · **Blocks:** F-2

**Context:** `core/mesh.carve_roads` applies carves shallow→deep with `-=`. Where two road buffers
overlap (e.g. a junction), both depths subtract, so the trench is the *sum*. The function's own
docstring and `tests/test_carve_order.py` state the intended rule is *deepest wins*. So code and
contract disagree. This is print geometry, so the rule is yours.

**Options:**

| Option | Upside | Downside | Reversibility |
| --- | --- | --- | --- |
| Deepest-wins (per-pixel max delta) | Matches the stated contract; no surprise pits at junctions | Junctions no deeper than the deepest road | Easy (small change) |
| Accumulate (current behaviour) | Junctions read as deeper cuts | Can punch through thin terrain/base; unstated | Easy |

**Recommendation:** Deepest-wins — it matches the documented intent and avoids accidental
punch-through at intersections. But confirm, since it changes carved geometry.

---

## Resolved

| ID | Decision | Chose | Date |
| --- | --- | --- | --- |
| — | Frontend framework for the rewrite | React + Vite (rewrite the vanilla-JS Leaflet app) | 2026-08-09 |
| — | Scope of the migration session | Full migration in one session (factory + restructure + React) | 2026-08-09 |

## Decision Log (made by Claude)

One line each — what makes autonomous tiers auditable. Newest first.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-08-09 | Pin backend lint via `backend/ruff.toml` (select E4/E7/E9/F/I, line-length 88) | Make `verify.py` reproducible regardless of any global/parent ruff config |
| 2026-08-09 | Split `dem_stl.py` → `core/{geometry,dem,roads,mesh,pipeline}.py` + `constants.py` + `config.py` | Factory backend layout; keep FastAPI out of `core/` so the science is testable |
| 2026-08-09 | Centralize domain constants in `constants.py` with sourced comments | science-integrity: no magic numbers in formulas |
| 2026-08-09 | Extract `build_carve_plan()` from `carve_roads` | Make the carve-ordering logic unit-testable directly |
| 2026-08-09 | Add `python-dotenv` + `config.py`; load `.env` from repo root | The old `.env` was never read; wire it up properly |
| 2026-08-09 | Rename API-key var to `OPEN_TOPO_API_KEY` (was `API_KEY`) | Match what the code reads; the old name was inert |
| 2026-08-09 | Drop dead DXF export (`write_roads_dxf`) and `ezdxf` + `osm2geojson` deps | Unused code/deps; prefer deleting |
| 2026-08-09 | Serve the built React app from FastAPI at `/`; Vite proxy for dev | Keeps local use a single command; standard Vite+FastAPI split |
| 2026-08-09 | Raw Leaflet + leaflet-draw (not react-leaflet) in the React port | Faithful 1:1 port of the existing map behaviour, less abstraction |
