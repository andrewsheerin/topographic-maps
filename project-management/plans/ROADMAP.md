# Roadmap

*Phases are lettered. Each ends at something runnable and reviewable. The current phase is planned
in detail; later phases stay sketches until their predecessor closes.*

## Phase A — Factory migration & as-built baseline

**Goal:** move the working app into the app-factory structure (backend/ + frontend/, React, PM
system) with behaviour preserved and the health check green.
**Done when:** `python verify.py` is green; app builds and serves; plans/register/decisions reflect
reality; leaked API key rotation is queued as a chore.
**Register items:** F-1

## Phase B — Correctness pass

**Goal:** resolve the domain decisions and the known defects surfaced during migration, so the
science is settled and the two broken paths work.
**Done when:** D-1 and D-2 answered and implemented; shapefile upload works end to end;
calculation logic (mesh scale, carve) has tests against a user-verified example.
**Register items:** F-2, F-3, F-4

## Phase C — Feature growth

**Goal:** the new features that motivated this cleanup (TBD with the user). Candidate themes:
expose more DEM datasets in the UI, in-browser 3D preview, saved presets.

## Later / unphased

Ideas live in `FEATURE_REGISTER.md` with Phase = `later`.
