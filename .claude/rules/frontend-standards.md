# Frontend standards

Two goals, in order: **(1) every screen in this app looks like it came from the same app;
(2) the app doesn't look like a template.** Consistency first, because inconsistency is the thing
that makes a self-built app feel unfinished.

## The design system is decided once, at the start, and then obeyed

Before writing any UI, `frontend/src/styles/tokens.css` must exist and be approved. It contains:

- **Color** — 4–6 named values, defined as CSS custom properties. Semantic names
  (`--color-surface`, `--color-accent`), never literal ones (`--blue-500`).
- **Type** — a display face and a body face, chosen deliberately, plus a fixed scale
  (`--text-xs` … `--text-3xl`). No arbitrary font sizes anywhere else in the codebase.
- **Space** — one scale (`--space-1` … `--space-8`), used for all margin, padding, and gap.
- **Radius, border, shadow** — one small set each.

**Deciding the token set is a stop gate.** Low-fi exploration (grayscale wireframes, throwaway
mockups) is allowed before approval — but nothing merges into a real page until the tokens are
approved, and exploration code doesn't graduate into product code by inertia. After that, no component introduces a new color, font size, or spacing value. If a component
seems to need one, that's a token-system gap — raise it, don't inline it.

A calibration note: AI-built UI has converged on a few looks — cream background with a serif
display and a terracotta accent; near-black with one acid-green accent; broadsheet layout with
hairline rules and zero radius. Any of them can be right, but pick them on purpose or not at all.

## CSS file organization

The structure is the standard across every app, even though each app's actual style differs:

```
frontend/src/styles/
  tokens.css        THE universal file: fonts, colors, weights, spacing, radius, shadow —
                    every design value in the app, defined once as CSS custom properties
  global.css        resets, base element styles, app shell — imports tokens.css first
frontend/src/pages/<Page>/<Page>.module.css       one stylesheet per page
frontend/src/components/<Section>/<Section>.module.css   one per section/component
```

One page or section = one stylesheet, named for it, next to it. No shared "misc.css", no styles
defined in a different page's file, no orphan stylesheets.

## CSS rules

1. **Tokens or nothing.** Every color, size, and space value in a component references a token.
   A hex code outside `tokens.css` is a bug. `px` literals are fine for the things `px` is for —
   1px borders, icon dimensions, minimum hit-target sizes — but recurring values still get
   promoted to tokens.
2. **One styling approach per repo.** CSS Modules by default. Do not mix in Tailwind, styled-
   components, and inline styles — pick one at project start and hold it.
3. **Watch selector specificity.** Section-level and element-level selectors that both set padding
   or margin will cancel each other out unpredictably. Keep spacing ownership in one layer:
   containers own outer spacing, components own inner spacing.
4. **No `!important`.** If you need it, the specificity is wrong.
5. **Mobile works.** Not "responsive" as an aspiration — the layout must be usable at 380px.
6. Visible keyboard focus. Respect `prefers-reduced-motion`.

## Component rules

- One component per file, named the same as the file.
- Shared primitives (`Button`, `Input`, `Card`, `Modal`) are built **once**, in
  `frontend/src/components/ui/`, and reused. A second Button implementation is never the answer.
- ~150 lines per component and one component per file are **review triggers, not laws** — crossing
  one means stop and justify to yourself, and the phase-report audit flags them; a tightly coupled
  190-line component can be the right call.
- Data fetching does not live inside presentational components.
- Loading, empty, and error states are part of building a view, not a follow-up task. An empty
  screen says what to do next; an error says what happened and how to fix it.

## Interface copy

Sentence case. Active voice. A button says what happens when it's clicked — "Save changes," not
"Submit" — and keeps that verb through the whole flow, so "Publish" produces "Published." Name
things by what the user controls, never by how the system is built.
