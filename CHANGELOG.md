# Changelog

## 0.3.2 — 2026-11-11

- Fixed `load_mask` for browser-exported RGBA masks: convert to luminance before
  shape checking and threshold at mid-grey. This makes a 2400x2400 RGBA PNG usable
  directly with `extract --mask`.
- The manual editor now binarizes the exported mask before PNG encoding.

## 0.3.1 — 2026-11-11

- Documentation fix for the manual editor: use an unscaled frame when exporting a
  mask; record `status=manual` in the README and schema notes.

## 0.3.0 — 2026-11-11

- Added `--shape irregular`: a non-convex contour fit for non-circular cells; `auto`
  now falls back to it when a Hough circle covers less than 80% of the foreground.
- Added manual masks: `mask --points boundary_points.json` writes a `status=manual`
  proposal with the same overlay/JSON contract; optional `holes_px` are supported.
- Added `tools/mask_editor.html`, a local no-upload polygon editor that exports
  `interior_mask.png` or `boundary_points.json`.
- Added `vibrotrack frame` for exporting one normalized PNG/JPEG frame for the editor.
- Kept `--shape polygon` as the convex-hull behavior for rectangular/hexagonal cells.
- Frozen boundary discs are detected from the temporal mean and subtracted as circular
  bites when a static ring is present.

## 0.2.1 — 2026-11-11

- Boundary proposal now detects a low-temporal-std frozen-particle/static ring: the
  green `interior_mask.png` stops inside the ring, and `mask_overlay.png` draws the
  detected outer ring in orange.
- Added `outer_radius_px`, `ring_detected`, `ring_inner_radius_px`, `active_std` and
  `std_threshold` to `mask.json`.
- `--margin-px` is now an inward margin from the detected boundary (negative values
  expand outward). For frozen-ring recordings, use extraction with
  `--wall-band 0 --wall-model off`.
- Synthetic frozen-ring test added.

## 0.2.0 — 2026-11-11

- New `vibrotrack mask` command: proposes a boundary mask from the recording, writes
  `interior_mask.png`, a real-frame + semi-transparent `mask_overlay.png`, and
  `mask.json` for human confirmation.
- Circle proposal uses a centred Hough fit with foreground-contour fallback; polygon
  proposal uses the convex hull for hex/rectangular cells.
- `mask --frames`, `--shape`, `--margin-px`, `--threshold`, `--preview-frame`, `--force`.
- Synthetic 8-bit boundary-proposal test added; no change to detector/linker behavior or
  the chunk schema.

## 0.1.1 — 2026-11-10

- `vibrotrack extract --mask <png>` overrides `boundary.mask` in the apparatus config.
  This keeps per-case masks in the data root while the apparatus config stays path-free.
- No schema or detector behavior change.

## 0.1.0 — 2026-11-10

- First public split from the private `vibrobot-experiment/extract` package.
- Config-driven apparatus: `fps`, `px_per_mm`, particle diameter, image shape,
  boundary mask, scale levels, detector/linker defaults.
- Removed all lab-specific case tables and absolute data paths.
- Added `vibrotrack` CLI: `extract`, `geometry`, `index`, `accept`, `verify`, `archive`.
- Added `dataset.json`, `index.json`, `acceptance.json` schemas and hash verification.
- Added synthetic end-to-end tests; the test suite needs no lab data.
- Kept the v1 detector frozen for archive reproducibility; v2 remains the default.
