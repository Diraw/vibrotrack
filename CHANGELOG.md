# Changelog

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
