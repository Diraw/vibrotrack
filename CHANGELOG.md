## 0.7.0 — 2026-11-11

- keep detector prior across shard boundaries with configurable warm-up frames
  (`run.shard_overlap`, default 5);
- use the last N frames of detections as prior seeds (`run.prior_frames`,
  default 5), instead of only the immediately previous frame;
- encode prior/overlap in the `dets_*` cache tag so stale caches cannot mix;
- add diagnostics in `docs/TROUBLESHOOTING.md`;
- test synthetic frame coverage and prior/overlap config defaults.

## 0.6.1 — 2026-11-11

- record `manual_rows` (frame, position, final segment id, link flag, gap) in
  `relink.json` so curated rows can be audited after re-linking.

## 0.6.0 — 2026-11-11

- add `vibrotrack relink`: recompute trajectory identities on an existing or
  curated case without re-detecting;
- preserve `xy_px`, `radius_px`, `theta_rad`, `angle_valid`, `manual_edit` and
  all detector quality fields;
- add `relink/v1` audit with source/output hashes and track statistics;
- add tests for manual rows across chunk boundaries and both v1/v2 linkers.

# Changelog

## 0.5.1 — 2026-11-11

- The review editor now uses a precise dot cursor over the canvas instead of the
  crosshair.
- `curate --raw --config` estimates the dark-marker orientation for human-added
  centres with the same local centroid rule as the detector. If the marker is not
  detectable, the manual row remains `angle_valid=False` with `theta_rad=NaN`.

## 0.5.0 — 2026-11-11

- Added an optional post-extraction review layer. `vibrotrack review` flags frames
  whose particle count differs from the run mode, exports the review queue and
  native-resolution frames.
- Added `tools/review_editor.html`: a local, no-upload editor for adding missing
  centres and removing false detections.
- Added `vibrotrack curate`: applies a `review-actions/v1` log to a separate
  curated copy, with `manual_edit`, `review.json`, source/action/output hashes and
  unchanged original chunks.
- Manual additions use the queue median radius, `segment_id=-1`, `angle_valid=False`
  and `theta_rad=NaN`; trajectory linker runs are intentionally separate.

## 0.4.0 — 2026-11-11

- Added `vibrotrack plot`: render selected raw frames with fitted circles, centres
  and orientation ticks from `chunk_*.npz`.
- Added `vibrotrack.visualize.theta_endpoint`: one tested, public helper for the
  y-up `theta_rad` convention. The image endpoint is
  `(cx + r*cos(theta), cy - r*sin(theta))`; using `+sin` mirrors every tick.
- Added `docs/OVERLAY.md` and a regression test that checks both the helper and a
  rendered tick.

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
