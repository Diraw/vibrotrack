# Changelog

## 0.10.0 — 2026-11-12

- `geometry` computes `phi` from per-detection radius classes when the chunk
  carries more than one radius: `sum(n_i * r_i^2) / R^2`, with `radius_classes`
  and `phi_method` recorded. A single-radius `phi` overcounts the large discs in
  a mixed bed. On the 306-frame 12/14 mm recording: classes 31.65 px / 481 and
  36.74 px / 359, phi 0.810 (the single-radius figure was 0.924).

- Each template can carry its own threshold (`small_template_thresh`). On the
  labelled 12/14 mm frame, lowering the 12 mm threshold from 0.20 to 0.10 only
  adds 2 of 487 small discs: the remaining misses have no peak under either
  template, so the threshold is not the bottleneck.
- A second template can be matched for the smaller radius class
  (`frame(..., small_template_radius=15.5)`, px in the 0.5x image). The 14 mm
  template only finds a peak on 49.5% of labelled 12 mm discs; a dedicated 12 mm
  template finds 95.9%. Peak sets from both templates are merged with a 15 px
  keep-the-higher-score suppression. On the 306-frame 12/14 mm recording, manual
  labels (360 large + 487/488 small per frame) give small-disc recall 38.8/42.7/
  45.1% before and 87.9/94.5/95.9% after; tracks 1193 -> 852 against 847 real
  particles and link_fraction 0.9676 -> 0.9961. Monodisperse G0 is unchanged
  (300 frames, every array identical) because the parameter defaults to None.

- Multi-radius detection for bidisperse beds. `detect2.frame` now infers one or two
  radius classes from the free-fit radii (1-D k-means, centres at least
  `radius_split_px` apart and the small class at least 5% of the solid peaks), or
  accepts an explicit `radii=[r1, r2]`. Each candidate is fitted at its class
  radius, the smaller class gets twice the search shift and twice `min_rays`
  because the 14 mm template does not match a 12 mm disc and a pocket between
  beads keeps only 18-31 of 64 rays, and NMS uses the larger of the two radii.
  Monodisperse output is unchanged: 300 paired G0 frames reproduce every detection
  array exactly.
- Small-class detections are not seeded from prior positions: pocket fits repeat at
  the same geometry, so a prior re-finds them and multiplies false positives
  (frame 100: 174 -> 290 small detections for about 173 real discs). The 14 mm
  class may still use priors.
- `refine` also fills head gaps: a track that starts after the first frame is
  extrapolated backwards from its first velocity and fitted like any other hole.
  On the 306-frame recording the first frame goes from 87.9% to 96.5% small-disc
  recall and 98.1% to 99.7% large-disc recall, matching the 98-99% of the
  interior frames without manual first-frame editing.
- `refine` fits each hole at the radius of its two neighbouring detections instead
  of one global median, and `--min-track-length N` drops identities shorter than N
  frames before gap finding. On the 306-frame 12/14 mm recording the radius change
  alone cuts refused attempts from 181 to 32 and leaves 0 gap rows; the length
  filter removes the fragments that pocket fits form (ml=50: 561 rows/frame,
  889 tracks; ml=100: 455 rows/frame, 536 tracks; about 526 particles).
- Measured limits on the 306-frame recording: the small-disc match rate rises from
  83/173 to 150/173, but pocket fits still account for roughly a third of the
  small-class detections, and the small discs break into about three track
  fragments each. The class is usable with review, not yet a drop-in default.

## 0.9.5 — 2026-11-12

- The candidate photometric veto now computes only the 80th-percentile
  brightness (`_brightness`) instead of running the full marker fit and
  discarding the angle/area/centroid fields. 300 paired real frames:
  detection 0.2235 -> 0.2001 s per frame (-10.4%), every detection array
  identical.

## 0.9.4 — 2026-11-12

- `refine --workers N` fits independent holes in parallel. Each worker decodes a
  frame once and returns only the fit fields, so no full frame crosses a process
  boundary. G0 (391 gaps / 468 attempts): 100.6 s -> 44.2 s; `refine.json`
  attempts and every chunk array are identical to the serial run.

## 0.9.3 — 2026-11-12

- Harden the exact assignment fast path: refuse float32 input, nonfinite or
  absurd (>1e9 px) coordinates, and any candidate whose distance sits within a
  numeric margin of the gate; those cases fall back to the dense reference
  assignment. Pixel-coordinate frames keep the measured 64% linking gain.
- `tools/benchmark_linker.py` now records the commit, dirty state, source and
  config hashes, the input-array fingerprint and the measured environment, warms
  both paths on a genuine prefix, and reports p95 run time plus process memory.
- Add `tools/benchmark_prior_dedupe.py`: a paired rolling-prior regression that
  reports detection counts, nearest-neighbour position offsets, angle-valid and
  quality-bit differences, radius-reference drift and paired frame timings.
- Document `detector.params.prior_dedupe`: a prior seed already covered by a
  template peak is redundant, and dropping it removes most of the ~5x seed
  overdraw. Its default is unchanged (0); projects opt in. On 300 paired real
  frames at 6 px it ran detection 0.365 -> 0.171 s per frame with a 0.009 px
  median position shift and 22/300 frames differing by one detection. A full
  9001-frame run with it still closes to 4,644,516 rows / 516 tracks / 0 gap
  rows and matches the existing product within 0.041 px median offset.
- `geometry` now prints a warning instead of silently omitting `n_particles` and
  `phi` when neither `--n-particles` nor `particle.nominal_n` is set. The load
  changes per experiment; the config example no longer suggests a fixed value.

- Add an exact unambiguous-assignment fast path: use a conservative spatial
  query, verify distances with the reference np.hypot strict gate, and bypass
  Hungarian only when every gated row and destination is unique. Competition,
  ties, unusual gates and nonfinite inputs retain the dense reference path.
  EMA, gap closing and full-state replay remain unchanged. G0 three-pair runs:
  linking 86.15 -> 32.07 s; complete refine 104.18 -> 48.99 s, with every output
  array identical. This supersedes withdrawn local-splice speed claims.

- Withdraw local identity splicing: a recovered detection changes EMA velocity
  and can alter subsequent Hungarian competition. G0 equivalence did not prove
  generic equivalence. Both the default path and compatibility splice entry point
  now replay the reference linker; record the fallback reason.
- Fix tail recovery using the maximum input frame, including empty trailing
  frames, instead of the last row after sorting by identity.
- Keep distinct boundary-target predictions until pixel fitting. Premature
  source/span deduplication could reject a valid recovery without attempting it.
- Add adversarial regressions for EMA competition, partial/long recovery,
  boundary chains, empty tails and alternate target seeds.

## 0.9.2 — 2026-11-12

- `refine` now writes a complete batch: `masks/`, `apparatus.yaml` and the batch
  `README.md` are copied next to `sequences/`, so `geometry`, `index`, `accept`
  and `verify` can run on the refined batch without a manual copy step
  (`index.json`/`acceptance.json` are deliberately not copied - they describe the
  pre-refine chunk hashes);
- the copied batch README gets a short "Refined copy" section with the source
  case, recovered/refused counts and the row/track totals, and `refine.json`
  records which batch assets were carried over;
- the copied summaries (`full_summary.json`, `progress.json`) are refreshed: segments, link_fraction,
  gap_closed, median_step_px, count_min/median/max, valid fractions and
  `refined_at`. The parent's numbers contradicted the refined chunk while still
  passing `accept`, which only hashes the file;
- the docs now state the default pipeline: extract -> **refine** -> geometry ->
  index/accept/verify. Manual `review`/`curate` is the fallback for the holes
  `refine` refused, not a required pass.

## 0.9.1 — 2026-11-12

- `refine` no longer re-links the whole case. The filled rows are spliced into
  the existing identities: a direct link when the hole is one frame, a
  gap-closing row otherwise, the two fragments of a boundary hole unioned under
  the earlier id, and every id renumbered in order of first appearance. Inserted
  rows shift the global row numbers that `previous_detection` points at, so those
  are remapped, and `dtheta_rad` is recomputed from the final predecessor for
  every direct link. G0: 109 s -> 21 s, bit-identical to a full re-link on all
  26 arrays of the chunk;
- add `refine --full-relink` to run the linker over the whole case, which is the
  audit path the splice is checked against;
- vectorise `find_gaps`: it looped over 4.6 M rows in Python and sorted one list
  per track (8 s), now it works on the concatenated arrays;
- `detect.files_for` lists the frame directory with `os.scandir`; `Path.is_file()`
  cost one stat per frame (3.4 ms x 9000 = 31 s on the USB archive, paid by every
  `extract`/`refine` run);
- `refine.json` and `dataset.json` record `link_mode` (`incremental` or `full`).

## 0.9.0 — 2026-11-12

- new `vibrotrack refine`: recovers particles that a linked track proves must
  exist but the detector dropped, without re-running detection. Each hole -
  an interior gap, a fragment end the linker could not join, or a track that
  stops at the last frames - seeds a 91x91 window with the track prediction and
  runs the same fixed-radius radial fit as the detector;
- a row is written only when the fit converges, keeps at least 14 of 64 rays,
  has edge rmse <= 1.2 px and lands within 12 px of the prediction. New rows
  carry `recovered=1`, existing rows are copied unchanged, and the case is
  re-linked before writing, so a filled frame joins the track it belongs to.
  Nothing is interpolated into the output;
- `refine.json` records every attempt (prediction, fitted centre, rays, rmse,
  offset, rejection reason) so a refusal is auditable;
- the fit restarts from a 3x3 grid around the prediction and follows the nearest
  fixed point instead of giving up when no candidate passes on the first
  iteration: a seed 6 px off at the wall used to end the attempt even though the
  disc was there;
- on the dense G0 case this recovered all 66 missing particles with a median
  1.04 px difference from the manual points (max 3.03 px) and reproduced the
  curated totals exactly: 4,644,516 rows, 516 tracks, 0 gap rows;
- `vibrotrack refine --max-gap 10` is needed on a pre-relink case whose longest
  hole (8 frames) exceeds the linker's own gap.

## 0.8.0 — 2026-11-11

- `vibrotrack review` now stores `hints` on every flagged frame: the likely
  position of each missing particle, taken from the neighbouring frames. A track
  that broke into two segments is reported once, with both ids, so a long gap is
  not mistaken for two missing particles;
- the review editor draws those hints in screen space: a filled halo, a dashed
  ring, a crosshair and a "missing" chip, so the marker stays readable at any
  zoom; it hides once a manual point is placed and can be toggled off;
- detections are drawn as rings from 30 % zoom up and as small dots below, so
  the fit view no longer turns into a thick green mesh;
- the editor canvas zooms (wheel, +/- keys, 适应 / 1:1 buttons) and pans by
  dragging; a drag never adds a point, and the overlay keeps a constant
  on-screen width at any zoom;
- the review editor loads a whole review directory in one step (directory
  picker or drag-and-drop): it finds `review_queue.json` and the frames it
  references, instead of asking for the queue and the folder separately;
- the review editor sidebar tracks progress per frame and overall (`缺 N`,
  `1/2`, `✓`), so a frame that is short two particles is not marked done
  after one click;
- the hint chip is clickable and pans the view to the reference position;
  when the target sits outside the viewport the chip is pinned to the edge
  with a direction arrow;
- a drag only swallows its own click, so the next click on a chip works;
- the long help paragraph is gone; the legend stays and the instructions
  live in an "操作说明" dialog opened from the header;
- hovering a hint chip just lightens its background, so it reads as a button;
- opening a review directory now restores its `review_actions.json` in the same
  gesture, so a half-finished review resumes without a separate import step;
- the editor writes that log back to the directory on every change (debounced),
  with local storage as a fallback, so closing the page loses nothing;
- fix `relink` reading a chunk column once per frame: `np.load` decompresses
  on every access, so the per-frame slice re-inflated the whole column and a
  full case took hours instead of ~3.5 min (read phase 20 s, link 3 min);
- add a test that hints point at the missing detection.

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
