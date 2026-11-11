# Review and curation (fallback)

The default pipeline is extract -> **refine** -> geometry -> index/accept/verify.
`refine` fills the holes a linked track proves must exist, so manual review is no
longer a required pass: it is the fallback for the holes `refine` refused and the
optional spot check. The review layer never calls the detector and never edits
the original `chunk_*.npz`.

## 0. The automatic recovery is the default

Most holes are not judgement calls. A linked track that is missing exactly one
frame still predicts the particle centre to a few pixels, and a fixed-radius fit
in a 91x91 window around that prediction recovers the true centre without any
human input. `refine` does this on a *linked* case and writes a new one:

```bash
vibrotrack refine \
  --case-dir relinked/case01 \
  --raw frames \
  --out refined/case01 \
  --max-gap 10
```

* only holes are attempted (interior gaps, fragment ends that the linker could
  not join, and tracks that stop at the last frames);
* a new row is written only if the fit converges, keeps >=14 of 64 rays, has
  edge rmse <=1.2 px and lands within 12 px of the prediction; otherwise the
  hole stays for a human;
* new rows carry `recovered=1`; existing rows are copied unchanged; the filled
  rows are linked by the full reference linker. Local splicing was withdrawn
  after EMA/competitive-assignment counterexamples; `--full-relink` remains
  compatible, and the actual mode/fallback reason is recorded;
* `refine.json` records every attempt (prediction, fitted centre, rays, rmse,
  offset, reason) so a rejected attempt is auditable.

On the dense G0 case this recovered 66/66 missing particles with a median
1.04 px difference from the manual points (max 3.03 px) and reproduced the
manually curated totals exactly (4,644,516 rows, 516 tracks, 0 gap rows).
The output is a complete batch (masks, apparatus, README are copied), so
`geometry`, `index`, `accept` and `verify` run on it directly.

Run `review` on the refined case afterwards: only the holes that `refine`
refused will be queued. Curation stays available as the fallback, not a step.

## 1. Build a review queue

```bash
vibrotrack review \
  --raw frames \
  --case-dir batch/sequences/case01 \
  --config apparatus.yaml \
  --out review/case01
```

By default the expected count is the **mode** of the per-frame counts. Use
`--expected-n N` when the nominal loading is known, and `--max-frames N` to cap a
large queue. The command writes:

```text
review/case01/
  review_queue.json
  frames/case01_frame00043.png
  ...
```

`review_queue.json` contains the count histogram, each flagged frame, its
detections (x, y, radius, theta, confidence, segment id), the exported frame
path, and `hints`: for every frame that is short of particles, the likely centre
of the missing one, taken from the neighbouring frames (up to 6 away). Two
segments that sit at the same place on either side of the gap are reported as a
single hint carrying both ids, so a track that broke in two is not counted
twice. Hints are a locating aid, not a detection: always confirm the particle on
the image before clicking. Images are exported at native resolution so editor
coordinates are in the same pixel frame as the detector.

## 2. Open the editor

Open `tools/review_editor.html` in a browser, then:

1. click **打开 review 目录** and pick `review/case01`: the editor finds
   `review_queue.json` and the frames it references in one step (dragging the
   same folder onto the page works too). Browsers cannot read a sibling folder
   just because a JSON was chosen, so the directory is the thing to grant;
2. zoom with the wheel and pan by dragging so the particle fills the canvas;
3. add a missing particle by clicking its centre (radius = queue median radius).
   The hinted position gets a cyan halo, a dashed ring, a crosshair and a `缺 1`
   chip drawn at a fixed screen size, so it is findable at any zoom. The chip is
   a button: hovering it lightens its background, and clicking it pans the
   view to that position (when the target is outside the viewport the
   chip sticks to the edge with an arrow). The hint disappears once a point is
   placed, and the **参考位置** button toggles all hints off;
4. remove a false detection by clicking its circle;
5. undo with <kbd>Ctrl</kbd>+<kbd>Z</kbd>, navigate with the arrow keys. The
   sidebar shows `缺 N` / `1/2` / `✓` per frame plus an overall count, so a frame
   that is short two particles is not signed off after one click;
6. click **保存 review_actions.json**; the header button **操作说明** explains
   all of the above without taking space on the page;
7. reopening the same directory restores its `review_actions.json` automatically,
   so a session resumes where it stopped; edits are written back to that file
   as you work (pick the directory with write access), with a local-storage
   copy as a fallback.

Nothing is uploaded. The editor writes only an action log; it does not write
back to the automatic chunks.

Example action log:

```json
{
  "schema": "review-actions/v1",
  "case": "manual100_G1",
  "actions": [
    {"frame": 43, "op": "add", "xy_px": [929.3, 90.7], "radius_px": 36.6,
     "reason": "manual_add"},
    {"frame": 64, "op": "remove", "xy_px": [1313.0, 770.5],
     "reason": "manual_remove"}
  ]
}
```

## 3. Apply actions to a separate copy

```bash
vibrotrack curate \
  --case-dir batch/sequences/case01 \
  --actions review/case01/review_actions.json \
  --out curated/case01 \
  --raw frames --config apparatus.yaml
```

`curate` writes:

```text
curated/case01/
  sequences/case01/chunk_*.npz
  sequences/case01/dataset.json       # qc.status = curated
  sequences/case01/review.json
  review.json
```

The original `chunk_*.npz` stays unchanged. The curated chunk contains an extra
`manual_edit` column:

- `0`: automatic detection, tracking fields preserved;
- `1`: human-added detection, with `segment_id=-1`.

If `--raw` and `--config` are supplied, `curate` estimates the dark-marker
direction from a 47×47 patch around each manually added centre. A valid estimate
sets `theta_rad` and `angle_valid=True`; if the marker is missing or ambiguous,
the row keeps `theta_rad=NaN`, `angle_valid=False`. This is an image-based
estimate, not a fabricated orientation. Without the two options, all manual rows
remain angle-invalid.

Removed detections are absent from the curated chunk. `review.json` records every
applied and skipped action, the action file hash, the source chunk hashes, frame
count changes and the curated output hashes.

## 4. Re-link trajectories

`curate` curates **detection rows**, not trajectories. Existing `segment_id`
values are preserved for untouched rows and manually added rows enter with
`segment_id=-1`. Before trajectory-level analysis, run the linker on the
curated case:

```bash
vibrotrack relink \
  --case-dir curated/case01 \
  --out relinked/case01
```

The input can be the batch directory (`curated/case01`) or its single
`sequences/<case>` directory. The output is a new batch-style tree:

```text
relinked/case01/
  sequences/case01/
    chunk_*.npz
    dataset.json          # qc.status = curated+relinked
    review.json           # copied audit
    relink.json
```

`relink` does not read raw frames and does not call the detector. It recomputes
only tracking fields:

- `segment_id`
- `previous_detection`
- `link_valid`
- `gap_frames`
- `link_margin_px`
- `step_px`
- `dtheta_rad`

Positions, radii, `theta_rad`, `angle_valid`, `manual_edit` and all detector
quality fields are unchanged. The linker and gap are read from
`run_config.json`; override with `--linker v1|v2` or `--link-gap N`.

`relink.json` records source chunk hashes, linker/gap, frame and row counts,
track count, linked/gap/manual row counts, and output hashes. It refuses to
write into the input case; use a new `--out` directory (or `--force` to replace
an existing output case).

```bash
# after relinking, the curated manual orientations are still present
vibrotrack plot \
  --raw frames \
  --case-dir relinked/case01/sequences/case01 \
  --config apparatus.yaml \
  --frames 43,64 --out overlays
```

## Safety rules

- do not run `curate` with the same output directory as the input case;
- do not commit raw frames or review images to the public package;
- keep `review_actions.json`, `review.json` and the curated hashes with the
  analysis record;
- a skipped action is visible in `review.json`; do not silently ignore it.
