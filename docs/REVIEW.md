# Review and curation (optional)

The review layer runs **after** automatic extraction. It never calls the detector
and never edits the original `chunk_*.npz`. It is meant for the small number of
frames where the automatic count differs from the run mode (or where a user
explicitly wants a second look).

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
detections (x, y, radius, theta, confidence, segment id), and the exported frame
path. Images are exported at native resolution so editor coordinates are in the
same pixel frame as the detector.

## 2. Open the editor

Open `tools/review_editor.html` in a browser, then:

1. choose `review_queue.json`;
2. choose the whole `review/case01/frames` directory;
3. add a missing particle by clicking its centre (radius = queue median radius);
4. remove a false detection by clicking its circle;
5. undo with <kbd>Ctrl</kbd>+<kbd>Z</kbd>, navigate with the arrow keys;
6. click **保存 review_actions.json**.

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

## Trajectory note

This first version curates **detection rows**, not trajectories. Existing
`segment_id` values are preserved for untouched rows; manually added rows are
unlinked. Run the linker on the curated chunks before doing trajectory-level
analysis. Manual rows can have an estimated marker direction, but if the marker
is not visible the row is explicitly angle-invalid rather than guessed.

## Safety rules

- do not run `curate` with the same output directory as the input case;
- do not commit raw frames or review images to the public package;
- keep `review_actions.json`, `review.json` and the curated hashes with the
  analysis record;
- a skipped action is visible in `review.json`; do not silently ignore it.
