# Troubleshooting

## `no frames found`

- `extract` looks for `.tiff` / `.tif` in the **top level** of `--raw`; it does not recurse.
  Point `--raw` at the directory that directly contains the frames, or pass
  `extensions: ['.tiff']` in the config.
- Check the filenames are not hidden or zero-byte. A trailing zero-byte frame is dropped
  with a warning; a zero-byte frame in the middle is a hard error.

## `boundary mask shape ... does not match image shape`

The mask must have the same `(height, width)` as the frames. Re-export it at the native
resolution; do not resize a mask from another ROI.

## Count is much lower than the nominal load

- Is `scale_levels` right? The default `[128, 28224]` is for the original Basler 16-bit
  data. A different camera may need `[0, 65535]` or its own black/white levels.
- Is the particle diameter/`px_per_mm` right? v2 estimates the frame radius from
  high-confidence detections; a wrong `px_per_mm` changes `particle_radius_px` and the
  wall-map exclusion.
- Are particles touching the wall? Set `boundary.band: 1` and `dilate_px: 40`; the
  empirical branch drops static outside-mask detections by temporal variance.
- Is the mask too tight? If it cuts into the bed, the detector cannot see the rim.

## Count is much higher than the nominal load

- The frozen particle ring is being detected. Use a mask that stops inside the ring, keep
  `boundary.band: 1` with `model: empirical` (temporal-std rule), or supply a measured
  `wall_disks.npy` and use `model: geometry`.
- `detector.params.thresh` may be too low; raise it (e.g. 0.24) and compare overlays.

## `geometry_warning: envelope_resid_rms_px > 30`

The bed is too sparse or the mask/centre seed is wrong. The envelope fit needs particles
around the full circumference. For a sparse case, inherit the centre shift from a dense
case recorded in the same container seating; do not trust the sparse fit.

## `center_valid` / `angle_valid` rate below the gate

- `center_valid` drops when the radial edge fit fails; check focus, exposure, and whether
  the mask cuts particles.
- `angle_valid` drops when the marker is not visible or the marker area/offset is outside
  the expected range. Inspect `marker_area_px` and the overlay images in `dets_*/`.
- The gate is configurable: `vibrotrack accept --min-center 0.85 --min-angle 0.85` for a
  pilot, but do not lower it silently for a production batch.

## `libGL.so.1: cannot open shared object file` (Linux)

```bash
sudo apt-get install -y libgl1
# or use a headless OpenCV build in a dedicated environment
```

## Multiprocessing on Windows / Jupyter

The CLI is the supported entry point; it guards the process pool. If you call
`pipeline.run_case` from a notebook, put it under `if __name__ == "__main__":` and make
sure the notebook kernel is not spawning itself.

## `verify` fails after copying a batch

Copy the whole batch directory, including `index.json` and `acceptance.json`. If files
were modified after `index`, re-run `vibrotrack index` (which re-hashes) before `verify`.
Do not edit a frozen archive.

## Count drops at regular intervals / short trajectory fragments

If low-count frames cluster near 500-frame boundaries, the detector lost its
previous-frame prior at a shard boundary. This is not a missing particle in the
image. The extractor now defaults to:

```yaml
run:
  prior_frames: 5      # use the last 5 frames as detector seeds
  shard_overlap: 5     # re-detect 5 warm-up frames before each shard
```

Check `run_config.json` for these values and the `dets_*` cache tag (`p5_o5`).
For a case extracted with an older cache, use a new output directory or remove
the old `dets_*` directory; changing these settings changes the detections and
therefore the chunk. Reproduce the old behavior with
`--prior-frames 1 --shard-overlap 0`.
