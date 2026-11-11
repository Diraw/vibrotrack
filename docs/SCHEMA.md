# Output schema (v1)

Four JSON files describe a dataset. They are written in this order:

```text
mask     -> interior_mask.png, mask_overlay.png, mask.json
extract  -> chunk_*.npz, run_config.json, full_summary.json, dataset.json
geometry -> geometry.json
index    -> index.json
accept   -> acceptance.json
```

## `mask.json` — boundary proposal

`vibrotrack mask` writes `schema: mask-proposal/v1`. Automatic proposals use
`status: proposed` and a `method` such as `hough_circle`, `contour_irregular` or
`contour_polygon`; human polygons use `status: manual` and `method: manual_polygon`.
Fields include `shape`, `center_px`/`radius_px`/`outer_radius_px`, `ring_detected`,
`frozen_discs`, sampled frame indices, and sha256 of the mask and overlay. It is not
part of `dataset.json` until the mask is explicitly passed to `extract --mask`.
See [`MASK.md`](MASK.md) and [`MANUAL_MASK.md`](MANUAL_MASK.md).

## `dataset.json` — one case

```json
{
  "schema_version": "dataset/v1",
  "dataset_id": "ds-20261110-case01",
  "project": "",
  "campaign": "",
  "batch": "20261110_140hz_14mm_P",
  "case": "case01",
  "apparatus": "shaker-2d",
  "acquired_at": null,
  "extracted_at": "2026-11-10T22:00:00+08:00",
  "operator": null,
  "fps": 10.0,
  "n_frames": 18000,
  "image": {"width": 2500, "height": 2496, "dtype": "uint16"},
  "particle": {"diameter_mm": 14.0, "nominal_n": 485},
  "conditions": {"frequency_hz": 140.0, "acceleration_g": 2.0},
  "calibration": {"px_per_mm": 5.1558, "source": "cad_affine_assumed", "uncertainty_px_per_mm": null},
  "boundary": {"mode": "mask", "mask": "...", "mask_sha256": "...", "wall_model": "empirical", "wall_band": 1},
  "code": {"name": "vibrotrack", "version": "0.3.0", "commit": "...", "dirty": false},
  "outputs": [{"file": "chunk_00000_17999.npz", "bytes": 314582971, "sha256": "..."}],
  "qc": {"status": "extracted"},
  "retention": {"raw_path": "<raw volume>/...", "raw_deleted": false},
  "config": {"path": "apparatus.yaml", "sha256": "..."}
}
```

`qc.status` is `extracted`, `detected_only` (if `--no-link`), or a project-specific value.
`acquired_at` is deliberately nullable: if the acquisition time is unknown, leave it
null rather than inventing one.

## `index.json` — one batch

```json
{
  "schema_version": "index/v1",
  "created_at": "...",
  "batch": "20261110_140hz_14mm_P",
  "gate": {"min_center_rate": 0.90, "min_angle_rate": 0.90},
  "n_sequences": 2,
  "total_frames": 36001,
  "total_rows": 17462411,
  "sequences": [
    {
      "sequence": "case01",
      "dataset_id": "ds-20261110-case01",
      "frames_expected": 18000,
      "frames_covered": 18000,
      "rows": 8730992,
      "tracks": 705,
      "center_valid_rate": 1.0,
      "angle_valid_rate": 0.9997,
      "link_fraction": 0.99988,
      "npz_file": "chunk_00000_17999.npz",
      "npz_sha256": "...",
      "geometry_sha256": "...",
      "summary_sha256": "...",
      "run_config_sha256": "...",
      "dataset_sha256": "..."
    }
  ]
}
```

## `acceptance.json` — gate report

`accepted` is true only if every sequence passes. `gate.checks` lists the exact checks.
A failed sequence keeps its `problems` array; the report is written even when rejected.

## Hash policy

- `index.json` hashes every product it lists.
- `verify` re-hashes `chunk_*.npz`, `geometry.json`, `full_summary.json`,
  `run_config.json`, `dataset.json` and compares with `index.json`.
- `archive` copies the batch and writes `FROZEN.json` with a fresh hash of every file.
  Never edit a frozen directory; create a new batch instead.
