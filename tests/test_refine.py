"""Tests for the local recovery pass (2026-11-12).

Each test pins one decision behind `vibrotrack refine`:
  * a hole in a track is found from the linked ids alone,
  * a fixed-radius fit seeded 7 px away converges to the true centre,
  * a fit that cannot pass the gates is refused instead of invented,
  * `refine_case` writes new rows, marks them recovered=1 and re-links the case.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vibrotrack import refine as RF

R = 37.0


def _frame(centers, size=700, seed=3, noise=6.0):
    rng = np.random.default_rng(seed)
    im = np.full((size, size), 45, np.float32)
    for x, y in centers:
        cv2.circle(im, (int(round(x)), int(round(y))), int(R), 220, -1)
        cv2.line(im, (int(round(x)), int(round(y))),
                 (int(round(x + 0.75 * R)), int(round(y - 0.5 * R))), 90, 5)
    im += rng.normal(0, noise, im.shape)
    return np.clip(im, 0, 255).astype(np.uint8)


def _frames(ids_and_rows):
    """Build the [(fid, record), ...] list used by relink._read_case."""
    out = []
    for fid, rows in ids_and_rows:
        xy = np.asarray([r[0] for r in rows], np.float32).reshape(-1, 2)
        rec = {
            'xy_px': xy,
            'radius_px': np.full(len(xy), R, np.float32),
            'theta_rad': np.full(len(xy), np.nan, np.float32),
            'angle_valid': np.zeros(len(xy), bool),
            'center_valid': np.ones(len(xy), bool),
            'confidence': np.full(len(xy), 0.8, np.float32),
            'edge_rmse_px': np.full(len(xy), 0.8, np.float32),
            'marker_offset_px': np.full(len(xy), np.nan, np.float32),
            'marker_area_px': np.zeros(len(xy), np.int16),
            'neighbor_distance_px': np.full(len(xy), 100.0, np.float32),
            'boundary_distance_px': np.full(len(xy), 200.0, np.float32),
            'quality_bits': np.zeros(len(xy), np.uint16),
            'outside_mask': np.zeros(len(xy), np.uint8),
            'manual_edit': np.zeros(len(xy), np.uint8),
            'segment_id': np.asarray([r[1] for r in rows], np.int32),
            'previous_detection': np.full(len(xy), -1, np.int64),
            'link_valid': np.zeros(len(xy), bool),
            'gap_frames': np.zeros(len(xy), np.int16),
            'link_margin_px': np.full(len(xy), np.nan, np.float32),
            'step_px': np.full(len(xy), np.nan, np.float32),
            'dtheta_rad': np.full(len(xy), np.nan, np.float32),
        }
        out.append((fid, rec))
    return out


def test_find_gaps_reads_holes_from_ids():
    frames = _frames([
        (10, [((100, 100), 7)]),
        (11, [((104, 101), 7)]),
        (13, [((112, 103), 7)]),      # hole at 12
        (14, [((116, 104), 7)]),
    ])
    gaps = RF.find_gaps(frames, max_gap=5)
    assert len(gaps) == 1
    g = gaps[0]
    assert g['track'] == 7 and g['f_lo'] == 11 and g['f_hi'] == 13
    assert g['frames'] == [12]
    pred = RF.predict(g, 12)
    assert np.allclose(pred, [108.0, 102.0])


def test_find_gaps_ignores_long_holes_and_unlinked_rows():
    frames = _frames([
        (10, [((100, 100), 7)]),
        (20, [((100, 100), 7)]),      # 10 frames: beyond max_gap
        (21, [((100, 100), -1)]),     # unlinked row
    ])
    assert RF.find_gaps(frames, max_gap=5) == []


def test_fit_local_converges_from_an_offset_seed():
    true = (350.0, 300.0)
    neighbour = (350.0 + 2 * R - 4, 300.0)     # hard-disc contact
    im = _frame([true, neighbour])
    fit = RF.fit_local(im, (true[0] + 7.0, true[1] - 2.0), R)
    assert fit is not None
    d = np.hypot(fit['xy_px'][0] - true[0], fit['xy_px'][1] - true[1])
    assert d < 2.0, d
    assert fit['rays'] >= 20
    assert fit['rmse'] <= 1.2


def test_fit_local_refuses_an_empty_patch():
    im = np.full((400, 400), 45, np.uint8)
    assert RF.fit_local(im, (200.0, 200.0), R) is None


def test_refine_case_end_to_end(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    tracks = {
        0: [(120, 120), (126, 122), (138, 126), (144, 128)],
        1: [(400, 120), (406, 122), (412, 124), (418, 126)],
    }
    for fid in range(4):
        centers = [tracks[0][fid], tracks[1][fid]]
        im = _frame(centers)
        assert cv2.imwrite(str(raw / ('f_%04d.png' % fid)), im)

    case = tmp_path / 'in' / 'sequences' / 'demo'
    case.mkdir(parents=True)
    rows = []
    for fid in range(4):
        r = []
        for tid in (0, 1):
            if fid == 2 and tid == 0:
                continue                      # the hole to recover
            r.append((tracks[tid][fid], tid))
        rows.append((fid, r))
    frames = _frames(rows)
    ids = np.asarray([f for f, _ in frames], np.int64)
    counts = [len(rec['xy_px']) for _, rec in frames]
    arrays = {'frame_ids': ids,
              'frame_offsets': np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)}
    keys = [k for k in frames[0][1]]
    for k in keys:
        arrays[k] = np.concatenate([rec[k] for _, rec in frames])
    arrays['schema_version'] = np.asarray('chunk/v1')
    np.savez_compressed(case / 'chunk_00000_0003.npz', **arrays)
    (case / 'run_config.json').write_text(
        json.dumps({'linker': 'v2', 'link_gap': 5, 'scale_levels': [0, 255]}), encoding='utf8')

    rec, out = RF.refine_case(case, tmp_path / 'out', raw, max_gap=5)
    assert rec['n_gaps'] == 1
    assert rec['n_recovered'] == 1
    assert rec['n_rejected'] == 0

    with np.load(Path(out) / 'chunk_00000_0003.npz') as z:
        assert int(z['recovered'].sum()) == 1
        offsets = z['frame_offsets']
        fid2 = int(np.where(z['frame_ids'] == 2)[0][0])
        lo, hi = int(offsets[fid2]), int(offsets[fid2 + 1])
        assert hi - lo == 2
        rec_xy = z['xy_px'][lo:hi][z['recovered'][lo:hi] == 1][0]
    assert np.hypot(rec_xy[0] - 138.0, rec_xy[1] - 126.0) < 3.0
    assert json.loads((Path(out) / 'refine.json').read_text(encoding='utf8'))[
        'n_recovered'] == 1
