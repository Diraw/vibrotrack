"""Tests for re-linking curated detection rows without a detector."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vibrotrack import cli, relink


def _chunk(path, frame_ids, rows):
    n = len(rows)
    fields = {
        'xy_px': np.asarray([r['xy'] for r in rows], np.float32),
        'radius_px': np.full(n, 5.0, np.float32),
        'theta_rad': np.asarray([r['theta'] for r in rows], np.float32),
        'center_valid': np.ones(n, bool),
        'angle_valid': np.asarray([r['angle_valid'] for r in rows], bool),
        'confidence': np.full(n, 0.8, np.float32),
        'quality_bits': np.zeros(n, np.uint16),
        'outside_mask': np.zeros(n, np.uint8),
        'segment_id': np.asarray([r['segment_id'] for r in rows], np.int32),
        'previous_detection': np.full(n, -1, np.int64),
        'link_valid': np.zeros(n, bool),
        'gap_frames': np.zeros(n, np.int16),
        'link_margin_px': np.full(n, np.nan, np.float32),
        'step_px': np.full(n, np.nan, np.float32),
        'dtheta_rad': np.full(n, np.nan, np.float32),
        'manual_edit': np.asarray([r['manual_edit'] for r in rows], np.uint8),
        'frame_ids': np.asarray(frame_ids, np.int64),
        'frame_offsets': np.asarray(range(len(frame_ids) + 1), np.int32),
        'schema_version': np.asarray('chunk/v1'),
    }
    np.savez(path, **fields)


def _make_case(tmp_path):
    case = tmp_path / 'curated' / 'sequences' / 'case'
    case.mkdir(parents=True)
    # The curated manual rows all start with segment_id=-1. They are split
    # across two chunk files so the relinker must carry state across files.
    rows = [dict(xy=[10, 10], theta=0.2, angle_valid=True,
                 segment_id=-1, manual_edit=1),
            dict(xy=[13, 10], theta=0.2, angle_valid=True,
                 segment_id=-1, manual_edit=1)]
    rows2 = [dict(xy=[16, 10], theta=0.2, angle_valid=True,
                  segment_id=-1, manual_edit=1)]
    _chunk(case / 'chunk_00000_00001.npz', [0, 1], rows)
    _chunk(case / 'chunk_00002_00002.npz', [2], rows2)
    (case / 'dataset.json').write_text(json.dumps({
        'case': 'case',
        'image': {'width': 60, 'height': 60, 'dtype': 'uint8'},
        'qc': {'status': 'curated'},
        'curation': {'schema': 'curation/v1'},
        'outputs': [],
    }), encoding='utf8')
    (case / 'run_config.json').write_text(json.dumps({
        'case': 'case', 'linker': 'v2', 'link_gap': 5,
    }), encoding='utf8')
    (case / 'review.json').write_text('{"schema": "curation/v1"}\n',
                                      encoding='utf8')
    (case / 'full_summary.json').write_text(json.dumps({
        'case': 'case', 'segments': 3, 'link_fraction': 0.0,
    }), encoding='utf8')
    return case


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_relink_manual_rows_across_chunks_and_preserves_fields(tmp_path):
    case = _make_case(tmp_path)
    rec, out = relink.relink_case(case, tmp_path / 'relinked')

    assert rec['schema'] == 'relink/v1'
    assert rec['linker'] == 'v2'
    assert rec['n_frames'] == 3
    assert rec['n_detections'] == 3
    assert rec['n_tracks'] == 1
    assert rec['n_manual_rows'] == 3
    assert rec['n_linked_rows'] == 2
    assert rec['n_gap_rows'] == 0
    assert len(rec['manual_rows']) == 3
    assert rec['manual_rows'][0]['segment_id'] == 0
    assert rec['manual_rows'][0]['link_valid'] is False
    assert rec['manual_rows'][1]['link_valid'] is True

    z1 = _load(out / 'chunk_00000_00001.npz')
    z2 = _load(out / 'chunk_00002_00002.npz')
    assert z1['segment_id'].tolist() == [0, 0]
    assert z2['segment_id'].tolist() == [0]
    assert z1['manual_edit'].tolist() == [1, 1]
    assert z2['manual_edit'].tolist() == [1]
    assert z1['link_valid'].tolist() == [False, True]
    assert z2['link_valid'].tolist() == [True]
    assert np.allclose(z1['theta_rad'], 0.2)
    assert np.allclose(z2['theta_rad'], 0.2)
    assert np.allclose(np.concatenate([z1['xy_px'], z2['xy_px']]),
                       [[10, 10], [13, 10], [16, 10]])

    ds = json.loads((out / 'dataset.json')
                    .read_text(encoding='utf8'))
    assert ds['qc']['status'] == 'curated+relinked'
    assert ds['relink']['n_tracks'] == 1
    assert (out / 'review.json').exists()
    assert (out / 'relink.json').exists()


def test_relink_v1_and_cli(tmp_path):
    case = _make_case(tmp_path)
    rec, _ = relink.relink_case(case, tmp_path / 'v1', linker='v1')
    assert rec['linker'] == 'v1'
    assert rec['n_tracks'] == 1

    out = tmp_path / 'cli'
    assert cli.main(['relink', '--case-dir', str(case),
                     '--out', str(out), '--link-gap', '2']) == 0
    assert (out / 'sequences' / 'case' / 'relink.json').exists()
