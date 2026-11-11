"""Tests for the optional review/curation layer (no detector calls)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from vibrotrack import cli
from vibrotrack import review


def _make_case(tmp_path):
    case = tmp_path / 'case'
    case.mkdir()
    # Frame 0 has two detections, frame 1 has one: frame 1 is the anomaly.
    fields = {
        'xy_px': np.array([[10, 10], [30, 10], [10, 30]], np.float32),
        'radius_px': np.array([5, 5, 5], np.float32),
        'theta_rad': np.array([0.0, np.nan, 0.5], np.float32),
        'center_valid': np.array([True, True, True]),
        'angle_valid': np.array([True, False, True]),
        'confidence': np.array([0.9, 0.8, 0.7], np.float32),
        'segment_id': np.array([0, 1, 0], np.int32),
        'frame_ids': np.array([0, 1], np.int64),
        'frame_offsets': np.array([0, 2, 3], np.int32),
        'schema_version': np.array('chunk/v1'),
        'wall_model': np.array('off'),
    }
    np.savez(case / 'chunk_00000_00001.npz', **fields)
    (case / 'dataset.json').write_text(json.dumps({
        'schema_version': 'dataset/v1',
        'image': {'width': 60, 'height': 60, 'dtype': 'uint8'},
        'outputs': [],
        'qc': {'status': 'extracted'},
    }, indent=2), encoding='utf8')
    return case


def test_queue_flags_count_mismatch_and_is_json_serialisable(tmp_path):
    case = _make_case(tmp_path)
    queue, selected = review.build_queue(case)
    assert queue['schema'] == 'review-queue/v1'
    assert queue['expected_count'] == 2
    assert [f['frame'] for f in queue['frames']] == [1]
    assert queue['frames'][0]['delta'] == -1
    assert len(selected[1]['xy_px']) == 1
    json.dumps(queue)


def test_curate_add_remove_is_separate_and_marks_manual(tmp_path):
    case = _make_case(tmp_path)
    actions = tmp_path / 'actions.json'
    actions.write_text(json.dumps({
        'schema': 'review-actions/v1',
        'case': 'case',
        'actions': [
            {'frame': 0, 'op': 'remove', 'xy_px': [10, 10]},
            {'frame': 1, 'op': 'add', 'xy_px': [25, 25], 'radius_px': 5},
        ],
    }), encoding='utf8')
    out = tmp_path / 'curated'
    original = (case / 'chunk_00000_00001.npz').read_bytes()
    rec = review.apply_actions(case, actions, out, tolerance=2.0)
    assert rec['n_applied'] == 2
    assert rec['n_skipped'] == 0
    assert (case / 'chunk_00000_00001.npz').read_bytes() == original
    with np.load(out / 'sequences' / 'case' / 'chunk_00000_00001.npz',
                 allow_pickle=False) as z:
        counts = np.diff(z['frame_offsets']).tolist()
        assert counts == [1, 2]
        assert int(z['manual_edit'].sum()) == 1
        manual = np.where(z['manual_edit'] == 1)[0][0]
        assert z['segment_id'][manual] == -1
        assert bool(z['angle_valid'][manual]) is False
        assert z['xy_px'][manual].tolist() == [25.0, 25.0]
    assert (out / 'review.json').exists()
    ds = json.loads((out / 'sequences' / 'case' / 'dataset.json').read_text(encoding='utf8'))
    assert ds['qc']['status'] == 'curated'


def test_review_cli_writes_queue_and_frame(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    Image.fromarray(np.full((60, 60), 20, np.uint8)).save(raw / 'frame_0000.tiff')
    Image.fromarray(np.full((60, 60), 20, np.uint8)).save(raw / 'frame_0001.tiff')
    cfg = tmp_path / 'apparatus.yaml'
    cfg.write_text(yaml.safe_dump({
        'apparatus': 'test', 'fps': 10.0, 'px_per_mm': 1.0,
        'particle': {'diameter_mm': 1.0},
        'image': {'width': 60, 'height': 60, 'dtype': 'uint8'},
        'scale_levels': [0, 255],
    }, sort_keys=False), encoding='utf8')
    case = _make_case(tmp_path)
    out = tmp_path / 'review'
    assert cli.main(['review', '--raw', str(raw), '--case-dir', str(case),
                     '--config', str(cfg), '--out', str(out)]) == 0
    q = json.loads((out / 'review_queue.json').read_text(encoding='utf8'))
    assert q['frames'][0]['frame'] == 1
    assert (out / q['frames'][0]['image']).exists()


def test_curate_cli(tmp_path):
    case = _make_case(tmp_path)
    actions = tmp_path / 'actions.json'
    actions.write_text(json.dumps({
        'schema': 'review-actions/v1',
        'actions': [{'frame': 1, 'op': 'add', 'xy_px': [20, 20]}],
    }), encoding='utf8')
    out = tmp_path / 'curated'
    assert cli.main(['curate', '--case-dir', str(case), '--actions', str(actions),
                     '--out', str(out), '--tolerance', '2']) == 0
    assert (out / 'review.json').exists()
