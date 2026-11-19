"""Unit tests for the void audit."""
import json

import numpy as np
import pytest

from vibrotrack import audit


def make_case(tmp_path, rows_per_frame, **fields):
    """rows_per_frame: list of per-frame dicts with the detection arrays."""
    xy, area, off, conf, seg = [], [], [0], [], []
    fid = 0
    for spec in rows_per_frame:
        n = spec['n']
        for k in range(n):
            xy.append(spec.get('xy', [[100.0 * k, 100.0]])[k % len(spec.get('xy', [[100.0, 100.0]]))])
            area.append(spec.get('area', [220])[k % len(spec.get('area', [220]))])
            conf.append(spec.get('conf', [0.5])[k % len(spec.get('conf', [0.5]))])
            seg.append(fid)
        off.append(off[-1] + n)
        fid += 1
    xy = np.asarray(xy, np.float32)
    offs = np.asarray(off, np.int64)
    moff = np.asarray(fields.get('moff', [11.0] * len(xy)), np.float32)
    case = tmp_path / 'case01'
    case.mkdir()
    np.savez_compressed(case / ('chunk_00000_%05d.npz' % (len(rows_per_frame) - 1)),
                        xy_px=xy, frame_offsets=offs,
                        marker_area_px=np.asarray(area, np.int16),
                        marker_offset_px=moff.astype(np.float32),
                        confidence=np.asarray(conf, np.float32),
                        segment_id=np.asarray(seg, np.int32))
    return case


def test_flags_a_void_and_keeps_a_real_particle(tmp_path):
    # frame 0: one real particle; frame 1: one void-like centre
    case = make_case(tmp_path, [{'n': 1, 'xy': [[10.0, 10.0]], 'area': [220], 'conf': [0.5]},
                                {'n': 1, 'xy': [[50.0, 50.0]], 'area': [121], 'conf': [0.30]}],
                     moff=[11.0, 8.4])
    res = audit.scan(case)
    assert len(res['rows']) == 1
    row = res['rows'][0]
    assert row['frame'] == 1
    assert row['marker_area_px'] == 121
    s = audit.summary(res)
    assert s['n_voids'] == 1 and s['n_frames_with_a_void'] == 1


def test_real_particle_with_a_large_slot_is_not_flagged(tmp_path):
    case = make_case(tmp_path, [{'n': 2, 'xy': [[10.0, 10.0], [90.0, 90.0]],
                                 'area': [220, 160], 'conf': [0.6, 0.5]}],
                     moff=[11.0, 12.0])
    assert audit.scan(case)['rows'] == []


def test_min_frame_skips_the_startup(tmp_path):
    case = make_case(tmp_path, [{'n': 1, 'xy': [[5.0, 5.0]], 'area': [90], 'conf': [0.3]},
                                {'n': 1, 'xy': [[5.0, 5.0]], 'area': [90], 'conf': [0.3]}],
                     moff=[8.6, 8.6])
    assert len(audit.scan(case)['rows']) == 2
    assert len(audit.scan(case, min_frame=1)['rows']) == 1


def test_write_produces_curate_ready_actions(tmp_path):
    case = make_case(tmp_path, [{'n': 1, 'xy': [[10.0, 10.0]], 'area': [110], 'conf': [0.3]}],
                     moff=[8.0])
    res = audit.scan(case)
    out = tmp_path / 'audit'
    audit_path, act_path, rec = audit.write(case, out, res)
    assert json.loads(audit_path.read_text(encoding='utf8'))['summary']['n_voids'] == 1
    acts = json.loads(act_path.read_text(encoding='utf8'))['actions']
    assert acts[0]['op'] == 'remove' and acts[0]['frame'] == 0
