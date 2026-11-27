"""A detection shard is reusable only for its recorded inputs."""

import os

import numpy as np
import pytest
from PIL import Image

from vibrotrack import shards as D


@pytest.fixture
def inputs(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    files = []
    for i in range(3):
        path = raw / f'frame_{i}.tiff'
        Image.fromarray(np.full((80, 80), 15 + i, np.uint8)).save(path)
        files.append(path)
    mask = tmp_path / 'interior_mask.png'
    Image.fromarray(np.full((80, 80), 255, np.uint8)).save(mask)
    case = {
        'id': 'case',
        'path': str(raw),
        'out_root': str(tmp_path / 'out'),
        'mask_path': str(mask),
        'shape': (80, 80),
        'scale_levels': [0., 255.],
        'detector': 'v2',
        'prior_mode': 'warmup',
        'prior_frames': 2,
        'shard_overlap': 1,
    }
    return case, files, mask


def prepare(case, files, shard_size=2):
    D.prepare_detection_cache(case, files, shard_size=shard_size)
    return case['detection_cache']['digest']


@pytest.mark.parametrize('damage', ['invalid_zip', 'wrong_digest', 'wrong_frames', 'missing_column'])
def test_reuse_then_rebuild_malformed_shard(inputs, damage):
    case, files, _ = inputs
    prepare(case, files)
    task = (case, 0, 2, files[:2], 0)
    assert D.shard_task(task)['status'] == 'ok'
    assert D.shard_task(task)['status'] == 'cached'

    dst = D.det_path(case, 0, 2)
    if damage == 'invalid_zip':
        dst.write_bytes(b'incomplete')
    else:
        with np.load(dst, allow_pickle=False) as archive:
            fields = {name: archive[name] for name in archive.files}
        if damage == 'wrong_digest':
            fields['_cache_digest'] = np.array('stale')
        elif damage == 'wrong_frames':
            fields['frame_ids'] = np.array([1, 2])
        else:
            fields.pop('angle_valid')
        np.savez_compressed(dst, **fields)

    assert D.shard_task(task)['status'] == 'ok'
    assert D.shard_task(task)['status'] == 'cached'
    assert '_cache_digest' not in D.load_det_part(case, 0, 2)


def test_digest_binds_frame_order_detector_settings_and_layout(inputs):
    case, files, _ = inputs
    baseline = prepare(case, files)
    assert prepare(dict(case, linker='v1'), files) == baseline

    changed = [
        prepare(dict(case, detector_params={'thresh': 0.45}), files),
        prepare(dict(case, scale_levels=[1.0, 254.0]), files),
        prepare(dict(case, prior_frames=3), files),
        prepare(dict(case, prior_mode='carry'), files),
        prepare(dict(case, shard_overlap=0), files),
        prepare(dict(case), files, shard_size=3),
        prepare(dict(case), files[::-1]),
        prepare(dict(case), files[:2]),
    ]
    assert len(set(changed + [baseline])) == len(changed) + 1


def test_digest_binds_same_metadata_wall_sidecar_content(inputs):
    case, files, mask = inputs
    boundary = mask.parent / 'boundary.json'
    info = mask.parent / 'mask.json'
    case.update(wall_band=1, wall_model='auto', avoid_r=10)
    initial = prepare(case, files)

    info.write_text('{"frozen_discs":[{"center_px":[1,2]}]}', encoding='utf8')
    empirical = prepare(case, files)
    assert empirical != initial
    before = info.stat()
    info.write_text('{"frozen_discs":[{"center_px":[2,3]}]}', encoding='utf8')
    os.utime(info, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert info.stat().st_size == before.st_size
    assert prepare(case, files) != empirical

    boundary.write_text('{"design_inferred":{"disks_px":[]}}', encoding='utf8')
    with_boundary_sidecar = prepare(case, files)
    assert with_boundary_sidecar != empirical
    wall = mask.parent / D.WALL_FILE
    np.save(wall, np.array([[12., 15., 4.]]))
    geometry = prepare(case, files)
    assert geometry != with_boundary_sidecar
    np.save(wall, np.array([[12., 16., 4.]]))
    assert prepare(case, files) != geometry


def test_detects_source_changes_and_mask_changes(inputs):
    case, files, mask = inputs
    baseline = prepare(case, files)
    task = (case, 2, 3, files[1:], 1)
    assert D.shard_task(task)['status'] == 'ok'

    touched = files[1].stat()
    os.utime(files[1], ns=(touched.st_atime_ns, touched.st_mtime_ns + 5_000_000_000))
    with pytest.raises(ValueError, match='source frames changed'):
        D.shard_task(task)
    with pytest.raises(ValueError, match='source frames changed'):
        D.validate_detection_cache(case, files=files)
    after_source = prepare(case, files)
    assert after_source != baseline
    assert D.shard_task(task)['status'] == 'ok'

    Image.fromarray(np.zeros((80, 80), np.uint8)).save(mask)
    with pytest.raises(ValueError, match='mask or wall inputs changed'):
        D.shard_task(task)
    assert prepare(case, files) != after_source
    assert D.shard_task(task)['status'] == 'ok'


def test_temporal_std_uses_sampled_files_outside_selected_range(inputs):
    case, files, _ = inputs
    prepare(case, files[:1])
    baseline = D.temporal_std(case, case['id'], n=2)
    np.testing.assert_array_equal(D.temporal_std(case, case['id'], n=2), baseline)
    # Even with one selected frame, the empirical wall rule samples across raw.
    before = files[-1].stat()
    Image.fromarray(np.full((80, 80), 50, np.uint8)).save(files[-1])
    os.utime(files[-1], ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))
    different = D.temporal_std(case, case['id'], n=2)
    assert different[0, 0] != baseline[0, 0]
