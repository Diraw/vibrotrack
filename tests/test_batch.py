"""Batch gates: geometry.json is a product, not an optional extra.

The 2026-11-19 incident: index.json recorded ``geometry_sha256 = null`` for a
batch whose geometry had been dropped by curate, so accept and verify had
nothing to compare and the batch froze without a wall radius or a phi.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vibrotrack import batch, case_output, schema


def _write_case(batch_dir, name='syn', counts=(2, 1)):
    case = Path(batch_dir) / 'sequences' / name
    case.mkdir(parents=True)
    n = sum(counts)
    fields = {
        'xy_px': np.asarray([[10.0 + 20 * i, 10.0] for i in range(n)], np.float32),
        'radius_px': np.full(n, 5.0, np.float32),
        'theta_rad': np.full(n, 0.3, np.float32),
        'center_valid': np.ones(n, bool),
        'angle_valid': np.ones(n, bool),
        'confidence': np.full(n, 0.8, np.float32),
        'edge_rmse_px': np.full(n, 0.5, np.float32),
        'marker_offset_px': np.full(n, 11.0, np.float32),
        'marker_area_px': np.full(n, 210.0, np.float32),
        'neighbor_distance_px': np.full(n, 40.0, np.float32),
        'boundary_distance_px': np.full(n, 100.0, np.float32),
        'quality_bits': np.zeros(n, np.uint16),
        'segment_id': np.arange(n, dtype=np.int32),
        'previous_detection': np.full(n, -1, np.int64),
        'link_valid': np.zeros(n, bool),
        'step_px': np.full(n, np.nan, np.float32),
        'dtheta_rad': np.full(n, np.nan, np.float32),
        'frame_ids': np.arange(len(counts), dtype=np.int64),
        'frame_offsets': np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
        'schema_version': np.asarray('chunk/v1'),
    }
    np.savez(case / 'chunk_00000_00001.npz', **fields)
    (case / 'dataset.json').write_text(json.dumps({
        'schema_version': 'dataset/v1',
        'case': name,
        'n_frames': len(counts),
        'image': {'width': 60, 'height': 60, 'dtype': 'uint8'},
        'qc': {'status': 'extracted'},
        'outputs': [],
    }), encoding='utf8')
    (case / 'run_config.json').write_text(json.dumps({'case': name}), encoding='utf8')
    return case


def _write_geometry(case, wall=500.0):
    (case / 'geometry.json').write_text(json.dumps({
        'schema': 'geometry/v1',
        'case_id': case.name,
        'wall_radius_px': wall,
        'n_particles': 2,
        'phi': 0.02,
        'n_detected_median': 2.0,
    }), encoding='utf8')


def test_index_refuses_a_case_without_geometry(tmp_path):
    batch_dir = tmp_path / 'batch'
    _write_case(batch_dir)
    try:
        batch.build_index(batch_dir)
    except FileNotFoundError as exc:
        assert 'geometry.json' in str(exc)
    else:
        raise AssertionError('index accepted a case without geometry.json')
    assert not (batch_dir / 'index.json').exists()


def test_allow_missing_geometry_indexes_but_accept_refuses(tmp_path):
    batch_dir = tmp_path / 'batch'
    _write_case(batch_dir)
    index = batch.build_index(batch_dir, allow_missing_geometry=True)
    rec = index['sequences'][0]
    assert rec['geometry'] is None
    assert rec['geometry_warning']

    report = batch.accept(batch_dir)
    assert report['accepted'] is False
    problems = [p for r in report['sequences'] for p in r['problems']]
    assert any('geometry' in p for p in problems)


def test_geometry_hash_is_part_of_the_gate(tmp_path):
    batch_dir = tmp_path / 'batch'
    case = _write_case(batch_dir)
    _write_geometry(case)
    batch.build_index(batch_dir)
    assert batch.accept(batch_dir)['accepted'] is True
    report, bad = batch.verify(batch_dir)
    assert report['accepted'] is True and bad == []

    _write_geometry(case, wall=501.0)          # file changed after indexing
    report, _bad = batch.verify(batch_dir)
    assert report['accepted'] is False
    problems = [p for r in report['sequences'] for p in r['problems']]
    assert any('geometry sha256' in p for p in problems)


def test_archive_holds_publication_locks_during_copy(tmp_path, monkeypatch):
    batch_dir = tmp_path / 'batch'
    case = _write_case(batch_dir)
    _write_geometry(case)
    chunk = next(case.glob('chunk_*.npz'))
    manifest = schema.read_json(case / 'dataset.json')
    manifest['outputs'] = schema.outputs_record(case, [chunk.name])
    schema.write_json(case / 'dataset.json', manifest)
    batch.build_index(batch_dir)
    assert batch.accept(batch_dir)['accepted']
    copytree = batch.shutil.copytree

    def interleave_copy(src, dest, *args, **kwargs):
        if Path(src) == batch_dir:
            with pytest.raises(RuntimeError, match='another extraction or archive'):
                with case_output.case_lock(case):
                    pass
            _write_case(batch_dir, name='late')
        return copytree(src, dest, *args, **kwargs)

    monkeypatch.setattr(batch.shutil, 'copytree', interleave_copy)
    dest, frozen = batch.archive(batch_dir, tmp_path / 'archives')

    archived = dest / 'sequences' / case.name
    assert schema.read_json(archived / 'dataset.json')['outputs'][0]['sha256'] == (
        schema.sha256(archived / chunk.name))
    assert not (dest / 'sequences' / 'late').exists()
    assert not any(case_output.LOCK_NAME in record['path'] for record in frozen['files'])


@pytest.mark.parametrize('interrupted', [False, True])
def test_archive_force_preserves_old_copy_when_source_case_is_busy(tmp_path, interrupted):
    batch_dir = tmp_path / 'batch'
    case = _write_case(batch_dir)
    archive_root = tmp_path / 'archives'
    previous = archive_root / batch_dir.name / 'FROZEN.json'
    schema.write_json(previous, {'generation': 'previous'})

    if interrupted:
        schema.write_json(case / case_output.STAGE_NAME / 'prepared.json', {})
        with pytest.raises(RuntimeError, match='publication interrupted'):
            batch.archive(batch_dir, archive_root, force=True)
    else:
        with case_output.case_lock(case):
            with pytest.raises(RuntimeError, match='another extraction or archive'):
                batch.archive(batch_dir, archive_root, force=True)
    assert schema.read_json(previous) == {'generation': 'previous'}

def test_index_locks_case_from_manifest_read_through_sidecar_hash(tmp_path, monkeypatch):
    batch_dir = tmp_path / 'batch'
    case = _write_case(batch_dir)
    _write_geometry(case)
    manifest_path = case / 'dataset.json'
    config_path = case / 'run_config.json'
    old = schema.read_json(manifest_path)
    schema.write_json(manifest_path, dict(old, dataset_id='before'))
    read_json, sha256 = schema.read_json, schema.sha256
    attempted_publications = []

    def try_publish(stage):
        attempted_publications.append(stage)
        with pytest.raises(RuntimeError, match='another extraction or archive'):
            with case_output.case_lock(case):
                with case_output.stage_case(case) as incoming:
                    schema.write_json(incoming / 'dataset.json', dict(old, dataset_id='after'))

    def pause_after_manifest(path):
        result = read_json(path)
        if path == manifest_path:
            try_publish('manifest')
        return result

    def pause_at_last_hash(path):
        if path == config_path:
            try_publish('config hash')
        return sha256(path)

    monkeypatch.setattr(schema, 'read_json', pause_after_manifest)
    monkeypatch.setattr(schema, 'sha256', pause_at_last_hash)
    record, = batch.build_index(batch_dir)['sequences']
    assert attempted_publications == ['manifest', 'config hash']
    assert record['dataset_id'] == 'before'
    assert record['dataset_sha256'] == sha256(manifest_path)
    assert record['npz_sha256'] == sha256(case / record['npz_file'])


@pytest.mark.parametrize('force', [False, True])
def test_archive_rejects_late_reindex_and_accept_without_replacing_destination(tmp_path, monkeypatch, force):
    batch_dir = tmp_path / 'batch'
    case = _write_case(batch_dir)
    _write_geometry(case)
    batch.build_index(batch_dir)
    assert batch.accept(batch_dir)['accepted']

    # Prepare the metadata a concurrent index/accept had already generated.
    newer = tmp_path / 'newer'
    for name in (case.name, 'late'):
        _write_geometry(_write_case(newer, name=name))
    new_index = batch.build_index(newer)
    new_acceptance = batch.accept(newer)
    assert new_acceptance['accepted']
    archive_root = tmp_path / 'archives'
    previous = archive_root / batch_dir.name / 'FROZEN.json'
    if force:
        schema.write_json(previous, {'generation': 'previous'})
    copytree = batch.shutil.copytree

    def interleave_copy(src, dest, *args, **kwargs):
        if Path(src) == batch_dir:
            _write_geometry(_write_case(batch_dir, name='late'))
            schema.write_json(batch_dir / 'index.json', new_index)
            schema.write_json(batch_dir / 'acceptance.json', new_acceptance)
        return copytree(src, dest, *args, **kwargs)

    monkeypatch.setattr(batch.shutil, 'copytree', interleave_copy)
    with pytest.raises(ValueError, match='archive snapshot case set disagrees'):
        batch.archive(batch_dir, archive_root, force=force)
    if force:
        assert schema.read_json(previous) == {'generation': 'previous'}
    else:
        assert not previous.parent.exists()
    assert not list(archive_root.glob('.vibrotrack-archive-*'))


@pytest.mark.parametrize('historical_mix', [False, True])
def test_archive_rejects_stale_or_mixed_index(tmp_path, historical_mix):
    batch_dir = tmp_path / 'batch'
    case = _write_case(batch_dir)
    _write_geometry(case)
    manifest = case / 'dataset.json'
    schema.write_json(manifest, dict(schema.read_json(manifest), dataset_id='old'))
    index = batch.build_index(batch_dir)
    assert batch.accept(batch_dir)['accepted']
    diagnostic = 'hash disagrees'
    if historical_mix:
        schema.write_json(manifest, dict(schema.read_json(manifest), dataset_id='new'))
        index['sequences'][0]['dataset_sha256'] = schema.sha256(manifest)
        schema.write_json(batch_dir / 'index.json', index)
        # Older acceptance reports did not record the index hash.
        report = schema.read_json(batch_dir / 'acceptance.json')
        del report['index_sha256']
        schema.write_json(batch_dir / 'acceptance.json', report)
        diagnostic = 'dataset ID disagrees'
    else:
        # Keep the ID stable so only the stale hash catches the old index.
        schema.write_json(manifest, dict(schema.read_json(manifest), changed=True))

    with pytest.raises(ValueError, match=diagnostic):
        batch.archive(batch_dir, tmp_path / 'archives', force=True)


def test_archive_force_allows_missing_index_and_acceptance(tmp_path):
    batch_dir = tmp_path / 'batch'
    case = _write_case(batch_dir)
    dest, frozen = batch.archive(batch_dir, tmp_path / 'archives', force=True)

    assert (dest / 'sequences' / case.name / 'dataset.json').exists()
    assert not (dest / 'index.json').exists()
    assert not (dest / 'acceptance.json').exists()
    assert schema.read_json(dest / 'FROZEN.json') == frozen

def test_archive_requires_acceptance_for_exact_index_bytes(tmp_path):
    batch_dir = tmp_path / 'batch'
    _write_geometry(_write_case(batch_dir))
    index = batch.build_index(batch_dir)
    report = batch.accept(batch_dir, min_angle=0.5)
    assert report['accepted']
    assert report['index_sha256'] == schema.sha256(batch_dir / 'index.json')
    index['created_at'] = '2026-11-19T00:00:00+00:00'
    schema.write_json(batch_dir / 'index.json', index)
    archive_root = tmp_path / 'archives'
    with pytest.raises(ValueError, match='index hash disagrees with acceptance.json'):
        batch.archive(batch_dir, archive_root)

    # Legacy reports remain valid when their case set and all product hashes agree.
    del report['index_sha256']
    schema.write_json(batch_dir / 'acceptance.json', report)
    dest, _ = batch.archive(batch_dir, archive_root)
    assert schema.read_json(dest / 'acceptance.json') == report


def test_archive_refuses_destination_that_contains_the_source(tmp_path):
    batch_dir = tmp_path / 'batch' / 'batch'
    case = _write_case(batch_dir)
    with pytest.raises(ValueError, match='outside the source'):
        batch.archive(batch_dir, tmp_path, force=True)
    assert (case / 'dataset.json').is_file()
