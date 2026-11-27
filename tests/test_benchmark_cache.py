"""Benchmark tools use the cache generation named by extraction metadata."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from test_pipeline_synthetic import _make_case
from tools import benchmark_prior_dedupe, benchmark_sequential
from vibrotrack import cli


TOOLS = Path(__file__).resolve().parents[1] / 'tools'
LEGACY_CACHE = 'dets_v2_p2_o4'


def write_seed(case, directory, x, digest=None):
    folder = case / directory
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        'frame_ids': np.array([0, 1]),
        'frame_offsets': np.array([0, 1, 2]),
        'xy_px': np.array([[x, 10.0], [x, 20.0]]),
    }
    if digest is not None:
        payload['_cache_digest'] = np.array(digest)
    np.savez(folder / 'det_00000_00001.npz', **payload)


@pytest.mark.parametrize('tool', [benchmark_prior_dedupe, benchmark_sequential])
def test_seed_loaders_select_current_or_legacy_cache(tmp_path, tool):
    digest = 'a' * 64
    old_digest = 'b' * 64
    current = LEGACY_CACHE + '_c' + digest[:24]
    stale = LEGACY_CACHE + '_c' + old_digest[:24]
    write_seed(tmp_path, LEGACY_CACHE, 10)
    write_seed(tmp_path, stale, 20, digest=old_digest)
    write_seed(tmp_path, current, 30, digest=digest)
    cfg = tmp_path / 'run_config.json'

    def loaded_x():
        seeds, _ = tool.load_seeds(tmp_path, [(2, 2)], 2, 4)
        if tool is benchmark_sequential:
            return [float(frame[0, 0]) for frame in seeds[2]]
        return [float(seeds[fid][0, 0]) for fid in (0, 1)]

    cfg.write_text('{}', encoding='utf8')
    assert loaded_x() == [10, 10]
    cfg.write_text(json.dumps({'detection_cache_digest': digest}), encoding='utf8')
    assert loaded_x() == [30, 30]

    write_seed(tmp_path, current, 40, digest=old_digest)
    with pytest.raises(ValueError, match='identity mismatch'):
        loaded_x()
    cfg.write_text(json.dumps({'detection_cache_digest': 'c' * 64}), encoding='utf8')
    with pytest.raises(FileNotFoundError, match='recorded detection cache'):
        loaded_x()


@pytest.mark.parametrize('prior_mode', ['warmup', 'carry', 'off'])
def test_benchmark_cli_reads_compatible_shards(tmp_path, prior_mode):
    frames, config = _make_case(tmp_path)
    out = tmp_path / 'batch'
    extract_args = [
        'extract', '--raw', str(frames), '--out', str(out), '--config', str(config),
        '--base-dir', str(tmp_path), '--case-id', 'synthetic', '--workers', '1',
    ]
    if prior_mode == 'off':
        apparatus = yaml.safe_load(config.read_text(encoding='utf8'))
        apparatus['run']['prior'] = False
        config.write_text(yaml.safe_dump(apparatus), encoding='utf8')
    if prior_mode == 'carry':
        extract_args += ['--prior-mode', 'carry']
    assert cli.main(extract_args) == 0
    case = out / 'sequences' / 'synthetic'
    run_config = json.loads((case / 'run_config.json').read_text(encoding='utf8'))
    assert run_config['prior_mode'] == prior_mode
    suffix = '_c' + run_config['detection_cache_digest'][:24]
    assert list(case.glob(f'dets*{suffix}/det_*.npz'))
    if prior_mode == 'off':
        # Exercise an effective mode even when no legacy boolean is supplied.
        run_config.pop('prior', None)
        (case / 'run_config.json').write_text(json.dumps(run_config), encoding='utf8')
    mask = Path(run_config['mask'])
    for script, extra, status, source_field in (
        ('benchmark_prior_dedupe', [], 'ok', 'seed_sources'),
        ('benchmark_sequential', ['--repeats', '1'], 'passed', 'stored_prior_sources'),
    ):
        report = tmp_path / (script + '.json')
        result = subprocess.run(
            [sys.executable, '-B', str(TOOLS / (script + '.py')),
             '--case', str(case), '--raw', str(frames), '--mask', str(mask),
             '--windows', '2:2', '--out', str(report), *extra],
            capture_output=True, text=True, encoding='utf8', errors='replace', timeout=60,
        )
        data = json.loads(report.read_text(encoding='utf8')) if report.exists() else {}
        if prior_mode == 'off':
            assert result.returncode != 0
            error = data.get('error', {}).get('message', result.stderr)
            assert 'rolling-prior regression requires prior enabled' in error
            continue
        assert result.returncode == 0, (result.stdout, result.stderr, data.get('error'))
        assert data['status'] == status, data.get('error')
        assert data[source_field]
        assert all(Path(source['path']).parent.name.endswith(suffix) for source in data[source_field])
        if script == 'benchmark_sequential':
            assert data['summary']['all_pairs_equal']
            assert data['summary']['detections_per_frame']['reference']['min'] > 0
