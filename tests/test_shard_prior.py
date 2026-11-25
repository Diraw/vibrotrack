"""Causal history, cache isolation and real sharded-vs-continuous extraction."""
from collections import deque
import json

import numpy as np
import pytest

from test_pipeline_synthetic import _make_case
from vibrotrack import cli, shards as D
from vibrotrack.config import load_config


@pytest.mark.parametrize('prefetch', ('0', '1'))
def test_carry_across_short_shards_including_empty_slot(tmp_path, monkeypatch, prefetch):
    monkeypatch.setenv('PREFETCH', prefetch)
    monkeypatch.setattr(D, 'load_mask', lambda *args, **kw: (np.ones((80, 80), np.uint8), None))
    monkeypatch.setattr(D.X, 'read_image', lambda i, *args, **kw: np.array([[i]]))
    seen = {}

    def detector(im, *args, prior_xy, **kwargs):
        f = int(im[0, 0])
        seen[f] = [] if prior_xy is None else prior_xy[:, 0].astype(int).tolist()
        # Output of each frame depends on its input history, and frame 1 is empty.
        n = f + 10 + sum(seen[f])
        return {'xy_px': np.empty((0, 2)) if f == 1 else np.array([[n, 0.0]])}, {}

    monkeypatch.setattr(D.D2, 'frame', detector)
    common = dict(id='unit', detector='v2', prior=True, out_root=str(tmp_path),
                  prior_frames=3, shard_overlap=2, scale_levels=[0, 256])
    old = dict(common, prior_mode='warmup')
    old_result = D.shard_task((old, 2, 4, [0, 1, 2, 3], 0))
    assert old_result['status'] == 'ok'
    case = dict(common, prior_mode='carry')
    hist = deque(maxlen=3)
    for lo, hi in [(0, 2), (2, 4), (4, 5)]:
        result = D.shard_task((case, lo, hi, list(range(lo, hi)), lo), history=hist)
        assert result['status'] == 'ok'
    assert seen == {0: [], 1: [10], 2: [10], 3: [10, 22], 4: [22, 45]}
    assert D.det_path(old, 2, 4) != D.det_path(case, 2, 4)
    assert D.load_det_part(case, 0, 2)['frame_offsets'].tolist() == [0, 1, 1]
    # Repeat with the existing carry cache; replay advances history, never accepts
    # cache hits from an unknown predecessor or stale input.
    seen.clear()
    result = D.shard_task((case, 0, 2, [0, 1], 0), history=deque(maxlen=3))
    assert result['status'] == 'ok' and sorted(seen) == [0, 1]
    with pytest.raises(ValueError, match='history'):
        D.shard_task((case, 2, 4, [2, 3], 2))


def test_cli_carry_matches_continuous_all_columns(tmp_path):
    frames, config = _make_case(tmp_path)
    args = ['extract', '--raw', str(frames), '--config', str(config),
            '--base-dir', str(tmp_path), '--case-id', 'syn']
    paths = {}
    for name, size, mode in (('continuous', '6', 'carry'), ('carry', '2', 'carry'),
                             ('warmup', '2', 'warmup'), ('default', '2', None)):
        extra = ['--prior-mode', mode] if mode else []
        assert cli.main(args + ['--out', str(tmp_path / name), '--shard', size] + extra) == 0
        case = tmp_path / name / 'sequences' / 'syn'
        path, = case.glob('chunk_*.npz')
        ds = json.loads((case / 'dataset.json').read_text(encoding='utf-8'))
        run = json.loads((case / 'run_config.json').read_text(encoding='utf-8'))
        expected = mode or 'warmup'
        assert run['prior_mode'] == ds['tracking']['prior_mode'] == expected
        assert run['effective_shard_overlap'] == (5 if expected == 'warmup' else 0)
        paths[name] = path
    for a, b in (('continuous', 'carry'), ('warmup', 'default')):
        with np.load(paths[a]) as left, np.load(paths[b]) as right:
            assert set(left.files) == set(right.files)
            for k in left.files:
                np.testing.assert_array_equal(left[k], right[k], err_msg=f'{a}/{b}/{k}')


def test_prior_mode_config_default_and_opt_in(tmp_path):
    cfg = tmp_path / 'apparatus.yaml'
    cfg.write_text('run: {}\n', encoding='utf-8')
    assert load_config(cfg).prior_mode == 'warmup'
    cfg.write_text('run: {prior_mode: carry}\n', encoding='utf-8')
    assert load_config(cfg).prior_mode == 'carry'


def test_prior_disabled_uses_cache_separate_from_prior():
    from vibrotrack.pipeline import _prior_mode
    assert _prior_mode('v1', True, 'carry') == 'off'
    assert _prior_mode('v2', False, 'carry') == 'off'
    assert D.det_tag({'detector': 'v2', 'prior_mode': 'off'}) != D.det_tag({'prior_mode': 'warmup'})
