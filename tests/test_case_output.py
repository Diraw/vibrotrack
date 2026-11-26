"""Failed publication preserves the previous dataset, including across a hard exit."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from test_pipeline_synthetic import _make_case
from vibrotrack import batch, case_output, pipeline
from vibrotrack.config import load_config


def _products(case_dir):
    names = ('dataset.json', 'run_config.json', 'full_summary.json', 'progress.json')
    paths = [case_dir / name for name in names] + list(case_dir.glob('chunk_*.npz'))
    return {path.name: path.read_bytes() for path in paths}


def test_extract_recovers_previous_products_on_publish_interrupt(tmp_path, monkeypatch):
    raw, config_path = _make_case(tmp_path)
    config = load_config(config_path, base_dir=tmp_path)
    out = tmp_path / 'batch'
    case_dir = out / 'sequences' / 'syn'
    pipeline.run_case(raw, out, config, case_id='syn', nframes=6)
    old = _products(case_dir)
    replace = Path.replace

    def interrupt_at_manifest(self, target):
        if self.name == 'dataset.json' and self.parent.name == 'new':
            raise KeyboardInterrupt
        return replace(self, target)

    with monkeypatch.context() as context:
        context.setattr(Path, 'replace', interrupt_at_manifest)
        with pytest.raises(KeyboardInterrupt):
            pipeline.run_case(raw, out, config, case_id='syn', nframes=4)
    assert _products(case_dir) == old
    assert not (case_dir / case_output.STAGE_NAME).exists()

    pipeline.run_case(raw, out, config, case_id='syn', nframes=4)
    assert [p.name for p in case_dir.glob('chunk_*.npz')] == ['chunk_00000_00003.npz']
    assert batch.build_index(out, allow_missing_geometry=True)['total_frames'] == 4


def test_extract_reuses_completed_shard_on_restart(tmp_path):
    raw, config_path = _make_case(tmp_path)
    config = load_config(config_path, base_dir=tmp_path)
    out = tmp_path / 'batch'
    pipeline.run_case(raw, out, config, case_id='syn', no_link=True, limit_shards=1)

    events = []
    pipeline.run_case(raw, out, config, case_id='syn', events=events.append)
    detected = {event['first_frame']: event['status'] for event in events
                if event['stage'] == 'detect'}
    assert detected == {0: 'cached', 3: 'ok'}

    config.detector_params['thresh'] = 0.3
    events.clear()
    pipeline.run_case(raw, out, config, case_id='syn', events=events.append)
    assert [event['status'] for event in events if event['stage'] == 'detect'] == ['ok', 'ok']


def test_extract_refuses_to_publish_when_an_earlier_frame_changes(tmp_path):
    raw, config_path = _make_case(tmp_path)
    config = load_config(config_path, base_dir=tmp_path)
    config.shard_overlap = 0
    out = tmp_path / 'batch'
    case_dir = out / 'sequences' / 'syn'
    pipeline.run_case(raw, out, config, case_id='syn')
    old = _products(case_dir)

    def change_used_frame(event):
        if event['stage'] == 'detect' and event['first_frame'] == 0:
            frame = sorted(raw.glob('*.tiff'))[0]
            before = frame.stat()
            os.utime(frame, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))
    with pytest.raises(ValueError, match='source frames changed during extraction'):
        pipeline.run_case(raw, out, config, case_id='syn', events=change_used_frame)
    assert _products(case_dir) == old


def test_case_lock_excludes_other_processes_and_release_after_exit(tmp_path):
    command = [sys.executable, '-c',
               'import sys; from vibrotrack.case_output import case_lock\n'
               'with case_lock(sys.argv[1]): pass', str(tmp_path)]
    with case_output.case_lock(tmp_path):
        with pytest.raises(RuntimeError, match='another extraction'):
            with case_output.case_lock(tmp_path):
                pass
        locked = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert locked.returncode != 0
    assert 'another extraction' in locked.stderr
    subprocess.run(command, check=True, timeout=15)


def test_case_recovers_publication_after_process_exit(tmp_path):
    case_dir = tmp_path / 'sequences' / 'syn'
    case_dir.mkdir(parents=True)
    old_data = {'dataset.json': 'previous manifest', 'chunk_00000_00005.npz': 'old chunk'}
    for name, value in old_data.items():
        (case_dir / name).write_text(value, encoding='utf8')
    worker = """
import os
import sys
from pathlib import Path
from vibrotrack.case_output import case_lock, stage_case

case_dir = Path(sys.argv[1])
replace = Path.replace

def interrupt_at_manifest(self, target):
    if self.name == 'dataset.json' and self.parent.name == 'new':
        os._exit(87)
    return replace(self, target)

Path.replace = interrupt_at_manifest
with case_lock(case_dir), stage_case(case_dir) as new:
    (new / 'dataset.json').write_text('incoming manifest')
    (new / 'chunk_00000_00003.npz').write_text('incoming chunk')
"""
    result = subprocess.run([sys.executable, '-c', worker, str(case_dir)], timeout=15)
    assert result.returncode == 87
    assert (case_dir / 'chunk_00000_00003.npz').exists()
    with pytest.raises(RuntimeError, match='publication interrupted'):
        batch.build_index(tmp_path, allow_missing_geometry=True)
    with case_output.case_lock(case_dir):
        case_output.recover_case(case_dir)
        case_output.recover_case(case_dir)
    assert not (case_dir / 'chunk_00000_00003.npz').exists()
    assert not (case_dir / case_output.STAGE_NAME).exists()
    for name, value in old_data.items():
        assert (case_dir / name).read_text(encoding='utf8') == value
