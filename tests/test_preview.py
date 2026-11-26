"""Preview and extraction share the detector and CLI reports machine events."""
import json
from io import BytesIO

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from test_pipeline_synthetic import _make_case
from vibrotrack import cli, preview


@pytest.mark.parametrize(('prior_mode', 'preview_count'), [('warmup', 5), ('carry', 4)])
def test_preview_matches_extractor_prefix(tmp_path, monkeypatch, prior_mode, preview_count):
    frames, config = _make_case(tmp_path)
    for path, frame_number in zip(sorted(frames.glob('*.tiff')), (2, 3, 8, 10, 12, 13)):
        path.rename(frames / f'sample_{frame_number}.tiff')
    first = frames / 'sample_2.tiff'
    with Image.open(BytesIO(first.read_bytes())) as image:
        pixels = np.array(image)
    cv2.circle(pixels, (650, 400), 37, 220 * 257, cv2.FILLED)
    Image.fromarray(pixels).save(first)
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    cfg['run']['prior_mode'] = prior_mode
    cfg['run']['strict_io'] = True
    cfg['boundary']['mask'] = 'unused.png'
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    mask = tmp_path / 'masks' / 'interior_mask.png'
    opts = ['--raw', str(frames), '--config', str(config), '--mask', str(mask)]

    preview_out = tmp_path / 'preview'
    if prior_mode == 'warmup':
        preview_out.mkdir()
    seen = []
    render_frame = preview._render_frame

    def capture(image, detection, contours, out):
        seen.append(detection)
        return render_frame(image, detection, contours, out)

    monkeypatch.setattr(preview, '_render_frame', capture)
    selected = [] if preview_count == 5 else ['--nframes', str(preview_count)]
    assert cli.main(['preview', '--out', str(preview_out), *opts, *selected]) == 0
    summary = json.loads((preview_out / 'preview_summary.json').read_text(encoding='utf-8'))
    assert summary['schema'] == 'preview/v1'
    assert summary['n_frames'] == preview_count
    assert summary['mask'] == str(mask)
    assert summary['prior_mode'] == prior_mode
    assert summary['frames'][0]['angle_invalid'] > 0
    assert [f['source_name'] for f in summary['frames']] == [
        f'sample_{number}.tiff' for number in (2, 3, 8, 10, 12)[:preview_count]]
    assert set(p.name for p in preview_out.iterdir()) == {
        'preview_summary.json', *(f['overlay'] for f in summary['frames'])}
    with Image.open(preview_out / summary['frames'][0]['overlay']) as overlay:
        assert overlay.size == (800, 800)
        assert overlay.getpixel((400, 80)) == (255, 84, 67)

    batch = tmp_path / 'batch'
    assert cli.main(['extract', '--out', str(batch), *opts]) == 0
    chunk, = (batch / 'sequences' / frames.name).glob('chunk_*.npz')
    with np.load(chunk, allow_pickle=False) as product:
        assert len(seen) == preview_count
        for i, detection in enumerate(seen):
            lo, hi = product['frame_offsets'][i:i + 2]
            for field, values in detection.items():
                np.testing.assert_array_equal(values, product[field][lo:hi])
            item = summary['frames'][i]
            assert item['detections'] == hi - lo
            assert item['angle_invalid'] == np.count_nonzero(~detection['angle_valid'])


def test_preview_refuses_to_replace_existing_results(tmp_path):
    out = tmp_path / 'preview'
    out.mkdir()
    marker = out / 'dataset.json'
    marker.write_text('existing data', encoding='utf-8')
    with pytest.raises(FileExistsError, match='new or empty'):
        preview.run_preview(tmp_path / 'raw', out, None)
    assert list(out.iterdir()) == [marker]
    assert marker.read_text(encoding='utf-8') == 'existing data'


def test_extract_jsonl_reports_progress_result_and_keeps_diagnostics_off_stdout(tmp_path, capsys):
    frames, config = _make_case(tmp_path)
    (frames / 'frame_9999.tiff').touch()
    out = tmp_path / 'batch'
    assert cli.main(['extract', '--jsonl', '--nframes', '2', '--raw', str(frames),
                     '--config', str(config), '--base-dir', str(tmp_path), '--out', str(out)]) == 0
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert {e['schema'] for e in events} == {'extract-event/v1'}
    assert events[0]['event'] == 'start'
    assert {'inputs', 'detect', 'link', 'publish'} <= {
        e['stage'] for e in events if e['event'] == 'progress'}
    assert events[-1]['event'] == 'result'
    assert events[-1]['qc_status'] == 'extracted'
    assert events[-1]['n_frames'] == 2
    assert 'dropping 1 trailing 0-byte frame' in captured.err


def test_extract_jsonl_reports_runtime_error(tmp_path, capsys):
    _, config = _make_case(tmp_path)
    assert cli.main(['extract', '--jsonl', '--raw', str(tmp_path / 'absent'),
                     '--config', str(config), '--out', str(tmp_path / 'batch')]) == 1
    captured = capsys.readouterr()
    start, error = [json.loads(line) for line in captured.out.splitlines()]
    assert start['event'] == 'start'
    assert (error['event'], error['status'], error['error_type']) == (
        'error', 'failed', 'FileNotFoundError')
