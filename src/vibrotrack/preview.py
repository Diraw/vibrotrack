"""Run the extractor's detection stage on initial frames for visual review."""
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from . import detect, pipeline, schema, shards, visualize
from .mask import load_mask


def _require_empty(out):
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise FileExistsError('preview output must be a new or empty directory: %s' % out)


def _render_frame(image, detection, contours, out):
    thickness = max(2, round(min(image.shape[:2]) / 800))
    overlay = visualize.draw_detections(
        image, detection['xy_px'], detection['radius_px'], detection['theta_rad'],
        center_valid=detection['center_valid'], angle_valid=detection['angle_valid'],
        thickness=thickness)
    cv2.drawContours(overlay, contours, -1, (255, 84, 67), thickness, cv2.LINE_AA)
    Image.fromarray(overlay).save(out)


def run_preview(raw, out, config, *, nframes=5):
    """Publish overlays and per-frame detection counts for up to ``nframes``."""
    if nframes < 1:
        raise ValueError('preview nframes must be positive')
    out = Path(out).expanduser().absolute()
    if out.resolve() == Path(raw).expanduser().resolve():
        raise ValueError('preview output must differ from raw frame directory')
    _require_empty(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix='.vibrotrack-preview-', dir=out.parent) as temp:
        temp = Path(temp)
        case, files, workers, shard = pipeline.prepare_case(
            raw, temp / 'cache', config, case_id='preview', nframes=nframes)
        selected = files[:case['nframes']]
        shards.prepare_detection_cache(case, selected, shard_size=shard)
        tasks = pipeline.detection_tasks(case, files, shard)
        parts = sorted(pipeline._detect_parts(case, tasks, workers), key=lambda part: part['lo'])
        interior, mask_source = load_mask(case['mask_path'], case['shape'], required=True)
        contours, _ = cv2.findContours(interior, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        stage = temp / 'products'
        stage.mkdir()
        frames = []
        for part in parts:
            data = shards.load_det_part(case, part['lo'], part['hi'])
            offsets = data['frame_offsets']
            for i, frame_id in enumerate(data['frame_ids']):
                source = files[int(frame_id)]
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                detection = {k: data[k][lo:hi] for k in (
                    'xy_px', 'radius_px', 'theta_rad', 'center_valid', 'angle_valid')}
                angles = int(np.count_nonzero(detection['angle_valid']))
                filename = 'overlay_%05d.png' % frame_id
                image = detect.read_image(source, case['scale_levels'], strict=case['strict_io'])
                _render_frame(image, detection, contours, stage / filename)
                frames.append({'frame_id': int(frame_id), 'source_name': source.name,
                               'overlay': filename, 'detections': hi - lo,
                               'center_valid': int(np.count_nonzero(detection['center_valid'])),
                               'angle_valid': angles, 'angle_invalid': hi - lo - angles})
        summary = {'schema': 'preview/v1', 'raw': case['path'], 'mask': mask_source,
                   'detector': case['detector'], 'prior_mode': case['prior_mode'],
                   'n_frames': len(frames), 'frames': frames}
        schema.write_json(stage / 'preview_summary.json', summary)
        shards.validate_detection_cache(case, files=selected)
        _require_empty(out)
        if out.exists():
            out.rmdir()
        stage.rename(out)

    return summary, out / 'preview_summary.json'
