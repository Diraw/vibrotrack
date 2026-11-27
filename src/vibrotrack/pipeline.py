"""Extract one case: detect shards, link in frame order, write products."""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import detect as X
from . import schema
from . import shards as D
from .case_output import case_lock, recover_case, stage_case
from .config import ApparatusConfig
from .mask import load_mask


def _dataset_id(case_id):
    from datetime import datetime
    return 'ds-%s-%s' % (datetime.now().strftime('%Y%m%d'), schema.slugify(case_id))


def _mask_sha(mask_path):
    if not mask_path:
        return None
    p = Path(mask_path)
    return schema.sha256(p) if p.exists() else None


def _write_dataset(case, config, output_dir, case_dir, dataset_id, project, campaign,
                   acquired_at, operator, *, detected_only=False):
    product_dir = case_dir if detected_only else output_dir
    chunk_files = sorted(product_dir.glob('chunk_*.npz'))
    names = [p.name for p in chunk_files] + ['full_summary.json', 'run_config.json']
    rec = {
        'schema_version': schema.SCHEMA_DATASET,
        'dataset_id': dataset_id or _dataset_id(case['id']),
        'project': project or '',
        'campaign': campaign or '',
        'batch': Path(case_dir).parent.parent.name,
        'case': case['id'],
        'apparatus': config.apparatus,
        'acquired_at': acquired_at,
        'extracted_at': schema.now_iso(),
        'operator': operator,
        'fps': config.fps,
        'n_frames': int(case['nframes']),
        'image': {
            'width': int(case['shape'][1]),
            'height': int(case['shape'][0]),
            'dtype': config.image_dtype or 'unknown',
        },
        'particle': {
            'diameter_mm': config.diameter_mm,
            'nominal_n': config.nominal_n,
        },
        'conditions': config.conditions,
        'tracking': {
            'detector': case['detector'],
            'detector_params': dict(config.detector_params),
            'prior_frames': int(config.prior_frames),
            'prior_mode': case['prior_mode'],
            'shard_overlap': int(config.shard_overlap),
            'effective_shard_overlap': int(config.shard_overlap) if case['prior_mode'] == 'warmup' else 0,
            'linker': case['linker'],
            'link_gap': int(config.link_gap),
        },
        'calibration': {
            'px_per_mm': config.px_per_mm,
            'source': config.calibration_source,
            'uncertainty_px_per_mm': config.calibration_uncertainty,
        },
        'boundary': {
            'mode': 'mask' if config.boundary_mask else 'full_frame',
            'mask': str(config.boundary_mask) if config.boundary_mask else None,
            'mask_sha256': _mask_sha(config.boundary_mask),
            'wall_model': config.wall_model,
            'wall_band': int(config.wall_band),
            'wall_dilate_px': int(config.wall_dilate),
            'wall_tol_px': float(config.wall_tol_px),
        },
        'code': {
            'name': 'vibrotrack',
            'version': schema.package_version(),
            **schema.git_info(),
        },
        'outputs': schema.outputs_record(product_dir, names),
        'qc': {'status': 'detected_only' if detected_only else 'extracted'},
        'retention': {
            'raw_path': case['path'],
            'raw_deleted': False,
        },
        'config': {
            'path': str(config.path) if config.path else None,
            'sha256': schema.sha256(config.path) if config.path and Path(config.path).exists() else None,
        },
    }
    schema.write_json(output_dir / 'dataset.json', rec)
    return rec


def _prior_mode(detector, prior, mode):
    if mode not in ('carry', 'warmup'):
        raise ValueError('run.prior_mode must be carry or warmup')
    return 'off' if detector == 'v1' or not prior else mode


def _detect_parts(case, tasks, workers):
    if case['prior_mode'] == 'carry':
        history = deque(maxlen=case['prior_frames'])
        for task in tasks:
            yield D.shard_task(task, history=history)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(D.shard_task, t) for t in tasks]
            for f in as_completed(futures):
                yield f.result()


def prepare_case(raw, seq_root, config: ApparatusConfig, *, case_id, nframes=None,
                 workers=None, shard=None, detector=None, linker=None, wall_band=None,
                 wall_model=None, strict_io=None):
    """Resolve the ordered input and effective detection settings for extraction."""
    raw = Path(raw).resolve()
    if not raw.is_dir():
        raise FileNotFoundError('raw frame directory not found: %s' % raw)
    workers = int(workers or config.workers)
    shard = int(shard or config.shard)
    detector = (detector or config.detector).lower()
    linker = (linker or config.linker).lower()
    wall_band = int(config.wall_band if wall_band is None else wall_band)
    wall_model = (wall_model or config.wall_model).lower()
    strict_io = config.strict_io if strict_io is None else bool(strict_io)
    mode = _prior_mode(detector, config.prior, config.prior_mode)
    if min(workers, shard, int(config.prior_frames)) < 1 or config.shard_overlap < 0:
        raise ValueError('workers, shard and prior frames must be positive, and overlap nonnegative')

    case = dict(
        id=case_id,
        nframes=0,
        path=str(raw),
        scale_levels=[float(x) for x in config.scale_levels],
        out_root=str(seq_root),
        mask_path=str(config.boundary_mask) if config.boundary_mask else None,
        detector=detector,
        linker=linker,
        wall_band=wall_band,
        wall_dilate=int(config.wall_dilate),
        wall_model=wall_model,
        wall_tol_px=float(config.wall_tol_px),
        avoid_r=float(config.avoid_r),
        link_gap=int(config.link_gap),
        strict_io=strict_io,
        extensions=tuple(config.extensions),
        detector_params=dict(config.detector_params),
        prior=bool(config.prior),
        prior_frames=int(config.prior_frames),
        prior_mode=mode,
        shard_overlap=int(config.shard_overlap),
        particle_radius_px=float(config.radius_px),
    )

    files = tuple(X.files_for(case))
    n = min(int(nframes), len(files)) if nframes else len(files)
    if n <= 0:
        raise ValueError('no frames found in %s (extensions=%s)' % (raw, config.extensions))
    case['nframes'] = n
    first = X.read_image(files[0], case['scale_levels'], strict=strict_io)
    case['shape'] = tuple(int(x) for x in first.shape)
    if config.image_width and config.image_height:
        if case['shape'] != (int(config.image_height), int(config.image_width)):
            raise ValueError(
                'image shape %r does not match config image %r (height, width)'
                % (case['shape'], (config.image_height, config.image_width)))
    _, mask_src = load_mask(case['mask_path'], case['shape'], required=True)
    case['mask_source'] = mask_src or 'full_frame'
    return case, files, workers, shard


def detection_tasks(case, files, shard):
    """Build the same detection shards for extraction and preview."""
    n = case['nframes']
    overlap = case['shard_overlap'] if case['prior_mode'] == 'warmup' else 0
    tasks = []
    for lo in range(0, n, shard):
        hi = min(lo + shard, n)
        file_lo = max(0, lo - overlap)
        # Legacy warm-up replays the overlap from cold. Carry takes the actual
        # preceding detections instead and reads every requested frame once.
        tasks.append((case, lo, hi, files[file_lo:hi], file_lo))
    return tasks


def run_case(raw, out, config: ApparatusConfig, *, case_id, nframes=None, workers=None,
             shard=None, detector=None, linker=None, wall_band=None, wall_model=None,
             strict_io=None, no_link=False, limit_shards=0, dataset_id=None,
             project='', campaign='', acquired_at=None, operator=None, progress=print,
             events=None):
    """Run one case and publish its dataset under out/sequences/<case_id>/."""
    if limit_shards < 0:
        raise ValueError('limit_shards must be nonnegative')
    if Path(case_id).name != case_id or case_id in ('', '.', '..'):
        raise ValueError('case_id must be a single directory name')
    seq_root = Path(out).resolve() / 'sequences'
    case_dir = seq_root / case_id
    prepare_options = dict(case_id=case_id, nframes=nframes, workers=workers, shard=shard,
                           detector=detector, linker=linker, wall_band=wall_band,
                           wall_model=wall_model, strict_io=strict_io)
    t0 = time.time()
    prepared = None
    if not case_dir.exists():
        # A failed preflight for a fresh case does not need an output directory.
        prepared = prepare_case(raw, seq_root, config, **prepare_options)
    with case_lock(case_dir):
        recover_case(case_dir)
        if prepared is None:
            prepared = prepare_case(raw, seq_root, config, **prepare_options)
        case, files, workers, shard = prepared
        progress('files: %d (%.2f s), shape=%s' % (len(files), time.time() - t0, case['shape']))
        D.prepare_detection_cache(case, files[:case['nframes']], shard_size=shard)
        return _run_prepared_case(
            case, files, workers, shard, config, case_dir, no_link=no_link,
            limit_shards=limit_shards, dataset_id=dataset_id, project=project,
            campaign=campaign, acquired_at=acquired_at, operator=operator,
            progress=progress, events=events)


def _run_prepared_case(case, files, workers, shard, config, case_dir, *,
                       no_link, limit_shards, dataset_id, project, campaign,
                       acquired_at, operator, progress, events):
    tasks = detection_tasks(case, files, shard)
    if limit_shards:
        tasks = tasks[:int(limit_shards)]
    frames = sum(hi - lo for _, lo, hi, _, _ in tasks)
    if events:
        events(dict(event='progress', stage='inputs', done=frames, total=frames,
                    available_frames=len(files), shape=case['shape']))
    link_state = None if no_link else D.new_link_state(case)
    pending = {}
    next_lo = 0
    detected = 0
    link_seconds = 0.0
    t0 = time.time()
    for r in _detect_parts(case, tasks, workers):
        detected += r['hi'] - r['lo']
        if events:
            events(dict(event='progress', stage='detect', done=detected, total=frames,
                        first_frame=r['lo'], end_frame_exclusive=r['hi'],
                        status=r['status']))
        if link_state is None:
            continue
        pending[r['lo']] = r
        while next_lo in pending:
            rr = pending.pop(next_lo)
            part = D.load_det_part(case, rr['lo'], rr['hi'])
            t = time.perf_counter()
            D.link_part(part, link_state)
            link_seconds += time.perf_counter() - t
            next_lo = rr['hi']
            if events:
                events(dict(event='progress', stage='link', done=next_lo, total=frames))
    dt = time.time() - t0
    mode = case['prior_mode']
    progress('detect: %d frames in %.1f s (%.1f frames/s, %d workers, shard %d, prior %s%s)'
             % (frames, dt, frames / dt, 1 if mode == 'carry' else workers, shard, mode,
                ', link interleaved' if link_state is not None and mode == 'carry'
                else ', link overlapped' if link_state is not None else ''))
    if link_state is not None and (pending or next_lo != frames):
        raise ValueError('detection shards did not cover all requested frames')
    if events:
        events(dict(event='progress', stage='publish', done=0, total=1))
    with stage_case(case_dir) as staging:
        if link_state is not None:
            t1 = time.perf_counter()
            status = D.finish_link(case, link_state, output_dir=staging, source_files=files)
            link_seconds += time.perf_counter() - t1
            progress('link: %.1f s measured %s' % (link_seconds, status))
        record = _write_dataset(case, config, staging, case_dir, dataset_id, project,
                                campaign, acquired_at, operator, detected_only=no_link)
        D.validate_detection_cache(case, files=files[:case['nframes']])
    if events:
        events(dict(event='progress', stage='publish', done=1, total=1))
    return record
