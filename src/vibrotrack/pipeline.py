"""Extract one case: detect shards in parallel, link in frame order, write products."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import detect as X
from . import schema
from . import shards as D
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


def _write_dataset(case, config, case_dir, raw, dataset_id, project, campaign,
                   acquired_at, operator):
    chunk_files = sorted(case_dir.glob('chunk_*.npz'))
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
        'outputs': schema.outputs_record(case_dir, names),
        'qc': {'status': 'extracted'},
        'retention': {
            'raw_path': str(raw),
            'raw_deleted': False,
        },
        'config': {
            'path': str(config.path) if config.path else None,
            'sha256': schema.sha256(config.path) if config.path and Path(config.path).exists() else None,
        },
    }
    schema.write_json(Path(case_dir) / 'dataset.json', rec)
    return rec


def run_case(raw, out, config: ApparatusConfig, *, case_id, nframes=None, workers=None,
             shard=None, detector=None, linker=None, wall_band=None, wall_model=None,
             strict_io=None, no_link=False, limit_shards=0, dataset_id=None,
             project='', campaign='', acquired_at=None, operator=None, progress=print):
    """Run one case and return the dataset.json record.

    ``out`` is the batch directory; products go to ``out/sequences/<case_id>/``.
    """
    raw = Path(raw).resolve()
    if not raw.is_dir():
        raise FileNotFoundError('raw frame directory not found: %s' % raw)
    out = Path(out).resolve()
    seq_root = out / 'sequences'
    seq_root.mkdir(parents=True, exist_ok=True)
    workers = int(workers or config.workers)
    shard = int(shard or config.shard)
    detector = (detector or config.detector).lower()
    linker = (linker or config.linker).lower()
    wall_band = int(config.wall_band if wall_band is None else wall_band)
    wall_model = (wall_model or config.wall_model).lower()
    strict_io = config.strict_io if strict_io is None else bool(strict_io)

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
        particle_radius_px=float(config.radius_px),
    )

    t0 = time.time()
    files = X.files_for(case)
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
    progress('files: %d (%.2f s), shape=%s' % (len(files), time.time() - t0, case['shape']))

    tasks = [(case, lo, min(lo + shard, n), files[lo:min(lo + shard, n)])
             for lo in range(0, n, shard)]
    if limit_shards:
        tasks = tasks[:int(limit_shards)]
    link_state = None if no_link else D.new_link_state(case)
    pending = {}
    next_lo = 0
    link_seconds = 0.0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(D.shard_task, t) for t in tasks]
        for f in as_completed(futures):
            r = f.result()
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
    dt = time.time() - t0
    frames = sum(len(t[3]) for t in tasks)
    progress('detect: %d frames in %.1f s (%.1f frames/s, %d workers, shard %d%s)'
             % (frames, dt, frames / dt, workers, shard,
                ', link overlapped' if link_state is not None else ''))
    if link_state is None:
        rec = _write_dataset(case, config, seq_root / case_id, raw, dataset_id,
                             project, campaign, acquired_at, operator)
        rec['qc']['status'] = 'detected_only'
        schema.write_json(seq_root / case_id / 'dataset.json', rec)
        return rec

    while next_lo in pending:
        rr = pending.pop(next_lo)
        part = D.load_det_part(case, rr['lo'], rr['hi'])
        t = time.perf_counter()
        D.link_part(part, link_state)
        link_seconds += time.perf_counter() - t
        next_lo = rr['hi']
    t1 = time.perf_counter()
    status = D.finish_link(case, link_state)
    link_seconds += time.perf_counter() - t1
    progress('link: %.1f s CPU (overlapped) %s' % (link_seconds, status))
    return _write_dataset(case, config, seq_root / case_id, raw, dataset_id,
                          project, campaign, acquired_at, operator)
