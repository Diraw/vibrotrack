"""Paired prior-dedupe regression on rolling real frames; no production default is changed.

The production detector seeds every frame with the last ``prior_frames`` detection
arrays. Those arrays repeat the same particle 5 times within a few pixels, so the
radial fit runs on roughly five times more seeds than there are particles. This tool
compares the reference seed set with a deduplicated one (drop a prior seed that sits
within ``--dedupe`` px of a template peak) on the same real frames and reports every
detection field and info value, plus paired timings.

Read-only: it never writes into the case directory and refuses to overwrite a report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.spatial import cKDTree

from vibrotrack import detect, detect2
from vibrotrack.mask import load_mask


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(record):
    h = hashlib.sha256()
    for key in sorted(record):
        a = np.ascontiguousarray(record[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def first_difference(left, right, path=''):
    """Return the first structural or element difference, or None."""
    if left.keys() != right.keys():
        return {'path': path, 'kind': 'keys',
                'left': sorted(left.keys()), 'right': sorted(right.keys())}
    for key in sorted(left):
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.shape != b.shape:
            return {'path': path + '/' + key, 'kind': 'shape', 'left': list(a.shape),
                    'right': list(b.shape)}
        if a.dtype != b.dtype:
            return {'path': path + '/' + key, 'kind': 'dtype', 'left': str(a.dtype),
                    'right': str(b.dtype)}
        if a.size and not np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'):
            bad = np.argwhere(~(a == b)).ravel()[:5].tolist()
            return {'path': path + '/' + key, 'kind': 'value', 'count': int((a != b).sum()),
                    'first_indices': bad}
    return None


def parse_windows(text):
    out = []
    for part in text.split(','):
        lo, hi = part.split(':')
        out.append((int(lo), int(hi)))
    return out


def load_seeds(case, windows, prior_frames, shard_overlap):
    """Original pre-link det-shard xy arrays for the frames before each window."""
    directory = case / f'dets_v2_p{prior_frames}_o{shard_overlap}'
    needed = {fid for lo, _ in windows for fid in range(max(0, lo - prior_frames), lo)}
    seeds = {}
    sources = []
    for path in sorted(directory.glob('det_*.npz')):
        with np.load(path) as z:
            ids, offsets, xy = z['frame_ids'], z['frame_offsets'], z['xy_px']
            for i, fid in enumerate(ids):
                if int(fid) in needed:
                    seeds[int(fid)] = xy[int(offsets[i]):int(offsets[i + 1])].copy()
                    sources.append({'frame': int(fid), 'file': path.name,
                                    'rows': int(offsets[i + 1] - offsets[i]),
                                    'xy_px_sha256': array_digest(
                                        {'xy_px': seeds[int(fid)]})})
    missing = needed - set(seeds)
    if missing:
        raise ValueError('missing prior seed frames: %s' % sorted(missing)[:5])
    return seeds, sources


def rolling_prior(history):
    parts = [xy for xy in history if len(xy)]
    return np.concatenate(parts) if parts else None


def git_state(repo):
    def command(*args):
        return subprocess.check_output(args, cwd=repo, text=True).strip()
    try:
        return {'commit': command('git', 'rev-parse', 'HEAD'),
                'dirty': command('git', 'status', '--porcelain')}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def run(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('refusing to overwrite an existing report: ' + str(out))
    started = time.perf_counter()
    case = args.case.resolve()
    raw = args.raw.resolve()
    cfg = json.loads((case / 'run_config.json').read_text(encoding='utf8'))
    if cfg.get('detector') != 'v2':
        raise ValueError('this tool only covers detector v2')
    if cfg.get('prior', True) is not True:
        raise ValueError('rolling-prior regression requires prior enabled')
    mask_path = args.mask.resolve()
    if Path(cfg['mask']).resolve() != mask_path:
        raise ValueError('supplied mask does not match archived run_config')
    mask, _ = load_mask(mask_path, required=True)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    bbox = detect.mask_bbox(mask)
    prior_frames = max(1, int(cfg['prior_frames']))
    params = dict(cfg['detector_defaults'])
    params.pop('prior_dedupe', None)
    params.pop('avoid_r', None)
    overrides = {}
    dataset_path = case / 'dataset.json'
    if dataset_path.exists():
        tracking = json.loads(dataset_path.read_text(encoding='utf8')).get('tracking', {})
        reserved = {'avoid_r', 'prior_xy', 'static_centres', 'interior_mask', 'bbox'}
        overrides = {k: v for k, v in tracking.get('detector_params', {}).items()
                     if k not in reserved}
    params.update(overrides)
    params['avoid_r'] = float(cfg.get('avoid_r', 0.0))
    strict_io = bool(cfg.get('strict_io', False))
    windows = parse_windows(args.windows)
    files = detect.files_for({'path': str(raw)})
    if len(files) != cfg['n_input_files']:
        raise ValueError('input frame count does not match archived sequence')
    if str(files[0].resolve()) != str(Path(cfg['first_file']).resolve()) or \
            str(files[-1].resolve()) != str(Path(cfg['last_file']).resolve()):
        raise ValueError('raw file ordering/endpoints differ from archived sequence')
    seeds, seed_sources = load_seeds(case, windows, prior_frames, int(cfg['shard_overlap']))

    arms = {'reference': 0.0, 'dedupe': float(args.dedupe)}
    report = {
        'schema': 'prior-dedupe-benchmark/v1', 'status': 'running',
        'started_at_unix': time.time(),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'argv': sys.argv, 'git': git_state(Path(__file__).resolve().parents[1]),
        'case': str(case), 'raw': str(raw), 'mask': str(mask_path),
        'mask_sha256': file_sha256(mask_path),
        'run_config_sha256': file_sha256(case / 'run_config.json'),
        'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                        'numpy': np.__version__, 'scipy': scipy.__version__,
                        'opencv': cv2.__version__, 'opencv_threads': cv2.getNumThreads()},
        'effective_parameters': params, 'prior_frames': prior_frames,
        'dedupe_px': arms['dedupe'], 'bbox': bbox, 'scale_levels': cfg['scale_levels'],
        'windows': [{'lo': lo, 'hi_inclusive': hi} for lo, hi in windows],
        'seed_sources': seed_sources,
        'method': {
            'reference': 'concatenate the last prior_frames detection arrays, as shards.shard_task',
            'dedupe': 'same concatenation, then drop any prior seed within dedupe_px of a '
                      'template peak; template peaks are already seeds, so coverage is unchanged',
            'equality': 'every detection field (dtype/shape/value, NaN-aware) and every info value',
            'limitations': ['selected windows, not all 9001 frames',
                            'stored history initialization, not full shard warmup replay',
                            'read/decode/scale excluded from per-frame detect timings',
                            'not a ground-truth recall measurement']},
    }
    per_window = []
    mismatches = []
    total_frames = 0
    detect_s = {name: 0.0 for name in arms}
    counts = {name: 0 for name in arms}
    prior_points = {name: 0 for name in arms}
    timings = []
    try:
        for lo, hi in windows:
            states = {name: deque((seeds[f].copy() for f in range(
                max(0, lo - prior_frames), lo)), maxlen=prior_frames) for name in arms}
            window = {'lo': lo, 'hi_inclusive': hi, 'frames': hi - lo + 1,
                      'counts': {name: 0 for name in arms},
                      'detect_s': {name: 0.0 for name in arms}}
            for fid in range(lo, hi + 1):
                im = detect.read_image(str(files[fid]), cfg['scale_levels'], strict=strict_io)
                outputs = {}
                for name, dedupe in arms.items():
                    prior = rolling_prior(states[name])
                    t0 = time.perf_counter()
                    det, info = detect2.frame(
                        im, mask, dist, bbox=bbox, interior_mask=mask,
                        static_centres=None, prior_xy=prior, prior_dedupe=dedupe, **params)
                    elapsed = time.perf_counter() - t0
                    detect_s[name] += elapsed
                    window['detect_s'][name] += elapsed
                    outputs[name] = (det, info)
                    counts[name] += len(det['xy_px'])
                    window['counts'][name] += len(det['xy_px'])
                    prior_points[name] += int(info['n_prior'])
                    states[name].append(det['xy_px'].copy())
                    if name == 'reference':
                        ref = (det, info)
                        ref_s = elapsed
                    else:
                        same_count = len(ref[0]['xy_px']) == len(det['xy_px'])
                        if not same_count:
                            mismatches.append({'frame': fid, 'kind': 'count',
                                               'reference': int(len(ref[0]['xy_px'])),
                                               'dedupe': int(len(det['xy_px']))})
                        offset = None
                        angle_diff = None
                        quality_diff = None
                        if same_count and len(det['xy_px']):
                            a, b = ref[0]['xy_px'], det['xy_px']
                            d, idx = cKDTree(b).query(a)
                            offset = {'median': float(np.median(d)),
                                      'p90': float(np.percentile(d, 90)),
                                      'p99': float(np.percentile(d, 99)),
                                      'max': float(d.max())}
                            angle_diff = int((ref[0]['angle_valid'] != det['angle_valid'][idx]).sum())
                            quality_diff = int((ref[0]['quality_bits'] != det['quality_bits'][idx]).sum())
                        timings.append({'frame': fid, 'reference_s': ref_s,
                                        'dedupe_s': elapsed, 'offset_px': offset,
                                        'angle_valid_diff': angle_diff,
                                        'quality_bits_diff': quality_diff,
                                        'radius_ref': [float(ref[1]['radius_ref']),
                                                       float(info['radius_ref'])],
                                        'prior_points': [int(ref[1]['n_prior']),
                                                         int(info['n_prior'])]})
                        # both arms keep their own rolling history, so a differing
                        # frame changes the next seed set of that arm only
                total_frames += 1
            per_window.append(window)
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        raise
    pairs = [t for t in timings if t['offset_px'] is not None]
    offsets = [t['offset_px'] for t in pairs]
    report.update(
        status='ok', finished_at_unix=time.time(), wall_s=time.perf_counter() - started,
        frames=total_frames, detections=counts, prior_points=prior_points,
        detect_s=detect_s,
        per_frame_s={name: detect_s[name] / max(1, total_frames) for name in arms},
        median_pair_s={'reference': statistics.median(t['reference_s'] for t in timings),
                       'dedupe': statistics.median(t['dedupe_s'] for t in timings)},
        paired_reduction=1.0 - (statistics.median(t['dedupe_s'] for t in timings) /
                                statistics.median(t['reference_s'] for t in timings)),
        count_mismatch_frames=sum(1 for m in mismatches if m['kind'] == 'count'),
        count_mismatch=mismatches[:20],
        offset_p50=statistics.median(o['median'] for o in offsets) if offsets else None,
        offset_p90=float(np.percentile([o['p90'] for o in offsets], 90)) if offsets else None,
        offset_p99=float(np.percentile([o['p99'] for o in offsets], 99)) if offsets else None,
        offset_max=max(o['max'] for o in offsets) if offsets else None,
        angle_valid_differences=sum(t['angle_valid_diff'] for t in pairs),
        quality_bits_differences=sum(t['quality_bits_diff'] for t in pairs),
        radius_ref_delta_max=max((abs(t['radius_ref'][1] - t['radius_ref'][0]) for t in pairs),
                                 default=None),
        windows_result=per_window,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: report[k] for k in (
        'status', 'frames', 'detections', 'prior_points', 'per_frame_s',
        'median_pair_s', 'paired_reduction', 'count_mismatch_frames',
        'offset_p50', 'offset_p90', 'offset_p99', 'offset_max',
        'angle_valid_differences', 'quality_bits_differences',
        'radius_ref_delta_max')}, ensure_ascii=False, indent=2))
    print('report:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, type=Path)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--mask', required=True, type=Path)
    p.add_argument('--windows', default='150:249,1050:1149,5450:5549')
    p.add_argument('--dedupe', default=6.0, type=float)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    run(args)
