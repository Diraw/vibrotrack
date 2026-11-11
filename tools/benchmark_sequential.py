"""Read-only rolling-prior detector regression; not a full extraction benchmark.

Only the explicitly named, NEW JSON report is written. Raw images, extraction
products, production modules and defaults are never modified. One <=100-frame
window is cached at a time. See the report's method/limitations before quoting
read+detect timings as an end-to-end result.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import scipy

from vibrotrack import detect, detect2
from vibrotrack.mask import load_mask


REPO = Path(__file__).resolve().parents[1]
ANCHOR = '    theta, angle_valid, offset, area, bright = _marker(im, xy_s)'
REPLACEMENT = (
    '    marker_indices = np.searchsorted(cand, sel)\n'
    '    theta, angle_valid, offset, area, bright = [\n'
    '        a[marker_indices] for a in (theta_t, ok_t, off_t, area_t, bright_t)]'
)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def json_sha256(value):
    data = json.dumps(json_value(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf8')
    return hashlib.sha256(data).hexdigest()


def array_digest(record, canonical=False):
    """Include names, dtype and shape; canonical mode treats NaNs/-0 as equal."""
    h = hashlib.sha256()
    for key in sorted(record):
        a = np.ascontiguousarray(record[key])
        if canonical and a.dtype.kind in 'fc':
            a = a.copy()
            a[a == 0] = 0
            a[np.isnan(a)] = np.nan
        for header in (key, a.dtype.str, str(a.shape)):
            h.update(header.encode('utf8') + b'\0')
        h.update(a.tobytes())
    return h.hexdigest()


def first_difference(left, right, path=''):
    """Exact values, keys, shape and dtype, except that colocated NaNs are equal."""
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return {'field': path, 'reason': 'mapping type differs'}
        if set(left) != set(right):
            return {'field': path, 'reason': 'keys differ',
                    'reference': sorted(left), 'candidate': sorted(right)}
        for key in sorted(left):
            diff = first_difference(left[key], right[key], f'{path}.{key}'.strip('.'))
            if diff is not None:
                return diff
        return None
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype != b.dtype or a.shape != b.shape:
        return {'field': path, 'reason': 'shape/dtype differs',
                'reference': {'shape': list(a.shape), 'dtype': str(a.dtype)},
                'candidate': {'shape': list(b.shape), 'dtype': str(b.dtype)}}
    same = a == b
    if a.dtype.kind in 'fc':
        same = same | (np.isnan(a) & np.isnan(b))
    if np.all(same):
        return None
    index = tuple(int(v) for v in np.unravel_index(int(np.flatnonzero(~same)[0]), a.shape))
    return {'field': path, 'reason': 'value differs', 'index': list(index),
            'reference': json_value(a[index]), 'candidate': json_value(b[index]),
            'unequal_elements': int(np.count_nonzero(~same))}


def make_candidate():
    """Apply only benchmark_detector.py's existing marker-reuse transformation."""
    reference_source = inspect.getsource(detect2.frame)
    if reference_source.count(ANCHOR) != 1:
        raise RuntimeError('prototype anchor changed; inspect detect2.frame first')
    candidate_source = reference_source.replace(ANCHOR, REPLACEMENT)
    namespace = dict(vars(detect2))
    namespace['__name__'] = 'vibrotrack._isolated_marker_reuse_benchmark'
    exec(compile(candidate_source, '<isolated-marker-reuse-prototype>', 'exec'), namespace)
    candidate = namespace['frame']
    if candidate.__globals__ is detect2.frame.__globals__:
        raise AssertionError('candidate must not share the production namespace')
    return candidate, {
        'reference_frame_source_sha256': hashlib.sha256(reference_source.encode()).hexdigest(),
        'candidate_frame_source_sha256': hashlib.sha256(candidate_source.encode()).hexdigest(),
        'old': ANCHOR, 'new': REPLACEMENT, 'isolated_globals': True,
        'production_function_unchanged': namespace['frame'] is not detect2.frame,
    }


def git_state():
    def command(*args):
        p = subprocess.run(['git', '-C', str(REPO), *args], capture_output=True,
                           text=True, encoding='utf8', errors='replace', timeout=20, check=True)
        return p.stdout.strip()
    try:
        status = command('status', '--porcelain', '--untracked-files=all')
        diff = command('diff', '--binary', 'HEAD')
        return {'commit': command('rev-parse', 'HEAD'), 'dirty': bool(status),
                'status_porcelain': status.splitlines(),
                'tracked_diff_sha256': hashlib.sha256(diff.encode()).hexdigest()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'error': str(exc), 'commit': None, 'dirty': None}


class MemoryMonitor:
    """Sample RSS, also use Windows' process-lifetime peak working set if exposed."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.result = {'available': False, 'sample_interval_s': 0.05}
        try:
            import psutil
            self.process = psutil.Process()
            self.result.update(available=True, psutil_version=psutil.__version__,
                               start_rss_bytes=self.process.memory_info().rss,
                               sampled_peak_rss_bytes=0, samples=0)
        except (ImportError, OSError) as exc:
            self.result['error'] = str(exc)

    def sample(self):
        try:
            m = self.process.memory_info()
            self.result['sampled_peak_rss_bytes'] = max(
                self.result['sampled_peak_rss_bytes'], int(m.rss))
            self.result['samples'] += 1
            if hasattr(m, 'peak_wset'):
                self.result['os_process_lifetime_peak_working_set_bytes'] = int(m.peak_wset)
        except Exception as exc:
            self.result['sample_error'] = str(exc)

    def start(self):
        if self.result['available']:
            def loop():
                while not self.stop_event.is_set():
                    self.sample()
                    self.stop_event.wait(self.result['sample_interval_s'])
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()

    def finish(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.sample()
            self.result['end_rss_bytes'] = int(self.process.memory_info().rss)
        return self.result


def parse_windows(text):
    result = []
    for part in text.split(','):
        lo, hi = (int(x) for x in part.split(':'))
        if lo <= 0 or hi < lo or hi - lo + 1 > 100:
            raise ValueError('windows must be positive inclusive ranges of <=100 frames')
        result.append((lo, hi))
    ids = [fid for lo, hi in result for fid in range(lo, hi + 1)]
    if len(set(ids)) != len(ids):
        raise ValueError('windows must not overlap')
    return result


def load_seeds(case, windows, prior_frames, shard_overlap):
    """Use pre-link det shards, not linked/filtered chunks, for all 5 prior frames."""
    directory = case / f'dets_v2_p{prior_frames}_o{shard_overlap}'
    paths = sorted(directory.glob('det_*.npz'))
    if not paths:
        raise FileNotFoundError('original detector shards required: ' + str(directory))
    needed = {fid for lo, _ in windows for fid in range(max(0, lo - prior_frames), lo)}
    records, sources = {}, []
    for path in paths:
        _, lo, hi = path.stem.split('_')
        if not any(int(lo) <= fid <= int(hi) for fid in needed):
            continue
        with np.load(path, allow_pickle=False) as z:
            ids, offsets, xy = z['frame_ids'], z['frame_offsets'], z['xy_px']
            selected = []
            for i, fid in enumerate(ids):
                fid = int(fid)
                if fid in needed:
                    records[fid] = xy[int(offsets[i]):int(offsets[i + 1])].copy()
                    selected.append(fid)
        sources.append({'path': str(path.resolve()), 'sha256': file_sha256(path),
                        'seed_frame_ids': selected})
    if set(records) != needed:
        raise ValueError('missing stored seed frames: ' + str(sorted(needed - set(records))))
    seeds = {lo: [records[fid] for fid in range(max(0, lo - prior_frames), lo)]
             for lo, _ in windows}
    return seeds, sources


def rolling_prior(history):
    # Same chronological non-empty concatenation as shards.shard_task.prior_xy.
    parts = [xy for xy in history if len(xy)]
    return np.concatenate(parts) if parts else None


def run_arm(fn, images, ids, seeds, prior_frames, mask, dist, kwargs):
    history = deque((xy.copy() for xy in seeds), maxlen=prior_frames)
    records, elapsed = [], []
    started = time.perf_counter()
    for fid, im in zip(ids, images):
        prior = rolling_prior(history)
        t0 = time.perf_counter()
        detection, info = fn(im, mask, dist, prior_xy=prior, **kwargs)
        elapsed.append(time.perf_counter() - t0)
        history.append(detection['xy_px'])
        records.append((detection, info))
    return records, {'detect_s': sum(elapsed), 'frame_detect_s': elapsed,
                     'arm_wall_s': time.perf_counter() - started}


def source_hashes():
    paths = [REPO / 'tools' / 'benchmark_sequential.py',
             REPO / 'tools' / 'benchmark_detector.py']
    paths.extend(REPO / 'src' / 'vibrotrack' / name for name in
                 ('detect.py', 'detect2.py', 'fast_tiff.py', 'mask.py', 'shards.py'))
    return {str(path.relative_to(REPO)): file_sha256(path) for path in paths}


def run(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('refusing to overwrite any existing report: ' + str(out))
    started_unix = time.time()
    started = time.perf_counter()
    monitor = MemoryMonitor()
    monitor.start()
    report = {'schema': 'detector-sequential-benchmark/v1', 'status': 'running',
              'started_at_unix': started_unix,
              'started_at_utc': datetime.fromtimestamp(started_unix, timezone.utc).isoformat(),
              'started_at_local': datetime.fromtimestamp(started_unix).astimezone().isoformat(),
              'argv': sys.argv, 'git_start': git_state(), 'source_sha256_start': source_hashes()}
    first_diff = None
    mismatches = []
    repeats = []
    input_manifest = {}
    stable_hashes = {}
    max_cached_bytes = 0
    total_pairs = 0
    warmup_done = False
    try:
        case = args.case.resolve()
        cfg_path, dataset_path = case / 'run_config.json', case / 'dataset.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf8'))
        dataset = json.loads(dataset_path.read_text(encoding='utf8'))
        if cfg.get('detector') != 'v2' or cfg.get('wall_model') != 'off' or cfg.get('wall_band'):
            raise ValueError('this deliberately narrow tool requires v2 + wall_model=off + wall_band=0')
        if cfg.get('prior', True) is not True:
            raise ValueError('rolling-prior regression requires prior enabled')
        windows = parse_windows(args.windows)
        prior_frames = max(1, int(cfg['prior_frames']))
        if cfg['detector_defaults'] != detect2.DEFAULTS:
            raise ValueError('archived detector_defaults differ from current implementation')
        if dataset['tracking']['prior_frames'] != prior_frames:
            raise ValueError('dataset and run_config disagree on prior_frames')
        params = dict(cfg['detector_defaults'])
        overrides = dataset['tracking'].get('detector_params', {})
        reserved = {'avoid_r', 'prior_xy', 'static_centres', 'interior_mask', 'bbox'}
        params.update({k: v for k, v in overrides.items() if k not in reserved})
        params['avoid_r'] = float(cfg.get('avoid_r', 0.0))
        mask_path = args.mask.resolve()
        if Path(cfg['mask']).resolve() != mask_path:
            raise ValueError('supplied mask does not match archived run_config')
        mask, _ = load_mask(mask_path, required=True)
        mask_hash = file_sha256(mask_path)
        if dataset['boundary']['mask_sha256'] != mask_hash:
            raise ValueError('mask content differs from dataset manifest')
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        bbox = detect.mask_bbox(mask)
        kwargs = dict(params, bbox=bbox, interior_mask=mask, static_centres=None)
        strict_io = bool(cfg.get('strict_io', False))
        cv2.setNumThreads(1)
        candidate, prototype = make_candidate()
        functions = {'reference': detect2.frame, 'candidate': candidate}
        scan_start = time.perf_counter()
        files = detect.files_for({'path': str(args.raw)})
        scan_s = time.perf_counter() - scan_start
        if len(files) != cfg['n_input_files'] or max(hi for _, hi in windows) >= len(files):
            raise ValueError('input frame count/window bounds do not match archived sequence')
        if str(files[0].resolve()) != str(Path(cfg['first_file']).resolve()) or \
                str(files[-1].resolve()) != str(Path(cfg['last_file']).resolve()):
            raise ValueError('raw file ordering/endpoints differ from archived sequence')
        seeds, seed_sources = load_seeds(case, windows, prior_frames, int(cfg['shard_overlap']))
        report.update(
            case=str(case), raw=str(args.raw.resolve()), mask=str(mask_path),
            config=cfg, run_config_sha256=file_sha256(cfg_path),
            dataset_sha256=file_sha256(dataset_path), mask_sha256=mask_hash,
            mask_array_sha256=array_digest({'mask': mask}),
            distance_array_sha256=array_digest({'dist': dist}), prototype=prototype,
            effective_parameters=params, bbox=bbox, scale_levels=cfg['scale_levels'],
            strict_io=strict_io, prior_frames=prior_frames, input_scan_s=scan_s,
            stored_prior_sources=seed_sources,
            windows=[{'lo': lo, 'hi_inclusive': hi, 'frames': hi - lo + 1,
                      'seed_frames': list(range(max(0, lo - prior_frames), lo)),
                      'seed_xy_sha256': [array_digest({'xy_px': xy}) for xy in seeds[lo]]}
                     for lo, hi in windows],
            selected_unique_frames=sum(hi - lo + 1 for lo, hi in windows),
            environment={'platform': platform.platform(), 'python': platform.python_version(),
                         'executable': sys.executable, 'numpy': np.__version__,
                         'scipy': scipy.__version__, 'opencv': cv2.__version__,
                         'opencv_threads': cv2.getNumThreads(),
                         'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
                         'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS')},
            method={
                'prior': 'At each window start, seed each arm independently from the last '
                         'prior_frames original pre-link det-shard xy arrays. Thereafter each '
                         'arm concatenates its OWN last 5 non-empty frame arrays (empty frames '
                         'still consume a history slot), exactly as shards.shard_task.prior_xy.',
                'timing': '3 full paired repetitions by default. Per repetition/window run '
                          'the two entire rolling sequences in alternating arm order (AB,BA,AB). '
                          'Read/decode/scale once per pair into a shared <=100-frame cache. '
                          'Detection timers enclose only frame(); prior concatenation, hashes '
                          'and comparisons are excluded. Actual arm and total wall time are separate.',
                'equality': 'Direct comparison of EVERY detection key, dtype, shape and element '
                            'and EVERY info key/value on every pair; no tolerance, NaNs equal. '
                            'Raw array hashes and NaN/-0-normalized stability hashes are separate.',
                'read_plus_detect': 'Shared measured read_s + arm detect_s is a serial '
                                    'read+detect estimate, NOT an independently measured full pipeline.',
                'limitations': ['Only selected windows, NOT all 9001 frames.',
                               'Not a ground-truth recall or precision measurement; checks exact '
                               'reference equivalence and detection-count preservation.',
                               'Stored history initialization replaces recomputing all preceding '
                               'frames/shard warmup. Windows roll continuously rather than resetting '
                               'at production shard boundaries (the default 5450:5549 window '
                               'crosses boundary 5500); shard-overlap warmup is not replayed.',
                               'No production prefetch overlap, linking, output writing, wall-track '
                               'filtering, multiprocessing or cold-cache guarantee.',
                               'Both arms share the same decoded images and fixed masks; separate '
                               'rolling priors/outputs. Three repetitions reuse OS disk cache.']})
        for repeat in range(args.repeats):
            order = ['reference', 'candidate'] if repeat % 2 == 0 else ['candidate', 'reference']
            repetition = {'repeat': repeat + 1, 'arm_order': order, 'windows': [],
                          'read_s': 0.0, 'detect_s': {'reference': 0.0, 'candidate': 0.0}}
            repetition_start = time.perf_counter()
            for lo, hi in windows:
                ids = list(range(lo, hi + 1))
                if not warmup_done:
                    t0 = time.perf_counter()
                    im = detect.read_image(files[lo], cfg['scale_levels'], strict=strict_io)
                    warm_read = time.perf_counter() - t0
                    prior = rolling_prior(seeds[lo])
                    warm_results, warm_seconds = {}, {}
                    for name in functions:
                        t0 = time.perf_counter()
                        warm_results[name] = functions[name](im, mask, dist, prior_xy=prior, **kwargs)
                        warm_seconds[name] = time.perf_counter() - t0
                    warm_diff = first_difference(warm_results['reference'][0],
                                                 warm_results['candidate'][0], 'detection')
                    if warm_diff is None:
                        warm_diff = first_difference(warm_results['reference'][1],
                                                     warm_results['candidate'][1], 'info')
                    report['warmup'] = {'frame': lo, 'calls_per_arm': 1, 'read_s': warm_read,
                                        'detect_s': warm_seconds, 'first_difference': warm_diff,
                                        'included_in_timing_totals': False}
                    if warm_diff is not None:
                        first_diff = dict(repeat=0, window=[lo, hi], frame=lo, kind='warmup', **warm_diff)
                    del im, prior, warm_results
                    warmup_done = True
                images, read_seconds = [], []
                window_start = time.perf_counter()
                for fid in ids:
                    t0 = time.perf_counter()
                    image = detect.read_image(files[fid], cfg['scale_levels'], strict=strict_io)
                    read_seconds.append(time.perf_counter() - t0)
                    if image.shape != mask.shape:
                        raise ValueError(f'image/mask shape differs at frame {fid}')
                    images.append(image)
                    image_hash = array_digest({'scaled_uint8_image': image})
                    if fid not in input_manifest:
                        stat = files[fid].stat()
                        input_manifest[fid] = {'frame': fid, 'file': str(files[fid]),
                                               'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                               'scaled_image_sha256': image_hash}
                    elif input_manifest[fid]['scaled_image_sha256'] != image_hash:
                        raise RuntimeError(f'decoded input changed across repetitions: {fid}')
                del image
                cached_bytes = sum(im.nbytes for im in images)
                max_cached_bytes = max(max_cached_bytes, cached_bytes)
                outputs, timings = {}, {}
                for name in order:
                    outputs[name], timings[name] = run_arm(
                        functions[name], images, ids, seeds[lo], prior_frames, mask, dist, kwargs)
                    repetition['detect_s'][name] += timings[name]['detect_s']
                paired_rows = []
                for i, fid in enumerate(ids):
                    rd, ri = outputs['reference'][i]
                    cd, ci = outputs['candidate'][i]
                    diff = first_difference(rd, cd, 'detection')
                    info_diff = first_difference(ri, ci, 'info')
                    total_pairs += 1
                    if diff is not None or info_diff is not None:
                        detail = dict(repeat=repeat + 1, window=[lo, hi], frame=fid,
                                      kind='reference_vs_candidate', **(diff or info_diff))
                        mismatches.append(detail)
                        if first_diff is None:
                            first_diff = detail
                    hashes = {}
                    for name, det, info in [('reference', rd, ri), ('candidate', cd, ci)]:
                        raw_hash = array_digest(det)
                        canonical_hash = array_digest(det, canonical=True)
                        info_hash = json_sha256(info)
                        hashes[name] = {'arrays_sha256': raw_hash,
                                        'nan_equal_arrays_sha256': canonical_hash,
                                        'info_sha256': info_hash}
                        key = (name, fid)
                        signature = (canonical_hash, info_hash)
                        if key in stable_hashes and stable_hashes[key] != signature:
                            detail = {'repeat': repeat + 1, 'frame': fid, 'arm': name,
                                      'kind': 'repetition_instability',
                                      'first_signature': stable_hashes[key], 'signature': signature}
                            mismatches.append(detail)
                            if first_diff is None:
                                first_diff = detail
                        stable_hashes.setdefault(key, signature)
                    paired_rows.append({'frame': fid, 'equal': diff is None and info_diff is None,
                                        'detections': {'reference': len(rd['xy_px']),
                                                       'candidate': len(cd['xy_px'])},
                                        'angle_valid': {'reference': int(rd['angle_valid'].sum()),
                                                        'candidate': int(cd['angle_valid'].sum())},
                                        'hashes': hashes, 'reference_info': json_value(ri),
                                        'candidate_info': json_value(ci),
                                        'read_s': read_seconds[i],
                                        'detect_s': {name: timings[name]['frame_detect_s'][i]
                                                     for name in functions}})
                read_s = sum(read_seconds)
                repetition['read_s'] += read_s
                for name in functions:
                    timings[name]['serial_read_plus_detect_s'] = read_s + timings[name]['detect_s']
                    del timings[name]['frame_detect_s']
                window_row = {'lo': lo, 'hi_inclusive': hi, 'cached_images': len(images),
                              'cached_image_bytes': cached_bytes, 'read_s': read_s,
                              'arms': timings, 'frames': paired_rows,
                              'pair_window_wall_s': time.perf_counter() - window_start}
                repetition['windows'].append(window_row)
                print(f'repeat {repeat + 1}/{args.repeats} [{lo},{hi}] order={order} '
                      f'read={read_s:.3f}s reference={timings["reference"]["detect_s"]:.3f}s '
                      f'candidate={timings["candidate"]["detect_s"]:.3f}s '
                      f'equal={all(row["equal"] for row in paired_rows)}', flush=True)
                images.clear()
                del images, outputs, rd, cd, ri, ci, det, info
            repetition['wall_s'] = time.perf_counter() - repetition_start
            repetition['serial_read_plus_detect_s'] = {
                name: repetition['read_s'] + repetition['detect_s'][name] for name in functions}
            repetition['detect_speedup_reference_over_candidate'] = (
                repetition['detect_s']['reference'] / repetition['detect_s']['candidate'])
            repeats.append(repetition)
        report['status'] = 'passed' if first_diff is None else 'failed_equivalence'
    except Exception as exc:
        report['status'] = 'error'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc),
                           'traceback': traceback.format_exc()}
    finally:
        report['memory'] = monitor.finish()
        report['source_sha256_end'] = source_hashes()
        report['git_end'] = git_state()
        report['source_files_unchanged'] = report['source_sha256_start'] == report['source_sha256_end']
        if not report['source_files_unchanged']:
            report['status'] = 'invalid_source_changed_during_run'
        report.update(repetitions=repeats, first_difference=first_diff,
                      differences=mismatches, paired_frames_compared=total_pairs,
                      input_manifest=[input_manifest[fid] for fid in sorted(input_manifest)],
                      max_cached_image_bytes=max_cached_bytes,
                      max_cached_images=max((w['cached_images'] for r in repeats
                                             for w in r['windows']), default=0))
        report['input_manifest_sha256'] = json_sha256(report['input_manifest'])
        if repeats:
            totals = {name: [r['detect_s'][name] for r in repeats]
                      for name in ('reference', 'candidate')}
            reference, candidate = (statistics.median(totals[name]) for name in totals)
            speedups = [r['detect_speedup_reference_over_candidate'] for r in repeats]
            report['summary'] = {
                'completed_repetitions': len(repeats), 'all_pairs_equal': first_diff is None,
                'detect_seconds_by_repeat': totals,
                'frames_per_repeat': report.get('selected_unique_frames'),
                'median_selected_frame_detect_s': {'reference': reference, 'candidate': candidate},
                'median_paired_detect_speedup': statistics.median(speedups),
                'paired_detect_speedups': speedups,
                'median_paired_detect_time_reduction_fraction': statistics.median(
                    1 - 1 / speedup for speedup in speedups),
                'read_seconds_by_repeat': [r['read_s'] for r in repeats],
                'serial_read_plus_detect_seconds_by_repeat': {
                    name: [r['serial_read_plus_detect_s'][name] for r in repeats] for name in totals},
                'detections_per_frame': {
                    name: {'min': min(counts), 'median': statistics.median(counts), 'max': max(counts)}
                    for name in totals
                    for counts in [[row['detections'][name] for r in repeats
                                    for w in r['windows'] for row in w['frames']]]}}
        ended_unix = time.time()
        report['ended_at_unix'] = ended_unix
        report['ended_at_utc'] = datetime.fromtimestamp(ended_unix, timezone.utc).isoformat()
        report['actual_total_wall_s'] = time.perf_counter() - started
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf8') as f:
            json.dump(json_value(report), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        print('status:', report['status'], 'paired frames:', total_pairs,
              'wall seconds:', report['actual_total_wall_s'], flush=True)
        print('report:', out, flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, type=Path)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--mask', required=True, type=Path)
    p.add_argument('--windows', default='150:249,1050:1149,5450:5549',
                   help='inclusive lo:hi ranges, each <=100 frames')
    p.add_argument('--repeats', default=3, type=int)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if args.repeats < 1:
        p.error('--repeats must be >= 1')
    raise SystemExit(run(args))
