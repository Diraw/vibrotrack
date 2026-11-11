"""Read-only, repeatable detector benchmark. No production defaults are changed."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import inspect
import json
import platform
import pstats
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import scipy

from vibrotrack import detect, detect2


def digest(record):
    h = hashlib.sha256()
    for key in sorted(record):
        a = np.ascontiguousarray(record[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def run(args):
    case = Path(args.case)
    cfg = json.loads((case / 'run_config.json').read_text(encoding='utf8'))
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError('unreadable mask')
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    selected = [int(x) for x in args.frames.split(',')]
    needed = {f - 1 for f in selected if f > 0}
    prior = {}
    for chunk in sorted(case.glob('chunk_*.npz')):
        with np.load(chunk, allow_pickle=False) as z:
            ids, offsets, xy = z['frame_ids'], z['frame_offsets'], z['xy_px']
            for i, fid in enumerate(ids):
                if int(fid) in needed:
                    prior[int(fid)] = xy[int(offsets[i]):int(offsets[i + 1])].copy()
    files = detect.files_for({'path': str(args.raw)})
    images = {}
    t0 = time.perf_counter()
    for fid in selected:
        images[fid] = detect.read_image(str(files[fid]), cfg.get('scale_levels'))
    read_s = time.perf_counter() - t0
    params = cfg.get('detector_defaults', {})
    frame_fn = detect2.frame
    if args.candidate == 'marker-reuse':
        # Isolated prototype: do not modify the production module or defaults.
        source = inspect.getsource(detect2.frame)
        old = '    theta, angle_valid, offset, area, bright = _marker(im, xy_s)'
        new = ('    marker_indices = np.searchsorted(cand, sel)\n'
               '    theta, angle_valid, offset, area, bright = [\n'
               '        a[marker_indices] for a in (theta_t, ok_t, off_t, area_t, bright_t)]')
        if source.count(old) != 1:
            raise RuntimeError('prototype anchor changed; inspect reference implementation')
        namespace = dict(vars(detect2))
        exec(compile(source.replace(old, new), '<marker-reuse-prototype>', 'exec'), namespace)
        frame_fn = namespace['frame']
    profile = cProfile.Profile()
    rows = []
    for fid in selected:
        def call():
            return frame_fn(images[fid], mask, dist,
                                 prior_xy=prior.get(fid - 1), **params)
        # Exclude warmup from timings; detect nondeterministic results explicitly.
        baseline, info = call()
        expected = digest(baseline)
        elapsed = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            result, _info = call()
            elapsed.append(time.perf_counter() - t0)
            if digest(result) != expected:
                raise AssertionError('nondeterministic detector frame %d' % fid)
        profile.enable()
        call()
        profile.disable()
        rows.append({'frame': fid, 'detections': len(baseline['xy_px']),
                     'sha256_arrays': expected, 'seconds': elapsed,
                     'median_s': statistics.median(elapsed),
                     'info': {k: v.item() if isinstance(v, np.generic) else v
                              for k, v in info.items()}})
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(30)
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {'schema': 'detector-benchmark/v1', 'created_at_unix': time.time(),
              'commit': commit, 'candidate': args.candidate,
              'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'],
                            cwd=Path(__file__).resolve().parents[1], text=True).strip()),
              'case': str(case.resolve()), 'raw': str(args.raw),
              'mask_sha256': hashlib.sha256(Path(args.mask).read_bytes()).hexdigest(),
              'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                              'numpy': np.__version__, 'scipy': scipy.__version__,
                              'opencv': cv2.__version__, 'opencv_threads': cv2.getNumThreads()},
              'method': 'stored previous-frame detections as fixed prior; not a sequential end-to-end regression',
              'repeats': args.repeats, 'image_read_s': read_s, 'frames': rows,
              'median_frame_s': statistics.median(row['median_s'] for row in rows),
              'profile': stream.getvalue()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(stream.getvalue())
    print('median frame seconds:', report['median_frame_s'])
    print('report:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, type=Path)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--mask', required=True, type=Path)
    p.add_argument('--frames', default='0,189,1085,3605,5500,8350')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--candidate', choices=['reference', 'marker-reuse'], default='reference')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if args.repeats < 2:
        p.error('--repeats must be >= 2')
    run(args)
