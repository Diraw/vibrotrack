"""Paired reference/fast full-state linker benchmark on stored real detections."""
import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
from vibrotrack import link2 as L, relink as R


def fingerprint(frames):
    h = hashlib.sha256()
    for fid, rec in frames:
        h.update(str(fid).encode())
        for k in sorted(rec):
            a = np.ascontiguousarray(rec[k])
            h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def main(args):
    t = time.perf_counter()
    chunks, grouped, scalars, keys, nf = R._read_case(args.case)
    source = [x for g in grouped for x in g][:args.frames]
    load_s = time.perf_counter() - t
    fast = L._assign
    results = []
    expected = None
    for repeat in range(args.repeats):
        row = {}
        for label in (('reference', 'fast') if repeat % 2 == 0 else ('fast', 'reference')):
            frames = copy.deepcopy(source)
            L._assign = L._assign_dense if label == 'reference' else fast
            t = time.perf_counter()
            try:
                stats = R._link_frames(frames, linker='v2', gap=5)
            finally:
                elapsed = time.perf_counter() - t
                L._assign = fast
            digest = fingerprint(frames)
            if expected is None:
                expected = digest
            if digest != expected:
                raise AssertionError('full-state output mismatch')
            row[label] = {'s': elapsed, 'arrays_sha256': digest, 'stats': stats}
        results.append(row)
    report = {'created_at_unix': time.time(), 'case': str(args.case),
              'frames': len(source), 'load_s': load_s, 'repeats': results,
              'median_reference_s': statistics.median(x['reference']['s'] for x in results),
              'median_fast_s': statistics.median(x['fast']['s'] for x in results),
              'all_arrays_exact': True,
              'limitations': 'Full-state linking only; excludes copy/load/hash/write timings. Not extraction end-to-end.'}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='repeats'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
