"""Paired full refine benchmark, creating new evidence outputs only."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
from vibrotrack import link2 as L, refine as R


def arrays_digest(case):
    h = hashlib.sha256()
    for p in sorted(case.glob('chunk_*.npz')):
        with np.load(p) as z:
            for k in sorted(z.files):
                a = np.ascontiguousarray(z[k])
                h.update(k.encode());h.update(str(a.dtype).encode())
                h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def main(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    fast = L._assign
    expected = None
    runs = []
    for i in range(3):
        pair = {}
        for label in (('reference', 'fast') if i % 2 == 0 else ('fast', 'reference')):
            L._assign = L._assign_dense if label == 'reference' else fast
            t = time.perf_counter()
            try:
                rec, case = R.refine_case(args.case, root / ('%d_%s' % (i,label)),
                                          args.raw, max_gap=10)
            finally:
                elapsed = time.perf_counter() - t
                L._assign = fast
            digest = arrays_digest(case)
            if expected is None:
                expected = digest
            if digest != expected:
                raise AssertionError('refine output differs')
            pair[label] = {'s':elapsed,'arrays_sha256':digest,
                           'case':str(case), 'stats': {k:rec[k] for k in
                           ['n_attempts','n_recovered','n_detections','n_tracks','n_gap_rows']}}
            print(i,label,elapsed,flush=True)
        runs.append(pair)
    repo=Path(__file__).resolve().parents[1]
    report={'created_at_unix':time.time(),'environment':platform.platform(),
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [repo/'src/vibrotrack/link2.py',repo/'src/vibrotrack/refine.py']},
            'repeats':runs,'all_output_arrays_exact':True,
            'median_reference_s':statistics.median(x['reference']['s'] for x in runs),
            'median_fast_s':statistics.median(x['fast']['s'] for x in runs),
            'method':'AB/BA/AB full refine including read, gap detection, local pixel fitting, full-state linking, hashing/write; output comparison excluded'}
    (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='repeats'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,type=Path)
    p.add_argument('--raw',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    main(p.parse_args())
