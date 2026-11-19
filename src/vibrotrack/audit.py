"""Audit a finished case for 'triangle void' false centres.

Three touching discs meet at a curvilinear-triangle gap whose three rims close a ring
that the fixed-radius fit accepts.  The resulting centre sits 1.0-1.3 R from the three
real particles; the hard-disc NMS then deletes them, so the frame loses 2-3 real
particles and gains one phantom.  The **count** stays plausible, which is why a
count-based gate cannot see it.

The phantom is identifiable from the marker pass on the *final* detection set: its
dark blob is the gap itself, i.e. centred and small, whereas a real particle's slot
sits ~11 px off centre and is about twice as large.

    46 user-confirmed voids: area 67-141, offset 0.5-8.4, confidence 0.22-0.37
    16.2 M real detections   : area p1 132, offset p1 7.0

So the audit flags `marker_area_px < 145 and marker_offset_px < 9 and
confidence < 0.40`.  It only *reports* (and writes a remove-action file); removing
them is `vibrotrack curate`, and the real neighbours come back through
`vibrotrack refine` (track prediction, 0.6 px from manual clicks on 39/39 points).

The rule must NOT go into the detector: applied to candidates before the NMS it cost
1,688 detections per 27,000-frame case for 13 fixed frames, because a real particle
whose slot is half covered has the same signature.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import schema

AUDIT_SCHEMA = 'audit/voids-v1'

DEFAULTS = dict(area_max=145.0, offset_max=9.0, conf_max=0.40)


def scan(case_dir, area_max=None, offset_max=None, conf_max=None, min_frame=0):
    """Return the void candidates of one case (no files written)."""
    case_dir = Path(case_dir)
    chunks = sorted(case_dir.glob('chunk_*.npz'))
    if not chunks:
        raise FileNotFoundError('no chunk_*.npz in %s' % case_dir)
    p = dict(DEFAULTS)
    if area_max is not None:
        p['area_max'] = float(area_max)
    if offset_max is not None:
        p['offset_max'] = float(offset_max)
    if conf_max is not None:
        p['conf_max'] = float(conf_max)

    rows = []
    counts = []
    for f in chunks:
        with np.load(f, allow_pickle=False) as z:
            off = np.asarray(z['frame_offsets']).astype(np.int64)
            xy = np.asarray(z['xy_px'], float)
            area = np.asarray(z['marker_area_px'])
            moff = np.asarray(z['marker_offset_px'])
            conf = np.asarray(z['confidence'])
            seg = np.asarray(z['segment_id'])
        counts.append(np.diff(off))
        hit = np.flatnonzero((area < p['area_max']) & (moff < p['offset_max']) &
                             (conf < p['conf_max']))
        for j in hit:
            j = int(j)
            frame = int(np.searchsorted(off, j, side='right') - 1)
            if frame < int(min_frame):
                continue
            lo, hi = int(off[frame]), int(off[frame + 1])
            rows.append({
                'frame': frame,
                'xy_px': [round(float(xy[j][0]), 2), round(float(xy[j][1]), 2)],
                'marker_area_px': int(area[j]),
                'marker_offset_px': round(float(moff[j]), 2),
                'confidence': round(float(conf[j]), 3),
                'segment_id': int(seg[j]),
                'frame_count': int(hi - lo),
            })
    counts = np.concatenate(counts) if counts else np.zeros(0, np.int64)
    return {'params': p, 'counts': counts, 'rows': rows, 'chunks': [c.name for c in chunks]}


def summary(result):
    counts = result['counts']
    rows = result['rows']
    frames = sorted({r['frame'] for r in rows})
    return {
        'n_frames': int(len(counts)),
        'count_min': int(counts.min()) if len(counts) else 0,
        'count_median': float(np.median(counts)) if len(counts) else 0,
        'count_max': int(counts.max()) if len(counts) else 0,
        'n_voids': len(rows),
        'n_frames_with_a_void': len(frames),
        'void_frames': frames,
    }


def write(case_dir, out_dir, result):
    """Write audit.json and a curate-ready remove-action file."""
    case_dir, out_dir = Path(case_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = case_dir.name
    rec = {
        'schema': AUDIT_SCHEMA,
        'created_at': schema.now_iso(),
        'case': case,
        'source_case': str(case_dir),
        'source_chunks': result['chunks'],
        'params': result['params'],
        'summary': summary(result),
        'voids': result['rows'],
        'note': ('remove these through `vibrotrack curate`, then run `vibrotrack refine`; '
                 'the three real neighbours come back from the track prediction'),
    }
    audit_path = out_dir / 'audit.json'
    schema.write_json(audit_path, rec)
    actions = {
        'schema': 'review-actions/v1',
        'case': case,
        'actions': [{'frame': r['frame'], 'op': 'remove', 'xy_px': r['xy_px'],
                     'reason': 'triangle_void area=%d offset=%.1f conf=%.2f'
                               % (r['marker_area_px'], r['marker_offset_px'], r['confidence'])}
                    for r in result['rows']],
    }
    act_path = out_dir / 'review_actions.json'
    schema.write_json(act_path, actions)
    return audit_path, act_path, rec
