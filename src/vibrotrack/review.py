"""Optional human review and curation after automatic extraction.

This module deliberately never calls the detector. It consumes an existing
``chunk_*.npz``, proposes frames whose particle count differs from the run mode,
and applies an explicit add/remove action log to a *copy* of the data. The
original extraction stays immutable.

The first version curates detection rows, not trajectories. Existing
``segment_id`` values are preserved for untouched rows; manually added rows get
``segment_id=-1`` and ``manual_edit=1``. A later linker pass can be run on the
curated chunks when trajectory-level analysis is needed.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .orientation import estimate_marker_orientation

QUEUE_SCHEMA = 'review-queue/v1'
ACTIONS_SCHEMA = 'review-actions/v1'
CURATION_SCHEMA = 'curation/v1'

_DETECTION_FIELDS = (
    'xy_px', 'radius_px', 'theta_rad', 'center_valid', 'angle_valid',
    'confidence', 'edge_rmse_px', 'marker_offset_px', 'marker_area_px',
    'neighbor_distance_px', 'boundary_distance_px', 'quality_bits',
    'outside_mask', 'segment_id', 'previous_detection', 'link_valid',
    'gap_frames', 'link_margin_px', 'step_px', 'dtheta_rad',
)


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _chunks(case_dir):
    return sorted(Path(case_dir).glob('chunk_*.npz'))


def _json_number(v):
    v = float(v)
    return None if not np.isfinite(v) else v


def _frame_rows(z, i):
    lo, hi = int(z['frame_offsets'][i]), int(z['frame_offsets'][i + 1])
    rec = {}
    for key in _DETECTION_FIELDS:
        if key in z:
            rec[key] = np.asarray(z[key][lo:hi])
    return rec


def count_stats(case_dir):
    """Return per-frame counts and robust summary statistics."""
    counts = []
    for f in _chunks(case_dir):
        with np.load(f, allow_pickle=False) as z:
            counts.extend(int(x) for x in np.diff(z['frame_offsets']))
    if not counts:
        raise ValueError('no frames found in %s' % case_dir)
    arr = np.asarray(counts, np.int64)
    c = Counter(int(x) for x in arr)
    top = max(c.values())
    modes = sorted(k for k, v in c.items() if v == top)
    expected = int(round(float(np.median(arr)))) if len(modes) > 1 else modes[0]
    return {
        'counts': [int(x) for x in arr],
        'n_frames': int(len(arr)),
        'min': int(arr.min()),
        'max': int(arr.max()),
        'median': float(np.median(arr)),
        'mode': int(expected),
        'mode_frames': int(top),
        'histogram': {str(k): int(v) for k, v in sorted(c.items())},
    }


_HINT_WINDOW = 6      # frames searched on each side of a flagged frame
_HINT_MERGE_PX = 25.0  # two ids closer than this are treated as one broken track


def _window_rows(case_dir, wanted):
    """Return ``{frame: (segment_id, xy_px)}`` for the requested frames only."""
    rows = {}
    for f in _chunks(case_dir):
        with np.load(f, allow_pickle=False) as z:
            ids = np.asarray(z['frame_ids']).reshape(-1)
            off = np.asarray(z['frame_offsets']).astype(np.int64)
            seg = np.asarray(z['segment_id'])
            xy = np.asarray(z['xy_px'], float)
            for i, fid in enumerate(ids):
                fid = int(fid)
                if fid in wanted and fid not in rows:
                    lo, hi = int(off[i]), int(off[i + 1])
                    rows[fid] = (seg[lo:hi].astype(np.int64), xy[lo:hi].astype(float))
        if len(rows) == len(wanted):
            break
    return rows


def missing_hints(flagged, rows, window=_HINT_WINDOW, merge_px=_HINT_MERGE_PX):
    """Guess where a missing particle is, for every frame in ``flagged``.

    Looks at neighbouring frames (up to ``window`` away), collects segment ids
    that appear there but not in the flagged frame, and merges ids sitting at
    almost the same place: a track that broke into two segments shows up twice.
    Returns ``{frame: [{'xy_px', 'segments', 'distance', 'ref_frame'}, ...]}``
    sorted by distance.
    """
    hints = {}
    empty = np.zeros(0, np.int64)
    for fid in flagged:
        present = set(int(v) for v in rows.get(fid, (empty,))[0])
        best = {}
        for g, (seg, xy) in rows.items():
            d = abs(g - fid)
            if d == 0 or d > window:
                continue
            for s, p in zip(seg, xy):
                s = int(s)
                if s < 0 or s in present:
                    continue
                if s not in best or d < best[s][0]:
                    best[s] = (d, [float(p[0]), float(p[1])], int(g))
        merged = []
        for s, (d, p, g) in sorted(best.items(), key=lambda kv: kv[1][0]):
            for m in merged:
                if math.hypot(p[0] - m['xy_px'][0], p[1] - m['xy_px'][1]) <= merge_px:
                    m['segments'].append(s)
                    if d < m['distance']:
                        m['distance'], m['xy_px'], m['ref_frame'] = d, p, g
                    break
            else:
                merged.append({'xy_px': p, 'segments': [s], 'distance': d,
                               'ref_frame': g})
        merged.sort(key=lambda m: m['distance'])
        hints[fid] = merged
    return hints


def build_queue(case_dir, expected_count=None, max_frames=0):
    """Build a review queue for frames whose count differs from the run mode.

    Returns ``(queue, selected)`` where ``selected`` maps frame id to detection
    arrays. The queue itself is JSON-serialisable.
    """
    case_dir = Path(case_dir)
    stats = count_stats(case_dir)
    expected = int(expected_count) if expected_count is not None else int(stats['mode'])
    counts = stats['counts']
    flagged = [i for i, n in enumerate(counts) if int(n) != expected]
    if max_frames and len(flagged) > int(max_frames):
        flagged = flagged[:int(max_frames)]

    selected = {}
    if flagged:
        wanted = set(flagged)
        for f in _chunks(case_dir):
            with np.load(f, allow_pickle=False) as z:
                ids = np.asarray(z['frame_ids']).reshape(-1)
                for i, fid in enumerate(ids):
                    fid = int(fid)
                    if fid in wanted and fid not in selected:
                        selected[fid] = _frame_rows(z, i)
            if len(selected) == len(wanted):
                break
    missing = sorted(set(flagged) - set(selected)) if flagged else []
    if missing:
        raise KeyError('flagged frames not found in chunks: %s' % missing)

    hints = {}
    if flagged:
        window = set()
        for fid in flagged:
            lo = max(0, fid - _HINT_WINDOW)
            hi = min(len(counts) - 1, fid + _HINT_WINDOW)
            window.update(range(lo, hi + 1))
        hints = missing_hints(flagged, _window_rows(case_dir, window))

    all_radii = []
    for rec in selected.values():
        if 'radius_px' in rec and len(rec['radius_px']):
            all_radii.append(np.asarray(rec['radius_px'], float))
    median_radius = float(np.nanmedian(np.concatenate(all_radii))) if all_radii else float('nan')

    frame_list = []
    for fid in sorted(selected):
        rec = selected[fid]
        n = len(rec.get('xy_px', []))
        dets = []
        for i in range(n):
            det = {}
            if 'xy_px' in rec:
                det['xy_px'] = [float(rec['xy_px'][i][0]), float(rec['xy_px'][i][1])]
            for key in ('radius_px', 'theta_rad', 'confidence'):
                if key in rec:
                    det[key] = _json_number(rec[key][i])
            if 'segment_id' in rec:
                det['segment_id'] = int(rec['segment_id'][i])
            if 'angle_valid' in rec:
                det['angle_valid'] = bool(rec['angle_valid'][i])
            if 'center_valid' in rec:
                det['center_valid'] = bool(rec['center_valid'][i])
            dets.append(det)
        frame_list.append({
            'frame': int(fid),
            'count': int(n),
            'expected': expected,
            'delta': int(n) - expected,
            'reason': 'count_mismatch',
            'detections': dets,
            'hints': hints.get(fid, [])[:max(0, expected - n)],
        })

    queue = {
        'schema': QUEUE_SCHEMA,
        'case': case_dir.name,
        'created_at': _now(),
        'expected_count': expected,
        'count_stats': {k: v for k, v in stats.items() if k != 'counts'},
        'median_radius_px': _json_number(median_radius),
        'frames': frame_list,
    }
    return queue, selected


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(path)


def _read_actions(path):
    p = Path(path)
    try:
        obj = json.loads(p.read_text(encoding='utf8'))
    except Exception as exc:
        raise ValueError('cannot read actions JSON: %s' % exc) from exc
    if obj.get('schema') != ACTIONS_SCHEMA:
        raise ValueError('actions schema must be %s' % ACTIONS_SCHEMA)
    actions = obj.get('actions')
    if not isinstance(actions, list):
        raise ValueError('actions JSON must contain an "actions" list')
    cleaned = []
    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            raise ValueError('action %d is not an object' % i)
        op = a.get('op')
        if op not in ('add', 'remove'):
            raise ValueError('action %d has unknown op %r' % (i, op))
        try:
            frame = int(a['frame'])
            x, y = [float(v) for v in a['xy_px']]
        except Exception as exc:
            raise ValueError('action %d needs frame and xy_px=[x,y]' % i) from exc
        aa = dict(a)
        aa.update(op=op, frame=frame, xy_px=[x, y])
        cleaned.append(aa)
    return obj, cleaned, _sha256(p)


def _nearest_row(xy, point):
    if not len(xy):
        return -1, float('inf')
    xy = np.asarray(xy, float).reshape(-1, 2)
    d = np.hypot(xy[:, 0] - point[0], xy[:, 1] - point[1])
    k = int(np.argmin(d))
    return k, float(d[k])


def _field_default(key, n, radius, action=None):
    action = action or {}
    if key == 'xy_px':
        raise ValueError('xy_px must be supplied explicitly')
    if key == 'radius_px':
        return np.full(n, float(radius), np.float32)
    if key == 'theta_rad':
        theta = action.get('theta_rad', np.nan)
        return np.full(n, theta if theta is not None else np.nan, np.float32)
    if key == 'center_valid':
        return np.ones(n, bool)
    if key == 'angle_valid':
        return np.full(n, bool(action.get('angle_valid', False)), bool)
    if key == 'confidence':
        return np.zeros(n, np.float32)
    if key == 'edge_rmse_px':
        return np.full(n, np.nan, np.float32)
    if key == 'marker_offset_px':
        return np.full(n, np.nan, np.float32)
    if key == 'marker_area_px':
        return np.zeros(n, np.int16)
    if key == 'neighbor_distance_px':
        return np.full(n, np.nan, np.float32)
    if key == 'boundary_distance_px':
        return np.full(n, np.nan, np.float32)
    if key == 'quality_bits':
        # bit 0: angle invalid
        return np.full(n, 0 if action.get('angle_valid', False) else 1, np.uint16)
    if key == 'outside_mask':
        return np.zeros(n, np.uint8)
    if key == 'recovered':
        # refine marks auto-recovered rows; a manually added row is not one
        return np.zeros(n, np.uint8)
    if key == 'segment_id':
        return np.full(n, -1, np.int32)
    if key == 'previous_detection':
        return np.full(n, -1, np.int64)
    if key == 'link_valid':
        return np.zeros(n, bool)
    if key == 'gap_frames':
        return np.zeros(n, np.int16)
    if key in ('link_margin_px', 'step_px', 'dtheta_rad'):
        return np.full(n, np.nan, np.float32)
    return np.zeros(n, np.float32)


def _image_shape(case_dir, override=None):
    if override is not None:
        return (int(override[0]), int(override[1]))
    dataset = Path(case_dir) / 'dataset.json'
    if dataset.exists():
        try:
            d = json.loads(dataset.read_text(encoding='utf8'))
            im = d.get('image') or {}
            if im.get('height') and im.get('width'):
                return (int(im['height']), int(im['width']))
        except Exception:
            pass
    return None


def apply_actions(case_dir, actions_path, out_dir, *, tolerance=8.0,
                  image_shape=None, force=False, image_loader=None):
    """Apply review actions to a **copy** of a case.

    The original chunk files are never modified.  Existing rows keep their
    original tracking fields; inserted rows get ``segment_id=-1`` and
    ``manual_edit=1``.  When ``image_loader(frame)`` is supplied, the marker
    direction is estimated from the raw frame around the manual centre; if the
    mark is not found, the row keeps ``angle_valid=False`` and ``theta_rad=NaN``.
    Returns the curation record.
    """
    import shutil

    case_dir = Path(case_dir).resolve()
    out_dir = Path(out_dir).resolve()
    chunks = _chunks(case_dir)
    if not chunks:
        raise FileNotFoundError('no chunk_*.npz in %s' % case_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise FileExistsError('output is not empty (use --force): %s' % out_dir)
        shutil.rmtree(out_dir)
    actions_obj, actions, actions_sha = _read_actions(actions_path)
    shape = _image_shape(case_dir, image_shape)
    by_frame = {}
    for a in actions:
        by_frame.setdefault(int(a['frame']), []).append(a)

    # A global radius fallback, used only if an action has no radius and the
    # target frame has no kept rows.
    radii = []
    for cf in chunks:
        with np.load(cf, allow_pickle=False) as z:
            if 'radius_px' in z and len(z['radius_px']):
                radii.append(np.asarray(z['radius_px'], float))
    global_radius = float(np.nanmedian(np.concatenate(radii))) if radii else 36.0
    image_cache = {}

    def _image_for(frame):
        if image_loader is None:
            return None
        if frame not in image_cache:
            image_cache[frame] = image_loader(int(frame))
        return image_cache[frame]

    out_dir.mkdir(parents=True, exist_ok=True)
    seq_out = out_dir / 'sequences' / case_dir.name
    seq_out.mkdir(parents=True, exist_ok=True)
    applied = []
    skipped = []
    frame_changes = {}
    seen_frames = set()
    outputs = []

    for cf in chunks:
        with np.load(cf, allow_pickle=False) as z:
            ids = np.asarray(z['frame_ids']).reshape(-1)
            offs = np.asarray(z['frame_offsets']).reshape(-1)
            arrays = {k: np.asarray(z[k]) for k in _DETECTION_FIELDS if k in z}
            scalar = {k: np.asarray(z[k]) for k in z.files
                      if k not in set(_DETECTION_FIELDS) | {'frame_ids', 'frame_offsets'}
                      and np.asarray(z[k]).ndim == 0}
        if 'xy_px' not in arrays:
            raise ValueError('chunk has no xy_px: %s' % cf)

        final_rows = []
        for i, fid in enumerate(ids):
            fid = int(fid)
            seen_frames.add(fid)
            lo, hi = int(offs[i]), int(offs[i + 1])
            original = list(range(lo, hi))
            acts = by_frame.get(fid, [])
            removed = set()

            for a in acts:
                if a['op'] != 'remove':
                    continue
                candidates = [j for j in original if j not in removed]
                if not candidates:
                    skipped.append({'frame': fid, 'op': 'remove',
                                    'xy_px': a['xy_px'],
                                    'reason': 'no detection left to remove'})
                    continue
                xy = arrays['xy_px'][candidates]
                k, d = _nearest_row(xy, a['xy_px'])
                if k < 0 or d > float(tolerance):
                    skipped.append({'frame': fid, 'op': 'remove',
                                    'xy_px': a['xy_px'],
                                    'reason': 'no detection within tolerance'})
                    continue
                j = candidates[k]
                removed.add(j)
                applied.append({'frame': fid, 'op': 'remove',
                                'xy_px': [float(x) for x in arrays['xy_px'][j]],
                                'matched_distance_px': d})

            kept = [j for j in original if j not in removed]
            additions = []
            for a in acts:
                if a['op'] != 'add':
                    continue
                p = np.asarray(a['xy_px'], float)
                if shape is not None:
                    h, w = shape
                    if not (0 <= p[0] < w and 0 <= p[1] < h):
                        skipped.append({'frame': fid, 'op': 'add',
                                        'xy_px': a['xy_px'],
                                        'reason': 'outside image'})
                        continue
                check_xy = []
                if kept:
                    check_xy.append(arrays['xy_px'][kept])
                if additions:
                    check_xy.append(np.asarray([x['xy_px'] for x in additions], float))
                if check_xy:
                    _, d = _nearest_row(np.concatenate(check_xy, axis=0), p)
                    if d <= float(tolerance):
                        skipped.append({'frame': fid, 'op': 'add',
                                        'xy_px': a['xy_px'],
                                        'reason': 'duplicate of existing detection'})
                        continue
                r = float(a.get('radius_px') or 0.0)
                if not np.isfinite(r) or r <= 0:
                    rr = arrays['radius_px'][kept] if kept and 'radius_px' in arrays else []
                    r = float(np.nanmedian(rr)) if len(rr) else global_radius
                theta = a.get('theta_rad')
                angle_valid = bool(a.get('angle_valid', False)) and theta is not None
                if theta is None and image_loader is not None:
                    try:
                        image = _image_for(fid)
                        th, ok, off, area, bright = estimate_marker_orientation(
                            image, [p])
                        if bool(ok[0]):
                            theta = float(th[0])
                            angle_valid = True
                    except Exception:
                        # A missing/unreadable raw frame must not lose the manual
                        # centre; keep it with an explicit invalid orientation.
                        angle_valid = False
                if theta is not None and not np.isfinite(float(theta)):
                    theta = None
                    angle_valid = False
                addition = {'frame': fid, 'op': 'add',
                            'xy_px': [float(p[0]), float(p[1])],
                            'radius_px': float(r),
                            'theta_rad': theta,
                            'angle_valid': bool(angle_valid),
                            'reason': a.get('reason', 'manual_add')}
                additions.append(addition)
                applied.append({'frame': fid, 'op': 'add',
                                'xy_px': addition['xy_px'],
                                'radius_px': addition['radius_px'],
                                'angle_valid': addition['angle_valid'],
                                'theta_rad': addition['theta_rad']})

            before = len(original)
            after = len(kept) + len(additions)
            if before != after:
                frame_changes[str(fid)] = {'before': before, 'after': after,
                                           'delta': after - before}
            for j in kept:
                final_rows.append(('auto', j))
            for addition in additions:
                final_rows.append(('add', addition))

        new_arrays = {}
        for key, arr in arrays.items():
            parts = []
            for kind, value in final_rows:
                if kind == 'auto':
                    parts.append(arr[value:value + 1])
                elif key == 'xy_px':
                    parts.append(np.asarray([value['xy_px']], np.float32))
                else:
                    default = _field_default(key, 1, value['radius_px'], value)
                    parts.append(default.reshape(1, *arr.shape[1:]))
            new_arrays[key] = np.concatenate(parts, axis=0) if parts else arr[:0]

        new_arrays['manual_edit'] = np.asarray(
            [1 if kind == 'add' else 0 for kind, _ in final_rows], np.uint8)
        # Rebuild offsets from the final row order.  Every source frame is kept,
        # even if a reviewer removed every detection in it.  Use an explicit
        # frame map so zero-row frames and inserted rows are unambiguous.
        frame_of = {}
        for i, fid in enumerate(ids):
            for j in range(int(offs[i]), int(offs[i + 1])):
                frame_of[('auto', j)] = int(fid)
        for kind, value in final_rows:
            if kind == 'add':
                frame_of[('add', id(value))] = int(value['frame'])
        counts = [0] * len(ids)
        for kind, value in final_rows:
            key = ('auto', value) if kind == 'auto' else ('add', id(value))
            frame_index = int(np.where(ids == frame_of[key])[0][0])
            counts[frame_index] += 1
        offsets = [0]
        for n in counts:
            offsets.append(offsets[-1] + n)
        new_arrays['frame_ids'] = np.asarray(ids, np.int64)
        new_arrays['frame_offsets'] = np.asarray(offsets, np.int32)
        for key, value in scalar.items():
            new_arrays[key] = value
        out_chunk = seq_out / cf.name
        np.savez_compressed(out_chunk, **new_arrays)
        outputs.append(out_chunk)

    # Action frames absent from every chunk are explicit skips, not silent.
    for fid, acts in sorted(by_frame.items()):
        if fid in seen_frames:
            continue
        for a in acts:
            skipped.append({'frame': fid, 'op': a['op'], 'xy_px': a['xy_px'],
                            'reason': 'frame not found in chunks'})

    for name in ('dataset.json', 'run_config.json'):
        src = case_dir / name
        if src.exists():
            (seq_out / name).write_bytes(src.read_bytes())
    review = {
        'schema': CURATION_SCHEMA,
        'case': case_dir.name,
        'created_at': _now(),
        'actions_file': Path(actions_path).name,
        'actions_sha256': actions_sha,
        'source_case': str(case_dir),
        'source_chunks': [{'file': p.name, 'sha256': _sha256(p)} for p in chunks],
        'tolerance_px': float(tolerance),
        'n_actions': len(actions),
        'n_applied': len(applied),
        'n_skipped': len(skipped),
        'applied': applied,
        'skipped': skipped,
        'frame_changes': frame_changes,
        'outputs': [{'file': p.name, 'bytes': p.stat().st_size,
                     'sha256': _sha256(p)} for p in outputs],
        'note': ('Curation preserves automatic segment ids for untouched rows; '
                 'manual rows have segment_id=-1; angle_valid is true only if the marker estimator succeeded. '
                 'Re-run the linker before using trajectories.'),
    }
    write_json(seq_out / 'review.json', review)
    write_json(out_dir / 'review.json', review)
    dataset = seq_out / 'dataset.json'
    if dataset.exists():
        try:
            d = json.loads(dataset.read_text(encoding='utf8'))
            d['outputs'] = [{'file': p.name, 'bytes': p.stat().st_size,
                             'sha256': _sha256(p)} for p in outputs]
            d['qc'] = dict(d.get('qc') or {})
            d['qc']['status'] = 'curated'
            d['curation'] = {'schema': CURATION_SCHEMA, 'review': 'review.json',
                             'actions_sha256': actions_sha,
                             'n_applied': len(applied), 'n_skipped': len(skipped)}
            write_json(dataset, d)
        except Exception:
            pass
    return review
