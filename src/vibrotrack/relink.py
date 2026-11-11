"""Recompute trajectory identities on an existing case without detecting again.

The curated data produced by :mod:`vibrotrack.review` keeps all detections and
orientations, but manual rows deliberately enter with ``segment_id=-1``.  This
module re-runs the identity linker over the ordered detection rows in a case and
writes a new copy.  It never reads raw frames, never calls the detector, and
never changes particle positions, radii or angles.

The input may be a case directory (``.../sequences/<case>``) or a batch
directory containing exactly one case.  Output uses the same
``<out>/sequences/<case>/chunk_*.npz`` layout as extraction/curation.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import detect as X
from . import link2 as L2

RELINK_SCHEMA = 'relink/v1'


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


def resolve_case_dir(path):
    """Accept a case directory or a batch directory with exactly one case."""
    p = Path(path).expanduser().resolve()
    if _chunks(p):
        return p
    seq = p / 'sequences'
    if seq.is_dir():
        cases = [d for d in sorted(seq.iterdir())
                 if d.is_dir() and _chunks(d)]
        if len(cases) == 1:
            return cases[0]
        if not cases:
            raise FileNotFoundError('no case with chunk_*.npz under %s' % seq)
        raise ValueError('%s contains %d cases; pass one sequence directory'
                         % (seq, len(cases)))
    raise FileNotFoundError('no chunk_*.npz in %s or %s/sequences' % (p, p))


def _case_defaults(case_dir):
    """Read linker/gap from run_config.json, then dataset.json."""
    linker, gap = 'v2', 5
    for name in ('run_config.json', 'dataset.json'):
        p = Path(case_dir) / name
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding='utf8'))
        except Exception:
            continue
        if 'linker' in obj:
            linker = str(obj['linker']).lower()
        if 'link_gap' in obj:
            gap = int(obj['link_gap'])
        # The first file is the authoritative run configuration.
        break
    return linker, gap


def _read_case(case_dir):
    """Load all per-frame detection records and scalar chunk metadata."""
    chunks = _chunks(case_dir)
    if not chunks:
        raise FileNotFoundError('no chunk_*.npz in %s' % case_dir)

    chunk_frames = []
    chunk_scalars = []
    all_keys = None
    last_frame = None
    all_ids = []

    for cf in chunks:
        with np.load(cf, allow_pickle=False) as z:
            if 'frame_ids' not in z or 'frame_offsets' not in z:
                raise ValueError('chunk missing frame_ids/frame_offsets: %s' % cf)
            ids = np.asarray(z['frame_ids']).reshape(-1).astype(np.int64)
            offs = np.asarray(z['frame_offsets']).reshape(-1).astype(np.int64)
            if len(offs) != len(ids) + 1:
                raise ValueError('frame_offsets do not match frame_ids: %s' % cf)
            n = int(offs[-1]) if len(offs) else 0
            if len(offs) and (offs[0] != 0 or offs[-1] != n or np.any(np.diff(offs) < 0)):
                raise ValueError('invalid frame_offsets: %s' % cf)
            # Detection arrays have one row per detection. Scalar per-chunk
            # metadata (schema_version, wall_model, ...) is copied unchanged.
            row_keys = [k for k in z.files
                        if k not in ('frame_ids', 'frame_offsets')
                        and np.asarray(z[k]).ndim >= 1
                        and np.asarray(z[k]).shape[0] == n]
            scalar = {k: np.asarray(z[k]).copy() for k in z.files
                      if k not in ('frame_ids', 'frame_offsets')
                      and k not in row_keys}
            keys = set(row_keys)
            if all_keys is None:
                all_keys = keys
            elif keys != all_keys:
                raise ValueError('chunk field set differs: %s' % cf)
            if 'xy_px' not in keys:
                raise ValueError('chunk has no xy_px: %s' % cf)
            for need in ('radius_px', 'theta_rad', 'angle_valid'):
                if need not in keys:
                    raise ValueError('chunk has no %s: %s' % (need, cf))

            frames = []
            for i, fid in enumerate(ids):
                fid = int(fid)
                if last_frame is not None and fid <= last_frame:
                    raise ValueError('frame ids must be strictly increasing; '
                                     'duplicate/backward frame %d in %s' % (fid, cf))
                last_frame = fid
                lo, hi = int(offs[i]), int(offs[i + 1])
                rec = {k: np.asarray(z[k][lo:hi]).copy() for k in row_keys}
                frames.append((fid, rec))
                all_ids.append(fid)
            chunk_frames.append(frames)
            chunk_scalars.append(scalar)

    if all_keys is None:
        all_keys = set()
    return chunks, chunk_frames, chunk_scalars, sorted(all_keys), len(all_ids)


def _link_frames(frames, linker, gap):
    """Return linked records and aggregate statistics.

    ``frames`` is a list of (fid, record). Records are modified in place only in
    the tracking fields; all other arrays are left untouched.
    """
    linker = str(linker).lower()
    if linker not in ('v1', 'v2'):
        raise ValueError('linker must be v1 or v2, got %r' % linker)
    if int(gap) < 1:
        raise ValueError('link_gap must be >= 1')

    state = L2.new_state(gap=int(gap)) if linker == 'v2' else None
    nextid = 0
    prev = None
    prev_start = 0
    global_start = 0
    total_rows = 0
    n_linked = 0
    n_gap = 0
    n_manual = 0
    manual_rows = []
    steps = []
    all_ids = []

    for fid, rec in frames:
        xy = np.asarray(rec['xy_px'], np.float64)
        n = len(xy)
        if 'radius_px' in rec and n:
            rmed = float(np.nanmedian(np.asarray(rec['radius_px'], float)))
            if not np.isfinite(rmed) or rmed <= 0:
                rmed = 36.9
        else:
            rmed = 36.9

        if linker == 'v2':
            ids, plocal, gaps, step = L2.link_frame(
                state, int(fid), xy, rmed)
            prev_det = np.where(plocal >= 0, prev_start + plocal, -1).astype(np.int64)
            link_valid = np.asarray(gaps == 1, bool)
            gap_frames = np.asarray(gaps, np.int16)
            margin = np.full(n, np.nan, np.float32)
            dtheta = np.full(n, np.nan, np.float32)
            if prev is not None and n:
                direct = link_valid & (plocal >= 0)
                if direct.any():
                    ptheta = np.asarray(prev['theta_rad'], float)[plocal[direct]]
                    ctheta = np.asarray(rec['theta_rad'], float)[direct]
                    pok = np.asarray(prev['angle_valid'], bool)[plocal[direct]]
                    cok = np.asarray(rec['angle_valid'], bool)[direct]
                    ok = pok & cok & np.isfinite(ptheta) & np.isfinite(ctheta)
                    dd = np.arctan2(np.sin(ctheta - ptheta),
                                    np.cos(ctheta - ptheta))
                    dd[~ok] = np.nan
                    dtheta[direct] = dd.astype(np.float32)
        else:
            cur = {k: rec[k] for k in
                   ('xy_px', 'radius_px', 'theta_rad', 'angle_valid')}
            # v1 has no gap closing; it only links adjacent frames with a
            # mutual nearest neighbour and a clear gap.
            nextid = X.link(prev, cur, nextid, prev_start)
            ids = np.asarray(cur['segment_id'])
            prev_det = np.asarray(cur['previous_detection'], np.int64)
            link_valid = np.asarray(cur['link_valid'], bool)
            gap_frames = np.where(link_valid, 1, 0).astype(np.int16)
            margin = np.asarray(cur['link_margin_px'], np.float32)
            step = np.asarray(cur['step_px'], np.float32)
            dtheta = np.asarray(cur['dtheta_rad'], np.float32)

        ids = np.asarray(ids, np.int32)
        rec['segment_id'] = ids
        rec['previous_detection'] = prev_det
        rec['link_valid'] = link_valid
        rec['gap_frames'] = gap_frames
        rec['link_margin_px'] = np.asarray(margin, np.float32)
        rec['step_px'] = np.asarray(step, np.float32)
        rec['dtheta_rad'] = np.asarray(dtheta, np.float32)

        if 'manual_edit' in rec:
            medit = np.asarray(rec['manual_edit'], np.uint8)
            n_manual += int(medit.sum())
            for j in np.where(medit == 1)[0]:
                manual_rows.append({
                    'frame': int(fid),
                    'xy_px': [float(v) for v in rec['xy_px'][j]],
                    'segment_id': int(rec['segment_id'][j]),
                    'link_valid': bool(rec['link_valid'][j]),
                    'gap_frames': int(rec['gap_frames'][j]),
                })
        n_linked += int(link_valid.sum())
        n_gap += int((gap_frames > 1).sum())
        valid_steps = np.asarray(step, float)
        steps.append(valid_steps[np.isfinite(valid_steps)])
        all_ids.append(ids[ids >= 0])
        prev = rec
        prev_start = global_start
        global_start += n
        total_rows += n

    track_ids = np.unique(np.concatenate(all_ids)) if all_ids else np.empty(0, np.int64)
    finite_steps = np.concatenate(steps) if steps else np.empty(0, np.float64)
    return {
        'n_frames': len(frames),
        'n_detections': int(total_rows),
        'n_tracks': int(len(track_ids)),
        'n_linked_rows': int(n_linked),
        'n_gap_rows': int(n_gap),
        'n_manual_rows': int(n_manual),
        'manual_rows': manual_rows,
        'link_fraction': float(n_linked / total_rows) if total_rows else 0.0,
        'median_step_px': float(np.median(finite_steps)) if len(finite_steps) else None,
    }


def _write_chunks(chunk_frames, source_chunks, seq_out, keys, chunk_scalars):
    """Write linked rows back into the source chunk layout."""
    outputs = []
    seq_out.mkdir(parents=True, exist_ok=True)
    for ci, (cf, frames) in enumerate(zip(source_chunks, chunk_frames)):
        out = seq_out / cf.name
        ids = np.asarray([fid for fid, _ in frames], np.int64)
        counts = [len(rec['xy_px']) for _, rec in frames]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        arrays = {'frame_ids': ids, 'frame_offsets': offsets}
        for key in keys:
            parts = []
            for _, rec in frames:
                if key not in rec:
                    raise KeyError('missing %s in one frame of %s' % (key, cf.name))
                parts.append(rec[key])
            if parts:
                arrays[key] = np.concatenate(parts, axis=0)
            else:
                # This should only occur for a frame-less chunk; retain the
                # source dtype and trailing shape rather than guessing.
                with np.load(cf, allow_pickle=False) as z:
                    a = np.asarray(z[key])
                arrays[key] = np.empty((0,) + a.shape[1:], dtype=a.dtype)
        arrays.update(chunk_scalars[ci])
        np.savez_compressed(out, **arrays)
        outputs.append(out)
    return outputs


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf8')
    tmp.replace(path)


def relink_case(case_dir, out_root, *, linker=None, link_gap=None, force=False):
    """Re-link one case and write a new batch-style output directory.

    Parameters
    ----------
    case_dir:
        Case directory containing ``chunk_*.npz``, or a batch directory with
        exactly one case.
    out_root:
        New batch root. Products are written to ``out_root/sequences/<case>``.
    linker, link_gap:
        Override the values recorded in ``run_config.json``.
    force:
        Replace an existing output case directory.
    """
    case_dir = resolve_case_dir(case_dir)
    out_root = Path(out_root).expanduser().resolve()
    seq_out = out_root / 'sequences' / case_dir.name
    if seq_out.resolve() == case_dir.resolve():
        raise ValueError('choose a new --out; in-place relinking is not allowed')
    if out_root == case_dir:
        raise ValueError('choose a new --out; in-place relinking is not allowed')
    if seq_out.exists() and any(seq_out.iterdir()):
        if not force:
            raise FileExistsError('output case is not empty (use --force): %s' % seq_out)
        shutil.rmtree(seq_out)
    seq_out.mkdir(parents=True, exist_ok=True)

    defaults_linker, defaults_gap = _case_defaults(case_dir)
    linker = str(linker or defaults_linker).lower()
    gap = int(defaults_gap if link_gap is None else link_gap)
    if linker not in ('v1', 'v2'):
        raise ValueError('linker must be v1 or v2, got %r' % linker)
    if gap < 1:
        raise ValueError('link_gap must be >= 1')

    chunks, chunk_frames, chunk_scalars, keys, n_frames = _read_case(case_dir)
    source_hashes = [{'file': p.name, 'sha256': _sha256(p)} for p in chunks]
    frames = [item for group in chunk_frames for item in group]
    stats = _link_frames(frames, linker=linker, gap=gap)

    outputs = _write_chunks(chunk_frames, chunks, seq_out, keys, chunk_scalars)
    output_records = [{'file': p.name, 'bytes': p.stat().st_size,
                       'sha256': _sha256(p)} for p in outputs]

    # Preserve dataset/run configuration, review audit and any sidecars.
    for src in case_dir.iterdir():
        if src.is_file() and not src.name.startswith('chunk_'):
            shutil.copy2(src, seq_out / src.name)

    record = {
        'schema': RELINK_SCHEMA,
        'case': case_dir.name,
        'created_at': _now(),
        'source_case': str(case_dir),
        'source_chunks': source_hashes,
        'linker': linker,
        'link_gap': gap,
        **stats,
        'outputs': output_records,
        'note': ('Only segment_id, previous_detection, link_valid, gap_frames, '
                 'link_margin_px, step_px and dtheta_rad are recomputed; '
                 'positions, radii, angles and manual_edit are unchanged.'),
    }
    relink_path = seq_out / 'relink.json'
    _write_json(relink_path, record)
    relink_entry = {'file': relink_path.name, 'bytes': relink_path.stat().st_size,
                    'sha256': _sha256(relink_path)}

    dataset_path = seq_out / 'dataset.json'
    if dataset_path.exists():
        try:
            d = json.loads(dataset_path.read_text(encoding='utf8'))
        except Exception:
            d = {}
        d['outputs'] = output_records + [relink_entry]
        d['qc'] = dict(d.get('qc') or {})
        d['qc']['status'] = ('curated+relinked' if d.get('curation')
                             else 'relinked')
        d['relink'] = {
            'schema': RELINK_SCHEMA,
            'file': relink_path.name,
            'linker': linker,
            'link_gap': gap,
            'n_tracks': stats['n_tracks'],
            'n_linked_rows': stats['n_linked_rows'],
            'n_gap_rows': stats['n_gap_rows'],
        }
        _write_json(dataset_path, d)

    full_path = seq_out / 'full_summary.json'
    if full_path.exists():
        try:
            s = json.loads(full_path.read_text(encoding='utf8'))
        except Exception:
            s = {}
        s['linker'] = linker
        s['link_fraction'] = stats['link_fraction']
        s['segments'] = stats['n_tracks']
        s['gap_closed'] = stats['n_gap_rows']
        s['median_step_px'] = stats['median_step_px']
        s['relinked_at'] = record['created_at']
        _write_json(full_path, s)
    return record, seq_out
