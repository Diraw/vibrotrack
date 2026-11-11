"""Recover particles that a linked track proves must exist but the detector dropped.

Why this exists
---------------
On the dense ``G0`` case 66 particles were missing from single frames.  All 66 sit
at hard-disc contact (nearest neighbour 62-75 px for a 73 px diameter) and move
6-8 px per frame, so the detector loses them twice over: the template peak of a
bead squeezed against a neighbour falls below ``thresh``, and the prior seed from
the previous frame is rejected by ``max_shift`` (5 px) before the fit can settle.
Fit quality was never the problem - a fixed-radius fit started at the true centre
returns 35+ of 64 rays with rmse 0.8 px.

A track with detections on both sides of the hole still knows where the particle
was to within ~8 px, because the gap is 1-3 frames and the velocity is smooth.
This module uses that prediction only as a *seed*: it crops a small window around
it, runs the same fixed-radius radial fit as the detector, and accepts the result
only when it converges, keeps 14+ rays and stays within 12 px of the prediction.
No position is ever interpolated into the output.

The pass is additive: existing detection values are preserved and new rows carry
``recovered=1``. Identities are recomputed by the reference linker. Former local
splicing was not equivalent to replaying velocity and competitive assignments;
``incremental`` is retained as a compatibility request but falls back to full
linking, with the reason recorded. A failed fit remains a hole for optional review.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import detect as X
from . import detect2 as D2
from . import relink as R

REFINE_SCHEMA = 'refine/v1'

DEFAULTS = dict(
    max_gap=5,        # only holes this many frames long or shorter (linker gap)
    min_rays=14,      # detector gate; the local window is much smaller than a full frame
    max_rmse=1.2,     # detector gate is 1.5
    max_offset=12.0,  # fit must stay this close to the prediction
    min_sep=1.6,      # reject a fit that duplicates an existing centre (<1.6 r)
    half_window=46,   # 91x91 patch: leaves 9 px around a 37 px disc rim
    peak_thresh=0.12,  # template peaks join the seed list at a lower bar
    iterations=4,     # restarts until the centre stops moving
    converge_px=0.5,
    boundary_gate=14.0,   # fragment ends/starts: prediction must land this close
    boundary_growth=4.0,  # extra slack per skipped frame beyond the first
)


# --------------------------------------------------------------------------- gaps
def _velocity(pts):
    """Per-frame velocity from the last two rows of a track."""
    if len(pts) < 2:
        return np.zeros(2)
    (f0, p0), (f1, p1) = pts[-2], pts[-1]
    if f1 <= f0:
        return np.zeros(2)
    return (np.asarray(p1, float) - np.asarray(p0, float)) / float(f1 - f0)


def find_gaps(frames, max_gap=5, boundary_gate=14.0, boundary_growth=4.0):
    """Return every place a track proves a particle must exist but none was detected.

    ``frames`` is the ``[(fid, record), ...]`` list produced by
    :func:`vibrotrack.relink._read_case`.  Two kinds of hole are returned:

    * **interior**: two consecutive rows of the same ``segment_id`` separated by
      2..max_gap+1 frames.  No pairing decision is needed - the identity is known.
    * **boundary**: an end of one fragment and the start of another that the
      linker could not join because the bridging detection is missing.  The
      pairing uses the same evidence the linker would: the end's velocity
      predicts the start's position to within ``boundary_gate`` px (plus
      ``boundary_growth`` per extra skipped frame).  This is what the raw
      (pre-relink) extract needs: 520 fragments, one broken at every hole.
    """
    if not frames:
        return []
    max_gap = int(max_gap)
    counts = np.asarray([len(rec['xy_px']) for _fid, rec in frames], np.int64)
    fid_all = np.repeat(np.asarray([int(fid) for fid, _ in frames], np.int64), counts)
    seg_all = np.concatenate([np.asarray(rec['segment_id'], np.int64)
                              for _fid, rec in frames])
    xy_all = np.concatenate([np.asarray(rec['xy_px'], float).reshape(-1, 2)
                             for _fid, rec in frames])
    keep = seg_all >= 0
    seg_all, fid_all, xy_all = seg_all[keep], fid_all[keep], xy_all[keep]
    if not len(seg_all):
        return []
    order = np.argsort(seg_all, kind='stable')
    seg_s, fid_s, xy_s = seg_all[order], fid_all[order], xy_all[order]
    uniq, first, npts = np.unique(seg_s, return_index=True, return_counts=True)
    last = first + npts - 1

    gaps = []
    # Interior holes: adjacent rows of one track separated by 2..max_gap+1 frames.
    same = seg_s[1:] == seg_s[:-1]
    span = fid_s[1:] - fid_s[:-1]
    for i in np.flatnonzero(same & (span > 1) & (span <= max_gap + 1)):
        i = int(i)
        gaps.append({'track': int(seg_s[i]), 'f_lo': int(fid_s[i]),
                     'f_hi': int(fid_s[i + 1]), 'xy_lo': xy_s[i].copy(),
                     'xy_hi': xy_s[i + 1].copy(),
                     'frames': list(range(int(fid_s[i]) + 1, int(fid_s[i + 1]))),
                     'kind': 'interior'})

    # Fragment ends and starts.
    first_fid, end_fid = fid_s[first], fid_s[last]
    first_xy, end_xy = xy_s[first], xy_s[last]
    vel = np.zeros((len(uniq), 2), float)
    two = npts >= 2
    if two.any():
        f1, f0 = fid_s[last[two]], fid_s[last[two] - 1]
        p1, p0 = xy_s[last[two]], xy_s[last[two] - 1]
        step = (f1 - f0).astype(float)
        good = step > 0
        v = np.zeros_like(p1)
        v[good] = (p1[good] - p0[good]) / step[good, None]
        vel[two] = v

    # Tail: a track that stops within max_gap frames of the last frame.  In a closed
    # cell a particle cannot leave, so the last frames are the place where a missed
    # detection has no "other side" to pair with.
    rows_by_frame = {int(fid): np.asarray(rec['xy_px'], float).reshape(-1, 2)
                     for fid, rec in frames}
    # fid_s is grouped by identity, not chronology; include empty final frames.
    last_fid = max(int(fid) for fid, _rec in frames)
    for i in np.flatnonzero((end_fid < last_fid) & (last_fid - end_fid <= max_gap)):
        i = int(i)
        tid, f0 = int(uniq[i]), int(end_fid[i])
        xy0, v = end_xy[i], vel[i]
        for k in range(1, last_fid - f0 + 1):
            pred = xy0 + v * k
            rows = rows_by_frame.get(f0 + k)
            if rows is not None and len(rows):
                # the particle may simply be an unlinked row in that frame
                if np.hypot(rows[:, 0] - pred[0], rows[:, 1] - pred[1]).min() < 25.0:
                    continue
            gaps.append({'track': tid, 'f_lo': f0, 'f_hi': f0 + k,
                         'xy_lo': xy0.copy(), 'xy_hi': (xy0 + v * k).copy(),
                         'frames': [f0 + k], 'kind': 'tail'})

    # Boundary holes: an end and a later start that the linker could not join.
    for i in range(len(uniq)):
        span = first_fid - end_fid[i]
        ok = (span > 1) & (span <= max_gap + 1) & (uniq != uniq[i])
        if not ok.any():
            continue
        sel = np.flatnonzero(ok)
        k = span[sel].astype(float)
        pred = end_xy[i] + vel[i] * k[:, None]
        gate = float(boundary_gate) + float(boundary_growth) * (k - 2)
        d = np.hypot(pred[:, 0] - first_xy[sel, 0], pred[:, 1] - first_xy[sel, 1])
        for j, kk, dd in zip(sel[d <= gate], k[d <= gate], d[d <= gate]):
            j = int(j)
            gaps.append({'track': int(uniq[i]), 'to_track': int(uniq[j]),
                         'f_lo': int(end_fid[i]), 'f_hi': int(first_fid[j]),
                         'xy_lo': end_xy[i].copy(), 'xy_hi': first_xy[j].copy(),
                         'frames': list(range(int(end_fid[i]) + 1, int(first_fid[j]))),
                         'kind': 'boundary'})

    gaps.sort(key=lambda g: (g['frames'][0], g.get('kind') != 'interior', g['track']))
    return gaps


def predict(gap, fid):
    """Linear prediction of the particle centre at ``fid`` inside ``gap``."""
    t = (fid - gap['f_lo']) / float(gap['f_hi'] - gap['f_lo'])
    return (1.0 - t) * gap['xy_lo'] + t * gap['xy_hi']


# ---------------------------------------------------------------------------- fit
def fit_local(image, predicted, radius, **kw):
    """Fit one disc in a patch around ``predicted``.

    Returns ``dict(xy_px, rays, rmse, offset_from_prediction_px, converged)`` or
    ``None`` when no seed yields a fit that passes the gates.
    """
    p = dict(DEFAULTS)
    p.update({k: v for k, v in kw.items() if k in DEFAULTS})

    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., 0]
    h, w = img.shape
    half = int(p['half_window'])
    xi, yi = int(round(float(predicted[0]))), int(round(float(predicted[1])))
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    if x1 - x0 < 25 or y1 - y0 < 25:
        return None
    patch = np.ascontiguousarray(img[y0:y1, x0:x1])
    origin = np.array([x0, y0], np.float32)
    local = np.asarray(predicted, np.float32) - origin

    peaks, _conf = D2._peaks(patch, None, float(p['peak_thresh']))
    seeds = np.zeros((0, 2), np.float32) if peaks is None else np.asarray(peaks, np.float32)
    seeds = seeds.reshape(-1, 2)
    # A single poor seed used to end the attempt: at the wall the fit from a 6 px-off
    # prediction locks onto the rim and no candidate passes, even though the disc is
    # there.  Start from a small grid as well, and follow the candidate that stays
    # closest to the prediction instead of giving up.
    grid = np.array([[dx, dy] for dx in (-6.0, 0.0, 6.0) for dy in (-6.0, 0.0, 6.0)],
                    np.float32)
    seeds = np.vstack([seeds, local[None, :], local[None, :] + grid]).astype(np.float32)

    best = None
    cur = seeds
    for _ in range(int(p['iterations'])):
        strength, rr = D2._profile(patch, cur)
        c, ns, rmse = D2._fit_fixed(strength, rr, float(radius))
        pos = cur + c
        off = np.hypot(pos[:, 0] - local[0], pos[:, 1] - local[1])
        moved = np.hypot(c[:, 0], c[:, 1])
        ok = (ns >= int(p['min_rays'])) & (rmse <= float(p['max_rmse'])) & \
             (off <= float(p['max_offset']))
        if ok.any():
            k = np.where(ok)[0][np.argmin(off[ok])]
            best = dict(xy_px=pos[k] + origin, rays=int(ns[k]), rmse=float(rmse[k]),
                        offset_from_prediction_px=float(off[k]),
                        converged=bool(moved[k] < float(p['converge_px'])))
            if best['converged']:
                return best
            cur = pos[k][None, :].astype(np.float32)
            continue
        # Nothing passes yet: walk the most plausible seed towards its fixed point.
        weak = ns >= 6
        if not weak.any():
            return best
        k = np.where(weak)[0][np.argmin(off[weak])]
        if moved[k] < float(p['converge_px']):
            return best
        cur = pos[k][None, :].astype(np.float32)
    return best


# ------------------------------------------------------------------------- rows
def _default_value(key, dtype):
    if dtype == np.bool_:
        return False
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.integer):
        return -1 if key in ('segment_id', 'previous_detection') else 0
    return 0


def push_row(rec, row):
    """Append one detection row to a frame record, preserving every field."""
    for key, arr in list(rec.items()):
        if key == 'xy_px':
            add = np.asarray([row['xy_px']], np.float32).reshape(1, 2)
        elif key in row:
            add = np.asarray([row[key]], dtype=arr.dtype).reshape((1,) + arr.shape[1:])
        else:
            add = np.full((1,) + arr.shape[1:], _default_value(key, arr.dtype),
                          dtype=arr.dtype)
        rec[key] = np.concatenate([arr, add], axis=0)


def _row_from(fit, radius, nearest, image):
    xy = np.asarray(fit['xy_px'], np.float32).reshape(1, 2)
    theta, angle_ok, offset, area, _bright = D2._marker(image, xy)
    bits = 0 if bool(angle_ok[0]) else 1
    if nearest < 1.85 * radius:
        bits |= 2
    th = float(theta[0]) if np.isfinite(theta[0]) else np.nan
    return dict(
        xy_px=xy[0], radius_px=float(radius), theta_rad=th,
        angle_valid=bool(angle_ok[0]), center_valid=True, confidence=0.5,
        edge_rmse_px=float(fit['rmse']), marker_offset_px=float(offset[0]),
        marker_area_px=int(area[0]), neighbor_distance_px=float(nearest),
        boundary_distance_px=np.nan, quality_bits=np.uint16(bits),
        outside_mask=np.uint8(0), manual_edit=np.uint8(0), recovered=np.uint8(1),
        segment_id=np.int32(-1), previous_detection=np.int64(-1),
        link_valid=False, gap_frames=np.int16(0),
        link_margin_px=np.nan, step_px=np.nan, dtheta_rad=np.nan)


# --------------------------------------------------------- reference linking

def splice_links(frames, recovered, *, gap=5):
    """Compatibility entry point using the authoritative v2 linker.

    Local identity splicing is not equivalent to replaying the EMA/assignment
    state: even an exact predicted insertion can change future competition.
    Keep callers safe rather than guessing a finite convergence window.
    ``recovered`` remains accepted for compatibility, but identities are never
    inferred from its candidate gap metadata.
    """
    return R._link_frames(frames, linker='v2', gap=gap)


# ------------------------------------------------------------------------ driver
def _case_settings(case_dir):
    cfg = {}
    p = Path(case_dir) / 'run_config.json'
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding='utf8'))
        except Exception:
            cfg = {}
    return cfg


def _copy_batch_assets(case_dir, out_root):
    """Make the refined output a complete batch (masks, apparatus, README).

    The case directory only holds chunk sidecars; ``geometry`` and ``accept``
    need the mask and the apparatus file, and a refined batch that silently
    lacks them cannot be archived without another manual copy step.
    """
    case_dir = Path(case_dir)
    if case_dir.parent.name != 'sequences':
        return []
    batch = case_dir.parent.parent
    copied = []
    for item in sorted(batch.iterdir()):
        if item.name in ('sequences', 'logs', 'index.json', 'acceptance.json'):
            continue
        dst = Path(out_root) / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        elif item.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
        else:
            continue
        copied.append(item.name)
    return copied


def _refresh_summaries(seq_out, frames, stats, record):
    """Refresh the copied summaries so they describe this chunk, not its parent.

    ``full_summary.json`` and ``progress.json`` are copied from the source case;
    their segment count and link fraction describe the pre-refine chunk and would
    contradict the rows in this batch.  Index/accept hash the summary, so a stale
    copy also survives the gates unnoticed.
    """
    seq_out = Path(seq_out)
    counts = np.asarray([len(rec['xy_px']) for _fid, rec in frames], float)
    angle_ok = np.concatenate([np.asarray(rec['angle_valid'], bool) for _fid, rec in frames])
    centre_ok = np.concatenate([np.asarray(rec['center_valid'], bool) for _fid, rec in frames])
    fields = {
        'linker': record['linker'],
        'link_fraction': stats['link_fraction'],
        'segments': stats['n_tracks'],
        'gap_closed': stats['n_gap_rows'],
        'median_step_px': stats['median_step_px'],
        'count_min': int(counts.min()) if len(counts) else 0,
        'count_median': float(np.median(counts)) if len(counts) else 0,
        'count_max': int(counts.max()) if len(counts) else 0,
        'angle_valid_fraction': float(angle_ok.mean()) if len(angle_ok) else None,
        'centre_valid_fraction': float(centre_ok.mean()) if len(centre_ok) else None,
        'refined_at': record['created_at'],
        'n_recovered': record['n_recovered'],
    }
    updated = []
    for name in ('full_summary.json', 'progress.json'):
        p = seq_out / name
        if not p.exists():
            continue
        try:
            summary = json.loads(p.read_text(encoding='utf8'))
        except Exception:
            summary = {}
        summary.update(fields)
        if name == 'full_summary.json':
            summary['status'] = 'refined'
        R._write_json(p, summary)
        updated.append(name)
    return updated


def _annotate_readme(out_root, record):
    """Append a short refinement note to the batch README that was copied over."""
    p = Path(out_root) / 'README.md'
    if not p.exists():
        return False
    text = p.read_text(encoding='utf8')
    if '## Refined copy' in text:
        return False
    block = (
        '\n## Refined copy (%s)\n\n'
        'Source case: `%s`.\n\n'
        '`vibrotrack refine` recovered %d of %d missing particles (holes up to %d '
        'frames; %d refused), re-linked with %s and wrote %d rows in %d tracks, '
        '%d gap rows, `link_mode=%s`. Every attempt, with its prediction and gate '
        'values, is in `sequences/%s/refine.json`.\n'
        % (record['created_at'][:10], record['source_case'], record['n_recovered'],
           record['n_attempts'], record['max_gap'], record['n_rejected'],
           record['linker'], record['n_detections'], record['n_tracks'],
           record['n_gap_rows'], record.get('link_mode'), record['case']))
    p.write_text(text.rstrip('\n') + '\n' + block, encoding='utf8')
    return True


def refine_case(case_dir, out_root, raw_dir, *, max_gap=None, radius=None,
                min_rays=None, max_rmse=None, max_offset=None, min_sep=None,
                limit=0, force=False, verbose=False, incremental=True):
    """Recover missing particles on one case and write a new, re-linked copy."""
    case_dir = R.resolve_case_dir(case_dir)
    out_root = Path(out_root).expanduser().resolve()
    seq_out = out_root / 'sequences' / case_dir.name
    if seq_out.resolve() == case_dir.resolve():
        raise ValueError('choose a new --out; in-place refinement is not allowed')

    cfg = _case_settings(case_dir)
    linker = str(cfg.get('linker', 'v2')).lower()
    link_gap = int(cfg.get('link_gap', 5))
    max_gap = link_gap if max_gap is None else int(max_gap)

    chunks, chunk_frames, chunk_scalars, keys, n_frames = R._read_case(case_dir)
    keys = set(keys) | {'recovered'}
    for frames in chunk_frames:
        for _fid, rec in frames:
            if 'recovered' not in rec:
                rec['recovered'] = np.zeros(len(rec['xy_px']), np.uint8)
    frames = [item for group in chunk_frames for item in group]
    if radius is None:
        radii = np.concatenate([np.asarray(rec['radius_px'], float)
                                for _fid, rec in frames])
        radius = float(np.nanmedian(radii))
    params = dict(DEFAULTS)
    if min_rays is not None:
        params['min_rays'] = int(min_rays)
    if max_rmse is not None:
        params['max_rmse'] = float(max_rmse)
    if max_offset is not None:
        params['max_offset'] = float(max_offset)
    if min_sep is not None:
        params['min_sep'] = float(min_sep)

    files = X.files_for({'path': str(raw_dir), 'extensions': ('.tiff', '.tif', '.png')})
    if not files:
        raise FileNotFoundError('no raw frames in %s' % raw_dir)
    last_fid = max(int(fid) for fid, _ in frames)
    if len(files) <= last_fid:
        raise ValueError('raw directory has %d frames but the case references frame %d'
                         % (len(files), last_fid))
    levels = cfg.get('scale_levels')

    cache = {}

    def image_for(fid):
        if fid in cache:
            return cache[fid]
        if len(cache) >= 3:
            cache.clear()
        img = X.read_image(str(files[int(fid)]), levels)
        cache[fid] = img
        return img

    gaps = find_gaps(frames, max_gap=max_gap,
                     boundary_gate=params['boundary_gate'],
                     boundary_growth=params['boundary_growth'])
    # Distinct boundary targets can yield distinct fitting seeds. Never discard
    # one before looking at pixels. The spatial duplicate-centre gate below
    # prevents two accepted predictions from inserting the same particle twice.
    if limit:
        gaps = gaps[:int(limit)]
    by_fid = {int(fid): rec for fid, rec in frames}

    attempts = []
    recovered_rows = []
    n_recovered = 0
    for gap in gaps:
        for fid in gap['frames']:
            pred = predict(gap, fid)
            fit = fit_local(image_for(fid), pred, radius, **params)
            reason = None
            nearest = None
            if fit is None:
                reason = 'no fit passed the gates'
            else:
                existing = np.asarray(by_fid[fid]['xy_px'], float)
                nearest = float(np.hypot(existing[:, 0] - fit['xy_px'][0],
                                         existing[:, 1] - fit['xy_px'][1]).min()) \
                    if len(existing) else float('inf')
                if nearest < float(params['min_sep']) * radius:
                    reason = 'duplicate centre (%.1f px)' % nearest
            if reason is None:
                push_row(by_fid[fid], _row_from(fit, radius, nearest, image_for(fid)))
                recovered_rows.append({'gap': gap, 'fid': int(fid),
                                       'row': len(by_fid[fid]['xy_px']) - 1})
                n_recovered += 1
            attempts.append({
                'track': int(gap['track']), 'kind': gap.get('kind', 'interior'),
                'frame': int(fid),
                'predicted': [round(float(v), 3) for v in pred],
                'fitted': None if fit is None else [round(float(v), 3) for v in fit['xy_px']],
                'rays': None if fit is None else fit['rays'],
                'rmse_px': None if fit is None else round(fit['rmse'], 3),
                'offset_from_prediction_px': None if fit is None
                else round(fit['offset_from_prediction_px'], 2),
                'nearest_existing_px': None if nearest is None else round(nearest, 2),
                'accepted': reason is None, 'reason': reason,
            })
            if verbose:
                print('%s frame %5d  rays %s  rmse %s  %s'
                      % (gap['track'], fid,
                         '-' if fit is None else fit['rays'],
                         '-' if fit is None else ('%.2f' % fit['rmse']),
                         'ok' if reason is None else reason))

    # Gap metadata supplies fitting seeds, not authoritative identities.
    # Replay the reference state after insertions (EMA changes can propagate).
    stats = R._link_frames(frames, linker=linker, gap=link_gap)
    link_mode = 'full'
    link_fallback_reason = ('incremental splice is not reference-equivalent'
                            if incremental else None)

    if seq_out.exists() and any(seq_out.iterdir()):
        if not force:
            raise FileExistsError('output case is not empty (use --force): %s' % seq_out)
        shutil.rmtree(seq_out)
    seq_out.mkdir(parents=True, exist_ok=True)
    batch_assets = _copy_batch_assets(case_dir, out_root)
    source_hashes = [{'file': p.name, 'sha256': R._sha256(p)} for p in chunks]
    outputs = R._write_chunks(chunk_frames, chunks, seq_out, keys, chunk_scalars)
    output_records = [{'file': p.name, 'bytes': p.stat().st_size,
                       'sha256': R._sha256(p)} for p in outputs]

    for src in case_dir.iterdir():
        if src.is_file() and not src.name.startswith('chunk_'):
            shutil.copy2(src, seq_out / src.name)

    record = {
        'schema': REFINE_SCHEMA,
        'case': case_dir.name,
        'created_at': R._now(),
        'source_case': str(case_dir),
        'source_chunks': source_hashes,
        'raw_dir': str(Path(raw_dir).resolve()),
        'radius_px': float(radius),
        'max_gap': int(max_gap),
        'params': {k: params[k] for k in ('min_rays', 'max_rmse', 'max_offset',
                                          'min_sep', 'half_window', 'peak_thresh',
                                          'iterations', 'converge_px',
                                          'boundary_gate', 'boundary_growth')},
        'n_gaps': len(gaps),
        'n_attempts': len(attempts),
        'n_recovered': n_recovered,
        'n_rejected': len(attempts) - n_recovered,
        'attempts': attempts,
        'linker': linker,
        'link_mode': link_mode,
        'link_fallback_reason': link_fallback_reason,
        'link_gap': link_gap,
        'n_frames': stats['n_frames'],
        'n_detections': stats['n_detections'],
        'n_tracks': stats['n_tracks'],
        'n_linked_rows': stats['n_linked_rows'],
        'n_gap_rows': stats['n_gap_rows'],
        'outputs': output_records,
        'batch_assets': batch_assets,
        'note': ('Additive pass: existing rows are copied unchanged, new rows carry '
                 'recovered=1 and were placed by a local fixed-radius fit seeded at the '
                 'track prediction. No position is interpolated.'),
    }
    refine_path = seq_out / 'refine.json'
    R._write_json(refine_path, record)
    _annotate_readme(out_root, record)
    _refresh_summaries(seq_out, frames, stats, record)
    refine_entry = {'file': refine_path.name, 'bytes': refine_path.stat().st_size,
                    'sha256': R._sha256(refine_path)}

    dataset_path = seq_out / 'dataset.json'
    if dataset_path.exists():
        try:
            d = json.loads(dataset_path.read_text(encoding='utf8'))
        except Exception:
            d = {}
        d['outputs'] = output_records + [refine_entry]
        d['qc'] = dict(d.get('qc') or {})
        d['qc']['status'] = 'refined'
        d['refine'] = {
            'schema': REFINE_SCHEMA, 'file': refine_path.name,
            'n_gaps': len(gaps), 'n_recovered': n_recovered,
            'n_rejected': len(attempts) - n_recovered,
            'radius_px': float(radius),
            'link_mode': link_mode,
        }
        R._write_json(dataset_path, d)
    return record, seq_out
