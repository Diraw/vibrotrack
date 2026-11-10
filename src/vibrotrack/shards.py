"""Sharded dense detection + per-case linking.

Stage A (parallel): detect every frame of a contiguous frame range, save arrays.
Stage B (serial per case): conservative short-range linking over the ordered detections
and write the existing chunk_*.npz contract consumed by the project analysis code.

Original TIFFs are read-only. Detection is identical to extract.detect; only the
parallel decomposition differs, so shard boundaries never enter the tracking because
linking happens after all detections of a case are ordered.
"""
from pathlib import Path
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue, threading

from .mask import load_mask

ROOT = Path(__file__).resolve().parent
if __package__:
    from . import detect as X
    from . import detect2 as D2
    from . import link2 as L2
else:                                   # flat execution (legacy driver style)
    sys.path.insert(0, str(ROOT))
    import detect as X
    import detect2 as D2
    import link2 as L2

# Output root: where <case>/dets and <case>/chunk_*.npz go. Never inside the package.
OUT_ROOT = Path.cwd()


def out_root(case=None):
    return Path((case or {}).get('out_root') or OUT_ROOT)

cv2.setNumThreads(1)
SHARD = 500
FIELD_KEYS = ['xy_px', 'radius_px', 'theta_rad', 'center_valid', 'angle_valid', 'confidence',
              'edge_rmse_px', 'marker_offset_px', 'marker_area_px', 'neighbor_distance_px',
              'boundary_distance_px', 'quality_bits']
# 2026-11-08: chunk/v2 adds outside_mask (u1) and gap_frames (i1). Old det shards do not
# have outside_mask, so it is requested only if present.
OPT_KEYS = ['outside_mask']


def det_tag(case):
    """Cache directory suffix encoding the detector configuration.

    Without this, switching detector/wall-band would silently reuse the previous
    run's cached det_*.npz and produce a mixed tracking.
    """
    parts = []
    if case.get('detector', 'v2') != 'v1':
        parts.append(case.get('detector', 'v2'))
    if int(case.get('wall_band', 0)):
        parts.append('w%d' % int(case.get('wall_dilate', 40)))
        parts.append('a%d' % int(float(case.get('avoid_r', 62.0))))
        wm = case.get('wall_model', 'auto')
        if wm != 'auto':
            parts.append('wm_' + str(wm))
        if not case.get('prior', True):
            parts.append('noprior')
    return '' if not parts else '_' + '_'.join(parts)


WALL_FILE = 'wall_disks.npy'


def load_wall_disks(case, src):
    """Measured wall map: (N,4) = x, y, disc_radius_px, px_per_mm."""
    if src is None:
        raise FileNotFoundError(
            'wall_model=geometry needs a boundary mask path so the measured %s can be found next to it; '
            'supply boundary.mask or use wall_model=empirical' % WALL_FILE)
    p = Path(str(src)).parent / WALL_FILE
    if not p.exists():
        raise FileNotFoundError(
            'wall_model=geometry needs a measured %s next to the boundary mask (%s); '
            'fall back to wall_model=empirical' % (WALL_FILE, p))
    a = np.load(p).astype(np.float64)
    if a.shape[1] < 3:
        raise ValueError('%s must have columns x,y,r' % p)
    return a


def resolve_wall_model(case, src):
    """auto: use the measured wall map when it exists, else the empirical variance rule."""
    wm = str(case.get('wall_model', 'auto')).lower()
    if wm == 'auto':
        return 'geometry' if (src is not None and (Path(str(src)).parent / WALL_FILE).exists()) else 'empirical'
    if wm not in ('geometry', 'empirical', 'off'):
        raise ValueError('wall_model must be auto|geometry|empirical|off, got %r' % wm)
    if wm == 'geometry' and src is None:
        raise FileNotFoundError('wall_model=geometry requires boundary.mask so wall_disks.npy can be located')
    return wm


def shard_task(args):
    case, lo, hi = args[:3]
    files = args[3] if len(args) > 3 else None
    cv2.setNumThreads(1)
    cid = case['id']
    if files is None:
        # Backward-compatible path for old drivers. New drivers pass files[lo:hi], so the
        # 9000-file directory scan/stat happens once in the parent instead of per shard.
        files = X.files_for(case)[lo:hi]
    out = out_root(case) / cid / ('dets' + det_tag(case))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f'det_{lo:05d}_{hi-1:05d}.npz'
    if dst.exists():
        return {'case': cid, 'lo': lo, 'hi': hi, 'status': 'cached'}
    interior, src = load_mask(case.get('mask_path'), case.get('shape'), required=True)
    case['mask_source'] = src or 'full_frame'
    wall_band = int(case.get('wall_band', 0))
    model = 'off' if not wall_band else resolve_wall_model(case, src)
    # Fixed apparatus discs (95 of them) are exact particle look-alikes. The search region of
    # the wall band reaches into their silhouettes, so their centres are passed to the detector
    # as exclusion points; a real particle cannot sit closer than one diameter to one.
    static_centres = None
    avoid_r = float(case.get('avoid_r', 0.0))
    if model == 'geometry':
        w = load_wall_disks(case, src)
        static_centres = w[:, :2]
        # measured wall map: keep the exclusion tight, only the residual registration
        # error (case['wall_tol_px']) is allowed for
        avoid_r = float(case.get('particle_radius_px', 36.98)) - float(case.get('wall_tol_px', 6.0))
    elif avoid_r:
        try:
            bjson = Path(str(src)).parent / 'boundary.json'
            if bjson.exists():
                discs = json.loads(bjson.read_text(encoding='utf8')).get('design_inferred', {}).get('disks_px', [])
                if discs:
                    static_centres = [[d['center_px'][0], d['center_px'][1]] for d in discs]
        except Exception:
            static_centres = None
    if model == 'geometry':
        walls = load_wall_disks(case, src)
        mask = D2.reachable_mask(interior.shape, walls[:, :2], walls[:, 2],
                                 float(case.get('particle_radius_px', 36.98)),
                                 float(walls[0, 3]) if walls.shape[1] > 3 else 1.0)
    elif model == 'empirical':
        mask = D2.search_mask(interior, int(case.get('wall_dilate', 40)))
    else:
        mask = interior
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    bbox = X.mask_bbox(mask)
    strict_io = bool(case.get('strict_io', False))
    detector = case.get('detector', 'v2')

    use_prior = bool(case.get('prior', True)) and detector != 'v1'

    def detect_frame(im, prior=None):
        if detector == 'v1':
            d = X.detect(im, mask, dist, bbox=bbox)
            d['outside_mask'] = np.zeros(len(d['xy_px']), np.uint8)
            return d
        params = {k: v for k, v in case.get('detector_params', {}).items()
                  if k not in ('avoid_r', 'prior_xy', 'static_centres', 'interior_mask', 'bbox')}
        d, _info = D2.frame(im, mask, dist, bbox=bbox, interior_mask=interior,
                            static_centres=static_centres, avoid_r=avoid_r,
                            prior_xy=prior, **params)
        return d
    recs = []
    t0 = time.time()
    if os.environ.get('PREFETCH', '1') == '1':
        # Reader thread fills a small queue while the main thread detects: the frame
        # read and the uint16->uint8 scaling overlap with the detector instead of
        # serialising. Depth 3 caps extra memory at ~35 MB per worker.
        q = queue.Queue(maxsize=3)
        def _reader():
            try:
                for fi, path in enumerate(files, start=lo):
                    q.put((fi, X.read_image(path, case['scale_levels'], strict=strict_io)))
            except Exception as exc:      # surface read errors in the consumer
                q.put(exc)
            finally:
                q.put(None)
        th = threading.Thread(target=_reader, daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            prev = recs[-1]['xy_px'][-800:] if (use_prior and recs) else None
            recs.append(detect_frame(item[1], prev))
        th.join()
    else:
        for fi, path in enumerate(files, start=lo):
            im = X.read_image(path, case['scale_levels'], strict=strict_io)
            prev = recs[-1]['xy_px'][-800:] if (use_prior and recs) else None
            recs.append(detect_frame(im, prev))
    payload = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    offsets = np.concatenate([[0], np.cumsum([len(r['xy_px']) for r in recs])])
    payload['frame_ids'] = np.arange(lo, lo + len(files))
    payload['frame_offsets'] = offsets
    tmp = dst.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **payload)
    tmp.replace(dst)
    return {'case': cid, 'lo': lo, 'hi': hi, 'status': 'ok', 'seconds': time.time() - t0,
            'frames': hi - lo, 'median_count': float(np.median([len(r['xy_px']) for r in recs]))}


def det_path(case, lo, hi):
    return out_root(case) / case['id'] / ('dets' + det_tag(case)) / f'det_{lo:05d}_{hi-1:05d}.npz'


def load_det_part(case, lo, hi):
    with np.load(det_path(case, lo, hi)) as z:
        return {k: z[k] for k in z.files}


def new_link_state(case=None):
    """State carried across shards while detection is still running."""
    case = case or {}
    st = {'prev': None, 'nextid': 0, 'offset': 0, 'records': [], 'frame_ids': [],
          'linker': case.get('linker', 'v2'), 'l2': None}
    if st['linker'] == 'v2':
        st['l2'] = L2.new_state(gap=int(case.get('link_gap', 5)))
    return st


def _keys(part):
    return [k for k in FIELD_KEYS if k in part] + [k for k in OPT_KEYS if k in part]


def link_part(part, state):
    """Link one det shard into the running state."""
    fids = part['frame_ids']
    bounds = part['frame_offsets']
    if len(bounds) != len(fids) + 1:
        raise ValueError('frame bounds do not match frame ids')
    keys = _keys(part)
    dets = {k: part[k] for k in keys}
    for j, fid in enumerate(fids):
        i0, i1 = int(bounds[j]), int(bounds[j + 1])
        cur = {k: dets[k][i0:i1] for k in keys}
        if 'outside_mask' not in cur:
            cur['outside_mask'] = np.zeros(len(cur['xy_px']), np.uint8)
        prev = state['prev']
        prev_offset = state['offset'] - (len(prev['xy_px']) if prev is not None else 0)
        if state['linker'] == 'v2':
            rmed = float(np.median(cur['radius_px'])) if len(cur['radius_px']) else 36.9
            ids, plocal, gapsz, step = L2.link_frame(state['l2'], int(fid), cur['xy_px'], rmed)
            cur['segment_id'] = ids.astype(np.int32)
            prev_g = np.where(plocal >= 0, prev_offset + plocal, -1).astype(np.int64)
            cur['previous_detection'] = prev_g
            cur['link_valid'] = gapsz == 1
            cur['gap_frames'] = gapsz.astype(np.int16)
            cur['link_margin_px'] = np.full(len(ids), np.nan, np.float32)
            cur['step_px'] = step.astype(np.float32)
            dt = np.full(len(ids), np.nan, np.float32)
            if prev is not None and len(prev['xy_px']):
                m = gapsz == 1
                if m.any():
                    p = prev['theta_rad'][plocal[m]]
                    c = cur['theta_rad'][m]
                    ok = np.isfinite(p) & np.isfinite(c)
                    dd = np.arctan2(np.sin(c - p), np.cos(c - p))
                    dd[~ok] = np.nan
                    dt[m] = dd
            cur['dtheta_rad'] = dt
        else:
            state['nextid'] = X.link(prev, cur, state['nextid'],
                                     prev_offset if prev is not None else 0)
        state['records'].append(cur)
        state['frame_ids'].append(int(fid))
        state['offset'] += len(cur['xy_px'])
        state['prev'] = cur
    return len(fids)



def drop_static_wall_tracks(payload, min_len=30, max_disp=3.0):
    """Remove tracks that are entirely outside the interior mask and never move.

    Those are the frozen particles that form the container wall. They only appear
    because the search mask is dilated to recover particles squeezed against the wall;
    a real particle, even a jammed one, moves far more than 3 px over 30 frames
    (measured median step at the wall: 2.26 px/frame).

    Returns (payload, n_dropped, n_tracks_dropped).
    """
    if 'outside_mask' not in payload or 'segment_id' not in payload:
        return payload, 0, 0
    seg = payload['segment_id'].astype(np.int64)
    om = payload['outside_mask']
    xy = payload['xy_px'].astype(np.float64)
    n = len(seg)
    if not n:
        return payload, 0, 0
    uniq, inv = np.unique(seg, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uniq))
    all_out = np.bincount(inv, weights=om.astype(np.float64), minlength=len(uniq)) == cnt
    mins = np.full((len(uniq), 2), np.inf)
    maxs = np.full((len(uniq), 2), -np.inf)
    for d in range(2):
        np.minimum.at(mins[:, d], inv, xy[:, d])
        np.maximum.at(maxs[:, d], inv, xy[:, d])
    disp = np.hypot(maxs[:, 0] - mins[:, 0], maxs[:, 1] - mins[:, 1])
    # Two staticness rules. The wall discs are often detected for only a few frames
    # (they flicker behind the search-mask edge), so a length-30 rule leaves ~16% of
    # the wall band static; a short-track rule catches those without touching real
    # particles, which jitter ~1.7 px/frame and therefore cannot stay within 1.5 px.
    drop_track = all_out & (((cnt >= min_len) & (disp < max_disp)) |
                            ((cnt >= 3) & (disp < 1.5)))
    if not drop_track.any():
        return payload, 0, 0
    keep = ~drop_track[inv]
    ndrop = int((~keep).sum())
    _apply_drop(payload, keep)
    return payload, ndrop, int(drop_track.sum())


def _apply_drop(payload, keep):
    """Remove the detections where `keep` is False and rebuild every derived array."""
    n = len(payload['segment_id'])
    fids = payload['frame_ids']
    nf = len(fids)
    fr = np.repeat(np.arange(nf), np.diff(payload['frame_offsets']))[keep]
    for k in list(payload.keys()):
        v = payload[k]
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            payload[k] = v[keep]
    counts = np.bincount(fr, minlength=nf)
    payload['frame_offsets'] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    seg2 = payload['segment_id']
    gap2 = payload['gap_frames']
    prev2 = np.full(len(seg2), -1, np.int64)
    last = {}
    last_f = {}
    for i in range(len(seg2)):
        sid = int(seg2[i])
        if gap2[i] == 1 and last_f.get(sid) == int(fr[i]) - 1:
            prev2[i] = last[sid]
        last[sid] = i
        last_f[sid] = int(fr[i])
    payload['previous_detection'] = prev2
    payload['link_valid'] = gap2 == 1
    return payload


_STD_KERNEL = {}


def _local_std(std_map, radius=25):
    import cv2
    key = (radius, std_map.shape)
    if key not in _STD_KERNEL:
        k = np.zeros((2 * radius + 1, 2 * radius + 1), np.float32)
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        k[np.hypot(xx, yy) < radius] = 1
        _STD_KERNEL[key] = k / k.sum()
    return cv2.filter2D(std_map.astype(np.float32), -1, _STD_KERNEL[key])


def drop_static_objects(payload, std_map, thresh=10.0, radius=25,
                        cluster_px=3.0, min_cluster=30):
    """Drop detections that sit on an apparatus disc which never changes.

    A frozen disc is a particle look-alike with one decisive difference: the image around
    it never changes. Measured on the production data the split is bimodal - 32.8% of the wall band sits
    on a neighbourhood with temporal std < 10 (sensor noise), the rest at 40-90, the same
    as the interior.

    This used to apply to the wall band only, assuming nothing static could be inside the
    interior mask. That assumption failed: the apparatus discs are 36 px in radius and the
    mask is inset by 24 px, so the inner edge of a disc is inside the mask. On the production data that
    left 6 656 v2 detections (0.245%) on frozen discs, concentrated in 18 cells of 20 px
    with ~370 detections each - the same bead, seen in every frame it is detected.

    A real particle that merely paused would not produce such a cluster, so a detection is
    only dropped when at least `min_cluster` detections share its 3 px neighbourhood.

    Outside the interior mask the cluster rule is too weak: the wall band exists only to
    recover particles pressed against the wall, and those move (local std 40-90). The
    frozen ring there is often detected in short bursts (1-26 frames in the 2026-11-09
    10 fps data), so a 30-hit cluster never forms and ~0.8 frozen hits per frame survived.
    A detection outside the mask is therefore dropped on temporal std alone; the cluster
    rule still protects detections inside the mask.

    Returns (payload, n_dropped).
    """
    if not len(payload['xy_px']):
        return payload, 0
    local = _local_std(std_map, radius)
    xy = payload['xy_px']
    xi = np.clip(xy[:, 0].astype(int), 0, local.shape[1] - 1)
    yi = np.clip(xy[:, 1].astype(int), 0, local.shape[0] - 1)
    cand = local[yi, xi] < thresh
    if not cand.any():
        return payload, 0
    bad = np.zeros(len(xy), bool)
    outside = payload.get('outside_mask')
    if outside is not None:
        bad |= cand & np.asarray(outside).astype(bool)
    from scipy.spatial import cKDTree
    P = xy[cand].astype(np.float64)
    nbr = cKDTree(P).query_ball_point(P, r=cluster_px, return_length=True)
    bad[np.where(cand)[0][nbr >= min_cluster]] = True
    if not bad.any():
        return payload, 0
    _apply_drop(payload, ~bad)
    return payload, int(bad.sum())


def temporal_std(case, cid, n=60):
    """Mean per-pixel temporal std over the recording, cached next to the chunk."""
    dst = out_root(case) / cid / 'temporal_std.npy'
    if dst.exists():
        return np.load(dst)
    files = X.files_for(case)
    if not files:
        return None
    idx = np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)
    acc = []
    levels = case.get('scale_levels', [128.0, 28224.0])
    for i in idx:
        acc.append(X.read_image(files[int(i)], levels).astype(np.float32))
    std = np.stack(acc).std(0)
    np.save(dst, std)
    return std

def _sha1(path):
    import hashlib
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()[:16]
    except Exception:
        return None


def write_run_config(case, cid, model):
    """Everything needed to answer "which code, which settings, which data made this chunk".

    Written next to the chunk as run_config.json. The git commit is included because the
    contract requires any quoted number to name it, and because a worktree can be dirty.
    """
    import subprocess
    out = out_root(case) / cid
    out.mkdir(parents=True, exist_ok=True)
    files = []
    try:
        files = [str(f) for f in X.files_for(case)]
    except Exception:
        pass
    mask, src = None, None
    try:
        _, src = load_mask(case.get('mask_path'), case.get('shape'))
        mask = str(src) if src else 'full_frame'
    except Exception:
        pass
    git = {}
    try:
        root = Path(__file__).resolve().parents[2]
        git['commit'] = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True, timeout=10).stdout.strip()
        git['dirty'] = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                                           capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        pass
    cfg = dict(
        case=cid, detector=case.get('detector', 'v2'), linker=case.get('linker', 'v2'),
        wall_model=model, wall_band=int(case.get('wall_band', 0)),
        wall_dilate=int(case.get('wall_dilate', 40)), wall_tol_px=float(case.get('wall_tol_px', 6.0)),
        avoid_r=float(case.get('avoid_r', 0.0)), link_gap=int(case.get('link_gap', 5)),
        wall_static_std=bool(case.get('wall_static_std', True)),
        wall_static_thresh=float(case.get('wall_static_thresh', 10.0)),
        detector_defaults={k: float(v) for k, v in D2.DEFAULTS.items()},
        nframes=int(case.get('nframes', 0)), scale_levels=list(case.get('scale_levels', [])),
        boundary=case.get('boundary'), mask=mask, mask_sha1=_sha1(mask) if mask else None,
        boundary_json_sha1=_sha1(Path(str(mask)).parent / 'boundary.json') if mask else None,
        wall_disks_sha1=_sha1(Path(str(mask)).parent / WALL_FILE) if mask else None,
        n_input_files=len(files), first_file=files[0] if files else None,
        last_file=files[-1] if files else None, git=git)
    (out / 'run_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf8')
    return cfg


def finish_link(case, state):
    cid = case['id']
    records = state['records']
    fids = np.asarray(state['frame_ids'], dtype=np.int64)
    if not len(records):
        return {'case': cid, 'status': 'no_detections'}
    if len(fids) != len(records) or not np.all(np.diff(fids) == 1):
        raise ValueError('shard frame ids are not contiguous after sorting')
    keys = list(records[0].keys())
    payload = {k: np.concatenate([r[k] for r in records]) for k in keys}
    offsets = np.concatenate([[0], np.cumsum([len(r['xy_px']) for r in records])])
    payload['frame_ids'] = fids
    payload['frame_offsets'] = offsets
    payload['global_detection_offset'] = np.array(0)
    payload['schema_version'] = np.array('chunk/v2')
    if 'gap_frames' not in payload:
        payload['gap_frames'] = np.ones(len(payload['segment_id']), np.int16)
    n_wall = n_wall_tracks = n_static = 0
    interior, mask_src = load_mask(case.get('mask_path'), case.get('shape'), required=True)
    case['mask_source'] = mask_src or 'full_frame'
    model = resolve_wall_model(case, mask_src) if int(case.get('wall_band', 0)) else 'off'
    payload['wall_model'] = np.array(model)
    if int(case.get('wall_band', 0)):
        payload, n_wall, n_wall_tracks = drop_static_wall_tracks(payload)
        # geometry branch: the discs are already excluded by the measured wall map, so the
        # statistical rule is not needed (kept available as an independent cross-check)
        if model == 'empirical' and case.get('wall_static_std', True):
            std_map = temporal_std(case, cid)
            if std_map is not None:
                payload, n_static = drop_static_objects(
                    payload, std_map, float(case.get('wall_static_thresh', 10.0)))
    for old in (out_root(case) / cid).glob('chunk_*.npz'):
        old.unlink()
    dst = out_root(case) / cid / f'chunk_{fids[0]:05d}_{fids[-1]:05d}.npz'
    tmp = dst.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **payload)
    tmp.replace(dst)
    write_run_config(case, cid, model)
    # payload['frame_offsets'] was rebuilt by the wall drops; the local `offsets` is stale
    counts_per_frame = np.diff(payload['frame_offsets'])
    status = dict(case=cid, mode='full', status='completed', source_frame_count=len(fids),
                  processed_frame_count=len(fids), mask_source=case.get('mask_source', 'full_frame'),
                  detector=case.get('detector', 'v2'), linker=case.get('linker', 'v2'),
                  wall_band=int(case.get('wall_band', 0)), wall_static_dropped=n_static, wall_model=model,
                  wall_detections_dropped=int(n_wall), wall_tracks_dropped=int(n_wall_tracks),
                  count_min=int(counts_per_frame.min()), count_median=float(np.median(counts_per_frame)),
                  count_max=int(counts_per_frame.max()),
                  angle_valid_fraction=float(np.mean(payload['angle_valid'])),
                  link_fraction=float(np.mean(payload['link_valid'])),
                  median_step_px=float(np.nanmedian(payload['step_px'])),
                  segments=int(len(np.unique(payload['segment_id']))),
                  gap_closed=int(np.sum(payload['gap_frames'] > 1)),
                  elapsed_seconds=None)
    X.dump(out_root(case) / cid / 'full_summary.json', status)
    X.dump(out_root(case) / cid / 'progress.json', status)
    return status


def link_case(case):
    cid = case['id']
    files = sorted((out_root(case) / cid / ('dets' + det_tag(case))).glob('det_*.npz'),
                   key=lambda p: int(p.stem.split('_')[1]))
    if not files:
        return {'case': cid, 'status': 'no_detections'}
    parts = []
    for f in files:
        with np.load(f) as z:
            parts.append({k: z[k] for k in z.files})
    parts.sort(key=lambda p: int(p['frame_ids'][0]))
    state = new_link_state(case)
    for part in parts:
        link_part(part, state)
    return finish_link(case, state)
