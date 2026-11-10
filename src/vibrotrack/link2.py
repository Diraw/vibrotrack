"""Linker v2: motion prediction + global assignment + gap closing (2026-11-08).

The production linker (`detect.link`) is mutual-nearest-neighbour only, has no motion
model, and drops a track the moment one frame is missing. Measured on the production dense data: 9 407
breaks for 286 particles, and 98% of them trace back to frames where the particle was
simply not detected (mostly at the wall). The information needed to keep identity is
usually still there -- 66-89% of fragment starts have a plausible predecessor within
5 frames -- but v1 cannot use it.

v2 keeps the same tolerance for the easy case (it reproduces every v1 link; verified on
2.57 M links) and adds:

  * prediction = last position + smoothed velocity, so assignment does not rely on
    mutual nearest;
  * Hungarian assignment over the gated sub-graph, so two particles competing for the
    same neighbour are resolved together instead of both failing;
  * tracks stay open for `gap` frames. A detection may rejoin a track that ended 1..gap
    frames earlier when (a) the predicted position is inside a gate that grows with the
    gap and (b) no detection sits near the interpolated path in the skipped frames --
    otherwise that detection is the particle and this one is somebody else.

Identity only: positions are never modified.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def new_state(gate=None, gap=5, vel_alpha=0.5):
    return dict(tracks={}, next_id=0, last_frame=None, recent={},
                gate=gate, gap=int(gap), vel_alpha=float(vel_alpha))


def _assign(pred, cur, gate):
    if not len(pred) or not len(cur):
        return np.zeros(0, int), np.zeros(0, int)
    d = np.hypot(pred[:, None, 0] - cur[None, :, 0], pred[:, None, 1] - cur[None, :, 1])
    big = 1e6
    cost = np.where(d < gate, d, big)
    rows = np.where((cost < big).any(1))[0]
    cols = np.where((cost < big).any(0))[0]
    if not len(rows) or not len(cols):
        return np.zeros(0, int), np.zeros(0, int)
    sub = cost[np.ix_(rows, cols)]
    r, c = linear_sum_assignment(sub)
    ok = sub[r, c] < big
    return rows[r[ok]], cols[c[ok]]


def link_frame(state, fid, xy, radius_median, base_gate=None):
    """Attach `xy` (current frame) to the running state.

    Returns ids, prev_local (index into the previous frame's array or -1),
    gap_frames (1 = direct, 0 = new track) and step_px.
    """
    xy = np.asarray(xy, float)
    m = len(xy)
    gate = base_gate if base_gate is not None else state['gate']
    if gate is None:
        gate = 0.9 * float(radius_median)
    gate = min(float(gate), 0.9 * float(radius_median))
    gap = state['gap']
    tracks = state['tracks']
    ids = np.full(m, -1, np.int64)
    prev_local = np.full(m, -1, np.int64)
    gapsz = np.zeros(m, np.int16)
    step = np.full(m, np.nan)
    used = np.zeros(m, bool)

    if state['last_frame'] is not None and fid == state['last_frame'] + 1 and m:
        # --- stage 1: tracks seen in the previous frame --------------------------
        one = [t for t in tracks.values() if t['last'] == fid - 1]
        if one:
            pred = np.array([t['xy'] + t['v'] for t in one])
            pi, ci = _assign(pred, xy, gate)
            for a, b in zip(pi, ci):
                t = one[a]
                d = xy[b] - t['xy']
                t['v'] = (1 - state['vel_alpha']) * t['v'] + state['vel_alpha'] * d
                t['xy'] = xy[b]
                t['last'] = fid
                t['idx'] = int(b)
                ids[b] = t['id']
                prev_local[b] = t['idx_prev']
                gapsz[b] = 1
                step[b] = float(np.hypot(*d))
                used[b] = True
        # --- stage 2: gap closing ------------------------------------------------
        free = np.where(~used)[0]
        if len(free) and gap > 1:
            cand = [t for t in tracks.values() if 1 < fid - t['last'] <= gap]
            if cand:
                k = np.array([fid - t['last'] for t in cand])
                pred = np.array([t['xy'] + t['v'] * kk for t, kk in zip(cand, k)])
                gate_k = gate + 6.0 * (k - 1)
                d = np.hypot(pred[:, None, 0] - xy[free][None, :, 0],
                             pred[:, None, 1] - xy[free][None, :, 1])
                cost = np.where(d < gate_k[:, None], d, 1e6)
                for i, t in enumerate(cand):
                    for j in range(1, int(k[i])):
                        mid = t['xy'] + t['v'] * j
                        past = state['recent'].get(t['last'] + j)
                        if past is not None and len(past):
                            if (np.hypot(past[:, 0] - mid[0], past[:, 1] - mid[1])
                                    < 0.6 * gate).any():
                                cost[i, :] = 1e6
                                break
                r, c = linear_sum_assignment(cost)
                ok = cost[r, c] < 1e6
                for a, b in zip(r[ok], c[ok]):
                    t = cand[a]
                    kk = int(k[a])
                    bb = int(free[b])
                    dvec = xy[bb] - t['xy']
                    t['v'] = dvec / kk
                    t['xy'] = xy[bb]
                    t['last'] = fid
                    t['idx'] = bb
                    ids[bb] = t['id']
                    gapsz[bb] = kk
                    step[bb] = float(np.hypot(*dvec))
                    used[bb] = True
    # --- new tracks --------------------------------------------------------------
    for b in np.where(~used)[0]:
        ids[b] = state['next_id']
        tracks[state['next_id']] = dict(id=state['next_id'], xy=xy[b], v=np.zeros(2),
                                        last=fid, idx=int(b), idx_prev=-1)
        state['next_id'] += 1
    # remember where each track sat in the previous frame, for previous_detection
    for b in range(m):
        t = tracks.get(int(ids[b]))
        if t is not None:
            t['idx_prev'] = int(b)
    if fid % 100 == 0:
        for kk in [k for k, t in tracks.items() if fid - t['last'] > gap + 2]:
            tracks.pop(kk, None)
    state['recent'][fid] = xy
    if len(state['recent']) > gap + 2:
        state['recent'].pop(min(state['recent']), None)
    state['last_frame'] = fid
    return ids, prev_local, gapsz, step
