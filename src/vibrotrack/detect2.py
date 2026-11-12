"""Detector v2: contact-robust disc detection (2026-11-08).

Why a second detector instead of editing `detect.py`
----------------------------------------------------
`detect.py` produced every archived chunk. Its output must stay reproducible bit for
bit, so v1 is frozen. v2 changes two decisions that the audit showed to be wrong:

1. **Radius is not a free parameter.** All discs are the same physical size
   (36.5-37.4 px measured over 2.6 M detections). Fitting (cx, cy, r) lets the fit lock
   onto a neighbour's rim or the frozen wall and then reject the true rim; with a
   missing arc (contact, wall) it inflates r and fails the ray-count gate.
   v2 estimates the frame radius from the high-confidence detections and then fits the
   CENTRE ONLY, which keeps 46-57 valid rays where v1 found 12-29.

2. **Hard-disc minimum separation is physics, not a quality flag.** v1 records
   `neighbor_distance < 1.85 r` as a quality bit but keeps the detection. v2 enforces
   >= 1.75 r (65 px < 74 px diameter) and keeps the more trustworthy of a pair, which
   is what makes a lower template threshold safe: at a midpoint between two touching
   discs the template score can reach 0.28, but the fixed-radius radial fit passes in
   only 0.4% of such points.

`detect.frame()` below returns the same field set as v1 plus `outside_mask`, so the
chunk contract only grows by one field.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import cv2
from scipy.spatial import cKDTree

from . import detect as V1

cv2.setNumThreads(1)

TEMPLATE = V1.TEMPLATE
CS, SN = V1.CS, V1.SN
RAD = V1.RAD
PX, PY = V1.PX, V1.PY
INNER = V1.INNER
A2 = np.column_stack((CS, SN)).astype(np.float32)

DEFAULTS = dict(thresh=0.20, min_rays=14, r_tol=1.5, max_shift=5.0,
                solid_ns=30, min_sep=1.75, bright_min=80.0, radius_fallback=36.9,
                avoid_r=0.0,
                radius_split_px=4.0,  # two inferred classes need centres at least this far apart
                prior_conf=0.5, prior_dedupe=0.0,   # 0 = keep every seed; the fit rejects stale ones   # a prior seed ranks below a real template peak
                )   # avoid_r 0 = no fixed-disc guard; only wall_model=geometry sets one


def _radius_classes(free_r, explicit=None, min_gap=4.0, min_share=0.05):
    """Radius classes for one frame, from the free-fit radii or an explicit list.

    A bidisperse bed produces two clean peaks in the free-fit radius histogram. One
    dimensional k-means with k=2 finds them; if the centres are closer than ``min_gap``
    or a class holds less than ``min_share`` of the solid peaks, the frame is treated
    as monodisperse and a single median radius is returned.
    """
    r = np.asarray(free_r, float)
    r = r[np.isfinite(r)]
    if explicit is not None:
        vals = np.sort(np.asarray(list(explicit), float).ravel())
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) >= 2:
            return vals
    if len(r) < 40:
        return np.array([np.median(r)]) if len(r) else np.array([])
    c = np.array([np.percentile(r, 10), np.percentile(r, 90)])
    if c[1] - c[0] < min_gap:
        return np.array([float(np.median(r))])
    for _ in range(20):
        lab = np.abs(r[:, None] - c[None, :]).argmin(1)
        new = np.array([r[lab == k].mean() if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    c = np.sort(c)
    if c[1] - c[0] < min_gap:
        return np.array([float(np.median(r))])
    lab = np.abs(r[:, None] - c[None, :]).argmin(1)
    if min((lab == 0).sum(), (lab == 1).sum()) < max(10, int(min_share * len(r))):
        return np.array([float(np.median(r))])
    return c


def _peaks(im, bbox, thresh):
    small = cv2.resize(im, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    if bbox is not None:
        x0, x1, y0, y1 = bbox
        cx0 = max(0, x0 // 2 - 24); cx1 = min(small.shape[1], x1 // 2 + 24)
        cy0 = max(0, y0 // 2 - 24); cy1 = min(small.shape[0], y1 // 2 + 24)
        sub = small[cy0:cy1, cx0:cx1]
    else:
        cx0 = cy0 = 0
        sub = small
    score = cv2.matchTemplate(sub.astype(np.float32) / 255, TEMPLATE, cv2.TM_CCOEFF_NORMED)
    pk = (score == cv2.dilate(score, np.ones((27, 27), np.uint8))) & (score > thresh)
    yy, xx = np.where(pk)
    conf = score[yy, xx]
    xy = np.column_stack(((xx + cx0 + 23) * 2, (yy + cy0 + 23) * 2)).astype(np.float32)
    return xy, conf.astype(np.float32)


def _profile(im, xy):
    mx = xy[:, 0, None, None] + CS[None, :, None] * RAD[None, None, :]
    my = xy[:, 1, None, None] + SN[None, :, None] * RAD[None, None, :]
    prof = cv2.remap(im, mx.reshape(len(xy), -1), my.reshape(len(xy), -1),
                     cv2.INTER_LINEAR).reshape(len(xy), 64, -1).astype(np.float32)
    grad = prof[:, :, :-2] - prof[:, :, 2:]
    k = grad.argmax(2)
    strength = np.take_along_axis(grad, k[:, :, None], 2)[:, :, 0]
    return strength, RAD[1:-1][k]


def _fit_free(strength, rr, min_strength=20, tol=2.7):
    A = np.column_stack((CS, SN, np.ones(64)))
    valid = strength > min_strength
    coef = np.zeros((len(strength), 3))
    for _ in range(3):
        w = valid.astype(float)
        lhs = np.einsum('ni,ij,ik->njk', w, A, A) + np.eye(3)[None] * 1e-5
        rhs = np.einsum('ni,ni,ij->nj', w, rr, A)
        coef = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        resid = rr - coef @ A.T
        valid = (strength > min_strength) & (np.abs(resid) < tol)
    ns = valid.sum(1)
    rmse = np.sqrt((resid ** 2 * valid).sum(1) / np.maximum(ns, 1))
    return coef, ns, rmse


def _fit_fixed(strength, rr, r0, min_strength=20, tol=2.2):
    """Centre-only fit at a known radius."""
    n = len(strength)
    c = np.zeros((n, 2))
    valid = (strength > min_strength) & (np.abs(rr - r0) < tol)
    for _ in range(3):
        w = valid.astype(np.float64)
        M = np.einsum('nr,ri,rj->nij', w, A2, A2) + np.eye(2)[None] * 1e-4
        b = np.einsum('nr,ri,nr->ni', w, A2, rr - r0)
        c = np.linalg.solve(M, b[..., None])[..., 0]
        resid = rr - (c @ A2.T + r0)
        valid = (strength > min_strength) & (np.abs(resid) < tol)
    ns = valid.sum(1)
    rmse = np.sqrt((resid ** 2 * valid).sum(1) / np.maximum(ns, 1))
    return c, ns, rmse


def _brightness(im, xy):
    """80th percentile of the inner disc patch, the photometric veto only.

    ``_marker`` also fits the dark radius mark; the candidate veto uses nothing but
    this brightness, so the first pass does not need the weight/centroid work.
    """
    n = len(xy)
    mx = xy[:, 0, None, None] + PX
    my = xy[:, 1, None, None] + PY
    patch = cv2.remap(im, mx.reshape(n, -1), my.reshape(n, -1),
                      cv2.INTER_LINEAR).reshape(n, 47, 47).astype(np.float32)
    return np.percentile(patch[:, INNER], 80, axis=1)


def _marker(im, xy):
    """Same estimator as v1: weighted centroid of the dark radius mark."""
    n = len(xy)
    mx = xy[:, 0, None, None] + PX
    my = xy[:, 1, None, None] + PY
    patch = cv2.remap(im, mx.reshape(n, -1), my.reshape(n, -1),
                      cv2.INTER_LINEAR).reshape(n, 47, 47).astype(np.float32)
    bright = np.percentile(patch[:, INNER], 80, axis=1)
    dark = (patch < bright[:, None, None] * .54) & INNER[None]
    weight = np.maximum(bright[:, None, None] * .54 - patch, 0) * dark
    area = dark.sum((1, 2))
    mass = weight.sum((1, 2))
    dx = (weight * PX).sum((1, 2)) / np.maximum(mass, 1)
    dy = (weight * PY).sum((1, 2)) / np.maximum(mass, 1)
    offset = np.hypot(dx, dy)
    ok = (area >= 100) & (area <= 380) & (offset >= 2) & (offset <= 16) & (bright > 85)
    theta = np.arctan2(-dy, dx).astype(np.float32)
    theta[~ok] = np.nan
    return theta, ok, offset.astype(np.float32), area.astype(np.int16), bright


def frame(im, search_mask, dist, bbox=None, interior_mask=None, static_centres=None,
          prior_xy=None, radii=None, **kw):
    """Detect discs. `search_mask` gates candidate centres (may be dilated);

    `interior_mask` (defaults to search_mask) defines `outside_mask` in the output.
    `static_centres` (N,2) are the fixed apparatus discs: a real particle centre cannot be
    closer than 2r to one of them, so candidates inside `avoid_r` are the discs themselves.

    `prior_xy` (N,2) are the positions detected in the previous frame. They join the
    template peaks as candidate seeds. This matters because the template is the fragile
    stage: a bead squeezed against a neighbour changes appearance enough that its
    correlation drops below `thresh` (measured 0.148 vs 0.20 on the reference dense data), while a fit
    seeded at the true position still returns 40 clean rays and rmse 0.85. The acceptance
    gates are unchanged, so a stale prediction simply fails the fit.
    Returns (detection dict, info dict).
    """
    p = dict(DEFAULTS)
    p.update({k: v for k, v in kw.items() if k in DEFAULTS})
    interior = search_mask if interior_mask is None else interior_mask

    xy0, conf0 = _peaks(im, bbox, p['thresh'])
    keep = search_mask[np.clip(xy0[:, 1].astype(int), 0, search_mask.shape[0] - 1),
                       np.clip(xy0[:, 0].astype(int), 0, search_mask.shape[1] - 1)] > 0
    xy0, conf0 = xy0[keep], conf0[keep]
    n_template = len(xy0)
    n_prior = 0
    if prior_xy is not None and len(prior_xy):
        pr = np.asarray(prior_xy, np.float32).reshape(-1, 2)
        pi = np.clip(pr[:, 1].astype(int), 0, search_mask.shape[0] - 1)
        pj = np.clip(pr[:, 0].astype(int), 0, search_mask.shape[1] - 1)
        pr = pr[search_mask[pi, pj] > 0]
        if len(pr) and len(xy0) and p['prior_dedupe'] > 0:
            # a prior that sits on a template peak adds nothing but cost
            dmin, _ = cKDTree(xy0.astype(np.float64)).query(pr.astype(np.float64))
            pr = pr[dmin > p['prior_dedupe']]
        if len(pr):
            n_prior = len(pr)
            xy0 = np.vstack([xy0, pr]).astype(np.float32)
            conf0 = np.r_[conf0, np.full(len(pr), p['prior_conf'], np.float32)]
    if not len(xy0):
        return _empty(), dict(radius_ref=p['radius_fallback'], n_peaks=0, n_recovered=0,
                              n_static=0, n_prior=0, radii=[])

    strength, rr = _profile(im, xy0)
    coef, ns1, rmse1 = _fit_free(strength, rr)
    shift1 = np.linalg.norm(coef[:, :2], axis=1)
    solid = (ns1 >= p['solid_ns']) & (rmse1 < 1.8) & (shift1 < p['max_shift']) & \
            (coef[:, 2] > 30) & (coef[:, 2] < 42)
    # The free-fit radius of a 12 mm disc can land just under 30 px, so the class
    # inference uses a wider window; the single-radius path keeps the strict `solid`
    # definition so monodisperse output is unchanged.
    broad = (ns1 >= p['solid_ns']) & (rmse1 < 1.8) & (shift1 < 10.0) & \
            (coef[:, 2] > 27) & (coef[:, 2] < 42)
    radius_ref = float(np.median(coef[solid, 2])) if solid.sum() >= 20 else p['radius_fallback']
    classes = _radius_classes(coef[broad, 2], radii,
                              min_gap=float(p['radius_split_px']))
    if len(classes) > 1:
        # Bidisperse: assign each candidate to its nearest free-fit radius class and run
        # the fixed-radius fit per class, so a 12 mm disc is not forced onto a 14 mm rim.
        lab = np.abs(coef[:, 2][:, None] - classes[None, :]).argmin(1)
        c = np.zeros((len(xy0), 2), np.float64)
        ns = np.zeros(len(xy0), np.int64)
        rmse = np.zeros(len(xy0), np.float64)
        rad_all = np.zeros(len(xy0), np.float32)
        for k, rk in enumerate(classes):
            m = lab == k
            if not m.any():
                continue
            ck, nsk, rmsek = _fit_fixed(strength[m], rr[m], float(rk))
            c[m], ns[m], rmse[m] = ck, nsk, rmsek
            rad_all[m] = float(rk)
        radius_ref = float(classes[int(np.argmax([np.sum(lab == k)
                                                  for k in range(len(classes))]))])
    else:
        rad_all = np.full(len(xy0), radius_ref, np.float32)
        c, ns, rmse = _fit_fixed(strength, rr, radius_ref)
    shift = np.linalg.norm(c, axis=1)
    xy = xy0 + c.astype(np.float32)
    shift_tol = np.full(len(xy0), float(p['max_shift']))
    min_rays_pt = np.full(len(xy0), float(p['min_rays']))
    smaller = np.zeros(len(xy0), bool)
    if len(classes) > 1:
        # A 12 mm disc does not match the 14 mm template, so its peak can sit several
        # pixels off centre; allow the smaller class twice the search shift. Its ray
        # count is also the strongest separator from a pocket between beads (real
        # small discs keep 31-53 rays, pocket fits 18-31), so require twice the rays.
        main_r = float(classes[int(np.argmax([np.sum(rad_all == rk) for rk in classes]))])
        smaller = rad_all < 0.9 * main_r
        shift_tol[smaller] = float(p['max_shift']) * 2.0
        min_rays_pt[smaller] = float(p['min_rays']) * 2.0
    ok = (ns >= min_rays_pt) & (rmse < p['r_tol']) & (shift < shift_tol)
    if len(classes) > 1 and n_prior:
        # A prior seed for the small class only re-finds the same pocket: pocket fits
        # are stable frame to frame, so seeding them multiplies false positives
        # (frame 100: 174 -> 290 small detections). Small discs must come from a
        # template peak; the 14 mm class may still use priors.
        prior_only = np.zeros(len(xy0), bool)
        prior_only[n_template:] = True
        ok &= ~(smaller & prior_only)

    # Fixed apparatus discs are perfect particle look-alikes. A real centre cannot sit closer
    # than 2r to one, so anything inside avoid_r is the disc, not a particle. Filter before the
    # NMS so a disc detection cannot suppress a genuine neighbour.
    n_static = 0
    if static_centres is not None and len(static_centres):
        sc = np.asarray(static_centres, np.float32).reshape(-1, 2)
        dmin = np.sqrt(((xy[:, None, :] - sc[None, :, :]) ** 2).sum(-1)).min(1)
        hit = ok & (dmin < p['avoid_r'])
        n_static = int(hit.sum())
        ok = ok & ~hit

    # photometric veto: gaps between discs are dark
    cand = np.where(ok)[0]
    if len(cand):
        bright_t = _brightness(im, xy[cand])
        ok[cand[bright_t <= p['bright_min']]] = False

    # hard-disc minimum separation, higher trust wins
    acc = np.where(ok)[0]
    if len(acc) > 1:
        rank = conf0[acc] + 10.0 * solid[acc]
        order = acc[np.argsort(-rank)]
        pts = xy[order]
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(d, 1e9)
        keep2 = np.ones(len(order), bool)
        rr_o = rad_all[order]
        # Use the larger of the two radii: a small disc must not be allowed to sit
        # inside a large one just because its own separation threshold is smaller.
        thr = p['min_sep'] * np.maximum(rr_o[:, None], rr_o[None, :])
        for i in range(len(order)):
            if keep2[i]:
                keep2[d[i] < thr[i]] = False
                keep2[i] = True
        ok[order[~keep2]] = False

    sel = np.where(ok)[0]
    if not len(sel):
        return _empty(), dict(radius_ref=radius_ref, n_peaks=int(len(xy0) - n_prior),
                              n_recovered=0, n_static=n_static, n_prior=n_prior,
                              radii=[float(x) for x in classes])
    xy_s = xy[sel]
    n = len(xy_s)
    theta, angle_valid, offset, area, bright = _marker(im, xy_s)
    nearest = cKDTree(xy_s).query(xy_s, k=2)[0][:, 1] if n > 1 else np.array([np.inf])
    bd = dist[np.clip(xy_s[:, 1].astype(int), 0, dist.shape[0] - 1),
              np.clip(xy_s[:, 0].astype(int), 0, dist.shape[1] - 1)] if dist is not None \
        else np.full(n, np.nan)
    rad = rad_all[sel]
    inside = interior[np.clip(xy_s[:, 1].astype(int), 0, interior.shape[0] - 1),
                      np.clip(xy_s[:, 0].astype(int), 0, interior.shape[1] - 1)] > 0
    bits = (~angle_valid).astype(np.uint16) | ((nearest < 1.85 * rad).astype(np.uint16) << 1) \
        | ((bd < rad + 4).astype(np.uint16) << 2)
    det = dict(xy_px=xy_s.astype(np.float32), radius_px=rad, theta_rad=theta,
               center_valid=np.ones(n, bool), angle_valid=angle_valid,
               confidence=conf0[sel].astype(np.float32),
               edge_rmse_px=rmse[sel].astype(np.float32),
               marker_offset_px=offset, marker_area_px=area,
               neighbor_distance_px=nearest.astype(np.float32),
               boundary_distance_px=bd.astype(np.float32),
               quality_bits=bits,
               outside_mask=(~inside).astype(np.uint8))
    return det, dict(radius_ref=radius_ref, n_peaks=int(len(xy0) - n_prior), n_recovered=n,
                     n_static=n_static, n_prior=n_prior,
                     radii=[float(x) for x in classes])


def _empty():
    z = V1.empty_detection()
    z['outside_mask'] = np.zeros(0, np.uint8)
    return z


def reachable_mask(shape, centres, disc_radii, particle_radius, scale=1.0):
    """Region a particle CENTRE can occupy, given the fixed discs.

    A centre must stay one particle radius away from every disc edge, i.e. at least
    `particle_radius + disc_radius` from each disc centre. Built as a distance transform
    from the union of the discs, so it handles overlapping exclusion zones exactly.
    """
    m = np.zeros(shape, np.uint8)
    for (x, y), r in zip(np.asarray(centres, float), np.asarray(disc_radii, float)):
        cv2.circle(m, (int(round(x)), int(round(y))), max(1, int(round(r))), 1, -1)
    clear = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    need = float(particle_radius) * float(scale)
    return (clear >= need).astype(np.uint8)


def search_mask(interior, dilate_px=40):
    """Candidate search region: the interior mask grown by `dilate_px`.

    The frozen wall ring sits just outside the interior mask; a particle squeezed
    against it is a real particle whose centre may fall outside. Growing the search
    region lets it be detected; the static-wall post-filter removes the wall discs
    themselves.
    """
    if dilate_px <= 0:
        return interior
    k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    return cv2.dilate(interior, k)
