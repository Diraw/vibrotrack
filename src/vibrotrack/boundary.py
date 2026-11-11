"""Propose a boundary mask from a recording, with a human-checkable overlay.

The mask is deliberately written as a *proposal*: automatic boundary detection is
not trusted blindly.  The command writes ``mask_overlay.png`` (the real frame plus a
semi-transparent green interior and a bright outline) and ``mask.json``.  A human
looks at the overlay before the mask is used for extraction.

The detector is independent of the lab's previous static masks.  It uses the
particle-filled interior as the signal:

- sample frames across the recording and take a temporal maximum (fills gaps left
  by moving particles);
- try a robust circular Hough fit first (the usual circular cell);
- fall back to the outer foreground contour / convex hull (rectangular or hex cells);
- add a small margin so particle centres at the rim are not clipped.

This module never writes into the package or the raw-frame directory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import detect as X
from . import schema

SCHEMA = 'mask-proposal/v1'
DEFAULT_FRAMES = 12
DEFAULT_MARGIN_PX = 2.0


def _sample_indices(n, k):
    if n <= 0:
        raise ValueError('no frames to sample')
    k = max(1, int(k))
    if n <= k:
        return list(range(n))
    return sorted(set(int(round(v)) for v in np.linspace(0, n - 1, k)))


def _probe(files, indices, levels):
    """Return (temporal mean, temporal maximum, temporal std).

    The maximum recovers the filled interior.  The standard deviation separates
    moving particles (high std) from frozen particles / a static wall (low std).
    The mean is used to find the individual frozen discs on the boundary.
    """
    probe = None
    total = None
    total_sq = None
    n = 0
    for i in indices:
        a = X.read_image(files[int(i)], levels)
        if a.ndim != 2:
            raise ValueError('boundary proposal needs single-channel frames, got shape %r' % (a.shape,))
        a = a.astype(np.float32)
        if probe is None:
            probe = a.copy()
            total = a.copy()
            total_sq = a * a
        else:
            np.maximum(probe, a, out=probe)
            total += a
            total_sq += a * a
        n += 1
    if probe is None or n == 0:
        raise ValueError('no readable frames')
    mean = total / n
    var = total_sq / n - mean * mean
    return (mean.astype(np.uint8), probe.astype(np.uint8),
            np.sqrt(np.maximum(var, 0.0)).astype(np.float32))


def _otsu(probe, threshold=None):
    blur = cv2.GaussianBlur(probe, (5, 5), 0)
    if threshold is None:
        level, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        level = float(threshold)
        _, fg = cv2.threshold(blur, level, 255, cv2.THRESH_BINARY)
    return float(level), fg


def _hough_circle(probe):
    """Return (cx, cy, r, score) in full-resolution coordinates, or None."""
    h, w = probe.shape
    scale = min(1.0, 1600.0 / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(probe, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = probe
    blur = cv2.GaussianBlur(small, (9, 9), 2)
    sh, sw = blur.shape
    short = min(sh, sw)
    min_r = max(12, int(0.32 * short))
    max_r = max(min_r + 1, int(0.58 * short))
    best = None
    for param2 in (100, 80, 60):
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=2,
            minDist=max(20, int(0.25 * short)),
            param1=80, param2=param2,
            minRadius=min_r, maxRadius=max_r)
        if circles is None:
            continue
        for x, y, r in np.asarray(circles[0], dtype=np.float64):
            d = math.hypot(x - sw / 2.0, y - sh / 2.0)
            # A container that is roughly centred; this rejects accidental circles
            # fitted to dense particle patches far from the optical axis.
            if d > 0.16 * short:
                continue
            score = float(r) - 0.35 * float(d)
            if best is None or score > best[3]:
                best = (x / scale, y / scale, r / scale, score)
        if best is not None:
            break
    return best


def _foreground_mask(probe, threshold):
    level, fg = _otsu(probe, threshold)
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Bright reflections at the image border are not part of the cell.  Remove a
    # narrow frame before selecting the main component; a real cell that touches
    # the border still keeps its interior.
    border = max(2, int(0.02 * min(fg.shape)))
    fg[:border, :] = 0
    fg[-border:, :] = 0
    fg[:, :border] = 0
    fg[:, -border:] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        raise ValueError('no foreground component found; lower --threshold or supply a mask manually')
    # Keep the largest component, but absorb other large components only if they
    # are close to it.  This avoids a stray bright object expanding the hull.
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    clean = (lab == k).astype(np.uint8) * 255
    return level, clean


def _contour_circle(probe, threshold):
    level, fg = _foreground_mask(probe, threshold)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('no contour found')
    c = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    return float(x), float(y), float(r), c


def _contour_polygon(probe, threshold):
    """Return a smoothed, non-convex boundary polygon and the threshold."""
    level, fg = _foreground_mask(probe, threshold)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('no contour found')
    c = max(contours, key=cv2.contourArea)
    perimeter = max(1.0, cv2.arcLength(c, True))
    # Keep the actual concavities; only remove pixel-level stair steps.  This is
    # deliberately not a convex hull: an L-shaped or keyhole cell must keep its
    # missing corner outside the mask.
    c = cv2.approxPolyDP(c, max(1.0, 0.0015 * perimeter), True)
    if len(c) < 3:
        c = cv2.convexHull(c)
    return c.reshape(-1, 2), float(level)


def _hull(contour):
    return cv2.convexHull(contour)


def _static_ring_inner_radius(std, center, outer_radius):
    """Find the inner edge of a frozen-particle ring, if one is present.

    A frozen ring has high mean brightness but low temporal std, while the active
    interior has high std.  Scan the radial std profile from the outer circle inward
    and find where it rises above a fraction of the interior level.

    Returns ``(radius, detected, profile_info)``.  When no clear drop is found,
    ``radius`` is the outer radius and ``detected`` is False.
    """
    h, w = std.shape
    yy, xx = np.indices(std.shape)
    r = np.hypot(xx - center[0], yy - center[1])
    hi = int(max(20, min(outer_radius, min(h, w) / 2.0 - 1)))
    lo = int(max(8, 0.45 * hi))
    bins = np.arange(0, hi + 2)
    idx = np.clip(r.astype(np.int32), 0, len(bins) - 2)
    counts = np.bincount(idx.ravel(), minlength=len(bins))[:-1]
    sums = np.bincount(idx.ravel(), weights=std.ravel(), minlength=len(bins))[:-1]
    prof = sums / np.maximum(counts, 1)
    interior = prof[int(0.45 * hi):int(0.80 * hi)]
    active_level = float(np.median(interior)) if len(interior) else 0.0
    if active_level < 3.0:
        return float(outer_radius), False, {
            'active_std': active_level, 'std_threshold': None,
            'ring_inner_radius_px': None,
        }
    threshold = max(2.0, 0.25 * active_level)
    # From the outer edge inward: the first bin above threshold is the last
    # moving-particle radius.  It is the inner edge of the frozen ring.
    transition = None
    for i in range(hi, lo, -1):
        if prof[i] > threshold:
            transition = i
            break
    if transition is None or transition >= 0.97 * hi:
        return float(outer_radius), False, {
            'active_std': active_level, 'std_threshold': threshold,
            'ring_inner_radius_px': None,
        }
    return float(transition), True, {
        'active_std': active_level, 'std_threshold': threshold,
        'ring_inner_radius_px': float(transition),
    }


def _frozen_disks(mean, std, *, center, base_radius, outer_radius,
                  particle_radius_px, active_std):
    """Find individual static/frozen discs on the boundary.

    A boundary is often a circular frame *plus* discrete frozen particles sitting
    on it.  A single radial cutoff cannot represent that: it cuts through the
    frozen discs and leaves crescents inside the mask.  Detect small circles in
    the temporal mean, keep those whose interior has low temporal std, and let
    the caller subtract them as circular bites.
    """
    if particle_radius_px is None or particle_radius_px <= 0:
        return []
    h, w = mean.shape
    scale = min(1.0, 1600.0 / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(mean, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = mean
    blur = cv2.GaussianBlur(small, (7, 7), 1.5)
    rmin = max(4, int(round(0.55 * particle_radius_px * scale)))
    rmax = max(rmin + 2, int(round(1.45 * particle_radius_px * scale)))
    min_dist = max(8, int(round(0.85 * particle_radius_px * scale)))
    circles = None
    for param2 in (30, 24, 18):
        found = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min_dist,
            param1=80, param2=param2, minRadius=rmin, maxRadius=rmax)
        if found is not None and len(found[0]):
            circles = found[0]
            break
    if circles is None:
        return []
    if active_std is None or not np.isfinite(active_std) or active_std <= 0:
        static_threshold = 6.0
    else:
        static_threshold = max(3.0, 0.25 * float(active_std))
    bright_min = max(15.0, 0.20 * float(np.percentile(mean, 95)))
    candidates = []
    for x, y, r in np.asarray(circles, dtype=np.float64):
        x /= scale
        y /= scale
        r /= scale
        rad = math.hypot(x - center[0], y - center[1])
        # Only discs that intersect the active-region boundary can create a
        # crescent.  The outer ring itself is also allowed; it usually lies just
        # outside the base circle and is harmless.
        if rad < base_radius - 2.2 * particle_radius_px:
            continue
        if rad > outer_radius + 0.5 * particle_radius_px:
            continue
        b = int(math.ceil(0.65 * r))
        x0, x1 = max(0, int(round(x)) - b), min(w, int(round(x)) + b + 1)
        y0, y1 = max(0, int(round(y)) - b), min(h, int(round(y)) + b + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        sy, sx = np.ogrid[y0:y1, x0:x1]
        inside = (sx - x) ** 2 + (sy - y) ** 2 <= (0.65 * r) ** 2
        if not inside.any():
            continue
        std_med = float(np.median(std[y0:y1, x0:x1][inside]))
        mean_med = float(np.median(mean[y0:y1, x0:x1][inside]))
        if std_med > static_threshold or mean_med < bright_min:
            continue
        candidates.append({
            'center_px': [float(x), float(y)],
            'radius_px': float(r),
            'std': std_med,
            'mean': mean_med,
        })
    # Hough can return the same disc more than once; keep the first candidate.
    candidates.sort(key=lambda q: (math.hypot(q['center_px'][0] - center[0],
                                               q['center_px'][1] - center[1]),
                                   q['center_px'][1], q['center_px'][0]))
    keep = []
    min_sep = 0.8 * particle_radius_px
    for q in candidates:
        if all(math.hypot(q['center_px'][0] - p['center_px'][0],
                          q['center_px'][1] - p['center_px'][1]) >= min_sep
               for p in keep):
            keep.append(q)
    return keep


def _circle_mask(shape, cx, cy, radius, margin):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))),
               int(round(radius + margin)), 255, -1)
    return mask


def _polygon_mask(shape, hull, margin):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.asarray(hull, dtype=np.int32)], 255)
    # Positive margin shrinks the mask inward (away from a frozen ring/wall);
    # a negative margin expands it for a deliberately generous search region.
    if margin:
        k = 2 * int(math.ceil(abs(margin))) + 1
        op = cv2.erode if margin > 0 else cv2.dilate
        mask = op(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def _overlay(frame, mask, *, shape, center=None, radius=None, outer_radius=None,
             ring_detected=False, frozen_disks=None, hull=None,
             label='proposed mask'):
    frozen_disks = list(frozen_disks or [])
    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    inside = mask > 0
    green = np.zeros_like(rgb)
    green[..., 1] = 255
    rgb[inside] = (0.65 * rgb[inside] + 0.35 * green[inside]).astype(np.uint8)
    if shape == 'circle' and center is not None and radius is not None:
        if frozen_disks:
            # The boundary is a circle with circular bites, not a plain circle.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (0, 255, 0), 4)
        else:
            cv2.circle(rgb, (int(round(center[0])), int(round(center[1]))),
                       int(round(radius)), (0, 255, 0), 4)
        if ring_detected and outer_radius is not None and outer_radius > radius + 1:
            cv2.circle(rgb, (int(round(center[0])), int(round(center[1]))),
                       int(round(outer_radius)), (255, 165, 0), 3)
    elif hull is not None:
        cv2.polylines(rgb, [np.asarray(hull, np.int32)], True, (0, 255, 0), 4)
    for disc in frozen_disks:
        x, y = disc['center_px']
        cv2.circle(rgb, (int(round(x)), int(round(y))),
                   int(round(disc['radius_px'])), (255, 140, 0), 2)
    text = '%s  %s' % (label, shape)
    if shape == 'circle' and center is not None and radius is not None:
        text += '  c=(%.1f, %.1f)  mask r=%.1f' % (center[0], center[1], radius)
        if ring_detected and outer_radius is not None:
            text += '  outer r=%.1f' % outer_radius
        if frozen_disks:
            text += '  frozen discs=%d' % len(frozen_disks)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(rgb, (18, 18), (28 + tw, 28 + th + 18), (0, 0, 0), -1)
    cv2.putText(rgb, text, (24, 24 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2, cv2.LINE_AA)
    return rgb


def load_points(path):
    """Load a manual polygon from JSON.

    Accepted forms::

        [[x, y], ...]
        {"points_px": [[x, y], ...], "holes_px": [[...], ...]}
    """
    rec = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(rec, list):
        points = rec
        holes = []
    elif isinstance(rec, dict):
        points = rec.get('points_px', rec.get('points'))
        holes = rec.get('holes_px', [])
    else:
        raise ValueError('manual points JSON must be a list or an object')
    if not points or len(points) < 3:
        raise ValueError('manual polygon needs at least three points')

    def clean(poly):
        out = []
        for p in poly:
            if len(p) != 2:
                raise ValueError('each point must be [x, y]')
            out.append([float(p[0]), float(p[1])])
        return out
    return clean(points), [clean(h) for h in holes if len(h) >= 3]


def _manual_mask(shape, points, holes, margin_px):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    for hole in holes:
        cv2.fillPoly(mask, [np.asarray(hole, dtype=np.int32)], 0)
    if margin_px:
        k = 2 * int(math.ceil(abs(margin_px))) + 1
        op = cv2.erode if margin_px > 0 else cv2.dilate
        mask = op(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def propose(files, *, levels=(128.0, 28224.0), n_frames=DEFAULT_FRAMES,
            shape='auto', margin_px=DEFAULT_MARGIN_PX, threshold=None,
            preview_frame=None, particle_radius_px=None):
    """Return ``(mask, info, overlay_rgb)`` for a list of frame paths."""
    if not files:
        raise ValueError('no frames supplied')
    indices = _sample_indices(len(files), n_frames)
    mean, probe, std = _probe(files, indices, levels)
    h, w = probe.shape
    shape = str(shape or 'auto').lower()
    if shape not in ('auto', 'circle', 'polygon', 'irregular'):
        raise ValueError("shape must be 'auto', 'circle', 'polygon' or 'irregular'")

    contour = None
    polygon = None
    threshold_used = None
    outer_radius = None
    ring_detected = False
    ring_info = {}
    frozen = []
    radius = None
    center = None
    hull = None
    info_shape = None

    if shape in ('polygon', 'irregular'):
        polygon, threshold_used = _contour_polygon(probe, threshold)
        if shape == 'polygon':
            polygon = _hull(polygon).reshape(-1, 2)
        method = 'contour_irregular' if shape == 'irregular' else 'contour_polygon'
        info_shape = 'irregular' if shape == 'irregular' else 'polygon'
    else:
        hough = _hough_circle(probe) if shape in ('auto', 'circle') else None
        if hough is None and shape == 'circle':
            cx, cy, r, contour = _contour_circle(probe, threshold)
            method = 'contour_circle'
        elif hough is None:
            polygon, threshold_used = _contour_polygon(probe, threshold)
            method = 'contour_irregular'
            info_shape = 'irregular'
        else:
            cx, cy, r, _ = hough
            method = 'hough_circle'

        if shape == 'auto' and method == 'hough_circle':
            # A circle is accepted only when it covers most of the foreground.
            # An L-shaped, keyhole, hex or rectangular cell would leave large
            # corners outside a fitted circle; use the concave contour instead.
            level, fg = _foreground_mask(probe, threshold)
            fg_area = float((fg > 0).sum())
            circle_area = math.pi * max(r, 1.0) ** 2
            if fg_area > 0 and circle_area / fg_area < 0.80:
                polygon, threshold_used = _contour_polygon(probe, threshold)
                method = 'contour_irregular'
                info_shape = 'irregular'
            else:
                info_shape = 'circle'
        elif shape == 'auto':
            if method == 'contour_circle':
                info_shape = 'circle'
            else:
                info_shape = 'irregular'
        elif shape == 'circle':
            info_shape = 'circle'

    if info_shape == 'circle':
        outer_radius = float(r)
        ring_radius, ring_detected, ring_info = _static_ring_inner_radius(
            std, (cx, cy), outer_radius)
        boundary_radius = ring_radius if ring_detected else outer_radius
        radius = max(1.0, boundary_radius - float(margin_px))
        mask = _circle_mask((h, w), cx, cy, radius, 0.0)
        center = [float(cx), float(cy)]
        if ring_detected:
            method += '+static_ring'
            frozen = _frozen_disks(
                mean, std, center=(cx, cy), base_radius=radius,
                outer_radius=outer_radius,
                particle_radius_px=particle_radius_px,
                active_std=ring_info.get('active_std'))
            if frozen:
                bite_radius = float(particle_radius_px)
                for disc in frozen:
                    cv2.circle(
                        mask,
                        (int(round(disc['center_px'][0])),
                         int(round(disc['center_px'][1]))),
                        int(round(bite_radius + 1.0)), 0, -1)
                method += '+frozen_disks'
    else:
        if polygon is None:
            polygon, threshold_used = _contour_polygon(probe, threshold)
            if info_shape == 'polygon':
                polygon = _hull(polygon).reshape(-1, 2)
        polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(polygon) < 3:
            raise ValueError('irregular boundary needs at least three points')
        mask = _polygon_mask((h, w), polygon, float(margin_px))
        hull = polygon.reshape(-1, 1, 2)
        # Mean of the boundary points; the polygon is already in image
        # coordinates and this is only a display/metadata centre.
        center = [float(polygon[:, 0].mean()), float(polygon[:, 1].mean())]
        info_shape = info_shape or 'irregular'

    preview_idx = indices[len(indices) // 2] if preview_frame is None else int(preview_frame)
    preview_idx = max(0, min(len(files) - 1, preview_idx))
    preview = X.read_image(files[preview_idx], levels)
    overlay = _overlay(preview, mask, shape=info_shape, center=center,
                       radius=radius, outer_radius=outer_radius,
                       ring_detected=ring_detected, frozen_disks=frozen, hull=hull)
    info = {
        'schema': SCHEMA,
        'status': 'proposed',
        'created_at': schema.now_iso(),
        'method': method,
        'shape': info_shape,
        'center_px': center,
        'radius_px': radius,
        'outer_radius_px': outer_radius,
        'ring_detected': bool(ring_detected),
        'particle_radius_px': (float(particle_radius_px)
                               if particle_radius_px is not None else None),
        'frozen_disc_count': int(len(frozen)),
        'frozen_discs': frozen,
        'margin_px': float(margin_px),
        'threshold': threshold_used,
        **ring_info,
        'image_shape': [int(h), int(w)],
        'frame_count': int(len(files)),
        'sample_indices': [int(i) for i in indices],
        'preview_index': int(preview_idx),
        'mask_coverage': float((mask > 0).mean()),
        'notes': 'inspect mask_overlay.png before using interior_mask.png',
    }
    return mask, info, overlay


def propose_manual(files, points, holes=None, *, levels=(128.0, 28224.0),
                    preview_frame=None, margin_px=0.0):
    """Build a mask from a human-drawn polygon; no automatic boundary fit."""
    if not files:
        raise ValueError('no frames supplied')
    holes = list(holes or [])
    first = X.read_image(files[0], levels)
    h, w = first.shape
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('manual polygon needs at least three [x, y] points')
    if points[:, 0].min() < 0 or points[:, 0].max() >= w or points[:, 1].min() < 0 or points[:, 1].max() >= h:
        raise ValueError('manual points fall outside the frame')
    for hole in holes:
        hole = np.asarray(hole, dtype=np.float64)
        if hole.ndim != 2 or hole.shape[1] != 2 or len(hole) < 3:
            raise ValueError('each manual hole needs at least three points')
        if hole[:, 0].min() < 0 or hole[:, 0].max() >= w or hole[:, 1].min() < 0 or hole[:, 1].max() >= h:
            raise ValueError('manual hole points fall outside the frame')
    mask = _manual_mask((h, w), points, holes, float(margin_px))
    preview_idx = len(files) // 2 if preview_frame is None else int(preview_frame)
    preview_idx = max(0, min(len(files) - 1, preview_idx))
    preview = X.read_image(files[preview_idx], levels)
    # Reuse the same green confirmation style as the automatic proposals.
    overlay = _overlay(preview, mask, shape='irregular', hull=points.reshape(-1, 1, 2),
                       label='manual mask')
    info = {
        'schema': SCHEMA,
        'status': 'manual',
        'created_at': schema.now_iso(),
        'method': 'manual_polygon',
        'shape': 'irregular',
        'center_px': [float(points[:, 0].mean()), float(points[:, 1].mean())],
        'radius_px': None,
        'outer_radius_px': None,
        'ring_detected': False,
        'margin_px': float(margin_px),
        'threshold': None,
        'image_shape': [int(h), int(w)],
        'frame_count': int(len(files)),
        'sample_indices': [],
        'preview_index': int(preview_idx),
        'mask_coverage': float((mask > 0).mean()),
        'points_px': points.tolist(),
        'holes_px': [np.asarray(h, dtype=np.float64).tolist() for h in holes],
        'notes': 'manually drawn mask; inspect mask_overlay.png before using interior_mask.png',
    }
    return mask, info, overlay


def propose_from_raw(raw, out, config, *, n_frames=DEFAULT_FRAMES, shape='auto',
                     margin_px=DEFAULT_MARGIN_PX, threshold=None,
                     preview_frame=None, force=False, points=None):
    """Create ``interior_mask.png``, ``mask_overlay.png`` and ``mask.json``."""
    raw = Path(raw).expanduser().resolve()
    if not raw.is_dir():
        raise FileNotFoundError('raw frame directory not found: %s' % raw)
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mask_path = out / 'interior_mask.png'
    overlay_path = out / 'mask_overlay.png'
    json_path = out / 'mask.json'
    if not force:
        existing = [p for p in (mask_path, overlay_path, json_path) if p.exists()]
        if existing:
            raise FileExistsError(
                'mask proposal already exists (%s); pass --force to overwrite'
                % ', '.join(str(p) for p in existing))
    files = X.files_for({'path': str(raw), 'extensions': tuple(config.extensions)})
    if not files:
        raise FileNotFoundError('no frames found in %s' % raw)
    if points is not None:
        manual_points, manual_holes = load_points(points)
        mask, info, overlay = propose_manual(
            files, manual_points, manual_holes, levels=config.scale_levels,
            preview_frame=preview_frame, margin_px=margin_px)
        info['points_file'] = str(Path(points).expanduser().resolve())
    else:
        mask, info, overlay = propose(
            files, levels=config.scale_levels, n_frames=n_frames, shape=shape,
            margin_px=margin_px, threshold=threshold, preview_frame=preview_frame,
            particle_radius_px=config.radius_px)
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)
    info.update({
        'raw': str(raw),
        'mask_file': mask_path.name,
        'mask_sha256': schema.sha256(mask_path),
        'overlay_file': overlay_path.name,
        'overlay_sha256': schema.sha256(overlay_path),
    })
    schema.write_json(json_path, info)
    return info, mask_path, overlay_path, json_path
