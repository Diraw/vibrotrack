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
    probe = None
    for i in indices:
        a = X.read_image(files[int(i)], levels)
        if a.ndim != 2:
            raise ValueError('boundary proposal needs single-channel frames, got shape %r' % (a.shape,))
        if probe is None:
            probe = a.copy()
        else:
            np.maximum(probe, a, out=probe)
    if probe is None:
        raise ValueError('no readable frames')
    return probe


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


def _hull(contour):
    return cv2.convexHull(contour)


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
    if margin > 0:
        k = 2 * int(math.ceil(margin)) + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def _overlay(frame, mask, *, shape, center=None, radius=None, hull=None,
             label='proposed mask'):
    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    inside = mask > 0
    green = np.zeros_like(rgb)
    green[..., 1] = 255
    rgb[inside] = (0.65 * rgb[inside] + 0.35 * green[inside]).astype(np.uint8)
    if shape == 'circle' and center is not None and radius is not None:
        cv2.circle(rgb, (int(round(center[0])), int(round(center[1]))),
                   int(round(radius)), (0, 255, 0), 4)
    elif hull is not None:
        cv2.polylines(rgb, [np.asarray(hull, np.int32)], True, (0, 255, 0), 4)
    text = '%s  %s' % (label, shape)
    if shape == 'circle' and center is not None and radius is not None:
        text += '  c=(%.1f, %.1f)  r=%.1f px' % (center[0], center[1], radius)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(rgb, (18, 18), (28 + tw, 28 + th + 18), (0, 0, 0), -1)
    cv2.putText(rgb, text, (24, 24 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2, cv2.LINE_AA)
    return rgb


def propose(files, *, levels=(128.0, 28224.0), n_frames=DEFAULT_FRAMES,
            shape='auto', margin_px=DEFAULT_MARGIN_PX, threshold=None,
            preview_frame=None):
    """Return ``(mask, info, overlay_rgb)`` for a list of frame paths."""
    if not files:
        raise ValueError('no frames supplied')
    indices = _sample_indices(len(files), n_frames)
    probe = _probe(files, indices, levels)
    h, w = probe.shape
    shape = str(shape or 'auto').lower()
    if shape not in ('auto', 'circle', 'polygon'):
        raise ValueError("shape must be 'auto', 'circle' or 'polygon'")

    hough = _hough_circle(probe) if shape in ('auto', 'circle') else None
    contour = None
    if hough is None:
        cx, cy, r, contour = _contour_circle(probe, threshold)
        method = 'contour_circle' if shape != 'polygon' else 'contour_polygon'
    else:
        cx, cy, r, _ = hough
        method = 'hough_circle'

    # For auto, a circle is accepted only when it covers most of the foreground.
    # A hex/rectangular cell would leave large corners outside a fitted circle.
    if shape == 'auto' and method == 'hough_circle':
        level, fg = _foreground_mask(probe, threshold)
        fg_area = float((fg > 0).sum())
        circle_area = math.pi * max(r, 1.0) ** 2
        if fg_area > 0 and circle_area / fg_area < 0.80:
            cx, cy, r, contour = _contour_circle(probe, threshold)
            method = 'contour_polygon'
            shape = 'polygon'
        else:
            shape = 'circle'
    elif shape == 'auto':
        shape = 'circle' if method == 'contour_circle' else 'polygon'

    threshold_used = float(threshold) if threshold is not None else None
    if method in ('contour_circle', 'contour_polygon'):
        threshold_used, _ = _otsu(probe, threshold)

    if shape == 'circle':
        mask = _circle_mask((h, w), cx, cy, r, float(margin_px))
        info_shape = 'circle'
        center = [float(cx), float(cy)]
        hull = None
        radius = float(r + margin_px)
    else:
        if contour is None:
            _, contour = _contour_circle(probe, threshold)
        hull = _hull(contour)
        mask = _polygon_mask((h, w), hull, float(margin_px))
        info_shape = 'polygon'
        center = [float(cx), float(cy)]
        radius = None

    preview_idx = indices[len(indices) // 2] if preview_frame is None else int(preview_frame)
    preview_idx = max(0, min(len(files) - 1, preview_idx))
    preview = X.read_image(files[preview_idx], levels)
    overlay = _overlay(preview, mask, shape=info_shape, center=center,
                       radius=radius, hull=hull)
    info = {
        'schema': SCHEMA,
        'status': 'proposed',
        'created_at': schema.now_iso(),
        'method': method,
        'shape': info_shape,
        'center_px': center,
        'radius_px': radius,
        'margin_px': float(margin_px),
        'threshold': threshold_used,
        'image_shape': [int(h), int(w)],
        'frame_count': int(len(files)),
        'sample_indices': [int(i) for i in indices],
        'preview_index': int(preview_idx),
        'mask_coverage': float((mask > 0).mean()),
        'notes': 'inspect mask_overlay.png before using interior_mask.png',
    }
    return mask, info, overlay


def propose_from_raw(raw, out, config, *, n_frames=DEFAULT_FRAMES, shape='auto',
                     margin_px=DEFAULT_MARGIN_PX, threshold=None,
                     preview_frame=None, force=False):
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
    mask, info, overlay = propose(
        files, levels=config.scale_levels, n_frames=n_frames, shape=shape,
        margin_px=margin_px, threshold=threshold, preview_frame=preview_frame)
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
