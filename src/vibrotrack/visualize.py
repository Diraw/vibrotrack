"""Raw-frame overlays for fitted particle centres, radii and orientations.

The one thing this module fixes is the angle convention.  ``theta_rad`` is
stored as ``atan2(-dy, dx)`` in image coordinates: x grows right, y grows down,
but theta is a y-up mathematical angle.  Therefore a direction of theta from a
centre is::

    (cx + r*cos(theta), cy - r*sin(theta))

Use :func:`theta_endpoint` whenever you draw an orientation.  Do not use
``cy + r*sin(theta)``: that mirrors every tick across the horizontal axis.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def theta_endpoint(cx, cy, radius, theta):
    """Return the pixel endpoint of a y-up ``theta_rad`` direction.

    Parameters are floats; the returned tuple is also float.  This function is
    intentionally tiny and tested directly so downstream plots cannot silently
    reintroduce the ``+sin`` sign error.
    """
    return (float(cx) + float(radius) * float(np.cos(theta)),
            float(cy) - float(radius) * float(np.sin(theta)))


def _display_uint8(image):
    """Convert a raw grayscale/RGB frame to displayable uint8.

    8-bit frames are returned unchanged.  16-bit frames are stretched using
    their observed 0.5-99.5 percentiles; extraction products normally use
    :func:`vibrotrack.detect.read_image`, but this keeps the plotting helper
    usable with an array loaded by another reader.
    """
    a = np.asarray(image)
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
        elif a.shape[2] == 3:
            pass
        else:
            raise ValueError('image must be 2-D grayscale or RGB/RGBA')
        if a.dtype == np.uint8:
            return a
        lo, hi = np.percentile(a, (0.5, 99.5))
    else:
        if a.dtype == np.uint8:
            return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        lo, hi = np.percentile(a, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros((a.shape[0], a.shape[1], 3), np.uint8)
    out = np.clip((a.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    out = out.astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def draw_detections(image, xy_px, radius_px, theta_rad, *,
                    center_valid=None, angle_valid=None,
                    circle_color=(0, 255, 0), tick_color=(0, 180, 255),
                    center_color=(0, 255, 255), thickness=4):
    """Draw fitted particle outlines and orientation ticks on one frame.

    ``theta_rad`` follows the stored convention and is drawn with
    :func:`theta_endpoint` (y-up).  Invalid centres are skipped; invalid angles
    keep the outline but omit the tick.  The input array is not modified.
    """
    rgb = _display_uint8(image).copy()
    xy = np.asarray(xy_px, dtype=float).reshape(-1, 2)
    rad = np.asarray(radius_px, dtype=float).reshape(-1)
    theta = np.asarray(theta_rad, dtype=float).reshape(-1)
    if center_valid is None:
        cok = np.ones(len(xy), bool)
    else:
        cok = np.asarray(center_valid, dtype=bool).reshape(-1)
    if angle_valid is None:
        aok = np.isfinite(theta)
    else:
        aok = np.asarray(angle_valid, dtype=bool).reshape(-1)
    h, w = rgb.shape[:2]
    for i, ((cx, cy), rr) in enumerate(zip(xy, rad)):
        if (i >= len(cok) or not cok[i] or not np.isfinite(cx)
                or not np.isfinite(cy) or not np.isfinite(rr) or rr <= 0):
            continue
        c = (int(round(float(cx))), int(round(float(cy))))
        ri = max(1, int(round(float(rr))))
        if c[0] < -ri or c[1] < -ri or c[0] >= w + ri or c[1] >= h + ri:
            continue
        cv2.circle(rgb, c, ri + 2, circle_color, int(thickness), cv2.LINE_AA)
        cv2.circle(rgb, c, max(1, int(thickness)), center_color, -1, cv2.LINE_AA)
        if i < len(aok) and aok[i] and np.isfinite(theta[i]):
            ex, ey = theta_endpoint(cx, cy, rr * 0.95, theta[i])
            cv2.line(rgb, c, (int(round(ex)), int(round(ey))),
                     tick_color, int(thickness), cv2.LINE_AA)
    return rgb


def render_frame(image, xy_px, radius_px, theta_rad, out_path, *,
                 center_valid=None, angle_valid=None, max_size=1200,
                 label=None, quality=92):
    """Write one raw-frame overlay as JPEG/PNG and return its path.

    ``max_size`` is the longest output side; 0 keeps the native resolution.
    Circle/tick widths are pre-scaled so they remain visible after downsampling.
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    scale = 1.0
    if max_size and max(h, w) > max_size:
        scale = float(max_size) / float(max(h, w))
    thickness = max(1, int(round(3.0 / max(scale, 1e-6))))
    rgb = draw_detections(image, xy_px, radius_px, theta_rad,
                          center_valid=center_valid, angle_valid=angle_valid,
                          thickness=thickness)
    if scale < 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.9 if min(rgb.shape[:2]) >= 500 else 0.65
        cv2.putText(rgb, str(label), (12, 32), font, fs, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(rgb, str(label), (12, 32), font, fs, (255, 255, 255), 2, cv2.LINE_AA)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == '.png':
        Image.fromarray(rgb).save(out)
    else:
        Image.fromarray(rgb).save(out, quality=int(quality))
    return out


def available_frame_ids(case_dir):
    """Return sorted frame ids present in ``case_dir/chunk_*.npz``.

    This reads only the small ``frame_ids`` arrays, not the detection rows.
    """
    ids = []
    for f in sorted(Path(case_dir).glob('chunk_*.npz')):
        with np.load(f, allow_pickle=False) as z:
            ids.extend(int(x) for x in np.asarray(z['frame_ids']).reshape(-1))
    return sorted(set(ids))


def load_frame_records(case_dir, frame_ids):
    """Load selected detection rows from one or more chunk files.

    Returns ``{frame_id: {field: array}}`` for the requested ids.  Raises
    ``KeyError`` if a requested frame is absent, so a typo cannot silently plot
    the wrong frame.
    """
    wanted = set(int(x) for x in frame_ids)
    found = {}
    for f in sorted(Path(case_dir).glob('chunk_*.npz')):
        with np.load(f, allow_pickle=False) as z:
            ids = np.asarray(z['frame_ids']).reshape(-1)
            offsets = np.asarray(z['frame_offsets']).reshape(-1)
            for i, fid in enumerate(ids):
                fid = int(fid)
                if fid not in wanted or fid in found:
                    continue
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                rec = {}
                for key in ('xy_px', 'radius_px', 'theta_rad',
                            'center_valid', 'angle_valid'):
                    if key in z:
                        rec[key] = np.asarray(z[key][lo:hi])
                found[fid] = rec
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - set(found))
    if missing:
        raise KeyError('frame(s) not found in %s: %s' % (case_dir, missing))
    return found
