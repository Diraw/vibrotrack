"""Estimate a particle marker direction from a small patch around its centre.

This is deliberately separate from the detector.  The review layer only calls it
for **human-added** centres, when raw frames are available in ``vibrotrack
curate``.  It uses the same dark-marker centroid rule as the production detector:
within the inner 22 px of a 47x47 patch, the darkest pixels are weighted and
their centroid gives the radial marker direction.

A manual centre is not allowed to invent an angle.  If the expected dark mark is
missing, the returned ``valid`` flag is false and the row should be written with
``angle_valid=False`` and ``theta_rad=NaN``.
"""
from __future__ import annotations

import cv2
import numpy as np

_PX, _PY = np.mgrid[-23:24, -23:24].astype(np.float32)
_INNER = (_PX * _PX + _PY * _PY) < 22 ** 2


def estimate_marker_orientation(image, centers, *, min_area=100, max_area=380,
                                min_offset=2.0, max_offset=16.0, min_bright=85.0):
    """Return ``(theta_rad, valid, offset_px, area_px, bright)`` for centres.

    Parameters
    ----------
    image:
        2-D grayscale frame (uint8 recommended; uint16 is scaled to uint8).
    centers:
        ``(N, 2)`` centres in image pixels.
    """
    a = np.asarray(image)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY if a.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
    if a.dtype != np.uint8:
        lo, hi = np.percentile(a, (0.5, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (np.full(len(centers), np.nan, np.float32),
                    np.zeros(len(centers), bool),
                    np.full(len(centers), np.nan, np.float32),
                    np.zeros(len(centers), np.int16),
                    np.zeros(len(centers), np.float32))
        a = np.clip((a.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    xy = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    n = len(xy)
    if n == 0:
        return (np.empty(0, np.float32), np.empty(0, bool),
                np.empty(0, np.float32), np.empty(0, np.int16),
                np.empty(0, np.float32))
    mx = xy[:, 0, None, None] + _PX
    my = xy[:, 1, None, None] + _PY
    patch = cv2.remap(a, mx.reshape(n, -1), my.reshape(n, -1),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    patch = patch.reshape(n, 47, 47).astype(np.float32)
    bright = np.percentile(patch[:, _INNER], 80, axis=1)
    dark = (patch < bright[:, None, None] * 0.54) & _INNER[None]
    weight = np.maximum(bright[:, None, None] * 0.54 - patch, 0) * dark
    area = dark.sum((1, 2)).astype(np.int16)
    mass = weight.sum((1, 2))
    dx = (weight * _PX).sum((1, 2)) / np.maximum(mass, 1)
    dy = (weight * _PY).sum((1, 2)) / np.maximum(mass, 1)
    offset = np.hypot(dx, dy).astype(np.float32)
    valid = ((area >= min_area) & (area <= max_area) &
             (offset >= min_offset) & (offset <= max_offset) &
             (bright > min_bright))
    theta = np.arctan2(-dy, dx).astype(np.float32)
    theta[~valid] = np.nan
    return theta, valid, offset, area, bright.astype(np.float32)
