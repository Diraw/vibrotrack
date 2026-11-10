"""Boundary mask loading, independent of any lab-specific case table.

The public package deliberately does not know where masks live.  A caller passes
either an explicit path or nothing:

- ``mask_path``: load that image, optionally assert its shape.
- ``None``: return an all-ones mask (the whole frame is searchable).  This is
  appropriate for a rectangular ROI; for a circular container, pass a mask.

The returned ``source`` is the path as a string, or ``None`` for a full frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def full_mask(shape):
    if shape is None:
        raise ValueError('shape is required when no boundary mask is supplied')
    return np.ones(tuple(int(x) for x in shape), np.uint8)


def load_mask(mask_path=None, shape=None, required=False):
    """Return ``(uint8 mask, source)``.

    ``required=True`` means a missing/unreadable explicit mask is an error rather
    than a silent full-frame fallback.  The pipeline uses ``required=True`` so a
    typo in the config cannot quietly change the search region.
    """
    if mask_path:
        p = Path(mask_path)
        if not p.exists():
            if required:
                raise FileNotFoundError('boundary mask not found: %s' % p)
            return full_mask(shape), None
        a = np.asarray(Image.open(p))
        if shape is not None and tuple(a.shape) != tuple(int(x) for x in shape):
            raise ValueError(
                'boundary mask shape %r does not match image shape %r (%s)'
                % (a.shape, tuple(shape), p))
        return (a > 0).astype(np.uint8), str(p)
    return full_mask(shape), None
