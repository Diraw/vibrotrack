"""Count the particles in one image, with the production detector.

Why this exists
---------------
Counting one frame by eye is slow and counting with a bespoke detector is wrong:
the 2026-11-19 attempt (a hand-written Hough in a browser) landed 0-8% off with
5-10% phantoms on real dense beds.  This module runs the same detector v2 as
`vibrotrack extract`, so the number agrees with the archived batches, and it
borrows the production trick for the hard beads: the neighbouring frames in the
same folder are detected first and their centres are passed as prior seeds.  A
bead squeezed against a neighbour loses template correlation but keeps a clean
ray fit once seeded, which is exactly why extract uses a 5-frame prior.

Measured on `590a10_3` frame 500 (dense bed, 601 particles): 599 found, no
phantoms, 0.3 px median centre error - versus 588 with no priors.

The output is a folder with `count.html` (viewer + hand editing), `frame.png`
and `detections.csv`.  It is a bench aid: nothing here writes into an archive.
"""
from __future__ import annotations

import base64
import csv
import io
import re
import time
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import detect as X
from . import detect2 as D2
from . import schema

_FRAME_RE = re.compile(r'^(?P<stem>.*?)(?P<num>\d+)(?P<ext>\.[^.]+)$')


def neighbour_paths(path, before=4, after=0):
    """Frames around ``path`` in the same folder, ordered oldest first.

    The name has to end in a number (`..._0042.tiff`), which is what the cameras
    write.  Candidate names are rebuilt from that number and stat'ed: listing a
    27,000-file folder on the lab NAS took 80 s, four stats take milliseconds.
    """
    path = Path(path)
    m = _FRAME_RE.match(path.name)
    if not m:
        return []
    stem, num_s, ext = m.group('stem'), m.group('num'), m.group('ext')
    width, num = len(num_s), int(num_s)
    out = []
    for k in range(max(0, int(before)), 0, -1):
        if num - k < 0:
            continue
        cand = path.with_name('%s%0*d%s' % (stem, width, num - k, ext))
        if cand.exists():
            out.append(cand)
    for k in range(1, max(0, int(after)) + 1):
        cand = path.with_name('%s%0*d%s' % (stem, width, num + k, ext))
        if cand.exists():
            out.append(cand)
    return out


def _levels(cfg):
    if cfg is not None and cfg.scale_levels:
        return tuple(cfg.scale_levels)
    return (0.0, 65535.0)


def _detector_params(cfg, thresh=None):
    params = dict(cfg.detector_params) if cfg is not None else {}
    if thresh is not None:
        params['thresh'] = float(thresh)
    params.setdefault('void_veto', True)
    return params


def _mask_from(path, shape, cfg):
    """Interior mask for the detector, or None for the whole frame."""
    mask_path = Path(path) if path else None
    if mask_path is None and cfg is not None and getattr(cfg, 'boundary_mask', None):
        mask_path = Path(cfg.boundary_mask)
    if mask_path is None or not mask_path.exists():
        return None
    m = cv2.imdecode(np.fromfile(str(mask_path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError('cannot read mask ' + str(mask_path))
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def count_image(path, cfg=None, mask_path=None, before=4, after=0,
                thresh=None, passes=None):
    """Detect the target frame and return ``(det, image, meta)``.

    ``det`` is the detector's frame record (``xy_px``, ``radius_px``, quality
    fields).  Priors come from the neighbouring frames when they exist.
    """
    path = Path(path).resolve()
    t0 = time.time()
    levels = _levels(cfg)
    neighbours = neighbour_paths(path, before=before, after=after)
    images = [(p, X.read_image(str(p), levels)) for p in neighbours]
    image = X.read_image(str(path), levels)
    shape = image.shape[:2]

    mask = _mask_from(mask_path, shape, cfg)
    if mask is None:
        search = np.ones(shape, np.uint8)
    else:
        search = mask
    dist = (cv2.distanceTransform(search, cv2.DIST_L2, 5)
            if mask is not None else np.full(shape, 1e4, np.float32))
    params = _detector_params(cfg, thresh)
    if passes is None:
        passes = 1 if neighbours else 2

    prior = None
    for _p, im in images:
        d, _info = D2.frame(im, search, dist, prior_xy=prior, **params)
        prior = d['xy_px'] if len(d['xy_px']) else prior
    det, info = D2.frame(image, search, dist, prior_xy=prior, **params)
    for _ in range(max(0, int(passes) - 1)):
        if not len(det['xy_px']):
            break
        det, info = D2.frame(image, search, dist, prior_xy=det['xy_px'],
                             **params)

    xy = np.asarray(det['xy_px'], float).reshape(-1, 2)
    radii = np.asarray(det['radius_px'], float).reshape(-1)
    meta = {
        'image': str(path),
        'width': int(shape[1]),
        'height': int(shape[0]),
        'n': int(len(xy)),
        'radius_px': float(np.median(radii)) if len(radii) else None,
        'neighbours': [p.name for p in neighbours],
        'prior_pass': bool(len(neighbours)) or passes > 1,
        'ms': int(round(1000 * (time.time() - t0))),
        'detector': 'v2',
    }
    return det, image, meta


def _png_data_uri(image):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format='PNG', optimize=False)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _viewer_template():
    return resources.files('vibrotrack').joinpath('count_viewer.html').read_text(encoding='utf8')


def write_bundle(out_dir, image, det, meta):
    """Write ``count.html`` (viewer), ``frame.png`` and ``detections.csv``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_path = out / 'frame.png'
    Image.fromarray(image).save(frame_path)

    xy = np.asarray(det['xy_px'], float).reshape(-1, 2)
    radii = np.asarray(det['radius_px'], float).reshape(-1)
    particles = [[round(float(x), 2), round(float(y), 2), round(float(r), 2)]
                 for (x, y), r in zip(xy, radii)]
    data = {
        'image': _png_data_uri(image),
        'width': meta['width'],
        'height': meta['height'],
        'particles': particles,
        'meta': meta,
    }
    html = _viewer_template().replace('/*__DATA__*/', schema_json(data))
    html_path = out / 'count.html'
    html_path.write_text(html, encoding='utf8')

    csv_path = out / 'detections.csv'
    with csv_path.open('w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['x_px', 'y_px', 'radius_px'])
        for x, y, r in particles:
            w.writerow([x, y, r])
    return html_path, frame_path, csv_path


def schema_json(obj):
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
