"""Bench counting with detector v2, separate from production tracking.

Presets target the lab camera scale. Both the loopback UI and CLI use the same
single-frame detector; only the CLI can explicitly request neighbouring frames.
The result is for manual review, not a substitute for a tracked, audited batch.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import re
import time
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import detect as X
from . import detect2 as D2

MODES = {
    'mono14': '14 mm 单分散',
    'bi12-14': '12 + 14 mm 双分散',
}
BIDISPERSE_FALLBACK = (31.7, 36.9)  # fitted radii in px for the lab camera
MAX_IMAGE_PIXELS = 16_000_000
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


def _detector_params(cfg, thresh=None, mode=None):
    params = dict(cfg.detector_params) if cfg is not None else {}
    for key in ('prior_xy', 'bbox', 'interior_mask', 'static_centres'):
        params.pop(key, None)
    if mode == 'mono14':
        for key in ('small_template_radius', 'small_template_thresh', 'radii', 'fallback_radii'):
            params.pop(key, None)
        params['infer_radii'] = False
    elif mode == 'bi12-14':
        params.setdefault('small_template_radius', 15.5)  # at the detector's 0.5x scale
        params.update(infer_radii=True, fallback_radii=BIDISPERSE_FALLBACK)
    elif mode is not None:
        raise ValueError('未知粒径模式：' + str(mode))
    if thresh is not None:
        params['thresh'] = float(thresh)
    params.setdefault('void_veto', True)
    return params


def _mask_from(path, shape, cfg):
    """Interior mask for the detector, or None for the whole frame."""
    mask_path = Path(path) if path else None
    if mask_path is None and cfg is not None and getattr(cfg, 'boundary_mask', None):
        mask_path = Path(cfg.boundary_mask)
    if mask_path is None:
        return None
    if not mask_path.is_file():
        raise FileNotFoundError('找不到内壁掩膜：' + str(mask_path))
    m = cv2.imdecode(np.fromfile(str(mask_path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError('cannot read mask ' + str(mask_path))
    if m.shape != shape:
        raise ValueError('掩膜尺寸必须与原图一致')
    return (m > 0).astype(np.uint8)


def count_image(path, cfg=None, mask_path=None, before=0, after=0,
                thresh=None, passes=None, mode=None):
    """Detect the target frame and return ``(det, image, meta)``.

    ``det`` is the detector's frame record (``xy_px``, ``radius_px``, quality
    fields).  Priors come from the neighbouring frames when they exist.
    """
    path = Path(path).resolve()
    t0 = time.time()
    neighbours = neighbour_paths(path, before=before, after=after) if before or after else []
    images = (decode_image(p.read_bytes(), cfg, mode) for p in neighbours)
    image = decode_image(path.read_bytes(), cfg, mode)
    return _count(image, path.name, cfg, mask_path, thresh, passes, mode,
                  images, [p.name for p in neighbours], t0)


def decode_image(raw, cfg=None, mode=None):
    # Inspect dimensions before decoding; use the same normalization for file
    # and browser input. Only the first page/frame is used, even if animated.
    with Image.open(io.BytesIO(raw), formats=['TIFF', 'PNG', 'JPEG']) as im:
        if im.width * im.height > MAX_IMAGE_PIXELS or min(im.size) < 64:
            raise ValueError('图片最短边须 ≥64 px，且总像素不得超过 1600 万')
    flags = cv2.IMREAD_UNCHANGED
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    if image is None:
        raise ValueError('无法解码图片，请使用 TIFF、PNG 或 JPEG')
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    if image.dtype == np.uint16:
        levels = X.PRODUCTION_LEVELS if cfg is None and mode in MODES else _levels(cfg)
        image = X.scale_to_u8(image, levels)
    elif image.dtype != np.uint8:
        raise ValueError('仅支持 8-bit 或 16-bit 整数图像')
    return image


def count_bytes(raw, name, mode, cfg=None, mask_path=None):
    t0 = time.time()
    if mode not in MODES:
        raise ValueError('请先选择粒径模式')
    image = decode_image(raw, cfg, mode)
    return _count(image, name, cfg, mask_path, None, None, mode, (), [], t0)


def _count(image, name, cfg, mask_path, thresh, passes, mode, images, neighbours, t0):
    shape = image.shape[:2]

    mask = _mask_from(mask_path, shape, cfg)
    if mask is None:
        search = np.ones(shape, np.uint8)
    else:
        search = mask
    dist = (cv2.distanceTransform(search, cv2.DIST_L2, 5)
            if mask is not None else np.full(shape, 1e4, np.float32))
    params = _detector_params(cfg, thresh, mode)
    if passes is None:
        passes = 1

    prior = None
    for im in images:
        if im.shape != shape:
            raise ValueError('先验帧与目标帧尺寸不一致')
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
        'image': name,
        'width': int(shape[1]),
        'height': int(shape[0]),
        'n': int(len(xy)),
        'radius_px': float(np.median(radii)) if len(radii) else None,
        'neighbours': neighbours,
        'prior_pass': bool(len(neighbours)) or passes > 1,
        'ms': int(round(1000 * (time.time() - t0))),
        'detector': 'v2',
    }
    classes = sorted(float(r) for r in info['radii'])
    if mode == 'bi12-14' and len(classes) != 2:
        classes = list(BIDISPERSE_FALLBACK) if not len(radii) else []
    if mode == 'mono14':
        classes = [meta['radius_px'] or D2.DEFAULTS['radius_fallback']]
    labels = (['12 mm 小颗粒', '14 mm 大颗粒'] if mode == 'bi12-14' and len(classes) == 2
              else ['14 mm 颗粒'] if mode == 'mono14' else [f'{r:.1f} px' for r in classes])
    meta['mode'] = mode
    meta['mode_label'] = MODES.get(mode, '自定义 / 原有配置')
    meta['groups'] = [dict(id=i, radius=r, label=label) for i, (r, label) in enumerate(zip(classes, labels))]
    return det, image, meta


def _png_data_uri(image):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format='PNG', optimize=False)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _display_uri(image, max_side=1200):
    """A 1200 px copy for the viewer, matching the review pack's frame size.

    The viewer redraws this every drag frame; a 2400 px source costs more per
    sample on a machine without GPU acceleration, and the editor does not need
    more than 1200 px to place a click.
    """
    h, w = image.shape[:2]
    k = min(1.0, max_side / max(w, h))
    if k >= 1.0:
        return _png_data_uri(image)
    small = np.asarray(Image.fromarray(image).resize(
        (max(1, round(w * k)), max(1, round(h * k))), Image.BILINEAR))
    return _png_data_uri(small)


def _viewer_template():
    return resources.files('vibrotrack').joinpath('count_viewer.html').read_text(encoding='utf8')


def grouped_particles(det, meta):
    groups = meta.get('groups', [])
    rows = []
    for (x, y), r in zip(det['xy_px'], det['radius_px']):
        group = min(groups, key=lambda g: abs(g['radius'] - r), default=None)
        rows.append([round(float(x), 2), round(float(y), 2), round(float(r), 2),
                     group['id'] if group is not None else None])
    return rows


def viewer_data(image, det, meta):
    return dict(image=_display_uri(image), full=_png_data_uri(image) if max(image.shape[:2]) > 1200 else None,
                width=meta['width'], height=meta['height'], particles=grouped_particles(det, meta),
                meta={**meta, 'groups': meta.get('groups', [])})


def viewer_page(data):
    return _viewer_template().replace('/*__DATA__*/', schema_json(data))


def write_bundle(out_dir, image, det, meta):
    """Write a standalone HTML viewer, the frame, an overlay and raw detection CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_path = out / 'frame.png'
    Image.fromarray(image).save(frame_path)
    data = viewer_data(image, det, meta)
    particles = [row[:3] for row in data['particles']]
    html = viewer_page(data)
    html_path = out / 'count.html'
    html_path.write_text(html, encoding='utf8')

    # A flat PNG with the rings burned in, so the count can be checked without a
    # browser at all.
    overlay_path = out / 'overlay.png'
    rgb = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(rgb)
    lw = max(2, round(min(image.shape[:2]) / 900))
    for x, y, r in particles:
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(32, 214, 122), width=lw)
    draw.rectangle([0, 0, 12 + 14 * len(str(len(particles))), 42], fill=(8, 12, 16))
    draw.text((10, 8), '%d' % len(particles), fill=(232, 238, 245))
    rgb.save(overlay_path)

    csv_path = out / 'detections.csv'
    with csv_path.open('w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['x_px', 'y_px', 'radius_px'])
        for x, y, r in particles:
            w.writerow([x, y, r])
    return html_path, frame_path, csv_path


def schema_json(obj):
    # User-controlled filenames must never terminate the inline script tag.
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
