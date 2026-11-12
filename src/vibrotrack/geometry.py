"""Geometry product for one case: the numbers every consumer must take from here.

Why this module exists
----------------------
On 2026-11-06 the container turned out to sit (+5, -20) px away from where the previous
session had left it, while the analysis scripts still used a hard-coded centre from the old
mask. Every new case was therefore scored against a wrong origin (-2.5% in |c|/R, 5 deg in
direction), and the error was only found by accident. The centre is a *product*, not a
constant.

Centre method
-------------
Outer-envelope circle fit: for each 1 deg bin take the largest particle-centre radius, then
least-squares fit R(theta) = c + a cos(theta) + b sin(theta). The container is axially
symmetric, so the offsets (a, b) estimate the centre error.

Do NOT select "particles that barely moved": in a dense vibrated bed ~70% of tracks move
<6 px over a whole run, so that test picks up jammed bulk particles and fits a centre that
is off by ~100 px.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCHEMA = 'geometry/v1'
NBINS = 360


def envelope_fit(xy_px, centre_px, nbins=NBINS, min_samples=None):
    """Outer-envelope circle fit about ``centre_px``.

    Returns (dx, dy, radius, n_bins, resid_rms). ``resid_rms`` is the quality gate: a
    sparse bed (few particles) has an envelope that is short of the wall and irregular, and
    the fit silently returns a wrong centre. Measured on the 2026-11-06 session:
    N >= 150 -> 17-19 px residual and 360/360 bins; N = 20 -> 88 px residual and 274 bins.
    Treat resid_rms > 30 px as a failed fit and inherit the shift from a dense case that
    shares the same container seating.
    """
    xy = np.asarray(xy_px, dtype=np.float64)
    x = xy[:, 0] - centre_px[0]
    y = xy[:, 1] - centre_px[1]
    th = np.arctan2(y, x)
    rr = np.hypot(x, y)
    b = ((th + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    if min_samples is None:
        min_samples = max(5, len(xy) // (nbins * 4))
    bins, radii = [], []
    for i in range(nbins):
        m = b == i
        if m.sum() >= min_samples:
            bins.append(i)
            radii.append(rr[m].max())
    bins = np.asarray(bins)
    radii = np.asarray(radii)
    if len(bins) < 8:
        return float('nan'), float('nan'), float('nan'), int(len(bins)), float('inf')
    t = (bins + 0.5) / nbins * 2 * np.pi - np.pi
    A = np.c_[np.cos(t), np.sin(t), np.ones(len(t))]
    sol, *_ = np.linalg.lstsq(A, radii, rcond=None)
    resid = radii - A @ sol
    return float(sol[0]), float(sol[1]), float(sol[2]), int(len(bins)), float(resid.std())


def envelope_shift(xy_px, centre_px, nbins=NBINS, min_samples=None):
    dx, dy, r, _, _ = envelope_fit(xy_px, centre_px, nbins, min_samples)
    return dx, dy, r


def equal_access_radius(mask_path, centre_px, shift_px=(0.0, 0.0), hi=1300.0):
    """Largest centred circle fully inside the trusted interior mask.

    ``shift_px`` re-registers a mask that was built for a different container seating:
    the mask belongs to the container, so when the container moved by (dx, dy) the mask
    moves with it. Without this, a 20 px reseating shrinks the trusted circle by 20 px and
    silently changes the normalisation.
    """
    mask = np.asarray(Image.open(mask_path)) > 0
    if abs(shift_px[0]) > 0.25 or abs(shift_px[1]) > 0.25:
        mask = np.roll(mask, (int(round(shift_px[1])), int(round(shift_px[0]))), axis=(0, 1))
    yy, xx = np.indices(mask.shape)
    r = np.hypot(xx - centre_px[0], centre_px[1] - yy)
    rad = np.linspace(20.0, hi, int(hi * 2) + 1)
    ok = [q for q in rad if mask[r <= q].mean() > 0.999]
    if not ok:
        raise ValueError('mask is empty around the given centre')
    return float(max(ok))


def build(case_id, xy_px, centre_seed_px, *, mask_path=None, px_per_mm=None, fps=None,
          n_particles=None, n_detected_median=None, radius_mm=None, source='experiment',
          container_radius_mm=None, px_per_mm_source='unspecified',
          centre_shift_px=None, centre_method=None,
          container_radius_source=None, radius_classes=None):
    if px_per_mm is None or radius_mm is None or fps is None:
        raise ValueError('px_per_mm, radius_mm and fps are required; read them from the apparatus config')
    if centre_shift_px is not None:
        dx, dy = float(centre_shift_px[0]), float(centre_shift_px[1])
        _, _, envelope_r, n_bins, resid_rms = envelope_fit(xy_px, centre_seed_px)
        method = centre_method or 'inherited'
        quality = {'envelope_resid_rms_px': resid_rms, 'envelope_bins': n_bins, 'inherited': True}
    else:
        dx, dy, envelope_r, n_bins, resid_rms = envelope_fit(xy_px, centre_seed_px)
        method = centre_method or 'envelope_circle_fit'
        quality = {'envelope_resid_rms_px': resid_rms, 'envelope_bins': n_bins,
                   'inherited': False, 'gate_passed': bool(resid_rms <= 30.0)}
    centre = [centre_seed_px[0] + dx, centre_seed_px[1] + dy]
    # ``envelope_r`` is where particle CENTRES reach, so the wall is one particle radius out.
    if radius_classes:
        # a bidisperse bed: the wall is one LARGEST particle radius beyond the envelope
        r_p_px = max(float(r) for r, _n in radius_classes)
    else:
        r_p_px = radius_mm * px_per_mm
    wall_r = envelope_r + r_p_px
    out = {
        'schema': SCHEMA,
        'case_id': case_id,
        'source': source,
        'center_px': centre,
        'center_seed_px': list(map(float, centre_seed_px)),
        'center_shift_px': [dx, dy],
        'centre_method': method,
        'centre_quality': quality,
        'wall_radius_px': wall_r,
        'envelope_radius_px': envelope_r,
        'px_per_mm': float(px_per_mm),
        'px_per_mm_source': px_per_mm_source,
        'fps': float(fps),
        'particle_radius_mm': float(radius_mm),
        'particle_radius_px': r_p_px,
    }
    if mask_path is not None:
        r_eq = equal_access_radius(mask_path, centre, shift_px=(dx, dy))
        out['equal_access_radius_px'] = r_eq
        import hashlib
        out['mask_file'] = Path(mask_path).name
        out['mask_sha256'] = hashlib.sha256(Path(mask_path).read_bytes()).hexdigest()
    else:
        # no mask: fall back to the wall radius minus one particle diameter of margin
        out['equal_access_radius_px'] = wall_r - 2.0 * radius_mm * px_per_mm
        out['mask_file'] = None
    if container_radius_mm is not None:
        r_container_m = float(container_radius_mm) / 1000.0
        out['container_radius_source'] = container_radius_source or 'supplied'
    else:
        r_container_m = out['wall_radius_px'] / px_per_mm / 1000.0
        out['container_radius_source'] = container_radius_source or 'image_envelope_plus_particle_radius'
    out['container_radius_m'] = r_container_m
    out['container_radius_px'] = r_container_m * 1000.0 * px_per_mm
    if radius_classes:
        # Occupy each class by its own area; one radius would overcount the large
        # discs and undercount the small ones.
        classes = [[float(r), int(n)] for r, n in radius_classes]
        area_px = sum(n * r * r for r, n in classes)
        out['radius_classes'] = classes
        out['radius_classes_source'] = 'per-detection radius_px'
        out['phi'] = float(area_px / wall_r ** 2)
        out['phi_method'] = 'sum(n_i * r_i^2) / R_container^2 (area-weighted)'
    if n_particles is not None:
        r_m = radius_mm / 1000.0
        out['n_particles'] = int(n_particles)
        if not radius_classes:
            out['phi'] = float(n_particles * r_m ** 2 / r_container_m ** 2)
            out['phi_method'] = 'N*pi*r^2 / pi*R_container^2'
    if n_detected_median is not None:
        out['n_detected_median'] = float(n_detected_median)
    if not mask_path:
        out['centre_uncertainty_px'] = None
    else:
        out['centre_uncertainty_px'] = 1.5
    return out


def write(path, record):
    Path(path).write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding='utf8')
    return path


def load(path):
    rec = json.loads(Path(path).read_text(encoding='utf8'))
    if rec.get('schema') != SCHEMA:
        raise ValueError('unexpected geometry schema: %r' % rec.get('schema'))
    return rec
