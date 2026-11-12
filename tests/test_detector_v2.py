"""Tests for the 2026-11-08 detector/linker changes.

Each test pins one decision that the audit showed to matter:
  * fixed-radius centre fit does not bias a clean field,
  * the hard-disc prior is enforced (a pair closer than 1.75 r cannot be two particles),
  * a particle whose centre lies outside the interior mask is still found, and flagged,
  * static wall discs are dropped while moving wall particles are kept,
  * a track survives a missing frame.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
from vibrotrack import detect as V1          # noqa: E402
from vibrotrack import detect2 as D2         # noqa: E402
from vibrotrack import link2 as L2           # noqa: E402
from vibrotrack import shards as S           # noqa: E402

R = 37.0


def _frame(centers, size=700, seed=3, noise=6.0):
    rng = np.random.default_rng(seed)
    im = np.full((size, size), 45, np.float32)
    for x, y in centers:
        cv2.circle(im, (int(x), int(y)), int(R), 220, -1)
        cv2.line(im, (int(x), int(y)), (int(x + 0.75 * R), int(y - 0.5 * R)), 90, 5)
    im += rng.normal(0, noise, im.shape)
    return np.clip(im, 0, 255).astype(np.uint8)


def _mask(shape=(700, 700)):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (shape[1] // 2, shape[0] // 2), 320, 1, -1)
    return m, cv2.distanceTransform(m, cv2.DIST_L2, 5)


def _detect2(im, search, interior, **kw):
    d, _ = D2.frame(im, search, cv2.distanceTransform(interior, cv2.DIST_L2, 5),
                    bbox=V1.mask_bbox(search), interior_mask=interior, **kw)
    return d


def test_detect2_clean_field_matches_v1():
    centers = [(350 + 110 * np.cos(t), 350 + 110 * np.sin(t))
               for t in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
    im = _frame(centers)
    mask, dist = _mask()
    v1 = V1.detect(im, mask, dist, bbox=V1.mask_bbox(mask))
    v2 = _detect2(im, mask, mask)
    assert len(v1['xy_px']) == len(centers)
    assert len(v2['xy_px']) == len(centers)
    # same field, same centres (order may differ)
    d = np.hypot(v2['xy_px'][:, None, 0] - v1['xy_px'][None, :, 0],
                 v2['xy_px'][:, None, 1] - v1['xy_px'][None, :, 1]).min(1)
    assert d.max() < 0.5, d
    assert np.all(v2['outside_mask'] == 0)
    assert np.allclose(v2['radius_px'], np.median(v2['radius_px']))


def test_hard_disc_prior_merges_impossible_pair():
    """Two equal discs 45 px apart cannot both exist (2r = 74 px). v2 keeps one."""
    im = _frame([(350, 350), (395, 350)])
    mask, dist = _mask()
    d = _detect2(im, mask, mask)
    assert len(d['xy_px']) == 1, d['xy_px']


def test_wall_band_recovers_particle_outside_interior_mask():
    """A particle squeezed against the wall has its centre outside the interior mask."""
    cy, cx = 400, 400
    particle = (cx + 240, cy)          # interior mask radius is 320 -> centre is inside
    outside = (cx + 340, cy)           # 20 px outside the mask
    im = _frame([particle, outside], size=800)
    interior = np.zeros((800, 800), np.uint8)
    cv2.circle(interior, (cx, cy), 320, 1, -1)
    tight = _detect2(im, interior, interior)
    assert (tight['outside_mask'] == 1).sum() == 0
    assert len(tight['xy_px']) == 1, 'particle outside the mask must be invisible'
    search = D2.search_mask(interior, 40)
    wide = _detect2(im, search, interior)
    assert len(wide['xy_px']) == 2
    assert (wide['outside_mask'] == 1).sum() == 1


def test_drop_static_wall_tracks_keeps_moving_particles():
    """Static outside-mask tracks are the frozen wall; moving ones are particles."""
    nf = 60
    frames, xy, om, seg = [], [], [], []
    rng = np.random.default_rng(0)
    for f in range(nf):
        frames.append(f)
        # static wall disc: same spot every frame
        xy.append([100.0, 100.0]); om.append(1); seg.append(0)
        # moving particle outside the mask: 2 px/frame
        xy.append([100.0 + 2.0 * f, 200.0]); om.append(1); seg.append(1)
        # interior particle
        xy.append([300.0 + 1.0 * f, 300.0]); om.append(0); seg.append(2)
    xy = np.asarray(xy, np.float32)
    n = len(xy)
    payload = dict(xy_px=xy, segment_id=np.asarray(seg, np.int32),
                   outside_mask=np.asarray(om, np.uint8),
                   frame_ids=np.arange(nf, dtype=np.int32),
                   frame_offsets=np.arange(0, 3 * nf + 1, 3, dtype=np.int32),
                   gap_frames=np.ones(n, np.int16),
                   previous_detection=np.full(n, -1, np.int64),
                   link_valid=np.ones(n, bool),
                   radius_px=np.full(n, R, np.float32))
    out, ndrop, ntracks = S.drop_static_wall_tracks(payload)
    assert (ndrop, ntracks) == (nf, 1)
    assert len(out['xy_px']) == 2 * nf
    assert not (out['segment_id'] == 0).any()
    assert np.all(np.diff(out['frame_offsets']) == 2)


def test_link2_keeps_identity_across_missing_frame():
    """A particle absent for one frame must keep its id, with gap_frames == 2."""
    st = L2.new_state(gap=5)
    pos = {}
    ids_seen = []
    for f in range(12):
        pts = []
        truth = []
        for k in range(3):
            x = 100 + 40 * k + 3.0 * f
            y = 200 + 2.0 * k
            if f == 6 and k == 1:
                continue          # detection missing this frame only
            truth.append(k)
            pts.append([x, y])
        pts = np.asarray(pts, float)
        ids, prev, gap, step = L2.link_frame(st, f, pts, R)
        ids_seen.append((f, truth, ids, gap))
    # particle 1 (k=1) should carry one id from frame 0 to 11
    k1 = [ids[truth.index(1)] for f, truth, ids, gap in ids_seen if 1 in truth]
    assert len(set(k1)) == 1, k1
    # the rejoining detection reports gap_frames == 2 and no direct predecessor
    f, truth, ids, gap = ids_seen[7]
    j = truth.index(1)
    assert gap[j] == 2


def test_link2_reproduces_direct_links():
    st = L2.new_state(gap=5)
    ids = None
    for f in range(20):
        pts = np.array([[100 + 5 * k + 2.0 * f, 150 + 30 * k] for k in range(4)], float)
        i, prev, gap, step = L2.link_frame(st, f, pts, R)
        if f == 0:
            ids = i.copy()
        else:
            assert np.array_equal(i, ids), 'straight motion must keep ids'
            assert np.all(gap == 1)


def _ring_frame(shape=(900, 900), n=8, ring_r=300, rp=R):
    """Synthetic wall ring: n fixed discs plus 4 free particles well inside."""
    cx = cy = shape[0] // 2
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    discs = np.c_[cx + ring_r * np.cos(th), cy + ring_r * np.sin(th)]
    parts = np.array([[cx - 120, cy - 90], [cx + 130, cy - 60],
                      [cx - 60, cy + 140], [cx + 90, cy + 120]], float)
    im = _frame(np.r_[discs, parts], size=shape[0], seed=7)
    return im, discs, parts


def test_wall_model_geometry_ignores_fixed_discs():
    """With a measured wall map the apparatus discs are excluded geometrically."""
    im, discs, parts = _ring_frame()
    wall = np.c_[discs, np.full(len(discs), R), np.ones(len(discs))]
    search = D2.reachable_mask(im.shape, wall[:, :2], wall[:, 2], R)
    interior = np.zeros(im.shape, np.uint8)
    cv2.circle(interior, (450, 450), 220, 1, -1)
    d = _detect2(im, search, interior, static_centres=wall[:, :2], avoid_r=R - 6.0)
    xy = d['xy_px']
    assert len(xy) == len(parts), 'expected the 4 particles, got %d' % len(xy)
    for (px, py) in xy:
        dmin = np.min(np.hypot(discs[:, 0] - px, discs[:, 1] - py))
        assert dmin > R - 6.0, 'detection sits on a fixed disc (d=%.1f)' % dmin


def test_wall_model_auto_falls_back_without_map(tmp_path):
    src = tmp_path / 'interior_mask.png'
    src.write_bytes(b'')
    assert S.resolve_wall_model({'wall_model': 'auto'}, str(src)) == 'empirical'
    np.save(tmp_path / S.WALL_FILE, np.zeros((3, 4)))
    assert S.resolve_wall_model({'wall_model': 'auto'}, str(src)) == 'geometry'
    assert S.resolve_wall_model({'wall_model': 'off'}, str(src)) == 'off'


def test_prior_seed_recovers_a_particle_but_cannot_invent_one():
    """A prediction may only confirm a disc that is really there.

    The template is the fragile stage: a bead squeezed against a neighbour drops below
    the correlation threshold (the production data: 0.148 vs 0.20) although a fit seeded at the
    right place still returns 40 clean rays. So the previous frame's positions are added
    as candidate seeds - but the acceptance gates are untouched, which is what this test
    pins: a seed on a particle finds it, a seed in empty space finds nothing.
    """
    centers = [(220, 220), (420, 220), (220, 430)]
    im = _frame(centers)
    mask, dist = _mask()
    d0, _ = D2.frame(im, mask, dist, bbox=V1.mask_bbox(mask))
    d1, info = D2.frame(im, mask, dist, bbox=V1.mask_bbox(mask),
                        prior_xy=np.array([[220.0, 220.0]], np.float32))
    assert info['n_prior'] == 1
    assert len(d1['xy_px']) == len(d0['xy_px'])
    hit = np.hypot(d1['xy_px'][:, 0] - 220, d1['xy_px'][:, 1] - 220) < 4
    assert hit.any()

    # a prior in a dark gap must produce nothing
    d2, _ = D2.frame(im, mask, dist, bbox=V1.mask_bbox(mask),
                     prior_xy=np.array([[320.0, 320.0]], np.float32))
    assert len(d2['xy_px']) == len(d0['xy_px'])
    assert not (np.hypot(d2['xy_px'][:, 0] - 320, d2['xy_px'][:, 1] - 320) < 4).any()


def _frame_radii(items, size=700, seed=3, noise=6.0):
    """Like _frame, but each item carries its own radius."""
    rng = np.random.default_rng(seed)
    im = np.full((size, size), 45, np.float32)
    for x, y, r in items:
        cv2.circle(im, (int(x), int(y)), int(round(r)), 220, -1)
        cv2.line(im, (int(x), int(y)), (int(x + 0.75 * r), int(y - 0.5 * r)), 90, 5)
    im += rng.normal(0, noise, im.shape)
    return np.clip(im, 0, 255).astype(np.uint8)


def test_detect2_bidisperse_infers_two_radius_classes():
    """A mixed field must not force the small disc onto the large radius.

    Ten discs on two rings: five at 37 px and five at 31 px, far enough apart that the
    hard-disc NMS cannot merge a small one into a large one. The detector should infer
    two radius classes and return a radius per detection, not one global value.
    """
    items = []
    for t in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        items.append((350 + 160 * np.cos(t), 350 + 160 * np.sin(t), 37.0))
    for t in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        items.append((350 + 65 * np.cos(t + 0.3), 350 + 65 * np.sin(t + 0.3), 31.0))
    im = _frame_radii(items)
    mask, _ = _mask()
    d, info = D2.frame(im, mask, cv2.distanceTransform(mask, cv2.DIST_L2, 5),
                       bbox=V1.mask_bbox(mask), interior_mask=mask,
                       radii=[31.0, 37.0])
    assert len(info['radii']) == 2, info['radii']
    small = d['radius_px'] < 34
    assert small.sum() >= 4, (int(small.sum()), np.unique(d['radius_px']))
    assert (~small).sum() >= 4
    for x, y, r in items:
        dd = np.hypot(d['xy_px'][:, 0] - x, d['xy_px'][:, 1] - y).min()
        assert dd < 2.5, (x, y, r, float(dd))


def test_detect2_monodisperse_keeps_one_radius_class():
    centers = [(350 + 110 * np.cos(t), 350 + 110 * np.sin(t))
               for t in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
    im = _frame(centers)
    mask, _ = _mask()
    d, info = D2.frame(im, mask, cv2.distanceTransform(mask, cv2.DIST_L2, 5),
                       bbox=V1.mask_bbox(mask), interior_mask=mask)
    assert len(info['radii']) == 1
    assert np.allclose(d['radius_px'], d['radius_px'][0])
