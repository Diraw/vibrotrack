"""Regression tests for the extraction hot paths.

The real-data equivalence is checked separately (fixed frames + archived chunks).
These tests keep the conversion and linker semantics from drifting.
"""
import sys
import tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
from vibrotrack import detect as X  # noqa: E402


def _reference_scale(a, levels):
    lo, hi = levels
    return np.clip((a.astype(np.float32) - lo) * (255 / (hi - lo)), 0, 255).astype(np.uint8)


def test_scale_to_u8_matches_reference():
    rng = np.random.default_rng(20261108)
    a = rng.integers(0, 65535, size=(64, 80), dtype=np.uint16)
    a[0, :4] = [0, 127, 128, 65535]
    for levels in [(128, 28224), (0, 65535), (100, 20000), (100.5, 20000.5)]:
        fast = X.scale_to_u8(a, levels)
        ref = _reference_scale(a, levels)
        assert np.array_equal(fast, ref), levels


def test_mask_bbox():
    mask = np.zeros((100, 120), np.uint8)
    mask[20:60, 30:70] = 1
    x0, x1, y0, y1 = X.mask_bbox(mask, margin=10)
    assert (x0, x1, y0, y1) == (20, 80, 10, 70)
    assert np.array_equal(X.mask_bbox(mask, margin=0), (30, 70, 20, 60))
    assert X.mask_bbox(np.zeros((10, 10), np.uint8)) is None


def test_files_for_skips_only_trailing_empty():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / 'f_0000.tiff').write_bytes(b'x')
        (root / 'f_0001.tiff').write_bytes(b'x')
        (root / 'f_0002.tiff').write_bytes(b'')
        files = X.files_for({'path': td})
        assert [f.name for f in files] == ['f_0000.tiff', 'f_0001.tiff']
        # A middle empty is still a hard error when requested explicitly.
        (root / 'f_0001.tiff').write_bytes(b'')
        (root / 'f_0002.tiff').write_bytes(b'x')
        try:
            X.files_for({'path': td}, strict=True)
        except IOError:
            pass
        else:
            raise AssertionError('middle 0-byte frame was not rejected')


def _dense_link(prev, cur, nextid, prevoffset):
    """The pre-2026-11-08 dense-Hungarian implementation, kept as a reference."""
    n = len(cur['xy_px'])
    tids = np.arange(nextid, nextid + n, dtype=np.int32)
    nextid += n
    pd = np.full(n, -1, np.int64)
    margin = np.full(n, np.nan, np.float32)
    step = np.full(n, np.nan, np.float32)
    dt = np.full(n, np.nan, np.float32)
    if prev is not None and len(prev['xy_px']) and n:
        a = prev['xy_px']
        b = cur['xy_px']
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        gate = float(min(np.median(prev['radius_px']), np.median(cur['radius_px'])) * .9)
        cost = np.where(d < gate, d, 1e6)
        rows, cols = linear_sum_assignment(cost)
        row2 = np.partition(d, min(1, n - 1), axis=1)[:, min(1, n - 1)] if n > 1 else np.full(len(a), np.inf)
        col2 = np.partition(d, min(1, len(a) - 1), axis=0)[min(1, len(a) - 1)] if len(a) > 1 else np.full(n, np.inf)
        for r, c in zip(rows, cols):
            dd = d[r, c]
            gap = min(row2[r], col2[c]) - dd
            margin[c] = gap
            if dd < gate and gap > max(5., .22 * gate) and dd == d[r].min() and dd == d[:, c].min():
                tids[c] = prev['segment_id'][r]
                pd[c] = prevoffset + r
                step[c] = dd
                if prev['angle_valid'][r] and cur['angle_valid'][c]:
                    dt[c] = np.arctan2(np.sin(cur['theta_rad'][c] - prev['theta_rad'][r]),
                                       np.cos(cur['theta_rad'][c] - prev['theta_rad'][r]))
    cur.update(segment_id=tids, previous_detection=pd, link_valid=pd >= 0,
               link_margin_px=margin, step_px=step, dtheta_rad=dt)
    return nextid


def test_link_matches_dense_reference():
    rng = np.random.default_rng(7)
    gx, gy = np.meshgrid(np.arange(8) * 72.0, np.arange(8) * 72.0)
    base = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    n = len(base)
    prev = dict(xy_px=base,
                radius_px=np.full(n, 36, np.float32),
                theta_rad=rng.uniform(-np.pi, np.pi, n).astype(np.float32),
                angle_valid=np.ones(n, bool),
                segment_id=np.arange(n, dtype=np.int32))
    # 10 fps-like jitter, plus one missing and one spurious detection.
    cur_xy = (base + rng.normal(0, 1.5, (n, 2))).astype(np.float32)
    keep = np.ones(n, bool); keep[17] = False
    cur_xy = cur_xy[keep]
    cur_xy = np.vstack([cur_xy, np.array([[2000., 2000.]], np.float32)])
    cur = dict(xy_px=cur_xy,
               radius_px=np.full(len(cur_xy), 36, np.float32),
               theta_rad=rng.uniform(-np.pi, np.pi, len(cur_xy)).astype(np.float32),
               angle_valid=np.ones(len(cur_xy), bool))
    prev_ref = {k: v.copy() for k, v in prev.items()}
    cur_ref = {k: v.copy() for k, v in cur.items()}
    X.link(prev, cur, 0, 0)
    _dense_link(prev_ref, cur_ref, 0, 0)
    for k in ('segment_id', 'previous_detection', 'link_valid', 'dtheta_rad'):
        assert np.array_equal(cur[k], cur_ref[k], equal_nan=True), k
    assert np.allclose(cur['step_px'], cur_ref['step_px'], atol=1e-6, equal_nan=True)


def test_link_without_previous():
    cur = dict(xy_px=np.array([[10., 20.]], np.float32), radius_px=np.array([36.], np.float32),
               theta_rad=np.array([0.], np.float32), angle_valid=np.array([True]))
    X.link(None, cur, 5, 0)
    assert cur['segment_id'][0] == 5
    assert cur['previous_detection'][0] == -1
    assert not cur['link_valid'][0]
    assert np.isnan(cur['step_px'][0])


if __name__ == '__main__':
    test_scale_to_u8_matches_reference()
    test_mask_bbox()
    test_files_for_skips_only_trailing_empty()
    test_link_matches_dense_reference()
    test_link_without_previous()
    print('extract core: 5/5 OK')


def test_read_image_uint8_passthrough(tmp_path):
    from PIL import Image
    a = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 256).astype(np.uint8)
    p = tmp_path / 'frame_0000.tif'
    Image.fromarray(a).save(p)
    out = X.read_image(str(p), (0, 65535))
    assert out.dtype == np.uint8
    assert np.array_equal(out, a)
