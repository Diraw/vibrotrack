"""Tests for manual-centre orientation estimation (no detector call)."""
from __future__ import annotations

import numpy as np

from vibrotrack.orientation import estimate_marker_orientation


def _marker_image(angle):
    im = np.full((180, 180), 25, np.uint8)
    cx = cy = 90
    yy, xx = np.mgrid[:180, :180]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= 23 ** 2
    im[disc] = 185
    mx = int(round(cx + np.cos(angle) * 9))
    my = int(round(cy - np.sin(angle) * 9))
    spot = (xx - mx) ** 2 + (yy - my) ** 2 <= 6 ** 2
    im[spot] = 35
    return im


def test_known_marker_angles_are_y_up():
    for angle in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        im = _marker_image(angle)
        theta, valid, offset, area, bright = estimate_marker_orientation(
            im, [[90.0, 90.0]])
        assert bool(valid[0]), angle
        # Angle wrapping is sufficient here: all test angles are cardinal.
        d = np.arctan2(np.sin(theta[0] - angle), np.cos(theta[0] - angle))
        assert abs(d) < 0.2, (angle, theta[0])
        assert 2 <= offset[0] <= 16
        assert 100 <= area[0] <= 380
        assert bright[0] > 85


def test_empty_disc_is_invalid():
    im = np.full((180, 180), 25, np.uint8)
    yy, xx = np.mgrid[:180, :180]
    im[(xx - 90) ** 2 + (yy - 90) ** 2 <= 23 ** 2] = 185
    theta, valid, *_ = estimate_marker_orientation(im, [[90.0, 90.0]])
    assert not bool(valid[0])
    assert np.isnan(theta[0])
