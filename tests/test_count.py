"""The one-image counter: neighbour discovery, detection and the viewer bundle."""
from __future__ import annotations

import csv

import numpy as np

from vibrotrack import count as C


def test_neighbour_paths_rebuilds_names(tmp_path):
    for i in range(6):
        (tmp_path / f'run_{i:04d}.png').write_bytes(b'')
    paths = C.neighbour_paths(tmp_path / 'run_0003.png', before=2, after=1)
    assert [p.name for p in paths] == ['run_0001.png', 'run_0002.png', 'run_0004.png']


def test_neighbour_paths_ignores_unnumbered(tmp_path):
    (tmp_path / 'frame.png').write_bytes(b'')
    assert C.neighbour_paths(tmp_path / 'frame.png') == []


def test_neighbour_paths_skips_missing_numbers(tmp_path):
    (tmp_path / 'run_0000.png').write_bytes(b'')
    (tmp_path / 'run_0002.png').write_bytes(b'')
    paths = C.neighbour_paths(tmp_path / 'run_0002.png', before=3, after=0)
    assert [p.name for p in paths] == ['run_0000.png']


def test_write_bundle_embeds_particles(tmp_path):
    image = np.zeros((40, 30), np.uint8)
    det = {'xy_px': np.array([[5.0, 6.0], [10.0, 20.0]], np.float32),
           'radius_px': np.array([3.0, 3.5], np.float32)}
    meta = {'image': 'x.tif', 'width': 30, 'height': 40, 'n': 2,
            'radius_px': 3.25, 'neighbours': [], 'prior_pass': False, 'ms': 1}
    html, frame, csv_path = C.write_bundle(tmp_path / 'out', image, det, meta)
    text = html.read_text(encoding='utf8')
    assert 'data:image/png;base64,' in text
    assert '"n":2' in text
    assert frame.exists()
    rows = list(csv.reader(csv_path.open(encoding='utf8')))
    assert rows[0] == ['x_px', 'y_px', 'radius_px']
    assert len(rows) == 3


def test_count_image_uses_neighbours_and_finds_every_disc(tmp_path):
    import cv2

    n, r = 8, 37
    centres = [(400 + 250 * np.cos(t), 400 + 250 * np.sin(t))
               for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    for fi in range(6):
        im = np.full((800, 800), 45, np.float32)
        for (x, y) in centres:
            cv2.circle(im, (int(x), int(y)), r, 220, -1)
            cv2.line(im, (int(x), int(y)), (int(x + 27), int(y - 18)), 90, 5)
        rng = np.random.default_rng(fi)
        im = np.clip(im + rng.normal(0, 4, im.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(tmp_path / f'frame_{fi:04d}.png'), im)

    det, _image, meta = C.count_image(tmp_path / 'frame_0005.png', before=4)
    assert len(meta['neighbours']) == 4
    assert meta['n'] == n, meta
    assert len(det['xy_px']) == n
