"""The one-image counter: neighbour discovery, detection and the viewer bundle."""
from __future__ import annotations

import csv
import io
import json
from http.client import HTTPConnection
from threading import Thread

import numpy as np
from PIL import Image
import pytest

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
    assert (html.parent / 'overlay.png').is_file()
    with csv_path.open(encoding='utf8') as stream:
        rows = list(csv.reader(stream))
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


def png_bytes(*, color=False):
    buf = io.BytesIO()
    Image.new('RGB' if color else 'L', (128, 96), 32).save(buf, format='PNG')
    return buf.getvalue()


def test_presets_and_single_image_no_neighbours(tmp_path, monkeypatch):
    target = tmp_path / 'frame_0042.png'
    target.write_bytes(png_bytes(color=True))
    calls = []

    def detect(image, search, dist, **kwargs):
        assert image.shape == (96, 128) and image.dtype == np.uint8
        assert kwargs['prior_xy'] is None
        calls.append(kwargs)
        if kwargs['infer_radii']:
            return (dict(xy_px=[[30, 30], [90, 30]], radius_px=[31.7, 36.9]),
                    dict(radii=[31.7, 36.9]))
        return dict(xy_px=[[90, 30]], radius_px=[36.9]), dict(radii=[36.9])

    def unexpected(*args, **kwargs):
        pytest.fail('single-image mode tried to discover neighbours')

    monkeypatch.setattr(C, 'neighbour_paths', unexpected)
    monkeypatch.setattr(C.D2, 'frame', detect)
    det, _, meta = C.count_image(target, mode='bi12-14')
    browser_det, _, browser_meta = C.count_bytes(target.read_bytes(), target.name, 'bi12-14')
    assert C.grouped_particles(det, meta) == C.grouped_particles(browser_det, browser_meta)
    assert [row[3] for row in C.grouped_particles(det, meta)] == [0, 1]
    assert meta['neighbours'] == [] and not meta['prior_pass']
    assert calls[0]['small_template_radius'] == 15.5
    assert len(calls) == 2
    _, _, mono = C.count_image(target, mode='mono14')
    assert mono['n'] == 1 and len(mono['groups']) == 1
    assert 'small_template_radius' not in calls[-1]


def test_grouped_empty_bed_and_script_safe_filename():
    det, img, meta = C.count_bytes(png_bytes(), '</script><p>not markup', 'bi12-14')
    assert len(meta['groups']) == 2 and meta['n'] == 0
    page = C.viewer_page(C.viewer_data(img, det, meta))
    assert page.count('</script>') == 1
    assert '\\u003c/script>' in page


def test_local_app_real_http_roundtrip():
    from vibrotrack import count_app

    with count_app.make_server() as server:
        origin = f'http://127.0.0.1:{server.server_port}'
        thread = Thread(target=server.serve_forever, kwargs={'poll_interval':0.01}, daemon=True)
        thread.start()
        def request(method, path, body=None, headers=None):
            with_connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
            try:
                with_connection.request(method, path, body=body, headers=headers or {})
                res = with_connection.getresponse()
                return res.status, res.read()
            finally:
                with_connection.close()
        try:
            assert request('GET', '/')[0] == 200
            path = '/detect?mode=bi12-14&name=frame.png'
            headers = {'Content-Type':'application/octet-stream', 'Origin':origin}
            assert request('POST', path, png_bytes(), {'Content-Type':'application/octet-stream'})[0] == 403
            assert request('POST', '/detect?mode=unknown', png_bytes(), headers)[0] == 400
            assert request('POST', path, b'not an image', headers)[0] == 422
            status, body = request('POST', path, png_bytes(), headers)
            assert status == 200, body
            data = json.loads(body)
            assert data['meta']['image'] == 'frame.png'
            assert data['meta']['mode'] == 'bi12-14'
            assert data['meta']['neighbours'] == [] and data['particles'] == []
            assert len(data['meta']['groups']) == 2
        finally:
            server.shutdown()
            thread.join(timeout=5)
