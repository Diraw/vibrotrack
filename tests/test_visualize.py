"""Regression tests for the raw-frame detection overlay."""
from __future__ import annotations

import numpy as np
from PIL import Image

from vibrotrack.visualize import _display_uint8, render_frame, theta_endpoint


def test_theta_endpoint_is_y_up_in_image_coordinates():
    # theta=+pi/2 must point toward the top of the image (smaller y).
    x, y = theta_endpoint(100.0, 100.0, 10.0, np.pi / 2)
    assert abs(x - 100.0) < 1e-6
    assert abs(y - 90.0) < 1e-6
    # theta=0 points right; theta=pi points left.
    assert theta_endpoint(100.0, 100.0, 10.0, 0.0) == (110.0, 100.0)
    assert theta_endpoint(100.0, 100.0, 10.0, np.pi) == (90.0, 100.0)


def test_display_accepts_grayscale_rgb_and_rgba():
    for shape in ((20, 20), (20, 20, 3), (20, 20, 4)):
        a = np.zeros(shape, np.uint8)
        assert _display_uint8(a).shape == (20, 20, 3)


def test_render_frame_draws_tick_toward_theta_endpoint(tmp_path):
    image = np.zeros((200, 200), np.uint8)
    out = tmp_path / 'overlay.png'
    render_frame(image, [[100.0, 100.0]], [20.0], [np.pi / 2],
                 out, max_size=0, label=None)
    a = np.asarray(Image.open(out).convert('RGB'))
    # Orange tick is (0, 180, 255) in RGB; the correct endpoint is above centre.
    assert a[82:88, 98:103, 2].max() > 100
    assert a[82:88, 98:103, 1].max() > 80
    # The mirrored (wrong) endpoint below the centre must remain background.
    assert a[113:121, 98:103, 2].max() < 30


def test_plot_cli_reads_a_chunk_and_writes_an_overlay(tmp_path):
    import yaml
    from vibrotrack import cli

    raw = tmp_path / 'frames'
    raw.mkdir()
    image = np.zeros((200, 200), np.uint8)
    image[80:120, 80:120] = 220
    Image.fromarray(image).save(raw / 'frame_0000.tiff')
    config = {
        'apparatus': 'test',
        'fps': 10.0,
        'px_per_mm': 1.0,
        'particle': {'diameter_mm': 1.0, 'nominal_n': 1},
        'image': {'width': 200, 'height': 200, 'dtype': 'uint8'},
        'scale_levels': [0, 255],
    }
    cfg = tmp_path / 'apparatus.yaml'
    cfg.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    case = tmp_path / 'batch' / 'sequences' / 'case'
    case.mkdir(parents=True)
    np.savez(case / 'chunk_00000_00000.npz',
             xy_px=np.array([[100.0, 100.0]], np.float32),
             radius_px=np.array([20.0], np.float32),
             theta_rad=np.array([np.pi / 2], np.float32),
             center_valid=np.array([True]),
             angle_valid=np.array([True]),
             frame_ids=np.array([0]),
             frame_offsets=np.array([0, 1], np.int32))
    out = tmp_path / 'overlays'
    rc = cli.main(['plot', '--raw', str(raw), '--case-dir', str(case),
                   '--config', str(cfg), '--frames', '0', '--out', str(out),
                   '--max-size', '200'])
    assert rc == 0
    produced = list(out.glob('*_frame00000_detections.jpg'))
    assert len(produced) == 1
    assert Image.open(produced[0]).size == (200, 200)
