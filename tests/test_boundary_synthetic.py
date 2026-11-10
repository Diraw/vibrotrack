"""Automatic boundary proposals on an 8-bit synthetic circular cell."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from vibrotrack import cli


def _make_case(tmp_path):
    frames = tmp_path / 'frames'
    frames.mkdir()
    size = 800
    cx = cy = 400
    # A filled, slightly moving bed: temporal max should recover the cell interior.
    rng = np.random.default_rng(7)
    pts = []
    for r in (0, 80, 150, 210, 250):
        n = max(1, int(2 * np.pi * r / 34)) if r else 1
        for k in range(n):
            t = 2 * np.pi * k / n
            pts.append((cx + r * np.cos(t), cy + r * np.sin(t)))
    for i in range(6):
        im = np.full((size, size), 12, np.uint8)
        for x, y in pts:
            # small collective rotation, so max over frames fills gaps
            t = 0.015 * i
            xx = cx + (x - cx) * np.cos(t) - (y - cy) * np.sin(t)
            yy = cy + (x - cx) * np.sin(t) + (y - cy) * np.cos(t)
            cv2.circle(im, (int(round(xx)), int(round(yy))), 18, 220, -1)
            cv2.circle(im, (int(round(xx)), int(round(yy))), 5, 90, -1)
        Image.fromarray(im).save(frames / f'frame_{i:04d}.tiff')
    cfg = {
        'apparatus': 'synthetic',
        'fps': 10.0,
        'px_per_mm': 5.0,
        'particle': {'diameter_mm': 7.0, 'nominal_n': len(pts)},
        'image': {'width': size, 'height': size, 'dtype': 'uint8'},
        'boundary': {},
        'detector': {'name': 'v2'},
        'linker': {'name': 'v2', 'max_gap': 5},
        'run': {'workers': 1, 'shard': 100},
    }
    cfg_path = tmp_path / 'apparatus.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return frames, cfg_path, pts


def test_mask_command_writes_proposal_and_overlay(tmp_path):
    frames, cfg_path, pts = _make_case(tmp_path)
    out = tmp_path / 'mask'
    rc = cli.main(['mask', '--raw', str(frames), '--out', str(out),
                   '--config', str(cfg_path), '--frames', '6'])
    assert rc == 0
    mask = np.asarray(Image.open(out / 'interior_mask.png')) > 0
    overlay = np.asarray(Image.open(out / 'mask_overlay.png'))
    assert mask.shape == (800, 800)
    assert mask.mean() > 0.15
    assert overlay.shape == (800, 800, 3)
    rec = json.loads((out / 'mask.json').read_text(encoding='utf-8'))
    assert rec['schema'] == 'mask-proposal/v1'
    assert rec['status'] == 'proposed'
    assert rec['mask_file'] == 'interior_mask.png'
    assert rec['overlay_file'] == 'mask_overlay.png'
    assert rec['mask_sha256']
    # every synthetic particle centre should be inside the proposed search mask
    for x, y in pts:
        assert mask[int(round(y)), int(round(x))], (x, y)

    # The polygon branch must at least produce a usable filled mask as well.
    out_poly = tmp_path / 'mask_poly'
    assert cli.main(['mask', '--raw', str(frames), '--out', str(out_poly),
                     '--config', str(cfg_path), '--frames', '6',
                     '--shape', 'polygon']) == 0
    mask_poly = np.asarray(Image.open(out_poly / 'interior_mask.png')) > 0
    assert mask_poly.mean() > 0.15


def _make_ring_case(tmp_path):
    frames = tmp_path / 'ring_frames'
    frames.mkdir()
    size = 800
    cx = cy = 400
    frozen = []
    for k in range(30):
        t = 2 * np.pi * k / 30
        frozen.append((cx + 290 * np.cos(t), cy + 290 * np.sin(t)))
    active = []
    for r in (40, 100, 160, 220):
        n = max(1, int(2 * np.pi * r / 34)) if r else 1
        for k in range(n):
            t = 2 * np.pi * k / n
            active.append((cx + r * np.cos(t), cy + r * np.sin(t)))
    for i in range(6):
        im = np.full((size, size), 12, np.uint8)
        # the physical circular frame around the frozen-particle ring
        cv2.circle(im, (cx, cy), 310, 180, 3)
        # frozen-particle ring: exactly the same pixels in every frame
        for x, y in frozen:
            cv2.circle(im, (int(round(x)), int(round(y))), 18, 220, -1)
        # active interior particles: small collective rotation
        t = 0.02 * i
        for x, y in active:
            xx = cx + (x - cx) * np.cos(t) - (y - cy) * np.sin(t)
            yy = cy + (x - cx) * np.sin(t) + (y - cy) * np.cos(t)
            cv2.circle(im, (int(round(xx)), int(round(yy))), 18, 220, -1)
        Image.fromarray(im).save(frames / f'frame_{i:04d}.tiff')
    cfg = {
        'apparatus': 'synthetic-ring',
        'fps': 10.0,
        'px_per_mm': 5.0,
        'particle': {'diameter_mm': 7.0, 'nominal_n': len(active)},
        'image': {'width': size, 'height': size, 'dtype': 'uint8'},
        'boundary': {},
        'detector': {'name': 'v2'},
        'linker': {'name': 'v2', 'max_gap': 5},
        'run': {'workers': 1, 'shard': 100},
    }
    cfg_path = tmp_path / 'ring_apparatus.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return frames, cfg_path, frozen, active


def test_mask_excludes_static_frozen_ring(tmp_path):
    frames, cfg_path, frozen, active = _make_ring_case(tmp_path)
    out = tmp_path / 'ring_mask'
    assert cli.main(['mask', '--raw', str(frames), '--out', str(out),
                     '--config', str(cfg_path), '--frames', '6']) == 0
    rec = json.loads((out / 'mask.json').read_text(encoding='utf-8'))
    assert rec['ring_detected'] is True
    assert rec['outer_radius_px'] > rec['radius_px'] + 5
    mask = np.asarray(Image.open(out / 'interior_mask.png')) > 0
    # frozen ring centres are outside the mask; active interior centres are inside
    for x, y in frozen:
        assert not mask[int(round(y)), int(round(x))], (x, y)
    for x, y in active:
        assert mask[int(round(y)), int(round(x))], (x, y)
