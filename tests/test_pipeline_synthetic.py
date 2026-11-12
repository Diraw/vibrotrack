"""End-to-end smoke test on a synthetic 12-disc field.

No lab data is required: the test writes 16-bit TIFF frames, a circular mask and
an apparatus config, then runs the real CLI (extract -> index -> accept -> verify).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from vibrotrack import cli
from vibrotrack import geometry as G
from vibrotrack.config import load_config


def _frame(centers, size=800, seed=3):
    rng = np.random.default_rng(seed)
    im = np.full((size, size), 45, np.float32)
    for x, y in centers:
        cv2.circle(im, (int(x), int(y)), 37, 220, -1)
        cv2.line(im, (int(x), int(y)), (int(x + 27), int(y - 18)), 90, 5)
    im += rng.normal(0, 6, im.shape)
    im = np.clip(im, 0, 255).astype(np.uint8)
    return (im.astype(np.uint16) * 257)


def _make_case(tmp_path):
    frames = tmp_path / 'frames'
    frames.mkdir()
    centres = [(400 + 250 * np.cos(t), 400 + 250 * np.sin(t))
               for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    for i in range(6):
        Image.fromarray(_frame(centres, seed=10 + i)).save(frames / f'frame_{i:04d}.tiff')
    masks = tmp_path / 'masks'
    masks.mkdir()
    mask = np.zeros((800, 800), np.uint8)
    cv2.circle(mask, (400, 400), 320, 255, -1)
    Image.fromarray(mask).save(masks / 'interior_mask.png')
    cfg = {
        'apparatus': 'synthetic',
        'fps': 10.0,
        'px_per_mm': 5.0,
        'scale_levels': [0, 65535],
        'particle': {'diameter_mm': 14.0, 'nominal_n': 12},
        'image': {'width': 800, 'height': 800, 'dtype': 'uint16'},
        'boundary': {'mask': 'masks/interior_mask.png', 'model': 'off', 'band': 0},
        'detector': {'name': 'v2', 'params': {}},
        'linker': {'name': 'v2', 'max_gap': 5},
        'run': {'workers': 2, 'shard': 3},
    }
    cfg_path = tmp_path / 'apparatus.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return frames, cfg_path


def test_cli_extract_index_accept_verify(tmp_path):
    frames, cfg_path = _make_case(tmp_path)
    batch = tmp_path / 'batch'
    rc = cli.main([
        'extract', '--raw', str(frames), '--out', str(batch),
        '--config', str(cfg_path), '--base-dir', str(tmp_path),
        '--case-id', 'syn', '--nframes', '6',
    ])
    assert rc == 0

    case_dir = batch / 'sequences' / 'syn'
    chunks = list(case_dir.glob('chunk_*.npz'))
    assert len(chunks) == 1
    with np.load(chunks[0], allow_pickle=False) as z:
        assert z['frame_ids'].tolist() == list(range(6))
        assert len(z['xy_px']) > 0
        assert len(z['frame_offsets']) == 7
    ds = json.loads((case_dir / 'dataset.json').read_text(encoding='utf-8'))
    assert ds['fps'] == 10.0
    assert ds['n_frames'] == 6
    assert ds['calibration']['px_per_mm'] == 5.0

    assert cli.main(['index', '--batch-dir', str(batch)]) == 0
    assert cli.main(['accept', '--batch-dir', str(batch),
                     '--min-center', '0', '--min-angle', '0']) == 0
    assert cli.main(['verify', '--batch-dir', str(batch),
                     '--min-center', '0', '--min-angle', '0']) == 0

    archive = tmp_path / 'archive'
    assert cli.main(['archive', '--batch-dir', str(batch),
                     '--archive-root', str(archive)]) == 0
    assert (archive / 'batch' / 'FROZEN.json').exists()



def test_cli_extract_mask_override(tmp_path):
    frames, cfg_path = _make_case(tmp_path)
    alt = tmp_path / 'alt_mask.png'
    shutil.copy(tmp_path / 'masks' / 'interior_mask.png', alt)
    batch = tmp_path / 'batch_override'
    rc = cli.main([
        'extract', '--raw', str(frames), '--out', str(batch),
        '--config', str(cfg_path), '--base-dir', str(tmp_path),
        '--mask', str(alt), '--case-id', 'syn_alt', '--nframes', '6',
    ])
    assert rc == 0
    ds = json.loads((batch / 'sequences' / 'syn_alt' / 'dataset.json').read_text(encoding='utf-8'))
    assert ds['boundary']['mask'] == str(alt.resolve())

def test_config_paths_and_radius(tmp_path):
    _, cfg_path = _make_case(tmp_path)
    cfg = load_config(cfg_path, base_dir=tmp_path)
    assert cfg.radius_px == 35.0
    assert cfg.boundary_mask == (tmp_path / 'masks' / 'interior_mask.png').resolve()


def test_geometry_envelope_fit_on_dense_ring():
    # 3600 points: 10 per 1-degree bin, enough for the envelope fit's min_samples.
    rng = np.random.default_rng(4)
    th = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    rr = 400.0 + rng.normal(0, 0.3, th.size)
    xy = np.c_[500 + rr * np.cos(th), 500 + rr * np.sin(th)]
    rec = G.build('syn', xy, [500, 500], px_per_mm=5.0, fps=10.0,
                  radius_mm=7.0, n_particles=1000)
    assert abs(rec['center_px'][0] - 500) < 1.0
    assert abs(rec['center_px'][1] - 500) < 1.0
    assert abs(rec['wall_radius_px'] - (400 + 35)) < 2.0
    assert rec['phi'] > 0


def test_wall_radius_override_and_phi_validity():
    # A dense bed whose binned envelope is irregular: the gate must fail and phi
    # must not be trusted unless the wall radius is supplied.
    th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    rr = 350 + 60 * np.sign(np.sin(5 * th))
    xy = np.c_[500 + rr * np.cos(th), 500 + rr * np.sin(th)]
    rec = G.build('syn', xy, [500, 500], px_per_mm=5.0, fps=10.0,
                  radius_mm=7.0, n_particles=500,
                  radius_classes=[[30.0, 300], [35.0, 200]])
    assert rec['centre_quality']['gate_passed'] is False
    assert rec['phi_valid'] is False
    assert rec['wall_radius_source'] == 'envelope_plus_particle_radius'

    rec2 = G.build('syn', xy, [500, 500], px_per_mm=5.0, fps=10.0,
                   radius_mm=7.0, n_particles=500,
                   radius_classes=[[30.0, 300], [35.0, 200]],
                   wall_radius_px=450.0)
    assert rec2['phi_valid'] is True
    assert rec2['wall_radius_px'] == 450.0
    assert abs(rec2['phi'] - (300 * 30 ** 2 + 200 * 35 ** 2) / 450.0 ** 2) < 1e-9


def test_prior_and_overlap_defaults():
    from vibrotrack import shards
    from vibrotrack.config import ApparatusConfig
    cfg = ApparatusConfig()
    assert cfg.prior_frames == 5
    assert cfg.shard_overlap == 5
    tag = shards.det_tag({'detector': 'v2', 'prior_frames': 7,
                          'shard_overlap': 3})
    assert 'p7_o3' in tag
