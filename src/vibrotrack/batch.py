"""Batch assembly, admission gates and hash verification.

A batch directory is::

    <batch>/
      sequences/<case>/dataset.json
      sequences/<case>/chunk_*.npz
      sequences/<case>/geometry.json
      sequences/<case>/full_summary.json
      sequences/<case>/run_config.json
      index.json
      acceptance.json

``index`` only reads and hashes; it never touches raw frames.  ``accept``
checks the chunk contract and writes ``acceptance.json``.  ``verify`` re-hashes
every product and re-runs ``accept``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from . import schema

REQUIRED = [
    'xy_px', 'radius_px', 'theta_rad', 'center_valid', 'angle_valid',
    'confidence', 'edge_rmse_px', 'marker_offset_px', 'marker_area_px',
    'neighbor_distance_px', 'boundary_distance_px', 'quality_bits',
    'segment_id', 'previous_detection', 'link_valid', 'step_px',
    'dtheta_rad', 'frame_ids', 'frame_offsets',
]
PER_DETECTION = [k for k in REQUIRED if k not in ('frame_ids', 'frame_offsets')]
DEFAULT_GATE = {'min_center_rate': 0.90, 'min_angle_rate': 0.90}


def _chunk(seq_dir):
    files = sorted(Path(seq_dir).glob('chunk_*.npz'))
    if len(files) != 1:
        raise ValueError('%s: expected exactly one chunk_*.npz, found %d' % (seq_dir, len(files)))
    return files[0]


def build_index(batch_dir, gate=None):
    batch_dir = Path(batch_dir).resolve()
    seq_root = batch_dir / 'sequences'
    if not seq_root.is_dir():
        raise FileNotFoundError('batch has no sequences/ directory: %s' % batch_dir)
    gate = dict(DEFAULT_GATE, **(gate or {}))
    records = []
    total_frames = total_rows = 0
    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        ds_path = seq_dir / 'dataset.json'
        if not ds_path.exists():
            raise FileNotFoundError('missing dataset.json in %s' % seq_dir)
        ds = schema.read_json(ds_path)
        npz = _chunk(seq_dir)
        with np.load(npz, allow_pickle=False) as z:
            frame_ids = z['frame_ids']
            rows = len(z['xy_px'])
            seg = z['segment_id']
            rec = {
                'sequence': seq_dir.name,
                'dataset_id': ds.get('dataset_id'),
                'case': ds.get('case', seq_dir.name),
                'source': (ds.get('retention') or {}).get('raw_path'),
                'frames_expected': ds.get('n_frames'),
                'frames_covered': int(len(frame_ids)),
                'first_frame': int(frame_ids[0]) if len(frame_ids) else None,
                'last_frame': int(frame_ids[-1]) if len(frame_ids) else None,
                'rows': int(rows),
                'tracks': int(len(np.unique(seg))),
                'tracks_linked': int(len(np.unique(seg[seg >= 0]))),
                'center_valid_rate': round(float(z['center_valid'].mean()), 6),
                'angle_valid_rate': round(float(z['angle_valid'].mean()), 6),
                'link_fraction': round(float(z['link_valid'].mean()), 6),
                'npz_file': npz.name,
                'npz_bytes': npz.stat().st_size,
                'npz_sha256': schema.sha256(npz),
                'dataset_sha256': schema.sha256(ds_path),
            }
        summary_path = seq_dir / 'full_summary.json'
        if summary_path.exists():
            summary = schema.read_json(summary_path)
            rec['count_median'] = summary.get('count_median')
            rec['summary_sha256'] = schema.sha256(summary_path)
        else:
            rec['count_median'] = None
            rec['summary_sha256'] = None
        geometry_path = seq_dir / 'geometry.json'
        if geometry_path.exists():
            geom = schema.read_json(geometry_path)
            rec['geometry_sha256'] = schema.sha256(geometry_path)
            rec['geometry'] = {
                'centre_method': geom.get('centre_method'),
                'envelope_resid_rms_px': (geom.get('centre_quality') or {}).get('envelope_resid_rms_px'),
                'wall_radius_px': geom.get('wall_radius_px'),
                'n_particles': geom.get('n_particles'),
                'phi': geom.get('phi'),
                'px_per_mm': geom.get('px_per_mm'),
                'fps': geom.get('fps'),
            }
            resid = (geom.get('centre_quality') or {}).get('envelope_resid_rms_px')
            if resid is not None and resid > 30:
                rec['geometry_warning'] = 'envelope_resid_rms_px > 30 px'
        else:
            rec['geometry_sha256'] = None
            rec['geometry'] = None
            rec['geometry_warning'] = 'geometry.json missing; run `vibrotrack geometry`'
        cfg_path = seq_dir / 'run_config.json'
        rec['run_config_sha256'] = schema.sha256(cfg_path) if cfg_path.exists() else None
        rec['status'] = 'ok'
        records.append(rec)
        total_frames += rec['frames_covered']
        total_rows += rec['rows']

    index = {
        'schema_version': schema.SCHEMA_INDEX,
        'created_at': schema.now_iso(),
        'created_by': 'vibrotrack index',
        'batch': batch_dir.name,
        'gate': gate,
        'n_sequences': len(records),
        'total_frames': total_frames,
        'total_rows': total_rows,
        'sequences': records,
    }
    schema.write_json(batch_dir / 'index.json', index)
    return index


def _check_sequence(batch_dir, rec, gate):
    problems = []
    seq_dir = Path(batch_dir) / 'sequences' / rec['sequence']
    npz_path = seq_dir / rec['npz_file']
    if not npz_path.exists():
        return {'sequence': rec['sequence'], 'ok': False, 'problems': ['missing npz']}

    with np.load(npz_path, allow_pickle=False) as d:
        missing = [k for k in REQUIRED if k not in d.files]
        if missing:
            return {'sequence': rec['sequence'], 'ok': False,
                    'problems': ['missing fields: ' + ', '.join(missing)]}
        n = len(d['xy_px'])
        for k in PER_DETECTION:
            if len(d[k]) != n:
                problems.append('length mismatch in %s: %d vs %d' % (k, len(d[k]), n))
                break
        if n == 0:
            return {'sequence': rec['sequence'], 'ok': False, 'problems': problems + ['no rows']}

        frame_ids = d['frame_ids']
        offsets = d['frame_offsets']
        if len(offsets) != len(frame_ids) + 1:
            problems.append('frame_offsets length %d != frames+1 %d' % (len(offsets), len(frame_ids) + 1))
        elif offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 0):
            problems.append('frame_offsets not a monotone partition of the rows')
        if not np.array_equal(frame_ids, np.arange(len(frame_ids), dtype=frame_ids.dtype)):
            problems.append('frame_ids are not 0..N-1 contiguous')

        center = d['center_valid'].astype(bool)
        angle = d['angle_valid'].astype(bool)
        if center.any():
            for k in ('xy_px', 'radius_px'):
                if not np.all(np.isfinite(d[k][center])):
                    problems.append('non-finite %s where center_valid' % k)
        if not angle.any():
            problems.append('no valid angle rows')
        else:
            th = d['theta_rad'][angle]
            if not np.all(np.isfinite(th)):
                problems.append('non-finite theta where angle_valid')
            elif th.min() < -np.pi - 1e-6 or th.max() > np.pi + 1e-6:
                problems.append('theta outside [-pi, pi]: %.4f..%.4f' % (th.min(), th.max()))

        center_rate = float(center.mean())
        angle_rate = float(angle.mean())
        if center_rate < gate['min_center_rate']:
            problems.append('center_valid rate %.4f < %.2f' % (center_rate, gate['min_center_rate']))
        if angle_rate < gate['min_angle_rate']:
            problems.append('angle_valid rate %.4f < %.2f' % (angle_rate, gate['min_angle_rate']))

        frame_of_det = np.repeat(frame_ids, np.diff(offsets))
        seg = d['segment_id']
        order = np.argsort(seg, kind='stable')
        seg_sorted = seg[order]
        frame_sorted = frame_of_det[order]
        same = seg_sorted[1:] == seg_sorted[:-1]
        if same.any() and not np.all(np.diff(frame_sorted)[same] > 0):
            problems.append('a segment has non-increasing frame index')
        n_frames = int(len(frame_ids))
        n_tracks = int(len(np.unique(seg)))

    if rec.get('frames_covered') != rec.get('frames_expected'):
        problems.append('frames_covered %s != frames_expected %s'
                        % (rec.get('frames_covered'), rec.get('frames_expected')))
    if rec.get('npz_sha256') and schema.sha256(npz_path) != rec['npz_sha256']:
        problems.append('npz sha256 does not match index')

    return {
        'sequence': rec['sequence'],
        'ok': not problems,
        'problems': problems,
        'frames_covered': n_frames,
        'rows': int(n),
        'tracks': n_tracks,
        'center_valid_rate': round(center_rate, 6),
        'angle_valid_rate': round(angle_rate, 6),
    }


def accept(batch_dir, min_center=None, min_angle=None):
    batch_dir = Path(batch_dir).resolve()
    index_path = batch_dir / 'index.json'
    if not index_path.exists():
        raise FileNotFoundError('missing index.json; run `vibrotrack index` first: %s' % index_path)
    index = schema.read_json(index_path)
    gate = dict(DEFAULT_GATE, **(index.get('gate') or {}))
    if min_center is not None:
        gate['min_center_rate'] = float(min_center)
    if min_angle is not None:
        gate['min_angle_rate'] = float(min_angle)
    results = [_check_sequence(batch_dir, rec, gate) for rec in index['sequences']]
    report = {
        'schema_version': schema.SCHEMA_ACCEPTANCE,
        'accepted': bool(results) and all(r['ok'] for r in results),
        'gate': {
            **gate,
            'required_fields': REQUIRED,
            'checks': [
                'all required chunk fields present',
                'per-detection arrays share one length',
                'frame_offsets partition rows, frame_ids contiguous 0..N-1',
                'finite centre where center_valid; theta finite and in [-pi,pi] where angle_valid',
                'valid-centre / valid-angle rates above thresholds',
                'segment frame indices strictly increasing',
                'frames_covered equals frames_expected',
                'npz sha256 matches index.json',
            ],
        },
        'n_checked': len(results),
        'n_failed': sum(1 for r in results if not r['ok']),
        'sequences': results,
    }
    schema.write_json(batch_dir / 'acceptance.json', report)
    return report


def verify(batch_dir):
    batch_dir = Path(batch_dir).resolve()
    index_path = batch_dir / 'index.json'
    if not index_path.exists():
        raise FileNotFoundError('missing index.json; run `vibrotrack index` first: %s' % index_path)
    index = schema.read_json(index_path)
    bad = []
    for rec in index['sequences']:
        seq_dir = batch_dir / 'sequences' / rec['sequence']
        checks = [
            ('npz_sha256', rec.get('npz_file')),
            ('geometry_sha256', 'geometry.json'),
            ('summary_sha256', 'full_summary.json'),
            ('run_config_sha256', 'run_config.json'),
            ('dataset_sha256', 'dataset.json'),
        ]
        for field, name in checks:
            if not name:
                continue
            p = seq_dir / name
            expected = rec.get(field)
            if expected is None:
                continue
            if not p.exists():
                bad.append((rec['sequence'], 'missing ' + name))
            elif schema.sha256(p) != expected:
                bad.append((rec['sequence'], 'hash mismatch: ' + name))
    report = accept(batch_dir)
    return report, bad


def archive(batch_dir, archive_root, force=False):
    """Copy a verified batch to an immutable archive directory and write FROZEN.json."""
    batch_dir = Path(batch_dir).resolve()
    archive_root = Path(archive_root).resolve()
    acc_path = batch_dir / 'acceptance.json'
    if not force:
        if not acc_path.exists():
            raise FileNotFoundError(
                'refusing to archive without acceptance.json; run `vibrotrack accept` first (%s)' % acc_path)
        if not schema.read_json(acc_path).get('accepted'):
            raise ValueError('refusing to archive a rejected batch; fix the gate or use --force')
    dest = archive_root / batch_dir.name
    if dest.exists() and not force:
        raise FileExistsError('archive target already exists: %s (use --force to replace)' % dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(batch_dir, dest)
    files = []
    for p in sorted(dest.rglob('*')):
        if p.is_file() and p.name != 'FROZEN.json':
            files.append({'path': str(p.relative_to(dest)).replace('\\', '/'),
                          'bytes': p.stat().st_size, 'sha256': schema.sha256(p)})
    frozen = {
        'schema_version': 'frozen/v1',
        'frozen_at': schema.now_iso(),
        'source': str(batch_dir),
        'n_files': len(files),
        'total_bytes': sum(f['bytes'] for f in files),
        'files': files,
    }
    schema.write_json(dest / 'FROZEN.json', frozen)
    return dest, frozen
