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

import hashlib
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from . import schema
from .case_output import LOCK_NAME, STAGE_NAME, case_lock, require_published

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


def _index_case(seq_dir, allow_missing_geometry):
    require_published(seq_dir)
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
        geometry_bytes = geometry_path.read_bytes()
        geom = json.loads(geometry_bytes)
        rec['geometry_sha256'] = hashlib.sha256(geometry_bytes).hexdigest()
        rec['geometry'] = {
            'centre_method': geom.get('centre_method'),
            'envelope_resid_rms_px': (geom.get('centre_quality') or {}).get('envelope_resid_rms_px'),
            'wall_radius_px': geom.get('wall_radius_px'),
            'wall_radius_source': geom.get('wall_radius_source'),
            'n_particles': geom.get('n_particles'),
            'phi': geom.get('phi'),
            'phi_valid': geom.get('phi_valid'),
            'px_per_mm': geom.get('px_per_mm'),
            'fps': geom.get('fps'),
        }
        resid = (geom.get('centre_quality') or {}).get('envelope_resid_rms_px')
        if resid is not None and resid > 30:
            if geom.get('phi_valid') is False:
                rec['geometry_warning'] = 'envelope fit failed; phi is not usable'
            else:
                rec['geometry_warning'] = ('envelope fit failed; wall radius supplied (%s)'
                                           % geom.get('wall_radius_source'))
    else:
        rec['geometry_sha256'] = None
        rec['geometry'] = None
        rec['geometry_warning'] = 'geometry.json missing; run `vibrotrack geometry`'
        if not allow_missing_geometry:
            raise FileNotFoundError(
                'missing geometry.json in %s; run `vibrotrack geometry --case-dir %s '
                '--config <apparatus.yaml>` before indexing, or pass '
                '--allow-missing-geometry for an exploratory batch'
                % (seq_dir, seq_dir))
    cfg_path = seq_dir / 'run_config.json'
    rec['run_config_sha256'] = schema.sha256(cfg_path) if cfg_path.exists() else None
    rec['status'] = 'ok'
    return rec


def build_index(batch_dir, gate=None, allow_missing_geometry=False):
    batch_dir = Path(batch_dir).resolve()
    seq_root = batch_dir / 'sequences'
    if not seq_root.is_dir():
        raise FileNotFoundError('batch has no sequences/ directory: %s' % batch_dir)
    gate = dict(DEFAULT_GATE, **(gate or {}))
    records = []
    total_frames = total_rows = 0
    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        with case_lock(seq_dir):
            rec = _index_case(seq_dir, allow_missing_geometry)
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
    try:
        require_published(seq_dir)
    except RuntimeError as exc:
        return {'sequence': rec['sequence'], 'ok': False, 'problems': [str(exc)]}
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
    geom_path = seq_dir / 'geometry.json'
    if not geom_path.exists():
        problems.append('missing geometry.json')
    elif rec.get('geometry_sha256') is None:
        # The index was built when the file was absent, so nothing tied the
        # batch to a wall radius or a phi.  Re-index, do not guess.
        problems.append('geometry.json is not in index.json; re-run `vibrotrack index`')
    elif schema.sha256(geom_path) != rec['geometry_sha256']:
        problems.append('geometry sha256 does not match index')

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
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    gate = dict(DEFAULT_GATE, **(index.get('gate') or {}))
    if min_center is not None:
        gate['min_center_rate'] = float(min_center)
    if min_angle is not None:
        gate['min_angle_rate'] = float(min_angle)
    results = [_check_sequence(batch_dir, rec, gate) for rec in index['sequences']]
    report = {
        'schema_version': schema.SCHEMA_ACCEPTANCE,
        'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
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
                'geometry.json present and its sha256 matches index.json',
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


def _archive_acceptance(path, force):
    if not path.exists():
        if force:
            return None
        raise FileNotFoundError(
            'refusing to archive without acceptance.json; run `vibrotrack accept` first (%s)' % path)
    report = schema.read_json(path)
    if not force and not report.get('accepted'):
        raise ValueError('refusing to archive a rejected batch; fix the gate or use --force')
    return report


def _validate_archive(snapshot, case_names, files, force):
    def check_cases(record, filename):
        names = [rec['sequence'] for rec in record['sequences']]
        if len(names) != len(case_names) or set(names) != case_names:
            raise ValueError(f'archive snapshot case set disagrees with {filename}. Re-index and retry archive.')

    if {p.name for p in (snapshot / 'sequences').glob('*') if p.is_dir()} != case_names:
        raise ValueError('archive snapshot is missing cases. Retry archive.')
    acceptance = _archive_acceptance(snapshot / 'acceptance.json', force)
    if acceptance is not None:
        check_cases(acceptance, 'acceptance.json')
    index_path = snapshot / 'index.json'
    if not index_path.exists():
        if force:
            return
        raise FileNotFoundError('archive snapshot is missing index.json. Re-index and retry archive.')
    index = schema.read_json(index_path)
    check_cases(index, 'index.json')
    hashes = {f['path']: f['sha256'] for f in files}
    accepted_index = (acceptance or {}).get('index_sha256')
    if accepted_index is not None and hashes['index.json'] != accepted_index:
        raise ValueError('archive snapshot index hash disagrees with acceptance.json. Re-run accept and retry archive.')
    for rec in index['sequences']:
        prefix = f"sequences/{rec['sequence']}"
        if rec.get('dataset_id') is not None:
            manifest = schema.read_json(snapshot / prefix / 'dataset.json')
            if manifest.get('dataset_id') != rec['dataset_id']:
                raise ValueError(f'archive snapshot dataset ID disagrees with index.json: {prefix}')
        checks = {
            'npz_sha256': rec.get('npz_file'),
            'dataset_sha256': 'dataset.json',
            'summary_sha256': 'full_summary.json',
            'run_config_sha256': 'run_config.json',
            'geometry_sha256': 'geometry.json',
        }
        for field, name in checks.items():
            expected = rec.get(field)
            relpath = f'{prefix}/{name}'
            if expected is not None and hashes.get(relpath) != expected:
                raise ValueError(f'archive snapshot hash disagrees with index.json: {relpath}. Re-index and retry archive.')


def archive(batch_dir, archive_root, force=False):
    """Copy a verified batch to an immutable archive directory and write FROZEN.json."""
    batch_dir = Path(batch_dir).resolve()
    archive_root = Path(archive_root).resolve()
    dest = archive_root / batch_dir.name
    resolved_dest = dest.resolve()
    if (archive_root.is_relative_to(batch_dir) or batch_dir.is_relative_to(resolved_dest)
            or resolved_dest.parent != archive_root or dest.is_symlink()):
        raise ValueError('archive destination must be outside the source batch and must not be a symlink')
    seq_root = batch_dir / 'sequences'
    cases = sorted(p for p in seq_root.glob('*') if p.is_dir())
    case_names = {case.name for case in cases}
    if not force:
        _archive_acceptance(batch_dir / 'acceptance.json', force)
        if dest.exists():
            raise FileExistsError('archive target already exists: %s (use --force to replace)' % dest)
    ignore_transient = shutil.ignore_patterns('dets_*', 'progress.json', '*.tmp.npz',
                                              '__pycache__', '.pytest_cache', LOCK_NAME, STAGE_NAME)

    def ignore(source, names):
        ignored = ignore_transient(source, names)
        if Path(source) == seq_root:
            # Only the captured case directories have publication locks.
            ignored.difference_update(case_names)
            ignored.update(name for name in names
                           if name not in case_names and (seq_root / name).is_dir())
        return ignored

    archive_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.vibrotrack-archive-', dir=archive_root))
    snapshot = temporary / 'snapshot'
    previous = temporary / 'previous'
    published = False
    try:
        with ExitStack() as locks:
            for case in cases:
                locks.enter_context(case_lock(case))
                require_published(case)
            if {p.name for p in seq_root.glob('*') if p.is_dir()} != case_names:
                raise RuntimeError(f'case set changed while preparing archive: {seq_root}. Retry archive.')
            shutil.copytree(batch_dir, snapshot, ignore=ignore)

        files = [{'path': p.relative_to(snapshot).as_posix(), 'bytes': p.stat().st_size,
                  'sha256': schema.sha256(p)}
                 for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name != 'FROZEN.json']
        _validate_archive(snapshot, case_names, files, force)
        frozen = {
            'schema_version': 'frozen/v1',
            'frozen_at': schema.now_iso(),
            'source': str(batch_dir),
            'n_files': len(files),
            'total_bytes': sum(f['bytes'] for f in files),
            'files': files,
        }
        schema.write_json(snapshot / 'FROZEN.json', frozen)
        try:
            if dest.exists():
                if not force:
                    raise FileExistsError('archive target already exists: %s (use --force to replace)' % dest)
                dest.rename(previous)
            snapshot.rename(dest)
        except BaseException:
            if previous.exists():
                try:
                    if not snapshot.exists() and dest.exists():
                        # The new rename completed before the interruption was delivered.
                        dest.rename(snapshot)
                    previous.rename(dest)
                except OSError as error:
                    raise RuntimeError(f'could not restore previous archive from {previous}') from error
            raise
        published = True
        return dest, frozen
    finally:
        # Leave the backup available if restoration failed.
        if published or not previous.exists():
            shutil.rmtree(temporary, ignore_errors=True)
