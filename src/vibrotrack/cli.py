"""Command line interface for vibrotrack."""
from __future__ import annotations

import argparse
import sys

import cv2
from pathlib import Path

import numpy as np
from PIL import Image

from . import batch as batch_mod
from . import boundary
from . import detect as X
from . import geometry as G
from . import pipeline
from . import schema
from .config import load_config


def _parse_centre(text):
    parts = [float(x) for x in str(text).replace(';', ',').split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('centre must be "x,y"')
    return parts


def _seed_centre(xy, mask_path):
    if mask_path:
        from PIL import Image
        a = np.asarray(Image.open(mask_path)) > 0
        ys, xs = np.where(a)
        if len(xs):
            return [float(xs.mean()), float(ys.mean())]
    xy = np.asarray(xy, float)
    return [float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))]


def _load_case_xy(case_dir):
    files = sorted(Path(case_dir).glob('chunk_*.npz'))
    if not files:
        raise FileNotFoundError('no chunk_*.npz in %s' % case_dir)
    xy, counts = [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            xy.append(z['xy_px'])
            counts.append(np.diff(z['frame_offsets']))
    return np.concatenate(xy), np.concatenate(counts)


def _cmd_extract(args):
    cfg = load_config(args.config, base_dir=args.base_dir)
    for key, value in (('fps', args.fps), ('px_per_mm', args.px_per_mm),
                       ('diameter_mm', args.diameter_mm), ('workers', args.workers),
                       ('shard', args.shard)):
        if value is not None:
            setattr(cfg, key, value)
    if args.n_particles is not None:
        cfg.nominal_n = int(args.n_particles)
    if args.wall_model is not None:
        cfg.wall_model = args.wall_model
    if args.wall_band is not None:
        cfg.wall_band = int(args.wall_band)
    if args.strict_io:
        cfg.strict_io = True
    if args.mask:
        cfg.boundary_mask = Path(args.mask).expanduser().resolve()
    case_id = args.case_id or Path(args.raw).name
    rec = pipeline.run_case(
        args.raw, args.out, cfg, case_id=case_id, nframes=args.nframes,
        workers=args.workers, shard=args.shard, detector=args.detector,
        linker=args.linker, wall_band=args.wall_band, wall_model=args.wall_model,
        strict_io=args.strict_io, no_link=args.no_link, limit_shards=args.limit_shards,
        dataset_id=args.dataset_id, project=args.project, campaign=args.campaign,
        acquired_at=args.acquired_at, operator=args.operator,
    )
    print('dataset: %s' % rec['dataset_id'])
    print('case:    %s' % rec['case'])
    print('frames:  %s' % rec['n_frames'])
    print('out:     %s' % (Path(args.out) / 'sequences' / case_id))
    return 0


def _cmd_mask(args):
    cfg = load_config(args.config, base_dir=args.base_dir)
    info, mask_path, overlay_path, json_path = boundary.propose_from_raw(
        args.raw, args.out, cfg, n_frames=args.frames, shape=args.shape,
        margin_px=args.margin_px, threshold=args.threshold,
        preview_frame=args.preview_frame, force=args.force,
        points=args.points)
    print('proposed: %s' % info['shape'])
    if info.get('method') == 'manual_polygon':
        print('manual:   %d boundary points, %d hole(s)'
              % (len(info.get('points_px', [])), len(info.get('holes_px', []))))
    if info.get('radius_px') is not None:
        print('center:   [%.2f, %.2f]' % tuple(info['center_px']))
        print('radius:   %.2f px (after %.2f px inward margin)'
              % (info['radius_px'], info['margin_px']))
        if info.get('ring_detected'):
            print('ring:     static/frozen ring detected; outer r=%.2f px, mask stops inside'
                  % info['outer_radius_px'])
        if info.get('frozen_disc_count'):
            print('frozen:   %d individual frozen disc(s) subtracted from the mask'
                  % info['frozen_disc_count'])
    print('mask:     %s' % mask_path)
    print('overlay:  %s' % overlay_path)
    print('meta:     %s' % json_path)
    print('inspect the overlay; if it covers the cell, use --mask %s' % mask_path)
    return 0


def _cmd_frame(args):
    """Export one normalized frame as PNG/JPEG for the manual mask editor."""
    cfg = load_config(args.config, base_dir=args.base_dir)
    files = X.files_for({'path': str(Path(args.raw).expanduser()),
                         'extensions': tuple(cfg.extensions)})
    if not files:
        raise FileNotFoundError('no frames found in %s' % args.raw)
    index = int(args.index)
    if index < 0:
        index += len(files)
    if index < 0 or index >= len(files):
        raise IndexError('frame index %s outside 0..%d' % (args.index, len(files) - 1))
    img = X.read_image(files[index], cfg.scale_levels)
    if args.max_size and max(img.shape) > args.max_size:
        scale = float(args.max_size) / max(img.shape)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    print('frame:   %s  (%dx%d) -> %s' % (files[index].name, img.shape[1], img.shape[0], out))
    return 0


def _cmd_geometry(args):
    cfg = load_config(args.config, base_dir=args.base_dir)
    for key, value in (('fps', args.fps), ('px_per_mm', args.px_per_mm),
                       ('diameter_mm', args.diameter_mm)):
        if value is not None:
            setattr(cfg, key, value)
    case_dir = Path(args.case_dir).resolve()
    xy, counts = _load_case_xy(case_dir)
    mask_path = Path(args.mask).resolve() if args.mask else cfg.boundary_mask
    seed = args.seed_centre or _seed_centre(xy, mask_path)
    n_particles = args.n_particles if args.n_particles is not None else cfg.nominal_n
    rec = G.build(
        args.case_id or case_dir.name, xy, seed, mask_path=mask_path,
        px_per_mm=cfg.px_per_mm, fps=cfg.fps, radius_mm=cfg.radius_mm,
        n_particles=n_particles, n_detected_median=float(np.median(counts)),
        px_per_mm_source=cfg.calibration_source,
    )
    out = Path(args.out) if args.out else case_dir / 'geometry.json'
    G.write(out, rec)
    print('geometry: %s' % out)
    print('  centre_px      %s' % (rec.get('center_px'),))
    print('  wall_radius_px %.2f' % rec['wall_radius_px'])
    print('  phi            %s' % (rec.get('phi'),))
    return 0


def _cmd_index(args):
    idx = batch_mod.build_index(args.batch_dir)
    print('index: %d sequences, %d frames, %d rows -> %s'
          % (idx['n_sequences'], idx['total_frames'], idx['total_rows'],
             Path(args.batch_dir) / 'index.json'))
    return 0


def _print_accept(report):
    for r in report['sequences']:
        print('%s %-42s frames=%s center=%.4f angle=%.4f%s'
              % ('OK  ' if r['ok'] else 'FAIL', r['sequence'], r.get('frames_covered'),
                 r.get('center_valid_rate', 0.0), r.get('angle_valid_rate', 0.0),
                 '' if r['ok'] else '  <- ' + '; '.join(r['problems'])))
    print('%s: %d checked, %d failed'
          % ('ACCEPTED' if report['accepted'] else 'REJECTED',
             report['n_checked'], report['n_failed']))


def _cmd_accept(args):
    report = batch_mod.accept(args.batch_dir, min_center=args.min_center,
                              min_angle=args.min_angle)
    _print_accept(report)
    return 0 if report['accepted'] else 1


def _cmd_verify(args):
    report, bad = batch_mod.verify(args.batch_dir)
    _print_accept(report)
    for seq, problem in bad:
        print('FAIL %s: %s' % (seq, problem))
    ok = report['accepted'] and not bad
    print('VERIFY %s' % ('OK' if ok else 'FAILED'))
    return 0 if ok else 1


def _cmd_archive(args):
    dest, frozen = batch_mod.archive(args.batch_dir, args.archive_root, force=args.force)
    print('archived: %s (%d files, %.1f MB)'
          % (dest, frozen['n_files'], frozen['total_bytes'] / 1e6))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog='vibrotrack',
        description='Dense particle tracking for quasi-2D vibrated granular monolayers.')
    ap.add_argument('--version', action='version', version='%(prog)s ' + schema.package_version())
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('extract', help='detect + link one case')
    p.add_argument('--raw', required=True, help='directory of TIFF frames')
    p.add_argument('--out', required=True, help='batch directory; products go to <out>/sequences/<case>/')
    p.add_argument('--config', required=True, help='apparatus YAML')
    p.add_argument('--base-dir', default=None, help='resolve relative config paths here (default: cwd)')
    p.add_argument('--mask', default=None, help='interior mask PNG; overrides boundary.mask in the config')
    p.add_argument('--case-id', default=None)
    p.add_argument('--nframes', type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--shard', type=int, default=None)
    p.add_argument('--detector', choices=['v1', 'v2'], default=None)
    p.add_argument('--linker', choices=['v1', 'v2'], default=None)
    p.add_argument('--wall-band', type=int, choices=[0, 1], default=None)
    p.add_argument('--wall-model', choices=['auto', 'geometry', 'empirical', 'off'], default=None)
    p.add_argument('--strict-io', action='store_true')
    p.add_argument('--no-link', action='store_true')
    p.add_argument('--limit-shards', type=int, default=0, help='pilot only: stop after N shards')
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--px-per-mm', type=float, default=None)
    p.add_argument('--diameter-mm', type=float, default=None)
    p.add_argument('--n-particles', type=int, default=None)
    p.add_argument('--dataset-id', default=None)
    p.add_argument('--project', default='')
    p.add_argument('--campaign', default='')
    p.add_argument('--acquired-at', default=None)
    p.add_argument('--operator', default=None)
    p.set_defaults(func=_cmd_extract)

    p = sub.add_parser('mask', help='propose a boundary mask from raw frames and write a confirmation overlay')
    p.add_argument('--raw', required=True, help='directory of TIFF frames')
    p.add_argument('--out', required=True, help='directory for interior_mask.png, mask_overlay.png, mask.json')
    p.add_argument('--config', required=True, help='apparatus YAML (image/extensions/scale levels)')
    p.add_argument('--base-dir', default=None)
    p.add_argument('--frames', type=int, default=12, help='number of frames sampled across the recording')
    p.add_argument('--shape', choices=['auto', 'circle', 'polygon', 'irregular'],
                   default='auto')
    p.add_argument('--points', default=None,
                   help='manual polygon JSON; bypasses automatic boundary fitting')
    p.add_argument('--margin-px', type=float, default=2.0)
    p.add_argument('--threshold', type=float, default=None, help='foreground threshold; default: Otsu')
    p.add_argument('--preview-frame', type=int, default=None, help='frame index used for the overlay')
    p.add_argument('--force', action='store_true', help='overwrite an existing proposal')
    p.set_defaults(func=_cmd_mask)

    p = sub.add_parser('frame', help='export one frame as PNG/JPEG for manual mask drawing')
    p.add_argument('--raw', required=True, help='directory of TIFF frames')
    p.add_argument('--config', required=True, help='apparatus YAML')
    p.add_argument('--base-dir', default=None)
    p.add_argument('--index', type=int, default=0, help='frame index; negative counts from the end')
    p.add_argument('--max-size', type=int, default=0, help='max output side in px; 0 = original (use original when exporting a mask)')
    p.add_argument('--out', required=True, help='output PNG/JPEG')
    p.set_defaults(func=_cmd_frame)

    p = sub.add_parser('geometry', help='write geometry.json for a tracked case')
    p.add_argument('--case-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--base-dir', default=None)
    p.add_argument('--case-id', default=None)
    p.add_argument('--seed-centre', type=_parse_centre, default=None,
                   help='x,y; default: mask centroid or median particle position')
    p.add_argument('--mask', default=None)
    p.add_argument('--n-particles', type=int, default=None, help='nominal N, not detections')
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--px-per-mm', type=float, default=None)
    p.add_argument('--diameter-mm', type=float, default=None)
    p.add_argument('--out', default=None)
    p.set_defaults(func=_cmd_geometry)

    p = sub.add_parser('index', help='assemble index.json for a batch')
    p.add_argument('--batch-dir', required=True)
    p.set_defaults(func=_cmd_index)

    for name, func, text in (
        ('accept', _cmd_accept, 'run the admission gate'),
        ('verify', _cmd_verify, 're-hash products and re-run accept'),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument('--batch-dir', required=True)
        p.add_argument('--min-center', type=float, default=None)
        p.add_argument('--min-angle', type=float, default=None)
        p.set_defaults(func=func)

    p = sub.add_parser('archive', help='copy a batch to an immutable archive and write FROZEN.json')
    p.add_argument('--batch-dir', required=True)
    p.add_argument('--archive-root', required=True)
    p.add_argument('--force', action='store_true')
    p.set_defaults(func=_cmd_archive)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
