"""Apparatus configuration for vibrotrack.

One YAML file describes one camera/rig calibration and the detector defaults.  It
contains no absolute paths: a relative ``boundary.mask`` is resolved against
``base_dir`` (the current working directory by default), so the same file works on
another machine as long as the project layout is preserved.

Example
-------
    apparatus: shaker-2d
    fps: 10
    px_per_mm: 5.1558
    particle:
      diameter_mm: 14
      # nominal_n is the load of THIS run, not a property of the apparatus:
      #   nominal_n: 516
      # Leave it out and pass --n-particles instead; the CLI value wins anyway.
    image:
      width: 2500
      height: 2496
      dtype: uint16
    boundary:
      mask: masks/interior_mask.png
      model: empirical
      band: 1
      dilate_px: 40
      tol_px: 6.0
    detector:
      name: v2
      params:
        thresh: 0.20
        min_rays: 14
    linker:
      name: v2
      max_gap: 5
    run:
      workers: 8
      shard: 500
      prior_frames: 5
      shard_overlap: 5
    scale_levels: [128, 28224]
    conditions:
      frequency_hz: 140
      acceleration_g: 2.0
      tilt_deg: 0.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ApparatusConfig:
    path: Path | None = None
    apparatus: str = 'shaker'
    fps: float = 10.0
    px_per_mm: float = 1.0
    diameter_mm: float = 1.0
    nominal_n: int | None = None
    scale_levels: tuple[float, float] = (128.0, 28224.0)
    image_width: int | None = None
    image_height: int | None = None
    image_dtype: str | None = None
    boundary_mask: Path | None = None
    wall_model: str = 'auto'
    wall_band: int = 1
    wall_dilate: int = 40
    wall_tol_px: float = 6.0
    avoid_r: float = 0.0
    detector: str = 'v2'
    detector_params: dict[str, Any] = field(default_factory=dict)
    linker: str = 'v2'
    link_gap: int = 5
    workers: int = 8
    shard: int = 500
    strict_io: bool = False
    prefetch: bool = True
    prior: bool = True
    prior_frames: int = 5
    shard_overlap: int = 5
    extensions: tuple[str, ...] = ('.tiff', '.tif')
    conditions: dict[str, Any] = field(default_factory=dict)
    calibration_source: str = 'unspecified'
    calibration_uncertainty: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    @property
    def radius_px(self) -> float:
        return self.radius_mm * self.px_per_mm


def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve(value, base_dir):
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = Path(base_dir) / p
    return p.resolve()


def _choice(value, default):
    if value is None:
        return default
    if value is False:          # YAML 1.1 parses bare `off` as False
        return 'off'
    if value is True:
        return 'on'
    return str(value).lower()


def load_config(path, base_dir=None) -> ApparatusConfig:
    """Load and validate a YAML/JSON apparatus config."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError('apparatus config not found: %s' % path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        import json
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError('apparatus config must be a mapping: %s' % path)

    base_dir = Path(base_dir or Path.cwd()).resolve()
    cfg = ApparatusConfig(path=path)
    cfg.apparatus = str(raw.get('apparatus', cfg.apparatus))
    cfg.fps = float(raw.get('fps', cfg.fps))
    cfg.px_per_mm = float(raw.get('px_per_mm', cfg.px_per_mm))
    cfg.diameter_mm = float(_get(raw, 'particle', 'diameter_mm', default=raw.get('diameter_mm', cfg.diameter_mm)))
    cfg.nominal_n = _get(raw, 'particle', 'nominal_n', default=raw.get('nominal_n', cfg.nominal_n))
    if cfg.nominal_n is not None:
        cfg.nominal_n = int(cfg.nominal_n)
    levels = raw.get('scale_levels', cfg.scale_levels)
    if not isinstance(levels, (list, tuple)) or len(levels) != 2:
        raise ValueError('scale_levels must be [lo, hi]')
    cfg.scale_levels = (float(levels[0]), float(levels[1]))
    cfg.image_width = _get(raw, 'image', 'width', default=raw.get('image_width'))
    cfg.image_height = _get(raw, 'image', 'height', default=raw.get('image_height'))
    cfg.image_dtype = _get(raw, 'image', 'dtype', default=raw.get('image_dtype'))
    cfg.boundary_mask = _resolve(_get(raw, 'boundary', 'mask', default=raw.get('boundary_mask')), base_dir)
    cfg.wall_model = _choice(_get(raw, 'boundary', 'model', default=cfg.wall_model), cfg.wall_model)
    cfg.wall_band = int(_get(raw, 'boundary', 'band', default=cfg.wall_band))
    cfg.wall_dilate = int(_get(raw, 'boundary', 'dilate_px', default=cfg.wall_dilate))
    cfg.wall_tol_px = float(_get(raw, 'boundary', 'tol_px', default=cfg.wall_tol_px))
    cfg.avoid_r = float(_get(raw, 'boundary', 'avoid_r', default=cfg.avoid_r))
    cfg.detector = str(_get(raw, 'detector', 'name', default=raw.get('detector', cfg.detector))).lower()
    cfg.detector_params = dict(_get(raw, 'detector', 'params', default={}) or {})
    cfg.linker = str(_get(raw, 'linker', 'name', default=raw.get('linker', cfg.linker))).lower()
    cfg.link_gap = int(_get(raw, 'linker', 'max_gap', default=cfg.link_gap))
    cfg.workers = int(_get(raw, 'run', 'workers', default=cfg.workers))
    cfg.shard = int(_get(raw, 'run', 'shard', default=cfg.shard))
    cfg.strict_io = bool(_get(raw, 'run', 'strict_io', default=cfg.strict_io))
    cfg.prefetch = bool(_get(raw, 'run', 'prefetch', default=cfg.prefetch))
    cfg.prior = bool(_get(raw, 'run', 'prior', default=cfg.prior))
    cfg.prior_frames = int(_get(raw, 'run', 'prior_frames', default=cfg.prior_frames))
    cfg.shard_overlap = int(_get(raw, 'run', 'shard_overlap', default=cfg.shard_overlap))
    exts = raw.get('extensions')
    if exts:
        cfg.extensions = tuple(str(e).lower() if str(e).startswith('.') else '.' + str(e).lower() for e in exts)
    cfg.conditions = dict(raw.get('conditions') or {})
    cal = raw.get('calibration') or {}
    cfg.calibration_source = str(cal.get('source', raw.get('px_per_mm_source', cfg.calibration_source)))
    unc = cal.get('uncertainty', raw.get('px_per_mm_uncertainty', cfg.calibration_uncertainty))
    cfg.calibration_uncertainty = float(unc) if unc is not None else None
    cfg.extra = {k: v for k, v in raw.items() if k not in {
        'apparatus', 'fps', 'px_per_mm', 'particle', 'nominal_n', 'scale_levels',
        'image', 'boundary', 'detector', 'linker', 'run', 'extensions',
        'conditions', 'calibration', 'px_per_mm_source', 'px_per_mm_uncertainty',
    }}
    _validate(cfg)
    return cfg


def _validate(cfg: ApparatusConfig):
    if cfg.fps <= 0:
        raise ValueError('fps must be > 0')
    if cfg.px_per_mm <= 0:
        raise ValueError('px_per_mm must be > 0')
    if cfg.diameter_mm <= 0:
        raise ValueError('particle.diameter_mm must be > 0')
    if cfg.detector not in ('v1', 'v2'):
        raise ValueError('detector.name must be v1 or v2, got %r' % cfg.detector)
    if cfg.linker not in ('v1', 'v2'):
        raise ValueError('linker.name must be v1 or v2, got %r' % cfg.linker)
    if cfg.wall_model not in ('auto', 'geometry', 'empirical', 'off'):
        raise ValueError('boundary.model must be auto|geometry|empirical|off')
    if cfg.wall_band not in (0, 1):
        raise ValueError('boundary.band must be 0 or 1')
    if cfg.prior_frames < 1:
        raise ValueError('run.prior_frames must be >= 1')
    if cfg.shard_overlap < 0:
        raise ValueError('run.shard_overlap must be >= 0')
    if cfg.scale_levels[1] <= cfg.scale_levels[0]:
        raise ValueError('scale_levels[1] must be greater than scale_levels[0]')
