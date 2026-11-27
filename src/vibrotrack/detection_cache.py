"""Identity and atomic storage for internal detection caches."""

from contextlib import contextmanager
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import PIL
import scipy

from . import __version__


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    return sha256(data.encode('ascii')).hexdigest()


def frame_stamps(files):
    """Use metadata for TIFFs so a cache check does not have to reread the movie."""
    stamps = []
    for file in files:
        path = os.path.abspath(file)
        stat = os.stat(path)
        stamps.append((path, stat.st_size, stat.st_mtime_ns))
    return stamps


def file_stamp(file):
    path = Path(file).absolute()
    with path.open('rb') as handle:
        checksum = sha256()
        for block in iter(lambda: handle.read(1 << 20), b''):
            checksum.update(block)
    return str(path), checksum.hexdigest()


@lru_cache(maxsize=1)
def implementation():
    sources = ('detect.py', 'detect2.py', 'fast_tiff.py', 'mask.py', 'shards.py')
    return {
        'version': __version__,
        'sources': {name: file_stamp(Path(__file__).with_name(name))[1] for name in sources},
        'opencv': cv2.__version__,
        'numpy': np.__version__,
        'pillow': PIL.__version__,
        'scipy': scipy.__version__,
    }


def recorded_shard_directory(case_dir, legacy_name, *, run_config=None):
    """Locate the archive named by run metadata, or the legacy layout."""
    if run_config is None:
        path = case_dir / 'run_config.json'
        run_config = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
    expected = run_config.get('detection_cache_digest')
    if expected is None:
        return case_dir / legacy_name, None
    if not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
        raise ValueError('invalid detection_cache_digest in %s' % (case_dir / 'run_config.json'))

    matches = [path for path in case_dir.glob(f'dets_*_c{expected[:24]}') if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f'recorded detection cache {expected} not found in {case_dir}')
    if len(matches) != 1:
        raise ValueError(f'ambiguous directories for detection cache {expected} in {case_dir}')
    return matches[0], expected


@contextmanager
def open_recorded_shard(path, expected):
    """Read a shard and verify its full digest when the run recorded one."""
    with np.load(path, allow_pickle=False) as shard:
        if expected is not None:
            stored = shard['_cache_digest'] if '_cache_digest' in shard.files else None
            if stored is None or stored.ndim != 0 or stored.item() != expected:
                raise ValueError(f'detection cache identity mismatch: {path}')
        yield shard


@contextmanager
def atomic_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.writing-', dir=path.parent) as tmpdir:
        temp = Path(tmpdir) / path.name
        yield temp
        temp.replace(path)
