"""Identity and atomic storage for internal detection caches."""

from contextlib import contextmanager
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
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


@contextmanager
def atomic_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.writing-', dir=path.parent) as tmpdir:
        temp = Path(tmpdir) / path.name
        yield temp
        temp.replace(path)
