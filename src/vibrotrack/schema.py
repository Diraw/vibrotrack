"""Small schema/provenance helpers shared by the CLI and batch tools."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_DATASET = 'dataset/v1'
SCHEMA_INDEX = 'index/v1'
SCHEMA_ACCEPTANCE = 'acceptance/v1'


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def slugify(text):
    text = str(text).strip().lower()
    text = re.sub(r'[^a-z0-9._-]+', '-', text)
    return text.strip('-_') or 'case'


def package_version():
    try:
        from importlib.metadata import version
        return version('vibrotrack')
    except Exception:
        return '0.0.0+unknown'


def git_info(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    info = {}
    try:
        info['commit'] = subprocess.run(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10).stdout.strip()
        info['dirty'] = bool(subprocess.run(
            ['git', '-C', str(root), 'status', '--porcelain'],
            capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        pass
    return info


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(p)
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def outputs_record(directory, names):
    """Hash the named files if they exist; return a list for dataset.json."""
    out = []
    directory = Path(directory)
    for name in names:
        p = directory / name
        if p.exists():
            out.append({'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    return out
