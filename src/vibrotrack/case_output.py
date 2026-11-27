"""Writer exclusion and recoverable publication of a case's flat products."""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from . import schema


LOCK_NAME = '.extract.lock'
STAGE_NAME = '.extract-publish'
_PRODUCTS = {'dataset.json', 'full_summary.json', 'run_config.json', 'progress.json'}
_locked = set()
_guard = Lock()


@contextmanager
def case_lock(case_dir):
    """Exclude extraction and archive copying for the same case."""
    case_dir = Path(case_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    message = f'another extraction or archive is running for {case_dir}'
    with _guard:
        if case_dir in _locked:
            raise RuntimeError(message)
        _locked.add(case_dir)
    try:
        with (case_dir / LOCK_NAME).open('a+b') as handle:
            try:
                if os.name == 'nt':
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    # Process-owned locks are released when the parent exits,
                    # even if a forked detection worker inherited the handle.
                    fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(message) from exc
            yield
    finally:
        with _guard:
            _locked.remove(case_dir)


def _stage_path(case_dir):
    stage = case_dir / STAGE_NAME
    if stage.resolve().parent != case_dir.resolve() or stage.is_symlink():
        raise ValueError(f'invalid extraction staging directory: {stage}')
    return stage


def require_published(case_dir):
    if (Path(case_dir) / STAGE_NAME / 'prepared.json').is_file():
        raise RuntimeError(f'case publication interrupted or in progress: {case_dir}; '
                           'rerun extract to recover before using the case')


def _product_name(name):
    return (Path(name).name == name and (name in _PRODUCTS
            or (name.startswith('chunk_') and name.endswith('.npz'))))


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def recover_case(case_dir):
    """Restore the previous products if a writer stopped during publication."""
    case_dir = Path(case_dir)
    stage = _stage_path(case_dir)
    journal = stage / 'prepared.json'
    if journal.is_file():
        record = schema.read_json(journal)
        old, new = set(record['previous']), set(record['incoming'])
        if not all(_product_name(name) for name in old | new):
            raise ValueError(f'invalid product name in {journal}')
        for name in sorted(old):
            destination = case_dir / name
            if destination.exists() and (name not in new or (stage / 'new' / name).exists()):
                continue  # This name was never changed, or is already restored.
            restore = stage / 'restore.tmp'
            restore.unlink(missing_ok=True)
            _link_or_copy(stage / 'previous' / name, restore)
            restore.replace(destination)
        for name in new - old:
            (case_dir / name).unlink(missing_ok=True)
        journal.unlink()
    if stage.exists():
        shutil.rmtree(stage)


@contextmanager
def stage_case(case_dir):
    """Build products off-path, then publish or roll back under the case lock."""
    case_dir = Path(case_dir)
    stage = _stage_path(case_dir)
    incoming = stage / 'new'
    incoming.mkdir(parents=True)
    try:
        yield incoming
        new = sorted(p.name for p in incoming.iterdir())
        if not new or not all(_product_name(name) for name in new):
            raise ValueError(f'unexpected extraction products in {incoming}')
        # A detection-only run publishes metadata without creating a chunk.
        old_chunks = {p.name for p in case_dir.glob('chunk_*.npz')} if (
            any(name.startswith('chunk_') for name in new)) else set()
        old = sorted(old_chunks | {name for name in new if (case_dir / name).exists()})
        previous = stage / 'previous'
        previous.mkdir()
        for name in old:
            _link_or_copy(case_dir / name, previous / name)
        journal = schema.write_json(stage / 'prepared.json', {'previous': old, 'incoming': new})
        # The dataset manifest goes last so it describes the products already in place.
        for name in new:
            if name == 'dataset.json':
                continue
            (incoming / name).replace(case_dir / name)
        for name in old_chunks - set(new):
            (case_dir / name).unlink()
        if 'dataset.json' in new:
            (incoming / 'dataset.json').replace(case_dir / 'dataset.json')
        journal.unlink()
    except BaseException:
        recover_case(case_dir)
        raise
    else:
        # The products are committed once the journal is removed. Temporary
        # debris can be cleaned on the next run if, for example, it is open.
        shutil.rmtree(stage, ignore_errors=True)
