"""Differential tests for provably unambiguous assignment fast paths."""
import numpy as np
import pytest
from vibrotrack import link2 as L


@pytest.mark.parametrize('seed', range(20))
def test_assignment_random_matches_reference(seed):
    rng = np.random.default_rng(seed)
    for n, m, gate in [(0, 10, 33.), (10, 0, 33.), (100, 100, 33.),
                       (90, 110, 33.), (110, 90, 33.), (80, 100, 120.)]:
        pred = rng.uniform(0, 2400, (n, 2))
        cur = rng.uniform(0, 2400, (m, 2))
        if n and m:
            k = min(n, m)
            cur[:k] = pred[:k] + rng.normal(0, 3, (k, 2))
        actual = L._assign(pred, cur, gate)
        expected = L._assign_dense(pred, cur, gate)
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize('gate', [0., -1., 1., 33., float('inf'), 1e7])
def test_assignment_ties_and_strict_gate(gate):
    pred = np.array([[0., 0.], [0., 0.], [100., 100.]])
    cur = np.array([[1., 0.], [-1., 0.], [133., 100.]])
    a = L._assign(pred, cur, gate)
    b = L._assign_dense(pred, cur, gate)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_large_isolated_assignment_skips_hungarian(monkeypatch):
    pred = np.column_stack([np.arange(100) * 80., np.zeros(100)])
    cur = pred + [3., 4.]
    def forbidden(*args):
        raise AssertionError('isolated edges do not need Hungarian')
    monkeypatch.setattr(L, 'linear_sum_assignment', forbidden)
    rows, cols = L._assign(pred, cur, 33.)
    assert np.array_equal(rows, np.arange(100))
    assert np.array_equal(cols, rows)


def test_large_competition_falls_back(monkeypatch):
    pred = np.column_stack([np.arange(100) * 80., np.zeros(100)])
    cur = pred.copy()
    cur[1] = cur[0] + [1., 0.]
    expected = L._assign_dense(pred, cur, 33.)
    original = L.linear_sum_assignment
    calls = []
    def counted(cost):
        calls.append(cost.shape)
        return original(cost)
    monkeypatch.setattr(L, 'linear_sum_assignment', counted)
    actual = L._assign(pred, cur, 33.)
    assert calls
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize('dtype,offset', [(np.float32, 0.), (np.float64, 1e100)])
def test_unusual_precision_uses_dense(monkeypatch, dtype, offset):
    pred = np.column_stack([np.arange(100) * 80., np.zeros(100)]).astype(dtype) + offset
    cur = pred.copy()
    original = L._assign_dense
    calls = []
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(L, '_assign_dense', counted)
    L._assign(pred, cur, 33.)
    assert calls


def test_near_gate_uses_reference(monkeypatch):
    pred = np.column_stack([np.arange(100) * 80., np.zeros(100)])
    cur = pred + [np.nextafter(33., 0.), 0.]
    original = L._assign_dense
    calls = []
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(L, '_assign_dense', counted)
    a = L._assign(pred, cur, 33.)
    b = original(pred, cur, 33.)
    assert calls
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
