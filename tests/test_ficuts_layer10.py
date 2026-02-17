"""Tests for FICUTS Layer 10: Multi-Instance Coordination"""

import time
import numpy as np
import pytest

from tensor.multi_instance import (
    GlobalManifoldAggregator,
    InstanceWorker,
    IsometricTransfer,
    MultiInstanceCoordinator,
)


# ── Task 10.1: Spawning ───────────────────────────────────────────────────────

def test_spawn_returns_id():
    coord = MultiInstanceCoordinator(max_instances=4)
    center = np.array([1.0, 0.0, 0.0])
    iid = coord.spawn_child(center, exploration_radius=0.3)
    assert iid == 0
    coord.shutdown_all()


def test_spawn_multiple():
    coord = MultiInstanceCoordinator(max_instances=4)
    centers = [np.array([float(i), 0., 0.]) for i in range(3)]
    ids = [coord.spawn_child(c) for c in centers]
    assert ids == [0, 1, 2]
    coord.shutdown_all()


def test_collect_results_after_wait():
    coord = MultiInstanceCoordinator(max_instances=4)
    coord.spawn_child(np.array([0.5, 0.5, 0.5]))
    time.sleep(1.5)
    results = coord.collect_results(timeout=0.5)
    assert len(results) > 0
    coord.shutdown_all()


def test_complete_message_received():
    coord = MultiInstanceCoordinator(max_instances=4)
    coord.spawn_child(np.array([0.0, 0.0, 0.0]))
    time.sleep(1.5)
    results = coord.collect_results(timeout=0.5)
    completions = [r for r in results if r.get('type') == 'complete']
    assert len(completions) >= 1
    coord.shutdown_all()


def test_should_spawn_false_at_max():
    coord = MultiInstanceCoordinator(max_instances=2)
    coord.spawn_child(np.array([1., 0., 0.]))
    coord.spawn_child(np.array([0., 1., 0.]))
    ign = np.array([0.9, 0.9, 0.9])
    pri = np.ones(3)
    assert coord.should_spawn_child(ign, pri) is False
    coord.shutdown_all()


def test_should_spawn_true_when_space():
    coord = MultiInstanceCoordinator(max_instances=4)
    ign = np.array([0.9, 0.1, 0.1])
    pri = np.ones(3)
    assert coord.should_spawn_child(ign, pri) is True


def test_shutdown_clears_instances():
    coord = MultiInstanceCoordinator(max_instances=4)
    coord.spawn_child(np.array([0., 0., 0.]))
    coord.shutdown_all()
    assert len(coord.instances) == 0
    assert len(coord.chart_centers) == 0


# ── Task 10.2: Isometric Transfer ─────────────────────────────────────────────

def test_transfer_value_preserved():
    tr = IsometricTransfer()
    ca = np.array([1., 0., 0.])
    cb = np.array([0., 1., 0.])
    disc = {'point': [1.1, 0.1, 0.0], 'value': 0.8, 'instance_id': 0}
    out = tr.transfer_discovery(disc, ca, cb)
    assert out['value'] == 0.8
    assert out['transferred'] is True


def test_transfer_coordinates_shifted():
    tr = IsometricTransfer()
    ca = np.array([1., 0., 0.])
    cb = np.array([0., 1., 0.])
    disc = {'point': [1.1, 0.1, 0.0], 'value': 0.8, 'instance_id': 0}
    out = tr.transfer_discovery(disc, ca, cb)
    expected = np.array([1.1, 0.1, 0.0]) + (cb - ca)
    assert np.allclose(out['point'], expected.tolist())


def test_verify_isometry_affine():
    tr = IsometricTransfer()
    ca = np.array([0., 0.])
    cb = np.array([3., 4.])
    pts = [np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.])]
    assert tr.verify_isometry(pts, ca, cb, tolerance=1e-9)


def test_verify_isometry_single_point():
    tr = IsometricTransfer()
    ca = np.array([0., 0.])
    cb = np.array([1., 1.])
    assert tr.verify_isometry([np.array([0., 0.])], ca, cb)


# ── Task 10.3: Global Aggregation ─────────────────────────────────────────────

def _make_discoveries():
    # instance 0 center=[0,0,0]; instance 1 center=[5,0,0]
    # after transfer, instance 1 point [5.5,0,0] → [0.5,0,0] (far from 0.1)
    return [
        {'instance_id': 0, 'type': 'discovery', 'point': [0.1, 0.0, 0.0], 'value': 0.8},
        {'instance_id': 0, 'type': 'discovery', 'point': [0.12, 0.0, 0.0], 'value': 0.75},  # near first
        {'instance_id': 1, 'type': 'discovery', 'point': [5.5, 0.0, 0.0], 'value': 0.9},
    ]


def test_aggregate_deduplicates():
    agg = GlobalManifoldAggregator()
    # instance 1 center far from instance 0 so transfer puts it at 0.5 (distinct)
    chart_centers = {0: np.array([0., 0., 0.]), 1: np.array([5., 0., 0.])}
    unified = agg.aggregate(_make_discoveries(), chart_centers, reference_instance=0)
    # [0.1,0,0] and [0.12,0,0] deduplicated → 1; transferred [0.5,0,0] → 1; total=2
    assert len(unified) == 2


def test_aggregate_top_value():
    agg = GlobalManifoldAggregator()
    chart_centers = {0: np.array([0., 0., 0.]), 1: np.array([5., 0., 0.])}
    agg.aggregate(_make_discoveries(), chart_centers)
    top = agg.get_top_discoveries(n=1)
    assert len(top) == 1
    assert top[0]['value'] == 0.9


def test_aggregate_skips_non_discovery():
    agg = GlobalManifoldAggregator()
    chart_centers = {0: np.array([0., 0., 0.])}
    discs = [{'instance_id': 0, 'type': 'complete', 'samples_evaluated': 100}]
    unified = agg.aggregate(discs, chart_centers)
    assert len(unified) == 0


def test_aggregate_transfers_coords():
    agg = GlobalManifoldAggregator()
    chart_centers = {0: np.array([0., 0.]), 1: np.array([10., 0.])}
    discs = [{'instance_id': 1, 'type': 'discovery', 'point': [10.5, 0.0], 'value': 0.7}]
    unified = agg.aggregate(discs, chart_centers, reference_instance=0)
    assert len(unified) == 1
    # Point should be transferred: [10.5,0] + ([0,0]-[10,0]) = [0.5, 0]
    assert abs(unified[0]['point'][0] - 0.5) < 1e-9


def test_get_top_discoveries_limit():
    agg = GlobalManifoldAggregator()
    agg.global_discoveries = [
        {'point': [i, 0.], 'value': float(i) / 10} for i in range(20)
    ]
    top = agg.get_top_discoveries(n=5)
    assert len(top) == 5
    assert top[0]['value'] == 1.9
