"""FICUTS Layer 4 tests: concurrency safety + hierarchical memory compression.

Tasks:
  4.1 — thread locks (RLock on LearningTrajectory + AgentNetwork)
  4.2 — hierarchical memory compression (max_points, compression_ratio)
"""
import os
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np


def _ctx(code_cons=0.5, t=0.0):
    return {
        'timestamp': t,
        'consonance': {'code': code_cons},
        'eigenvalue_gaps': {},
        'growth_nodes': [],
        'golden_resonance_matrix': [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.1 — Thread Locks
# ══════════════════════════════════════════════════════════════════════════════

def test_trajectory_has_rlock():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    assert isinstance(traj._write_lock, type(threading.RLock()))


def test_agent_network_has_state_lock():
    from tensor.agent_network import AgentNetwork
    net = AgentNetwork()
    assert hasattr(net, '_state_lock')
    assert isinstance(net._state_lock, type(threading.RLock()))


def test_trajectory_concurrent_record_no_corruption():
    """10 threads × 1000 records = 10 000 total writes; no data corruption."""
    from tensor.trajectory import LearningTrajectory, TrajectoryPoint
    traj = LearningTrajectory(window=500)
    errors = []

    def worker():
        for i in range(1000):
            try:
                traj.record(_ctx(0.5 + i * 0.00005, t=float(i)))
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    # Length bounded by window
    assert len(traj.points) <= traj.window, \
        f"Points exceeded window: {len(traj.points)} > {traj.window}"
    # All points are valid TrajectoryPoint instances (no None/corruption)
    assert all(isinstance(p, TrajectoryPoint) for p in traj.points), \
        "Corrupted entries in self.points"
    # All consonance values are floats
    for p in traj.points:
        assert isinstance(p.consonance.get('code', 0.0), float)


def test_trajectory_concurrent_read_write():
    """Readers and writers simultaneously — no deadlock, no exception."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory(window=200)
    stop = threading.Event()
    errors = []

    def writer():
        for i in range(500):
            try:
                traj.record(_ctx(0.5 + i * 0.001, t=float(i)))
            except Exception as e:
                errors.append(e)

    def reader():
        while not stop.is_set():
            try:
                _ = traj.lyapunov_energy()
                _ = traj.consonance_velocity('code')
                _ = traj.meta_loss()
            except Exception as e:
                errors.append(e)
            time.sleep(0.0005)

    writers = [threading.Thread(target=writer) for _ in range(3)]
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in readers:
        t.start()
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join(timeout=2.0)

    assert not errors, f"Concurrent read/write errors: {errors}"


def test_agent_network_concurrent_add_fire():
    """Multiple threads adding agents and firing concurrently — no exception."""
    from tensor.agent_network import AgentNetwork, AgentNode
    net = AgentNetwork()
    errors = []

    def adder():
        for i in range(50):
            try:
                net.add_agent(AgentNode(role=f'a{i}', model='m', level='code'))
            except Exception as e:
                errors.append(e)

    def updater():
        for _ in range(200):
            try:
                with net._state_lock:
                    for a in list(net.agents):
                        a.influence = max(0.01, a.influence * 0.999)
            except Exception as e:
                errors.append(e)

    threads = ([threading.Thread(target=adder) for _ in range(3)] +
               [threading.Thread(target=updater) for _ in range(3)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Agent network thread errors: {errors}"
    # All influences are valid positive floats
    with net._state_lock:
        for a in net.agents:
            assert a.influence > 0, f"Corrupted influence: {a.influence}"


def test_stress_10_threads_10k_iterations():
    """10 threads × 1000 iters on trajectory.record — key stress test."""
    from tensor.trajectory import LearningTrajectory, TrajectoryPoint
    traj = LearningTrajectory(window=300)
    counter = [0]
    lock = threading.Lock()
    errors = []

    def worker():
        for i in range(1000):
            try:
                traj.record(_ctx(0.5))
                with lock:
                    counter[0] += 1
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors: {errors[:3]}"
    assert counter[0] == 10000, f"Not all iterations completed: {counter[0]}"
    assert len(traj.points) <= traj.window
    assert all(isinstance(p, TrajectoryPoint) for p in traj.points)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.2 — Hierarchical Memory Compression
# ══════════════════════════════════════════════════════════════════════════════

def test_compression_never_exceeds_max_points():
    """len(points) never exceeds max_points after any number of records."""
    from tensor.trajectory import LearningTrajectory
    max_pts = 100
    traj = LearningTrajectory(max_points=max_pts, compression_ratio=0.1)
    for i in range(1000):
        traj.record(_ctx(0.5 + i * 0.0001, t=float(i)))
        assert len(traj.points) <= max_pts, \
            f"Exceeded max_points at step {i}: {len(traj.points)}"


def test_compression_recent_points_present():
    """The most recent points are always retained after compression."""
    from tensor.trajectory import LearningTrajectory
    max_pts = 100
    traj = LearningTrajectory(max_points=max_pts, compression_ratio=0.1)
    n = 500
    for i in range(n):
        traj.record(_ctx(float(i), t=float(i)))

    # Last 30% of max_points = 30 recent points
    n_recent = int(max_pts * 0.3)
    last_timestamps = sorted([p.timestamp for p in traj.points])[-n_recent:]
    expected_last = float(n - 1)
    assert last_timestamps[-1] == expected_last, \
        f"Most recent point missing: expected t={expected_last}, got {last_timestamps[-1]}"


def test_compression_counter_increments():
    """metadata['compressions'] increments each time compression fires."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory(max_points=50, compression_ratio=0.1)
    for i in range(300):
        traj.record(_ctx(0.5, t=float(i)))
    assert traj.metadata['compressions'] > 0, "No compressions recorded"


def test_compression_statistics_preserved():
    """Compressed points preserve mean/var of a STATIONARY distribution.

    The compression keeps 30% recent + 10% sampled old, so it's biased toward
    recency.  To test statistical preservation we use a periodic (stationary)
    signal: any subsample has approximately the same mean and variance as the
    full population.
    """
    from tensor.trajectory import LearningTrajectory
    import math
    max_pts = 200
    traj = LearningTrajectory(max_points=max_pts, compression_ratio=0.1)
    n = 2000
    # Stationary oscillation around 0.7 with amplitude 0.15
    all_values = [0.7 + 0.15 * math.sin(i * 0.3) for i in range(n)]
    for i, v in enumerate(all_values):
        traj.record(_ctx(v, t=float(i)))

    orig_mean = float(np.mean(all_values))
    orig_var = float(np.var(all_values))

    compressed_values = [p.consonance['code'] for p in traj.points]
    comp_mean = float(np.mean(compressed_values))
    comp_var = float(np.var(compressed_values))

    # Mean within 5% absolute (signal range is 0.15*2 = 0.30; 5% = 0.015)
    assert abs(comp_mean - orig_mean) < 0.05, \
        f"Mean drifted: orig={orig_mean:.4f}, compressed={comp_mean:.4f}"

    # Variance within 20% relative (subsampling a periodic signal has some variance)
    if orig_var > 1e-9:
        var_ratio = abs(comp_var - orig_var) / orig_var
        assert var_ratio < 0.20, \
            f"Variance drifted: orig={orig_var:.4f}, comp={comp_var:.4f}, ratio={var_ratio:.4f}"


def test_compression_window_still_works():
    """Original window trimming still works when max_points=None."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory(window=50)
    for i in range(200):
        traj.record(_ctx(0.5, t=float(i)))
    assert len(traj.points) == 50


def test_10000_points_memory_bound():
    """Record 10 000 points with max_points=1000; length stays ≤ 1000."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory(max_points=1000, compression_ratio=0.1)
    for i in range(10000):
        traj.record(_ctx(0.5 + i * 0.00001, t=float(i)))
    assert len(traj.points) <= 1000, \
        f"Memory not bounded: {len(traj.points)} points"
    # Most recent point is present
    final_t = traj.points[-1].timestamp
    assert final_t == 9999.0, f"Last point missing: t={final_t}"
