"""FICUTS Layer 1 tests: Lyapunov energy, meta_loss_stable, write-ahead journal.

Tasks:
  1.1 — lyapunov_energy()
  1.2 — meta_loss_stable()
  1.3 — WAL + atomic checkpoint + recovery
"""
import json
import math
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np

# ── helpers ────────────────────────────────────────────────────────────────

def _ctx(code_cons, t=0.0):
    return {
        'timestamp': t,
        'consonance': {'code': code_cons},
        'eigenvalue_gaps': {},
        'growth_nodes': [],
        'golden_resonance_matrix': [],
    }


def _fill(traj, values):
    for i, v in enumerate(values):
        traj.record(_ctx(v, t=float(i)))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1.1 — Lyapunov Energy
# ══════════════════════════════════════════════════════════════════════════════

def test_lyapunov_energy_empty_returns_zero():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    assert traj.lyapunov_energy() == 0.0


def test_lyapunov_energy_finite():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    _fill(traj, [0.5 + 0.01 * i for i in range(20)])
    E = traj.lyapunov_energy()
    assert math.isfinite(E), f"E = {E} is not finite"


def test_lyapunov_energy_stored_in_metadata():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    _fill(traj, [0.6] * 15)
    assert 'lyapunov_energy' in traj.points[-1].metadata
    assert math.isfinite(traj.points[-1].metadata['lyapunov_energy'])


def test_lyapunov_energy_flat_is_constant():
    """Flat trajectory → E is approximately constant (zero drift)."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    _fill(traj, [0.7] * 30)
    energies = [p.metadata['lyapunov_energy'] for p in traj.points[10:]]
    drifts = [abs(energies[i + 1] - energies[i]) / max(abs(energies[i]), 1e-9)
              for i in range(len(energies) - 1)]
    assert all(d < 1e-6 for d in drifts), f"Flat trajectory drifted: max={max(drifts):.6f}"


def test_lyapunov_energy_bounded_drift():
    """Over a smooth sigmoid trajectory, consecutive E values drift < 5%."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    # Sigmoid approaching 0.9, settling smoothly
    values = [0.9 - 0.4 * math.exp(-i / 30) for i in range(100)]
    _fill(traj, values)
    energies = [p.metadata['lyapunov_energy'] for p in traj.points[10:]]
    drifts = [abs(energies[i + 1] - energies[i]) / max(abs(energies[i]), 1e-9)
              for i in range(len(energies) - 1)]
    max_drift = max(drifts)
    assert max_drift < 0.05, f"Bounded drift violated: max drift = {max_drift:.4f}"


def test_lyapunov_energy_decreases_for_damped_trajectory():
    """E decreases when system converges from high energy to lower steady state."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    # Start at 1.0, converge toward 0.3 (damped overshoot scenario)
    values = [0.3 + 0.7 * math.exp(-i / 20) for i in range(100)]
    _fill(traj, values)
    energies = [p.metadata['lyapunov_energy'] for p in traj.points]
    # Energy at end should be lower than at start (after warmup)
    E_start = np.mean(energies[5:15])
    E_end = np.mean(energies[-15:])
    assert E_end < E_start, (
        f"E did not decrease: E_start={E_start:.4f}, E_end={E_end:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1.2 — meta_loss_stable
# ══════════════════════════════════════════════════════════════════════════════

def test_meta_loss_stable_stable_trajectory():
    """Truly stable trajectory (tiny slope) → near-zero penalty, loss ≈ raw."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    # Very slow linear increase: vel > 0, var << 0.01, energy drift << 5%
    _fill(traj, [0.70 + 0.0001 * i for i in range(30)])
    loss = traj.meta_loss_stable()
    raw = traj.meta_loss()
    assert math.isfinite(loss)
    # Penalty should be near zero; loss ≈ raw (both near 0 for linear trajectory)
    assert abs(loss - raw) < 1.0, \
        f"Unexpectedly large penalty: stable_loss={loss:.4f}, raw={raw:.4f}"


def test_meta_loss_stable_penalizes_non_monotonic():
    """Non-monotonic (vel ≤ 0) → +10 penalty; clearly worse than stable."""
    from tensor.trajectory import LearningTrajectory

    # Stable: tiny positive slope, all penalties suppressed
    traj_stable = LearningTrajectory()
    _fill(traj_stable, [0.70 + 0.0001 * i for i in range(30)])
    loss_stable = traj_stable.meta_loss_stable()

    # Non-monotonic: slowly decreasing → vel < 0 → +10 penalty
    traj_bad = LearningTrajectory()
    _fill(traj_bad, [0.70 - 0.0001 * i for i in range(30)])
    loss_bad = traj_bad.meta_loss_stable()

    assert loss_bad > loss_stable, (
        f"Non-monotonic not penalized: stable={loss_stable:.4f}, non-mono={loss_bad:.4f}")
    assert loss_bad - loss_stable > 5.0, (
        f"Penalty gap too small: diff={loss_bad - loss_stable:.4f}")


def test_meta_loss_stable_penalizes_oscillating():
    """Oscillating (var > 0.01) → large positive penalty."""
    from tensor.trajectory import LearningTrajectory

    # Stable
    traj_stable = LearningTrajectory()
    _fill(traj_stable, [0.5 + 0.001 * i * i for i in range(30)])
    loss_stable = traj_stable.meta_loss_stable()

    # Oscillating: variance >> 0.01
    traj_osc = LearningTrajectory()
    _fill(traj_osc, [0.7 + 0.2 * math.sin(i * 0.8) for i in range(30)])
    loss_osc = traj_osc.meta_loss_stable()
    var_osc = np.var([0.7 + 0.2 * math.sin(i * 0.8) for i in range(10, 30)])

    assert var_osc > 0.01, f"Test setup: var={var_osc:.4f} should be > 0.01"
    assert loss_osc > loss_stable, (
        f"Oscillating not penalized: stable={loss_stable:.3f}, osc={loss_osc:.3f}")


def test_meta_loss_stable_returns_finite():
    """meta_loss_stable always returns a finite float."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    _fill(traj, [0.5] * 5)
    assert math.isfinite(traj.meta_loss_stable())


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1.3 — Write-Ahead Journal
# ══════════════════════════════════════════════════════════════════════════════

def test_wal_disabled_works_normally():
    """LearningTrajectory() without journal_path operates normally."""
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    _fill(traj, [0.5 + 0.01 * i for i in range(20)])
    assert len(traj.points) == 20


def test_wal_writes_all_points():
    """WAL file has one line per recorded point."""
    from tensor.trajectory import LearningTrajectory
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'traj.wal')
        traj = LearningTrajectory(journal_path=path)
        n = 50
        _fill(traj, [0.5 + 0.005 * i for i in range(n)])
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == n, f"Expected {n} WAL lines, got {len(lines)}"


def test_wal_lines_are_valid_json():
    """Every WAL line deserializes to a valid point dict."""
    from tensor.trajectory import LearningTrajectory
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'traj.wal')
        traj = LearningTrajectory(journal_path=path)
        _fill(traj, [0.6] * 10)
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                assert 'timestamp' in obj
                assert 'consonance' in obj
                assert 'metadata' in obj
                assert 'lyapunov_energy' in obj['metadata']


def test_wal_recovery_all_points():
    """recover_from_wal() restores all written points."""
    from tensor.trajectory import LearningTrajectory
    with tempfile.TemporaryDirectory() as d:
        wal_path = os.path.join(d, 'traj.wal')
        n = 200
        traj = LearningTrajectory(journal_path=wal_path)
        _fill(traj, [0.5 + 0.001 * i for i in range(n)])

        # Recover into fresh trajectory
        traj2 = LearningTrajectory()
        traj2.recover_from_wal(wal_path)
        assert len(traj2.points) == n, \
            f"Expected {n} recovered points, got {len(traj2.points)}"
        # Spot-check consonance
        assert abs(traj2.points[0].consonance['code'] - 0.5) < 1e-9
        assert abs(traj2.points[-1].consonance['code'] - (0.5 + 0.001 * (n - 1))) < 1e-9


def test_wal_atomic_checkpoint_created():
    """After 100 points, a .json checkpoint file exists and is valid JSON."""
    from tensor.trajectory import LearningTrajectory
    with tempfile.TemporaryDirectory() as d:
        wal_path = os.path.join(d, 'traj.wal')
        traj = LearningTrajectory(journal_path=wal_path)
        _fill(traj, [0.5 + 0.001 * i for i in range(105)])

        ckpt_path = wal_path.replace('.wal', '.json')
        assert os.path.exists(ckpt_path), "Checkpoint .json not created after 100 points"

        with open(ckpt_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 100


def test_wal_recovery_stops_at_corrupt_line():
    """recover_from_wal() returns points up to (not including) corrupt line."""
    from tensor.trajectory import LearningTrajectory
    with tempfile.TemporaryDirectory() as d:
        wal_path = os.path.join(d, 'crash.wal')
        # Write 5 valid lines then a corrupt one
        with open(wal_path, 'w') as f:
            for i in range(5):
                obj = {
                    'timestamp': float(i),
                    'consonance': {'code': 0.5},
                    'eigenvalue_gaps': {},
                    'growth_regime_count': 0,
                    'golden_resonance_matrix': [],
                    'metadata': {},
                }
                f.write(json.dumps(obj) + '\n')
            f.write('{"broken": true, "incomplete...\n')  # corrupt line

        traj = LearningTrajectory()
        traj.recover_from_wal(wal_path)
        assert len(traj.points) == 5, \
            f"Expected 5 points before corrupt line, got {len(traj.points)}"


def test_wal_1000_points_no_data_loss():
    """Write 1000 points; full recovery from WAL, all 500 recent in window."""
    from tensor.trajectory import LearningTrajectory
    with tempfile.TemporaryDirectory() as d:
        wal_path = os.path.join(d, 'big.wal')
        traj = LearningTrajectory(window=500, journal_path=wal_path)
        _fill(traj, [0.5 + 0.0005 * i for i in range(1000)])

        # self.points trimmed to window
        assert len(traj.points) == 500

        # WAL has all 1000 lines
        with open(wal_path) as f:
            n_lines = sum(1 for l in f if l.strip())
        assert n_lines == 1000, f"Expected 1000 WAL lines, got {n_lines}"

        # Checkpoint exists (triggered at 100, 200, ..., 500 points in window)
        ckpt = wal_path.replace('.wal', '.json')
        assert os.path.exists(ckpt)

        # Recovery gives window-sized slice
        traj2 = LearningTrajectory(window=500)
        traj2.recover_from_wal(wal_path)
        assert len(traj2.points) == 500
