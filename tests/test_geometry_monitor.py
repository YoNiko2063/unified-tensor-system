"""Tests for tensor/geometry_monitor.py — GeometryMonitor."""

import time
import numpy as np
import pytest

from tensor.geometry_monitor import GeometryMonitor
from tensor.lca_patch_detector import PatchClassification
from tensor.koopman_edmd import EDMDKoopman


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classification(curvature: float, trust: float) -> PatchClassification:
    n = 2
    basis = np.eye(n)[np.newaxis, :, :]
    return PatchClassification(
        patch_type='lca',
        operator_rank=1,
        commutator_norm=0.0,
        curvature_ratio=curvature,
        spectral_gap=0.1,
        basis_matrices=basis,
        eigenvalues=np.array([-0.5 + 0.0j, -0.5 + 0.0j]),
        centroid=np.zeros(n),
        koopman_trust=trust,
    )


def _edmd(degree: int = 2) -> EDMDKoopman:
    """Unfitted EDMDKoopman — only .degree is needed for monitoring."""
    return EDMDKoopman(observable_degree=degree)


def _record_n(monitor: GeometryMonitor, n: int,
               curvature: float = 0.1, trust: float = 0.8,
               degree: int = 2, patch_count: int = 5,
               n_equiv: int = 0) -> None:
    """Helper: record the same observation n times."""
    for _ in range(n):
        monitor.record(
            _classification(curvature, trust),
            _edmd(degree),
            patch_count,
            n_equiv,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        m = GeometryMonitor()
        assert m._window == 100
        assert m._max_degree_jump == 1
        assert m._min_trust == 0.2
        assert m._max_growth_rate == 2.0
        assert m._spike_sigma == 3.0

    def test_custom(self):
        m = GeometryMonitor(window_size=50, min_trust=0.3)
        assert m._window == 50
        assert m._min_trust == 0.3


# ---------------------------------------------------------------------------
# is_unstable — early (< 3 observations)
# ---------------------------------------------------------------------------

class TestEarlyStability:
    def test_no_observations_stable(self):
        m = GeometryMonitor()
        assert m.is_unstable() is False

    def test_one_observation_stable(self):
        m = GeometryMonitor()
        m.record(_classification(0.5, 0.5), _edmd(), 5, 0)
        assert m.is_unstable() is False

    def test_two_observations_stable(self):
        m = GeometryMonitor()
        _record_n(m, 2)
        assert m.is_unstable() is False


# ---------------------------------------------------------------------------
# Condition 1: Curvature monotone increasing
# ---------------------------------------------------------------------------

class TestCurvatureMonotone:
    def test_monotone_increasing_triggers(self):
        m = GeometryMonitor(window_size=20)
        # First half: low curvature ~0.1; second half: high curvature ~0.5
        for i in range(20):
            c = 0.05 * (i + 1)   # 0.05, 0.10, ..., 1.00 — strictly increasing
            m.record(_classification(c, 0.8), _edmd(), 5, 0)
        assert m.is_unstable() is True
        assert 'curvature_monotone_increasing' in m._fired_conditions()

    def test_stable_curvature_no_trigger(self):
        m = GeometryMonitor(window_size=20)
        _record_n(m, 20, curvature=0.2)
        assert 'curvature_monotone_increasing' not in m._fired_conditions()

    def test_decreasing_curvature_no_trigger(self):
        m = GeometryMonitor(window_size=20)
        for i in range(20):
            c = 1.0 - 0.04 * i   # decreasing
            m.record(_classification(c, 0.8), _edmd(), 5, 0)
        assert 'curvature_monotone_increasing' not in m._fired_conditions()


# ---------------------------------------------------------------------------
# Condition 2: Mean trust too low
# ---------------------------------------------------------------------------

class TestMeanTrustTooLow:
    def test_low_trust_triggers(self):
        m = GeometryMonitor(min_trust=0.3)
        _record_n(m, 10, trust=0.1)
        assert m.is_unstable() is True
        assert 'mean_trust_too_low' in m._fired_conditions()

    def test_high_trust_no_trigger(self):
        m = GeometryMonitor(min_trust=0.3)
        _record_n(m, 10, trust=0.9)
        assert 'mean_trust_too_low' not in m._fired_conditions()

    def test_boundary_trust(self):
        m = GeometryMonitor(min_trust=0.3)
        _record_n(m, 10, trust=0.35)  # clearly above threshold
        assert 'mean_trust_too_low' not in m._fired_conditions()


# ---------------------------------------------------------------------------
# Condition 3: Degree jump > max_degree_jump
# ---------------------------------------------------------------------------

class TestDegreeJump:
    def test_large_degree_jump_triggers(self):
        m = GeometryMonitor(max_degree_jump=1)
        # degrees: 1, 1, ..., 1, 3 → jump = 2 > 1
        _record_n(m, 5, degree=1)
        m.record(_classification(0.1, 0.8), _edmd(3), 5, 0)
        _record_n(m, 4, degree=3)
        assert m.is_unstable() is True
        assert any('degree_jump' in c for c in m._fired_conditions())

    def test_small_degree_jump_ok(self):
        m = GeometryMonitor(max_degree_jump=1)
        # degrees: 2, ..., 2, 3 → jump = 1 ≤ 1 → no trigger
        _record_n(m, 9, degree=2)
        m.record(_classification(0.1, 0.8), _edmd(3), 5, 0)
        assert not any('degree_jump' in c for c in m._fired_conditions())


# ---------------------------------------------------------------------------
# Condition 4: Patch count growth excessive
# ---------------------------------------------------------------------------

class TestPatchGrowth:
    def test_excessive_growth_triggers(self):
        m = GeometryMonitor(max_patch_growth_rate=2.0)
        # Start at 10 patches, grow to 25 (> 2×10 = 20)
        for i in range(10):
            m.record(_classification(0.1, 0.8), _edmd(), 10 + i, 0)
        m.record(_classification(0.1, 0.8), _edmd(), 25, 0)
        assert m.is_unstable() is True
        assert 'patch_growth_excessive' in m._fired_conditions()

    def test_moderate_growth_ok(self):
        m = GeometryMonitor(max_patch_growth_rate=2.0)
        for i in range(10):
            m.record(_classification(0.1, 0.8), _edmd(), 10 + i, 0)
        # Total = 19 ≤ 2×10 = 20 → no trigger
        m.record(_classification(0.1, 0.8), _edmd(), 19, 0)
        assert 'patch_growth_excessive' not in m._fired_conditions()


# ---------------------------------------------------------------------------
# Condition 5: Equivalence spike
# ---------------------------------------------------------------------------

class TestEquivalenceSpike:
    def test_spike_triggers(self):
        m = GeometryMonitor(equiv_spike_sigma=2.0, window_size=20)
        # Fill with baseline with slight variation (std > 0 needed)
        for i in range(19):
            m.record(_classification(0.1, 0.8), _edmd(), 5, 1 + (i % 3))
        # Sudden spike: 30 equivalences (far above mean ~2)
        m.record(_classification(0.1, 0.8), _edmd(), 5, 30)
        assert 'equivalence_spike' in m._fired_conditions()

    def test_no_spike_stable(self):
        m = GeometryMonitor(equiv_spike_sigma=3.0, window_size=20)
        _record_n(m, 20, n_equiv=1)
        assert 'equivalence_spike' not in m._fired_conditions()

    def test_no_variation_no_spike(self):
        m = GeometryMonitor(equiv_spike_sigma=3.0, window_size=20)
        # All same value: std=0, no spike can be computed
        _record_n(m, 19, n_equiv=5)
        m.record(_classification(0.1, 0.8), _edmd(), 5, 5)
        assert 'equivalence_spike' not in m._fired_conditions()


# ---------------------------------------------------------------------------
# snapshot_state and rollback
# ---------------------------------------------------------------------------

class TestSnapshotAndRollback:
    def test_rollback_no_snapshot_returns_default(self):
        m = GeometryMonitor()
        snap = m.rollback()
        assert snap['degree'] == 1
        assert snap['patch_summary'] == {}

    def test_rollback_returns_saved_snapshot(self):
        m = GeometryMonitor()
        m.snapshot_state(basis_degree=3, patch_summary={'n_patches': 10, 'n_edges': 5})
        snap = m.rollback()
        assert snap['degree'] == 3
        assert snap['patch_summary']['n_patches'] == 10

    def test_rollback_logs_event(self):
        m = GeometryMonitor()
        _record_n(m, 5)
        m.snapshot_state(2, {})
        m.rollback()
        assert len(m._rollback_log) == 1
        assert 'timestamp' in m._rollback_log[0]

    def test_rollback_updates_last_rollback_time(self):
        m = GeometryMonitor()
        _record_n(m, 5)
        m.snapshot_state(2, {})
        before = time.time()
        m.rollback()
        after = time.time()
        assert m._last_rollback_time is not None
        assert before <= m._last_rollback_time <= after

    def test_snapshot_is_copied_not_referenced(self):
        m = GeometryMonitor()
        summary = {'n_patches': 5}
        m.snapshot_state(basis_degree=2, patch_summary=summary)
        summary['n_patches'] = 999   # modify original
        snap = m.rollback()
        assert snap['patch_summary']['n_patches'] == 5  # snapshot unchanged


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self):
        m = GeometryMonitor()
        s = m.summary()
        assert s['n_observations'] == 0
        assert s['is_unstable'] is False
        assert s['has_stable_snapshot'] is False

    def test_summary_after_records(self):
        m = GeometryMonitor()
        _record_n(m, 5, curvature=0.3, trust=0.7, degree=2, patch_count=8)
        m.snapshot_state(2, {'n_patches': 8})
        s = m.summary()
        assert s['n_observations'] == 5
        assert s['mean_curvature'] == pytest.approx(0.3)
        assert s['mean_trust'] == pytest.approx(0.7)
        assert s['current_degree'] == 2
        assert s['current_patch_count'] == 8
        assert s['has_stable_snapshot'] is True
        assert s['n_rollbacks'] == 0

    def test_summary_counts_rollbacks(self):
        m = GeometryMonitor()
        _record_n(m, 5)
        m.snapshot_state(2, {})
        m.rollback()
        m.rollback()
        s = m.summary()
        assert s['n_rollbacks'] == 2


# ---------------------------------------------------------------------------
# Rolling window cap
# ---------------------------------------------------------------------------

class TestWindowCap:
    def test_window_caps_history(self):
        m = GeometryMonitor(window_size=5)
        _record_n(m, 20)
        assert len(m._curvatures) == 5
        assert len(m._trusts) == 5
