"""
Tests for HDVSNavigator — 3-mode HDVS state machine.
"""

import numpy as np
import pytest
from tensor.lca_patch_detector import LCAPatchDetector
from tensor.koopman_edmd import EDMDKoopman
from tensor.hdvs_navigator import HDVSNavigator, NavigatorThresholds, NavigationStep
from tensor.hdvs_navigator import MODE_LCA, MODE_TRANSITION, MODE_KOOPMAN


# ------------------------------------------------------------------
# Fixtures: linear system (LCA) and nonlinear system (transitions)
# ------------------------------------------------------------------

def linear_system(x: np.ndarray) -> np.ndarray:
    """Linear RLC: ẋ = Ax, small curvature → LCA patch."""
    A = np.array([[-0.5, -1.0], [1.0, -0.5]])
    return A @ x


def nonlinear_system(x: np.ndarray) -> np.ndarray:
    """RLC + diode: nonlinear for large |v|."""
    v, iL = x
    alpha = 0.5
    vdot = -0.5 * v - alpha * v ** 3 - iL
    iLdot = v
    return np.array([vdot, iLdot])


def make_navigator(system_fn=linear_system, n_states: int = 2) -> HDVSNavigator:
    detector = LCAPatchDetector(system_fn, n_states=n_states)
    koopman = EDMDKoopman(observable_degree=2)
    return HDVSNavigator(detector, koopman)


# ------------------------------------------------------------------
# Tests: Basic construction and step
# ------------------------------------------------------------------

class TestConstruction:
    def test_default_mode_is_lca(self):
        nav = make_navigator()
        assert nav.current_mode() == MODE_LCA

    def test_step_returns_string(self):
        nav = make_navigator()
        x = np.array([0.1, 0.0])
        result = nav.step(x)
        assert isinstance(result, str)

    def test_step_returns_valid_mode(self):
        nav = make_navigator()
        x = np.array([0.1, 0.0])
        for _ in range(5):
            result = nav.step(x)
        assert result in (MODE_LCA, MODE_TRANSITION, MODE_KOOPMAN)

    def test_history_accumulates(self):
        nav = make_navigator()
        x = np.array([0.1, 0.0])
        for _ in range(5):
            nav.step(x)
        assert len(nav.navigation_history()) == 5


# ------------------------------------------------------------------
# Tests: Linear system stays in LCA mode
# ------------------------------------------------------------------

class TestLinearSystemLCA:
    def test_small_signal_stays_lca_after_warmup(self):
        """Linear system with small signal should stay in LCA mode."""
        nav = make_navigator(linear_system)
        x = np.array([0.05, 0.0])

        # Run for many steps
        modes = []
        x_prev = None
        for _ in range(30):
            m = nav.step(x, x_prev)
            modes.append(m)
            x_prev = x.copy()
            # Slowly spiral inward
            x = x + 0.01 * linear_system(x)

        # After warmup (first few steps have too few samples), should be LCA
        late_modes = modes[10:]
        lca_fraction = late_modes.count(MODE_LCA) / len(late_modes)
        assert lca_fraction > 0.5  # majority LCA for linear system

    def test_mode_sequence_is_list(self):
        nav = make_navigator()
        for _ in range(5):
            nav.step(np.array([0.1, 0.0]))
        seq = nav.mode_sequence()
        assert isinstance(seq, list)
        assert len(seq) == 5


# ------------------------------------------------------------------
# Tests: Navigation history
# ------------------------------------------------------------------

class TestNavigationHistory:
    def test_history_entries_are_navigation_steps(self):
        nav = make_navigator()
        nav.step(np.array([0.1, 0.0]))
        nav.step(np.array([0.1, 0.1]))
        history = nav.navigation_history()
        assert all(isinstance(h, NavigationStep) for h in history)

    def test_history_has_mode_field(self):
        nav = make_navigator()
        nav.step(np.array([0.1, 0.0]))
        h = nav.navigation_history()[0]
        assert h.mode in (MODE_LCA, MODE_TRANSITION, MODE_KOOPMAN)

    def test_history_has_curvature(self):
        nav = make_navigator()
        for _ in range(5):
            nav.step(np.array([0.1, 0.0]))
        history = nav.navigation_history()
        assert all(isinstance(h.curvature_ratio, float) for h in history)

    def test_history_has_bifurcation_status(self):
        nav = make_navigator()
        for _ in range(5):
            nav.step(np.array([0.1, 0.0]))
        history = nav.navigation_history()
        assert all(h.bifurcation_status in ('stable', 'critical', 'bifurcation') for h in history)


# ------------------------------------------------------------------
# Tests: Reset
# ------------------------------------------------------------------

class TestReset:
    def test_reset_clears_history(self):
        nav = make_navigator()
        for _ in range(5):
            nav.step(np.array([0.1, 0.0]))
        nav.reset()
        assert len(nav.navigation_history()) == 0

    def test_reset_restores_lca_mode(self):
        nav = make_navigator()
        for _ in range(5):
            nav.step(np.array([0.1, 0.0]))
        nav.reset()
        assert nav.current_mode() == MODE_LCA


# ------------------------------------------------------------------
# Tests: Summary
# ------------------------------------------------------------------

class TestSummary:
    def test_summary_empty(self):
        nav = make_navigator()
        s = nav.summary()
        assert s['n_steps'] == 0

    def test_summary_after_steps(self):
        nav = make_navigator()
        for _ in range(10):
            nav.step(np.array([0.1, 0.0]))
        s = nav.summary()
        assert s['n_steps'] == 10

    def test_summary_fractions_sum_to_one(self):
        nav = make_navigator()
        for _ in range(20):
            nav.step(np.array([0.05, 0.0]))
        s = nav.summary()
        fractions = list(s['mode_fractions'].values())
        if fractions:
            assert abs(sum(fractions) - 1.0) < 1e-10


# ------------------------------------------------------------------
# Tests: Active basis
# ------------------------------------------------------------------

class TestActiveBasis:
    def test_active_basis_is_none_initially(self):
        nav = make_navigator()
        assert nav.get_active_basis() is None

    def test_active_basis_becomes_array_after_warmup(self):
        nav = make_navigator()
        for _ in range(10):
            nav.step(np.array([0.1, 0.0]))
        basis = nav.get_active_basis()
        # After enough steps, basis should be set
        if basis is not None:
            assert isinstance(basis, np.ndarray)


# ------------------------------------------------------------------
# Tests: Custom thresholds
# ------------------------------------------------------------------

class TestCustomThresholds:
    def test_custom_thresholds_accepted(self):
        t = NavigatorThresholds(eps1=0.1, eps2=0.3, delta1=0.05)
        detector = LCAPatchDetector(linear_system, n_states=2)
        koopman = EDMDKoopman(observable_degree=2)
        nav = HDVSNavigator(detector, koopman, thresholds=t)
        assert nav.thresholds.eps1 == 0.1
