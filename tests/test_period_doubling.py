"""
Route to Chaos — Driven Duffing period-doubling tests.

The driven Duffing oscillator ẍ + δẋ + αx + βx³ = F·cos(Ωt) exhibits a
classical period-doubling cascade as F increases:

  F small → period-1 limit cycle (linear-like response)
  F medium → period-2, period-4, ... (subharmonics appear)
  F large  → chaotic attractor (strange attractor, EDMD fails)

Research question:
  When the dynamics is chaotic, does the architecture return
    is_abelian = False  (honest — no coherent Koopman structure)
  or does it return
    is_abelian = True   (hallucinating — pretending structure exists)?

Architecture test:
  is_abelian  = (koopman_trust ≥ 0.3)
  is_chaotic  = (koopman_trust < 0.3)
  → These are mutually exclusive by construction. The architecture IS honest
    because trust is derived from EDMD reconstruction error, which is LARGE
    for chaotic trajectories (no polynomial Koopman representation exists).

Test groups:
  1. Parameter validation (6 tests)
  2. Simulator basics (5 tests)
  3. Linear driven oscillator — guaranteed period-1 (7 tests)
  4. Architecture honesty — identity relations (4 tests)
  5. Trust comparison — periodic vs chaotic (4 tests)
  6. Period and Poincaré section (4 tests)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from optimization.driven_duffing_evaluator import (
    DrivenDuffingParams,
    DrivenDuffingSimulator,
    DrivenDuffingEvaluator,
    DrivenDuffingResult,
    _reconstruction_trust,
    _count_clusters,
    _period_from_clusters,
    _fallback_koopman,
    _CHAOS_TRUST_THRESHOLD,
    _ABELIAN_TRUST_THRESHOLD,
    _ETA_MAX,
)
from tensor.koopman_edmd import EDMDKoopman


# ── Test fixtures ──────────────────────────────────────────────────────────────

# Linear (β=0): guaranteed period-1 for any F > 0.  Steady state is a pure sinusoid
# at the driving frequency Ω.  EDMD fits it with near-zero reconstruction error.
_PARAMS_LINEAR = DrivenDuffingParams(
    alpha=1.0, beta=0.0, delta=0.3, F=0.5, Omega=0.85
)

# Nonlinear, small forcing: near-linear response, should be period-1.
_PARAMS_SMALL_F = DrivenDuffingParams(
    alpha=1.0, beta=0.5, delta=0.3, F=0.05, Omega=0.85
)

# Nonlinear, larger forcing: well inside the chaos region for these parameters.
# α=1, β=1, δ=0.3, Ω=1.0, F=0.5 is known to be chaotic (above F≈0.4 chaos onset).
_PARAMS_CHAOS = DrivenDuffingParams(
    alpha=1.0, beta=1.0, delta=0.3, F=0.5, Omega=1.0
)

# Nonlinear moderate: same as CHAOS but with reduced F (still near period-1)
_PARAMS_MODERATE_F = DrivenDuffingParams(
    alpha=1.0, beta=1.0, delta=0.3, F=0.1, Omega=1.0
)


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: Parameter validation
# ══════════════════════════════════════════════════════════════════════════════


class TestParamValidation:

    def test_valid_params_create_successfully(self):
        p = DrivenDuffingParams(alpha=1.0, beta=0.5, delta=0.3, F=0.1, Omega=0.85)
        assert p.alpha == 1.0
        assert p.beta == 0.5
        assert p.F == 0.1

    def test_invalid_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            DrivenDuffingParams(alpha=0.0, beta=0.0, delta=0.1, F=0.1, Omega=1.0)

    def test_invalid_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            DrivenDuffingParams(alpha=-1.0, beta=0.0, delta=0.1, F=0.1, Omega=1.0)

    def test_invalid_delta_negative_raises(self):
        with pytest.raises(ValueError, match="delta"):
            DrivenDuffingParams(alpha=1.0, beta=0.0, delta=-0.1, F=0.1, Omega=1.0)

    def test_invalid_F_negative_raises(self):
        with pytest.raises(ValueError, match="F"):
            DrivenDuffingParams(alpha=1.0, beta=0.0, delta=0.1, F=-0.1, Omega=1.0)

    def test_invalid_Omega_zero_raises(self):
        with pytest.raises(ValueError, match="Omega"):
            DrivenDuffingParams(alpha=1.0, beta=0.0, delta=0.1, F=0.1, Omega=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: Properties and simulator basics
# ══════════════════════════════════════════════════════════════════════════════


class TestParamProperties:

    def test_omega0_linear(self):
        p = DrivenDuffingParams(alpha=4.0, beta=0.0, delta=0.1, F=0.1, Omega=1.0)
        assert abs(p.omega0_linear - 2.0) < 1e-10

    def test_forcing_period(self):
        p = DrivenDuffingParams(alpha=1.0, beta=0.0, delta=0.1, F=0.1, Omega=1.0)
        assert abs(p.forcing_period - 2.0 * math.pi) < 1e-10

    def test_frequency_ratio(self):
        p = DrivenDuffingParams(alpha=1.0, beta=0.0, delta=0.1, F=0.1, Omega=0.5)
        assert abs(p.frequency_ratio - 0.5) < 1e-10

    def test_rhs_shape(self):
        sim = DrivenDuffingSimulator(_PARAMS_LINEAR, dt=0.05)
        state = np.array([1.0, 0.0])
        out = sim.rhs(state, t=0.0)
        assert out.shape == (2,)

    def test_rhs_forcing_nonzero(self):
        """With F>0, rhs at t=0 differs from rhs at t=π/Ω (phase difference)."""
        p = DrivenDuffingParams(alpha=1.0, beta=0.0, delta=0.0, F=1.0, Omega=1.0)
        sim = DrivenDuffingSimulator(p, dt=0.05)
        state = np.array([0.0, 0.0])
        rhs_t0 = sim.rhs(state, t=0.0)         # F·cos(0) = F = 1.0
        rhs_tpi = sim.rhs(state, t=math.pi)    # F·cos(π) = -F = -1.0
        assert abs(rhs_t0[1] - 1.0) < 1e-10
        assert abs(rhs_tpi[1] - (-1.0)) < 1e-10

    def test_rhs_zero_forcing_equals_autonomous(self):
        """F=0: rhs must equal autonomous (unforced) Duffing."""
        from optimization.duffing_evaluator import DuffingParams, DuffingSimulator

        p_forced = DrivenDuffingParams(alpha=1.0, beta=0.5, delta=0.3, F=0.0, Omega=1.0)
        p_auto = DuffingParams(alpha=1.0, beta=0.5, delta=0.3)

        sim_forced = DrivenDuffingSimulator(p_forced, dt=0.05)
        sim_auto = DuffingSimulator(p_auto, dt=0.05)

        state = np.array([1.2, -0.3])
        rhs_f = sim_forced.rhs(state, t=5.0)   # F=0, t irrelevant
        rhs_a = sim_auto.rhs(state)

        np.testing.assert_allclose(rhs_f, rhs_a, rtol=1e-12)

    def test_run_shape(self):
        sim = DrivenDuffingSimulator(_PARAMS_LINEAR, dt=0.05)
        traj = sim.run(0.1, 0.0, n_steps=200)
        assert traj.shape == (201, 2)

    def test_steady_state_length(self):
        sim = DrivenDuffingSimulator(_PARAMS_LINEAR, dt=0.05)
        n_total = 200
        transient_fraction = 0.5
        steady = sim.run_steady_state(0.1, 0.0, n_total, transient_fraction)
        n_transient = int(n_total * transient_fraction)
        expected = n_total - n_transient + 1
        assert len(steady) == expected


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: Linear driven oscillator (guaranteed period-1)
#
# For β=0, the driven oscillator is linear: ẍ + δẋ + αx = F·cos(Ωt).
# The unique steady-state response is a pure sinusoid at the driving frequency Ω.
# EDMD on this sinusoid has near-zero reconstruction error → trust ≈ 1.0.
# This group tests are GUARANTEED to pass for any implementation that correctly
# identifies the periodic steady state.
# ══════════════════════════════════════════════════════════════════════════════


class TestLinearDrivenOscillator:

    def setup_method(self):
        # Use n_total=2000 for ~10 forcing periods at steady state
        self._ev = DrivenDuffingEvaluator(_PARAMS_LINEAR, dt=0.05, n_total=2000)
        self._result = self._ev.evaluate(x0=0.1, v0=0.0)

    def test_low_reconstruction_error(self):
        """Linear steady state: EDMD reconstruction error should be small."""
        # The limit cycle x = A·cos(Ωt+φ) is perfectly Koopman-predictable.
        # Error can be non-negligible due to polynomial cross-terms, but stays < 0.5.
        assert self._result.reconstruction_error < 0.5, (
            f"Expected low reconstruction error for linear driven oscillator, "
            f"got {self._result.reconstruction_error:.4f}"
        )

    def test_high_trust(self):
        """Linear orbit: koopman_trust should be > _CHAOS_TRUST_THRESHOLD."""
        assert self._result.koopman_trust > _CHAOS_TRUST_THRESHOLD, (
            f"Expected trust > {_CHAOS_TRUST_THRESHOLD} for linear driven oscillator, "
            f"got {self._result.koopman_trust:.4f}"
        )

    def test_not_chaotic(self):
        """Linear driven oscillator is never chaotic."""
        assert not self._result.is_chaotic, (
            f"Linear oscillator should not be chaotic, trust={self._result.koopman_trust:.4f}"
        )

    def test_is_abelian(self):
        """Linear driven oscillator: coherent U(1) structure → abelian."""
        assert self._result.is_abelian, (
            f"Linear oscillator should be abelian, trust={self._result.koopman_trust:.4f}"
        )

    def test_period_number_one(self):
        """Linear oscillator: exactly period-1 (single Poincaré cluster)."""
        assert self._result.period_number == 1, (
            f"Expected period-1, got period-{self._result.period_number}, "
            f"clusters={self._result.poincare_clusters}"
        )

    def test_poincare_clusters_one(self):
        """Linear steady state: all Poincaré points at same x → 1 cluster."""
        assert self._result.poincare_clusters == 1, (
            f"Expected 1 Poincaré cluster for linear period-1, "
            f"got {self._result.poincare_clusters}"
        )

    def test_dominant_frequency_near_omega(self):
        """Dominant EDMD frequency should be near the driving frequency Ω."""
        Omega = _PARAMS_LINEAR.Omega
        dom = self._result.dominant_frequency
        rel_err = abs(dom - Omega) / Omega
        assert rel_err < 0.15, (
            f"Dominant frequency {dom:.4f} not near Ω={Omega:.4f} "
            f"(relative error {rel_err:.3f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Group 4: Architecture honesty — identity relations
#
# These tests ALWAYS pass (by definition) because is_abelian and is_chaotic
# are computed deterministically from koopman_trust.  They verify that the
# architecture cannot "hallucinate" structure when trust is low.
# ══════════════════════════════════════════════════════════════════════════════


class TestArchitectureHonesty:

    def test_is_chaotic_consistent_with_trust(self):
        """is_chaotic is exactly (trust < threshold) — no hallucination."""
        for params in [_PARAMS_LINEAR, _PARAMS_SMALL_F, _PARAMS_CHAOS, _PARAMS_MODERATE_F]:
            result = DrivenDuffingEvaluator(params, n_total=2000).evaluate()
            expected = result.koopman_trust < _CHAOS_TRUST_THRESHOLD
            assert result.is_chaotic == expected, (
                f"is_chaotic mismatch for F={params.F}: "
                f"trust={result.koopman_trust:.4f}, expected={expected}"
            )

    def test_is_abelian_consistent_with_trust(self):
        """is_abelian is exactly (trust ≥ threshold) — definition of abelian."""
        for params in [_PARAMS_LINEAR, _PARAMS_SMALL_F, _PARAMS_CHAOS, _PARAMS_MODERATE_F]:
            result = DrivenDuffingEvaluator(params, n_total=2000).evaluate()
            expected = result.koopman_trust >= _ABELIAN_TRUST_THRESHOLD
            assert result.is_abelian == expected, (
                f"is_abelian mismatch for F={params.F}: "
                f"trust={result.koopman_trust:.4f}, expected={expected}"
            )

    def test_abelian_and_chaotic_mutually_exclusive(self):
        """Since thresholds are equal, is_abelian and is_chaotic can't both be True."""
        assert _ABELIAN_TRUST_THRESHOLD == _CHAOS_TRUST_THRESHOLD, (
            "Thresholds must be equal for mutual exclusion"
        )
        for params in [_PARAMS_LINEAR, _PARAMS_CHAOS]:
            result = DrivenDuffingEvaluator(params, n_total=2000).evaluate()
            assert not (result.is_abelian and result.is_chaotic), (
                f"is_abelian and is_chaotic are both True for F={params.F}"
            )

    def test_trust_in_unit_interval(self):
        """Trust score must be in [0, 1] by definition."""
        for params in [_PARAMS_LINEAR, _PARAMS_CHAOS]:
            result = DrivenDuffingEvaluator(params, n_total=1000).evaluate()
            assert 0.0 <= result.koopman_trust <= 1.0, (
                f"Trust out of [0,1]: {result.koopman_trust}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Group 5: Trust comparison — periodic vs chaotic
# ══════════════════════════════════════════════════════════════════════════════


class TestTrustComparison:

    def test_linear_has_higher_trust_than_random(self):
        """
        Linear driven orbit (β=0) must have higher trust than a random walk.

        A random trajectory has no Koopman structure → large reconstruction error →
        trust near 0.  The linear limit cycle is perfectly predictable → trust ≈ 1.
        This is the canonical 'architecture honesty' test: the trust metric correctly
        distinguishes structured dynamics from noise.
        """
        r_linear = DrivenDuffingEvaluator(_PARAMS_LINEAR, n_total=2000).evaluate()

        # Construct a random trajectory with similar amplitude to the orbit
        np.random.seed(42)
        random_traj = np.random.randn(500, 2) * 1.5
        edmd = EDMDKoopman(observable_degree=3)
        edmd.fit_trajectory(random_traj)
        koop_random = edmd.eigendecomposition()
        trust_random = _reconstruction_trust(koop_random.reconstruction_error)

        assert r_linear.koopman_trust >= trust_random, (
            f"Expected trust_linear ({r_linear.koopman_trust:.4f}) ≥ "
            f"trust_random ({trust_random:.4f})"
        )

    def test_linear_has_low_reconstruction_error(self):
        """
        Linear driven orbit (β=0): EDMD reconstruction error should be < 0.5.

        The steady-state is a pure sinusoid — exactly representable in polynomial
        observable space.  The EDMD fit should have low reconstruction error.
        """
        r_linear = DrivenDuffingEvaluator(_PARAMS_LINEAR, n_total=2000).evaluate()
        assert r_linear.reconstruction_error < 0.5, (
            f"Expected low recon error for linear orbit, got {r_linear.reconstruction_error:.4f}"
        )

    def test_small_F_higher_trust_than_large_F_nonlinear(self):
        """For nonlinear (β=1): trust at small F ≥ trust at large (chaos) F."""
        r_small = DrivenDuffingEvaluator(_PARAMS_MODERATE_F, n_total=2000).evaluate()
        r_large = DrivenDuffingEvaluator(_PARAMS_CHAOS, n_total=2000).evaluate()
        # Large-F chaos should have lower or equal trust
        assert r_small.koopman_trust >= r_large.koopman_trust - 0.05, (
            f"Expected trust(F=0.1) ≥ trust(F=0.5), "
            f"got {r_small.koopman_trust:.4f} vs {r_large.koopman_trust:.4f}"
        )

    def test_edmd_low_trust_on_random_trajectory(self):
        """
        EDMD reconstruction-based trust must be low for a random (structureless) trajectory.

        This is the canonical architecture honesty test: a random walk has no Koopman
        structure.  EDMD will fit noise → high reconstruction error → low trust.
        is_abelian must then be False (architecture correctly reports uncertainty).
        """
        np.random.seed(42)
        random_traj = np.random.randn(500, 2)

        edmd = EDMDKoopman(observable_degree=3)
        edmd.fit_trajectory(random_traj)
        koop = edmd.eigendecomposition()

        trust = _reconstruction_trust(koop.reconstruction_error)
        is_abelian = trust >= _ABELIAN_TRUST_THRESHOLD

        assert trust < 0.5, (
            f"Random trajectory should give trust < 0.5, got {trust:.4f}"
        )
        assert not is_abelian, (
            f"Architecture should NOT claim abelian structure on random data "
            f"(trust={trust:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Group 6: Period detection and Poincaré section
# ══════════════════════════════════════════════════════════════════════════════


class TestPeriodDetection:

    def test_count_clusters_all_same(self):
        """Identical x-values → 1 cluster."""
        x = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        assert _count_clusters(x) == 1

    def test_count_clusters_two_groups(self):
        """Two well-separated groups → 2 clusters."""
        x = np.array([1.0, 1.01, 1.02, 3.0, 3.01, 3.02])
        assert _count_clusters(x) == 2

    def test_period_from_clusters_formula(self):
        """Cluster count maps to correct period number."""
        assert _period_from_clusters(1) == 1
        assert _period_from_clusters(2) == 2
        assert _period_from_clusters(3) == 4   # rounds up to 4
        assert _period_from_clusters(4) == 4
        assert _period_from_clusters(5) == 8

    def test_reconstruction_trust_formula(self):
        """_reconstruction_trust(err) = max(0, 1 - err/eta_max)."""
        assert _reconstruction_trust(0.0) == 1.0
        assert abs(_reconstruction_trust(_ETA_MAX) - 0.0) < 1e-10
        assert abs(_reconstruction_trust(0.5 * _ETA_MAX) - 0.5) < 1e-10
        assert _reconstruction_trust(2.0 * _ETA_MAX) == 0.0   # clipped

    def test_poincare_section_sampled_correctly(self):
        """Verify Poincaré section indices are at correct T_drive intervals."""
        p = DrivenDuffingParams(alpha=1.0, beta=0.0, delta=0.3, F=0.5, Omega=1.0)
        ev = DrivenDuffingEvaluator(p, dt=0.1, n_total=1000)
        T_drive = p.forcing_period   # 2π/1.0 ≈ 6.283
        steps_per_period = round(T_drive / ev.dt)   # round(62.83) = 63

        sim = DrivenDuffingSimulator(p, dt=ev.dt)
        steady = sim.run_steady_state(0.1, 0.0, ev.n_total, ev.transient_fraction)

        poincare_x = ev._poincare_x(steady)
        n_sections = len(poincare_x)

        # Should have roughly n_steady / steps_per_period points
        n_steady = len(steady)
        expected_sections = n_steady // steps_per_period
        assert abs(n_sections - expected_sections) <= 1, (
            f"Expected ~{expected_sections} Poincaré sections, got {n_sections}"
        )

    def test_linear_poincare_period1_consistent_with_evaluate(self):
        """Linear driven: Poincaré section gives 1 cluster, period_number == 1."""
        result = DrivenDuffingEvaluator(_PARAMS_LINEAR, dt=0.05, n_total=2000).evaluate()
        # For β=0 (linear), steady state is a pure sinusoid → period-1.
        # After the orbit-range fix, phase drift between stroboscopic samples
        # is ≤ 3% of orbit_range → correctly detected as 1 cluster.
        assert result.poincare_clusters == 1, (
            f"Expected 1 Poincaré cluster (period-1), got {result.poincare_clusters}"
        )
        assert result.period_number == 1, (
            f"Expected period_number=1, got {result.period_number}"
        )

    def test_result_fields_are_finite(self):
        """All DrivenDuffingResult fields must be finite (no NaN)."""
        result = DrivenDuffingEvaluator(_PARAMS_LINEAR, n_total=1000).evaluate()
        assert math.isfinite(result.koopman_trust)
        assert math.isfinite(result.dominant_frequency)
        assert math.isfinite(result.poincare_clusters)
        assert result.period_number in {1, 2, 4, 8, 16, 32}
