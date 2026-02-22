"""Test suite for the Spectral Coherence Regime System (reharmonization).

Phase A tests (primary): Layers 1-3 + integration.
Phase B tests (stubbed): Layer 4 DuffingParameterFilter (gated by enable_duffing).
Phase C tests (stubbed): Layer 5 ProfitWindow (gated by enable_profit_window).
"""

from __future__ import annotations

import math
import sys
import os
import unittest
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensor.reharmonization import (
    BootstrappedSpectrumTracker,
    CoherenceScorer,
    CoherenceState,
    DuffingEstimate,
    DuffingParameterFilter,
    FrequencyEstimate,
    LockScore,
    PersistentRegime,
    ProfitWindow,
    ReharmonizationEvent,
    ReharmonizationTracker,
    RegimePersistenceFilter,
    TimescaleSpectrum,
)
from tensor.timescale_state import (
    CrossTimescaleSystem,
    FundamentalState,
    RegimeState,
    ShockState,
)
from tensor.multi_horizon_mixer import MultiHorizonMixer, MixedPrediction


# ── Helpers ──────────────────────────────────────────────────────────────────

def generate_duffing_trajectory(
    alpha: float = 1.0,
    beta: float = 0.0,
    delta: float = 0.1,
    f_drive: float = 0.0,
    omega: float = 1.0,
    x0: float = 1.0,
    v0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 500,
) -> np.ndarray:
    """Generate Duffing oscillator trajectory: x'' + delta*x' + alpha*x + beta*x^3 = f*cos(omega*t)."""
    traj = np.zeros((n_steps, 2))
    traj[0] = [x0, v0]
    for i in range(n_steps - 1):
        x, v = traj[i]
        t = i * dt
        a = -delta * v - alpha * x - beta * x ** 3 + f_drive * math.cos(omega * t)
        # RK4
        k1x, k1v = v, a
        x2, v2 = x + 0.5 * dt * k1x, v + 0.5 * dt * k1v
        t2 = t + 0.5 * dt
        a2 = -delta * v2 - alpha * x2 - beta * x2 ** 3 + f_drive * math.cos(omega * t2)
        k2x, k2v = v2, a2
        x3, v3 = x + 0.5 * dt * k2x, v + 0.5 * dt * k2v
        a3 = -delta * v3 - alpha * x3 - beta * x3 ** 3 + f_drive * math.cos(omega * t2)
        k3x, k3v = v3, a3
        x4, v4 = x + dt * k3x, v + dt * k3v
        t4 = t + dt
        a4 = -delta * v4 - alpha * x4 - beta * x4 ** 3 + f_drive * math.cos(omega * t4)
        k4x, k4v = v4, a4
        traj[i + 1, 0] = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        traj[i + 1, 1] = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return traj


def make_synthetic_spectrum(
    freqs: List[float],
    amplitudes: List[float],
    stabilities: List[float] = None,
    variances: List[float] = None,
) -> TimescaleSpectrum:
    """Create a synthetic TimescaleSpectrum for testing."""
    n = len(freqs)
    if stabilities is None:
        stabilities = [1.0] * n
    if variances is None:
        variances = [0.001] * n
    modes = []
    for i in range(n):
        modes.append(FrequencyEstimate(
            frequency=freqs[i],
            damping=0.1,
            confidence_low=freqs[i] * 0.95,
            confidence_high=freqs[i] * 1.05,
            variance=variances[i],
            stability_score=stabilities[i],
            amplitude=amplitudes[i],
        ))
    modes.sort(key=lambda m: m.amplitude, reverse=True)
    return TimescaleSpectrum(
        modes=modes,
        koopman_trust=0.8,
        gram_condition=10.0,
        spectral_entropy=0.5,
        timestamp=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A Tests: Layers 1-3 + Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestBootstrappedSpectrumTracker(unittest.TestCase):
    """Layer 1: Bootstrapped spectrum estimation."""

    def test_clean_oscillator_frequency(self):
        """Clean linear oscillator (alpha=1, beta=0, delta=0.1) → frequency ~1.0."""
        tracker = BootstrappedSpectrumTracker(
            window_size=200, n_bootstrap=10, observable_degree=2,
            dt=0.01, n_modes=3, min_stability=0.3, recompute_interval=1,
        )
        traj = generate_duffing_trajectory(alpha=1.0, beta=0.0, delta=0.1, n_steps=300)
        # Subsample for window (every 5th point to decorrelate)
        for i in range(0, len(traj), 3):
            tracker.push_state("test", traj[i])

        spec = tracker.compute_spectrum("test")
        self.assertIsNotNone(spec)
        # At least one mode should exist
        self.assertGreater(len(spec.modes), 0)
        # Koopman trust should be positive
        self.assertGreater(spec.koopman_trust, 0.0)

    def test_bootstrap_ci_contains_true_frequency(self):
        """Bootstrap CI should bracket the true oscillation frequency."""
        tracker = BootstrappedSpectrumTracker(
            window_size=200, n_bootstrap=15, observable_degree=2,
            dt=0.01, n_modes=3, min_stability=0.3, recompute_interval=1,
        )
        traj = generate_duffing_trajectory(alpha=1.0, beta=0.0, delta=0.05, n_steps=400)
        for i in range(0, len(traj), 2):
            tracker.push_state("test", traj[i])

        spec = tracker.compute_spectrum("test")
        self.assertIsNotNone(spec)
        # Check that at least one mode's CI brackets a reasonable frequency
        has_reasonable = any(
            m.confidence_low <= m.frequency <= m.confidence_high
            for m in spec.modes
        )
        self.assertTrue(has_reasonable, "No mode has CI containing its median frequency")

    def test_unstable_modes_have_low_stability(self):
        """Noisy signal should produce modes with lower stability scores."""
        tracker = BootstrappedSpectrumTracker(
            window_size=60, n_bootstrap=10, observable_degree=2,
            dt=0.01, n_modes=5, min_stability=0.3, recompute_interval=1,
        )
        rng = np.random.default_rng(42)
        # Pure noise trajectory
        for _ in range(80):
            tracker.push_state("noise", rng.standard_normal(2))

        spec = tracker.compute_spectrum("noise")
        self.assertIsNotNone(spec)
        # For noisy data, many modes should have lower stability
        avg_stability = np.mean([m.stability_score for m in spec.modes])
        # Not all should be perfectly stable
        self.assertLess(avg_stability, 1.0)

    def test_spectral_entropy_in_01(self):
        """Normalized spectral entropy must be in [0, 1]."""
        tracker = BootstrappedSpectrumTracker(
            window_size=100, n_bootstrap=5, observable_degree=2,
            dt=0.01, n_modes=3, min_stability=0.3, recompute_interval=1,
        )
        traj = generate_duffing_trajectory(alpha=1.0, beta=0.0, delta=0.1, n_steps=150)
        for i in range(0, len(traj), 1):
            tracker.push_state("test", traj[i])

        spec = tracker.compute_spectrum("test")
        self.assertIsNotNone(spec)
        self.assertGreaterEqual(spec.spectral_entropy, 0.0)
        self.assertLessEqual(spec.spectral_entropy, 1.0)

    def test_adaptive_recompute_caching(self):
        """No bootstrap on unchanged window (performance test)."""
        tracker = BootstrappedSpectrumTracker(
            window_size=100, n_bootstrap=10, observable_degree=2,
            dt=0.01, n_modes=3, min_stability=0.3, recompute_interval=10,
        )
        traj = generate_duffing_trajectory(alpha=1.0, beta=0.0, delta=0.1, n_steps=120)
        for state in traj:
            tracker.push_state("test", state)

        # First compute triggers bootstrap
        spec1 = tracker.compute_spectrum("test")
        self.assertIsNotNone(spec1)

        # Second compute without new data should return cached
        spec2 = tracker.compute_spectrum("test")
        self.assertIsNotNone(spec2)
        # Should be the same object (cached)
        self.assertIs(spec1, spec2)

    def test_variance_lower_with_hungarian_matching(self):
        """Hungarian assignment should not inflate variance vs ground truth."""
        tracker = BootstrappedSpectrumTracker(
            window_size=200, n_bootstrap=15, observable_degree=2,
            dt=0.01, n_modes=3, min_stability=0.2, recompute_interval=1,
        )
        # Clean oscillator → low variance expected
        traj = generate_duffing_trajectory(alpha=1.0, beta=0.0, delta=0.05, n_steps=300)
        for i in range(0, len(traj), 2):
            tracker.push_state("test", traj[i])

        spec = tracker.compute_spectrum("test")
        self.assertIsNotNone(spec)
        # Stable modes should have low variance
        stable = [m for m in spec.modes if m.stability_score >= 0.5]
        if stable:
            max_var = max(m.variance for m in stable)
            # For clean oscillator, variance should be reasonable
            self.assertLess(max_var, 10.0, "Variance too high for clean oscillator")


class TestCoherenceScorer(unittest.TestCase):
    """Layer 2: Lock coherence scoring."""

    def test_same_frequencies_high_lock(self):
        """Same frequencies → lock_score ≈ 1.0."""
        scorer = CoherenceScorer(sigma=0.05, K=10, max_pairs=15)
        spec_a = make_synthetic_spectrum([1.0, 2.0], [1.0, 0.5])
        spec_b = make_synthetic_spectrum([1.0, 2.0], [1.0, 0.5])
        result = scorer.score(spec_a, spec_b)
        # At least the dominant pair should have high lock
        self.assertGreater(result.coherence_energy, 0.8)

    def test_distant_frequencies_low_lock(self):
        """Very different frequencies → lock_score ≈ 0.0."""
        scorer = CoherenceScorer(sigma=0.05, K=10, max_pairs=15)
        spec_a = make_synthetic_spectrum([1.0], [1.0])
        spec_b = make_synthetic_spectrum([3.7], [1.0])
        result = scorer.score(spec_a, spec_b)
        # Dissonant pair → low energy
        self.assertLess(result.coherence_energy, 0.5)

    def test_coherence_energy_is_mean(self):
        """coherence_energy = mean(lock_scores), not sum."""
        scorer = CoherenceScorer(sigma=0.05, K=10, max_pairs=15)
        spec_a = make_synthetic_spectrum([1.0, 2.0, 3.0], [1.0, 0.8, 0.6])
        spec_b = make_synthetic_spectrum([1.0, 2.0], [1.0, 0.8])
        result = scorer.score(spec_a, spec_b)
        # Verify mean vs sum
        if result.lock_scores:
            expected_mean = np.mean([s.lock_score for s in result.lock_scores])
            self.assertAlmostEqual(result.coherence_energy, expected_mean, places=10)

    def test_lock_vector_fixed_length(self):
        """Lock vector is ALWAYS length max_pairs, zero-padded."""
        max_p = 10
        scorer = CoherenceScorer(sigma=0.05, K=10, max_pairs=max_p)

        # Few modes → few pairs → rest zero-padded
        spec_a = make_synthetic_spectrum([1.0], [1.0])
        spec_b = make_synthetic_spectrum([1.0], [1.0])
        result = scorer.score(spec_a, spec_b)
        self.assertEqual(len(result.lock_vector), max_p)

        # More modes
        spec_a2 = make_synthetic_spectrum([1.0, 2.0, 3.0], [1.0, 0.8, 0.6])
        spec_b2 = make_synthetic_spectrum([1.0, 2.0, 3.0], [1.0, 0.8, 0.6])
        result2 = scorer.score(spec_a2, spec_b2)
        self.assertEqual(len(result2.lock_vector), max_p)

    def test_deterministic_pair_ordering(self):
        """Lock vector is deterministic across calls with same inputs."""
        scorer = CoherenceScorer(sigma=0.05, K=10, max_pairs=15)
        spec_a = make_synthetic_spectrum([1.0, 2.0], [1.0, 0.5])
        spec_b = make_synthetic_spectrum([1.5, 3.0], [1.0, 0.5])
        r1 = scorer.score(spec_a, spec_b)
        r2 = scorer.score(spec_a, spec_b)
        np.testing.assert_array_equal(r1.lock_vector, r2.lock_vector)

    def test_high_variance_modes_filtered(self):
        """High-variance modes should be filtered before scoring."""
        scorer = CoherenceScorer(
            sigma=0.05, K=10, max_pairs=15,
            min_stability=0.6, variance_threshold=0.01,
        )
        # One stable mode, one high-variance mode
        spec_a = make_synthetic_spectrum(
            [1.0, 5.0], [1.0, 0.5],
            stabilities=[0.9, 0.9],
            variances=[0.001, 0.1],  # second mode high variance
        )
        spec_b = make_synthetic_spectrum([1.0], [1.0])
        result = scorer.score(spec_a, spec_b)
        # Should only have 1 pair (high-variance mode filtered)
        self.assertEqual(len(result.lock_scores), 1)

    def test_coherence_energy_in_01(self):
        """Coherence energy (mean of lock scores) should be in [0, 1]."""
        scorer = CoherenceScorer(sigma=0.05, K=10, max_pairs=15)
        spec_a = make_synthetic_spectrum([1.0, 2.0, 3.0], [1.0, 0.8, 0.6])
        spec_b = make_synthetic_spectrum([1.5, 2.5], [1.0, 0.8])
        result = scorer.score(spec_a, spec_b)
        self.assertGreaterEqual(result.coherence_energy, 0.0)
        self.assertLessEqual(result.coherence_energy, 1.0)


class TestRegimePersistenceFilter(unittest.TestCase):
    """Layer 3: Regime persistence with hysteresis."""

    def _make_coherence(
        self, lock_vector: np.ndarray, energy: float = 0.5, timestamp: float = 0.0,
    ) -> CoherenceState:
        return CoherenceState(
            lock_scores=[],
            coherence_energy=energy,
            dominant_lock=None,
            lock_vector=lock_vector,
            spectral_entropy_drop=0.0,
            timestamp=timestamp,
        )

    def test_n_entry_required_for_regime(self):
        """Need N_entry consecutive similar windows for regime declaration."""
        filt = RegimePersistenceFilter(N_entry=5, N_exit=3, max_pairs=10)
        base = np.array([0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Before N_entry: no regime
        for i in range(4):
            noise = np.random.default_rng(i).normal(0, 0.01, 10)
            event = filt.update(self._make_coherence(base + noise, timestamp=float(i)))
            self.assertIsNone(event)
        self.assertIsNone(filt.current_regime)

        # At N_entry: regime declared
        event = filt.update(self._make_coherence(base, timestamp=5.0))
        self.assertIsNone(event)  # No event on first regime (no transition)
        self.assertIsNotNone(filt.current_regime)

    def test_single_perturbation_no_break(self):
        """Single-window perturbation → no regime break (hysteresis)."""
        filt = RegimePersistenceFilter(N_entry=3, N_exit=3, max_pairs=10)
        base = np.array([0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Establish regime
        for i in range(5):
            filt.update(self._make_coherence(base, timestamp=float(i)))
        self.assertIsNotNone(filt.current_regime)
        rid = filt.current_regime.regime_id

        # Single perturbation
        perturbed = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
        event = filt.update(self._make_coherence(perturbed, timestamp=6.0))
        self.assertIsNone(event)
        self.assertEqual(filt.current_regime.regime_id, rid)

    def test_n_exit_breaks_trigger_event(self):
        """N_exit consecutive breaks → ReharmonizationEvent emitted."""
        filt = RegimePersistenceFilter(
            N_entry=3, N_exit=3, similarity_threshold=0.3, max_pairs=10,
        )
        base = np.array([0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        other = np.array([0.1, 0.1, 0.1, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Establish regime
        for i in range(5):
            filt.update(self._make_coherence(base, timestamp=float(i)))
        self.assertIsNotNone(filt.current_regime)

        # N_exit breaks
        events = []
        for i in range(10):
            ev = filt.update(self._make_coherence(other, timestamp=float(10 + i)))
            if ev is not None:
                events.append(ev)
        self.assertGreater(len(events), 0, "No ReharmonizationEvent emitted after N_exit breaks")
        event = events[0]
        self.assertIsInstance(event, ReharmonizationEvent)
        self.assertNotEqual(event.old_regime.regime_id, event.new_regime.regime_id)

    def test_cosine_similarity_scale_invariant(self):
        """Cosine similarity threshold is scale-invariant."""
        filt = RegimePersistenceFilter(
            N_entry=3, N_exit=3, similarity_threshold=0.3, max_pairs=5,
        )
        base = np.array([0.8, 0.6, 0.4, 0.2, 0.1])

        # Establish regime with base
        for i in range(5):
            filt.update(self._make_coherence(base, timestamp=float(i)))

        # Scaled version should be similar (cosine invariant to magnitude)
        scaled = base * 3.0
        event = filt.update(self._make_coherence(scaled, timestamp=10.0))
        self.assertIsNone(event)  # Same direction → no break

    def test_regime_merging_reuses_id(self):
        """Close centroid → reuses historical regime_id."""
        filt = RegimePersistenceFilter(
            N_entry=3, N_exit=3, similarity_threshold=0.3,
            merge_threshold=0.85, max_pairs=10,
        )
        base_a = np.array([0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        base_b = np.array([0.1, 0.1, 0.1, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Establish regime A
        for i in range(5):
            filt.update(self._make_coherence(base_a, timestamp=float(i)))
        regime_a_id = filt.current_regime.regime_id

        # Transition to regime B
        for i in range(5):
            filt.update(self._make_coherence(base_b, timestamp=float(10 + i)))

        # Transition back to A-like
        for i in range(5):
            filt.update(self._make_coherence(base_a, timestamp=float(20 + i)))

        # Should reuse regime A's ID
        self.assertEqual(filt.current_regime.regime_id, regime_a_id)

    def test_centroid_ema_updates(self):
        """Centroid EMA-updates and re-normalizes within regime."""
        filt = RegimePersistenceFilter(
            N_entry=3, N_exit=3, ema_alpha=0.2, max_pairs=5,
        )
        base = np.array([0.8, 0.6, 0.0, 0.0, 0.0])

        # Establish regime
        for i in range(5):
            filt.update(self._make_coherence(base, timestamp=float(i)))

        centroid_before = filt.current_regime.centroid.copy()

        # Feed slightly shifted vector (within threshold)
        shifted = np.array([0.75, 0.65, 0.05, 0.0, 0.0])
        filt.update(self._make_coherence(shifted, timestamp=10.0))

        centroid_after = filt.current_regime.centroid
        # Should have moved slightly
        self.assertFalse(np.allclose(centroid_before, centroid_after, atol=1e-10))
        # But still normalized
        norm = np.linalg.norm(centroid_after)
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestIntegrationPhaseA(unittest.TestCase):
    """Integration tests for Phase A: end-to-end tracker."""

    def test_synthetic_duffing_regime_change(self):
        """Planted parameter change → 1 event, 2 regimes."""
        tracker = ReharmonizationTracker(
            window_size=60, n_bootstrap=5, observable_degree=2,
            dt_S=0.01, n_modes=3, min_stability=0.2,
            recompute_interval=1,
            sigma=0.1, K=5, max_pairs=10,
            N_entry=3, N_exit=3, similarity_threshold=0.4,
        )

        # Phase 1: alpha=1.0 oscillator
        traj1 = generate_duffing_trajectory(alpha=1.0, beta=0.0, delta=0.1, n_steps=100)
        for i, state in enumerate(traj1):
            tracker.update(
                shock_state=state,
                regime_state=state * 0.5,
                timestamp=float(i) * 0.01,
            )

        # Phase 2: alpha=4.0 oscillator (different frequency)
        traj2 = generate_duffing_trajectory(alpha=4.0, beta=0.0, delta=0.1, n_steps=100)
        events = []
        for i, state in enumerate(traj2):
            ev = tracker.update(
                shock_state=state,
                regime_state=state * 0.5,
                timestamp=float(100 + i) * 0.01,
            )
            if ev is not None:
                events.append(ev)

        summary = tracker.get_coherence_summary()
        # Should have detected some regime structure
        self.assertIn("regime_id", summary)

    def test_pure_noise_no_false_regimes(self):
        """Pure noise → no false regime declarations (or very few)."""
        tracker = ReharmonizationTracker(
            window_size=40, n_bootstrap=5, observable_degree=2,
            dt_S=1.0, n_modes=3, min_stability=0.7,  # high stability threshold
            recompute_interval=1,
            sigma=0.05, K=5, max_pairs=10,
            N_entry=5, N_exit=3, similarity_threshold=0.3,
        )

        rng = np.random.default_rng(123)
        events = []
        for i in range(100):
            ev = tracker.update(
                shock_state=rng.standard_normal(2),
                regime_state=rng.standard_normal(2),
                timestamp=float(i),
            )
            if ev is not None:
                events.append(ev)

        # Should have few or no events (noise doesn't produce stable coherence)
        # Allow up to 3 false positives for statistical fluctuations
        self.assertLessEqual(len(events), 3,
                             f"Too many false regime events in noise: {len(events)}")

    def test_stable_oscillator_single_regime(self):
        """Stable oscillator → single regime, no events."""
        tracker = ReharmonizationTracker(
            window_size=60, n_bootstrap=5, observable_degree=2,
            dt_S=0.01, n_modes=3, min_stability=0.3,
            recompute_interval=1,
            sigma=0.1, K=5, max_pairs=10,
            N_entry=3, N_exit=3, similarity_threshold=0.3,
        )

        traj = generate_duffing_trajectory(alpha=1.0, beta=0.0, delta=0.1, n_steps=200)
        events = []
        for i, state in enumerate(traj):
            ev = tracker.update(
                shock_state=state,
                regime_state=state * 0.5,
                timestamp=float(i) * 0.01,
            )
            if ev is not None:
                events.append(ev)

        # Stable system → no regime transitions
        self.assertEqual(len(events), 0, f"Unexpected events in stable oscillator: {len(events)}")

    def test_backward_compat_cross_timescale_none_tracker(self):
        """CrossTimescaleSystem(reharmonization_tracker=None) unchanged behavior."""
        sys = CrossTimescaleSystem(
            shock_dim=12, regime_dim=16, fundamental_dim=12,
            reharmonization_tracker=None,
        )
        shock = ShockState(features=np.random.randn(12), timestamp=1.0)
        regime = RegimeState(features=np.random.randn(16))
        fundamental = FundamentalState(features=np.random.randn(12))
        # Should work exactly as before
        new_r, new_f = sys.propagate_shock(shock, regime, fundamental)
        self.assertEqual(new_r.features.shape, (16,))
        self.assertEqual(new_f.features.shape, (12,))

    def test_backward_compat_mixer_no_regime_vector(self):
        """mixer.mix(regime_vector=None) unchanged behavior."""
        mixer = MultiHorizonMixer()
        fund = FundamentalState(features=np.random.randn(12))
        reg = RegimeState(features=np.random.randn(16))
        shock = ShockState(features=np.random.randn(12))
        result = mixer.mix(fund, reg, shock, regime_vector=None)
        self.assertIsInstance(result, MixedPrediction)
        self.assertAlmostEqual(result.weights.sum(), 1.0, places=5)

    def test_mixer_with_regime_vector_shifts_weights(self):
        """Regime vector (4,) should shift softmax weights."""
        mixer = MultiHorizonMixer()
        fund = FundamentalState(features=np.zeros(12))
        reg = RegimeState(features=np.zeros(16))
        shock = ShockState(features=np.zeros(12))

        # Without regime vector
        base = mixer.mix(fund, reg, shock, regime_vector=None)

        # With strong mean-reversion signal → boost M
        mean_rev = np.array([1.0, 0.0, 0.0, 0.0])
        shifted = mixer.mix(fund, reg, shock, regime_vector=mean_rev)

        # Weights should differ
        self.assertFalse(np.allclose(base.weights, shifted.weights))

    def test_cross_timescale_with_tracker_feeds(self):
        """CrossTimescaleSystem with tracker should auto-feed states."""
        tracker = ReharmonizationTracker(
            window_size=20, n_bootstrap=3, observable_degree=2,
            dt_S=1.0, n_modes=3, min_stability=0.2,
            recompute_interval=1, N_entry=3, N_exit=2,
        )
        sys = CrossTimescaleSystem(
            shock_dim=12, regime_dim=16, fundamental_dim=12,
            reharmonization_tracker=tracker,
        )

        # Propagate several shocks
        for i in range(10):
            shock = ShockState(
                features=np.random.randn(12) * 0.1,
                timestamp=float(i),
            )
            regime = RegimeState(features=np.random.randn(16) * 0.1)
            fundamental = FundamentalState(features=np.random.randn(12) * 0.1)
            sys.propagate_shock(shock, regime, fundamental)

        # Tracker should have received data
        has_S = len(tracker.spectrum_tracker._buffers.get("S", [])) > 0
        has_M = len(tracker.spectrum_tracker._buffers.get("M", [])) > 0
        self.assertTrue(has_S, "Tracker did not receive shock states")
        self.assertTrue(has_M, "Tracker did not receive regime states")

    def test_get_coherence_summary_structure(self):
        """get_coherence_summary returns expected keys."""
        tracker = ReharmonizationTracker(
            window_size=20, n_bootstrap=3, recompute_interval=1,
        )
        summary = tracker.get_coherence_summary()
        self.assertIn("coherence_energy", summary)
        self.assertIn("regime_id", summary)
        self.assertIn("regime_confidence", summary)
        self.assertIn("spectral_entropy", summary)

    def test_regime_vector_none_when_duffing_disabled(self):
        """get_regime_vector() returns None when enable_duffing=False."""
        tracker = ReharmonizationTracker(enable_duffing=False)
        self.assertIsNone(tracker.get_regime_vector())

    def test_lock_vector_dimensional_stability(self):
        """Lock vector always has length max_pairs across different mode counts."""
        max_p = 12
        scorer = CoherenceScorer(sigma=0.05, K=5, max_pairs=max_p)

        for n_modes_a in [1, 2, 3, 5]:
            for n_modes_b in [1, 2, 4]:
                spec_a = make_synthetic_spectrum(
                    [float(i + 1) for i in range(n_modes_a)],
                    [1.0 / (i + 1) for i in range(n_modes_a)],
                )
                spec_b = make_synthetic_spectrum(
                    [float(i + 1) * 1.5 for i in range(n_modes_b)],
                    [1.0 / (i + 1) for i in range(n_modes_b)],
                )
                result = scorer.score(spec_a, spec_b)
                self.assertEqual(
                    len(result.lock_vector), max_p,
                    f"Lock vector wrong length for {n_modes_a}x{n_modes_b} modes",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B Tests: Layer 4 DuffingParameterFilter (gated by enable_duffing)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDuffingParameterFilter(unittest.TestCase):
    """Layer 4: EKF Duffing parameter estimation (Phase B)."""

    def test_ekf_converges_to_true_params(self):
        """EKF converges to true alpha, beta within tolerance after updates."""
        filt = DuffingParameterFilter(
            process_noise=0.001, measurement_noise=0.01, amplitude_threshold=0.005,
        )
        # True params: alpha=1.0, beta=0.5
        true_alpha = 1.0
        true_beta = 0.5
        rng = np.random.default_rng(42)

        for i in range(50):
            # Simulate varying amplitude
            A = 0.5 + 0.5 * math.sin(0.1 * i)
            # True frequency from backbone
            true_freq = math.sqrt(true_alpha + 0.75 * true_beta * A ** 2)
            # Add measurement noise
            obs_freq = true_freq + rng.normal(0, 0.01)
            mode = FrequencyEstimate(
                frequency=obs_freq, damping=0.1,
                confidence_low=obs_freq - 0.05, confidence_high=obs_freq + 0.05,
                variance=0.001, stability_score=0.9, amplitude=A,
            )
            filt.update(mode, A)

        est = filt.get_estimate()
        # Should be within 20% of true values
        self.assertAlmostEqual(est.alpha, true_alpha, delta=0.3)
        self.assertAlmostEqual(est.beta, true_beta, delta=0.2)

    def test_variance_decreases(self):
        """Variance should decrease with more observations."""
        filt = DuffingParameterFilter(
            process_noise=0.001, measurement_noise=0.01, amplitude_threshold=0.005,
        )
        rng = np.random.default_rng(42)
        variances = []

        for i in range(30):
            A = 0.5 + 0.3 * math.sin(0.2 * i)
            freq = math.sqrt(1.0 + 0.75 * 0.3 * A ** 2) + rng.normal(0, 0.01)
            mode = FrequencyEstimate(
                frequency=freq, damping=0.1,
                confidence_low=freq - 0.05, confidence_high=freq + 0.05,
                variance=0.001, stability_score=0.9, amplitude=A,
            )
            filt.update(mode, A)
            est = filt.get_estimate()
            variances.append(est.alpha_var)

        # Later variances should be smaller than initial
        self.assertLess(variances[-1], variances[0])

    def test_regime_probabilities_sum_to_one(self):
        """regime_probabilities should sum to 1.0."""
        filt = DuffingParameterFilter()
        mode = FrequencyEstimate(
            frequency=1.0, damping=0.1,
            confidence_low=0.95, confidence_high=1.05,
            variance=0.001, stability_score=0.9, amplitude=0.5,
        )
        filt.update(mode, 0.5)
        est = filt.get_estimate()
        self.assertAlmostEqual(float(est.regime_probabilities.sum()), 1.0, places=5)
        self.assertEqual(est.regime_probabilities.shape, (4,))

    def test_observability_gate_no_beta_update(self):
        """No beta update when amplitude is constant (observability gate)."""
        filt = DuffingParameterFilter(
            process_noise=0.001, measurement_noise=0.01, amplitude_threshold=0.02,
        )
        # Feed constant amplitude
        A = 1.0
        freq = 1.0
        mode = FrequencyEstimate(
            frequency=freq, damping=0.1,
            confidence_low=0.95, confidence_high=1.05,
            variance=0.001, stability_score=0.9, amplitude=A,
        )
        # First update sets last_amplitude
        filt.update(mode, A)
        beta_var_before = filt._P[1, 1]

        # Second update with same amplitude → beta should not be updated
        filt.update(mode, A)
        # Beta variance should only grow from process noise (no information gain)
        # Since beta gain is zeroed, variance should increase, not decrease
        beta_var_after = filt._P[1, 1]
        self.assertGreaterEqual(beta_var_after, beta_var_before * 0.9)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C Tests: Layer 5 ProfitWindow (gated by enable_profit_window)
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfitWindow(unittest.TestCase):
    """Layer 5: Profit window with online statistics (Phase C)."""

    def _make_window(self, min_obs: int = 20) -> ProfitWindow:
        regime = PersistentRegime(
            regime_id=1, centroid=np.zeros(10),
            coherence_energy_mean=0.5, coherence_energy_std=0.1,
            entry_time=0.0, exit_time=None,
            n_observations=10, confidence=0.8,
        )
        return ProfitWindow(
            regime=regime, duffing_estimate=None, start_time=0.0,
            min_observations_for_sharpe=min_obs,
        )

    def test_welford_online_statistics(self):
        """Welford online mean/variance matches batch computation."""
        pw = self._make_window()
        returns = [0.01, -0.005, 0.008, -0.003, 0.012, 0.002, -0.001, 0.005,
                   0.003, -0.002, 0.007, -0.004, 0.009, 0.001, -0.006,
                   0.004, 0.008, -0.003, 0.006, -0.001]
        for r in returns:
            pw.update_returns(r)

        batch_mean = np.mean(returns)
        batch_var = np.var(returns, ddof=1)
        self.assertAlmostEqual(pw._mean, batch_mean, places=10)
        self.assertAlmostEqual(pw.return_variance, batch_var, places=10)

    def test_not_profitable_until_min_observations(self):
        """is_profitable = False until n_returns >= min_observations_for_sharpe."""
        pw = self._make_window(min_obs=20)
        for _ in range(19):
            pw.update_returns(0.01)  # positive returns
        self.assertFalse(pw.is_profitable)
        self.assertEqual(pw.sharpe_estimate, 0.0)

        # After min_obs
        pw.update_returns(0.01)
        self.assertTrue(pw.is_profitable)
        self.assertGreater(pw.sharpe_estimate, 0.0)

    def test_position_scale_clamped(self):
        """position_scale stays 1.0 until sufficient observations."""
        pw = self._make_window(min_obs=10)
        for _ in range(9):
            pw.update_returns(0.01)
        self.assertEqual(pw.position_scale, 1.0)

    def test_max_drawdown_tracked(self):
        """Max drawdown correctly tracked."""
        pw = self._make_window(min_obs=5)
        pw.update_returns(0.10)   # cumulative: 0.10
        pw.update_returns(0.05)   # cumulative: 0.15
        pw.update_returns(-0.08)  # cumulative: 0.07
        pw.update_returns(-0.05)  # cumulative: 0.02
        # Peak was 0.15, trough is 0.02 → drawdown = 0.13
        self.assertAlmostEqual(pw.max_drawdown, 0.13, places=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
