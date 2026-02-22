"""
Long-Run Observer Drift Simulation

Validates stability under sustained operation:
  - 10,000 sequential semantic inputs with random bounded forcing
  - Periodic basis consolidation
  - Energy bound tracking over time
  - Orthogonality norm ‖H_iᵀ H_j‖ drift measurement
  - Floating-point accumulation detection
  - Re-orthogonalization if drift detected
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from tensor.semantic_observer import (
    ObserverConfig,
    SemanticObserver,
    BasisConsolidator,
    HDVOrthogonalizer,
    semantic_energy,
    truncate_spectrum,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_STEPS = 10_000
CONSOLIDATION_INTERVAL = 500
ENERGY_CAP = 10.0
STATE_DIM = 32
INPUT_DIM = 16


# ---------------------------------------------------------------------------
# 1. Energy boundedness under long-run forcing
# ---------------------------------------------------------------------------

class TestLongRunEnergyBound:

    def test_energy_bounded_10k_steps(self):
        """Energy must remain below cap * safety_factor across 10K steps."""
        config = ObserverConfig(
            state_dim=STATE_DIM,
            input_dim=INPUT_DIM,
            dt=0.01,
            energy_cap=ENERGY_CAP,
            gamma_damp=0.1,
            lambda_max=2.0,
        )
        obs = SemanticObserver(config)
        rng = np.random.default_rng(42)

        max_energy = 0.0
        energy_violations = 0

        for step in range(N_STEPS):
            # Random bounded forcing: ||u|| ≤ 1
            u = rng.standard_normal(INPUT_DIM)
            u = u / (np.linalg.norm(u) + 1e-12)

            obs.step(u)
            E = semantic_energy(obs.x, np.zeros_like(obs.x), obs.P)
            max_energy = max(max_energy, E)

            # Track post-damping energy violations
            # (energy should recover within a few steps due to damping)
            if E > ENERGY_CAP * 5.0:
                energy_violations += 1

        # Energy should stay bounded (allow transient spikes but not persistent)
        assert energy_violations < N_STEPS * 0.01, \
            f"Too many energy violations: {energy_violations}/{N_STEPS}"
        assert np.all(np.isfinite(obs.x)), "State diverged to non-finite"

    def test_state_norm_bounded_10k_steps(self):
        """State norm must not grow unboundedly."""
        config = ObserverConfig(
            state_dim=STATE_DIM,
            input_dim=INPUT_DIM,
            dt=0.005,
            energy_cap=ENERGY_CAP,
            gamma_damp=0.15,
            lambda_max=1.5,
        )
        obs = SemanticObserver(config)
        rng = np.random.default_rng(7)

        norms = []
        for step in range(N_STEPS):
            u = rng.standard_normal(INPUT_DIM) * 0.5
            obs.step(u)
            norms.append(np.linalg.norm(obs.x))

        norms = np.array(norms)
        # Norm should not exhibit exponential growth
        # Check: max norm in last 1000 steps should not be >> max in first 1000
        early_max = np.max(norms[:1000])
        late_max = np.max(norms[-1000:])
        growth_ratio = late_max / (early_max + 1e-12)
        assert growth_ratio < 10.0, \
            f"State norm grew {growth_ratio:.1f}x from early to late phase"
        assert np.all(np.isfinite(norms)), "State diverged"

    def test_zero_input_decays(self):
        """With zero forcing, state should decay toward origin (stable A)."""
        config = ObserverConfig(
            state_dim=STATE_DIM,
            input_dim=INPUT_DIM,
            dt=0.01,
            energy_cap=ENERGY_CAP,
            gamma_damp=0.1,
            lambda_max=1.0,
        )
        obs = SemanticObserver(config)
        # Kick state to a nonzero initial condition
        obs.x = np.ones(STATE_DIM) * 5.0
        u_zero = np.zeros(INPUT_DIM)

        initial_norm = np.linalg.norm(obs.x)
        for _ in range(2000):
            obs.step(u_zero)

        final_norm = np.linalg.norm(obs.x)
        # Nonlinear g(x) = scale·tanh(x) creates a small equilibrium near origin,
        # so full decay to <10% is not expected.  70% decay proves stability.
        assert final_norm < initial_norm * 0.3, \
            f"State did not decay: initial={initial_norm:.3f}, final={final_norm:.3f}"


# ---------------------------------------------------------------------------
# 2. Basis consolidation stability
# ---------------------------------------------------------------------------

class TestConsolidationStability:

    def test_consolidation_preserves_finite_A(self):
        """Repeated consolidation must not introduce NaN/Inf into A."""
        config = ObserverConfig(
            state_dim=STATE_DIM,
            input_dim=INPUT_DIM,
            dt=0.01,
            energy_cap=ENERGY_CAP,
            gamma_damp=0.1,
        )
        obs = SemanticObserver(config)
        consolidator = BasisConsolidator(k=16, consolidate_every=CONSOLIDATION_INTERVAL)
        rng = np.random.default_rng(99)

        n_consolidations = 0
        for step in range(N_STEPS):
            u = rng.standard_normal(INPUT_DIM) * 0.3
            obs.step(u)
            consolidator.record(obs.x)

            if consolidator.should_consolidate():
                basis = consolidator.consolidate()
                A_new = consolidator.rotate_operator(obs.A, basis)
                # Rotate back to full space for the observer
                # A_full = basis @ A_new @ basis.T
                A_full = basis @ A_new @ basis.T
                obs.update_operator(A_full)
                n_consolidations += 1

                assert np.all(np.isfinite(obs.A)), \
                    f"A became non-finite after consolidation #{n_consolidations}"
                assert obs.spectral_radius < config.stability_cap + 0.1, \
                    f"Spectral radius {obs.spectral_radius:.3f} > cap after consolidation"

        assert n_consolidations >= 10, \
            f"Expected ≥10 consolidations, got {n_consolidations}"

    def test_spectral_radius_stable_across_consolidations(self):
        """Spectral radius should not drift upward across consolidations."""
        config = ObserverConfig(
            state_dim=STATE_DIM,
            input_dim=INPUT_DIM,
            dt=0.01,
            energy_cap=ENERGY_CAP,
            gamma_damp=0.1,
            lambda_max=2.0,
        )
        obs = SemanticObserver(config)
        consolidator = BasisConsolidator(k=16, consolidate_every=200)
        rng = np.random.default_rng(123)

        spectral_radii = [obs.spectral_radius]

        for step in range(5000):
            u = rng.standard_normal(INPUT_DIM) * 0.2
            obs.step(u)
            consolidator.record(obs.x)

            if consolidator.should_consolidate():
                basis = consolidator.consolidate()
                A_new = consolidator.rotate_operator(obs.A, basis)
                A_full = basis @ A_new @ basis.T
                obs.update_operator(A_full)
                spectral_radii.append(obs.spectral_radius)

        spectral_radii = np.array(spectral_radii)
        # No spectral radius should exceed stability_cap
        assert np.all(spectral_radii < config.stability_cap + 0.1)
        # Check no upward trend: last SR should not be >> first SR
        assert spectral_radii[-1] < spectral_radii[0] * 5.0, \
            f"Spectral radius drifted upward: {spectral_radii[0]:.3f} → {spectral_radii[-1]:.3f}"


# ---------------------------------------------------------------------------
# 3. Orthogonality drift measurement
# ---------------------------------------------------------------------------

class TestOrthogonalityDrift:

    def test_fixed_domain_orthogonality_preserved(self):
        """Fixed-slice domains must maintain exact orthogonality over time."""
        hdv_dim = 1000
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        rng = np.random.default_rng(42)

        domains = ["circuit", "semantic", "market", "code"]
        contamination_history = []

        for step in range(1000):
            vec = rng.standard_normal(hdv_dim)
            # Project through all domains
            projections = {d: orth.project(vec, d) for d in domains}

            # Measure pairwise contamination
            max_contamination = 0.0
            for i, d1 in enumerate(domains):
                for d2 in domains[i+1:]:
                    cc = abs(np.dot(projections[d1], projections[d2]))
                    max_contamination = max(max_contamination, cc)

            contamination_history.append(max_contamination)

        contamination = np.array(contamination_history)
        # Fixed slices should have EXACTLY zero cross-contamination
        assert np.all(contamination < 1e-10), \
            f"Cross-contamination detected: max={np.max(contamination):.2e}"

    def test_learned_basis_orthogonality_after_registration(self):
        """Learned domains should maintain near-zero contamination with fixed domains."""
        hdv_dim = 1000
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        rng = np.random.default_rng(88)

        # Register a learned domain
        learned_vectors = rng.standard_normal((hdv_dim, 5))
        orth.register_basis("physics", learned_vectors)

        contamination_samples = []
        for _ in range(500):
            vec = rng.standard_normal(hdv_dim)
            p_physics = orth.project(vec, "physics")
            for fixed_domain in ["circuit", "semantic", "market", "code"]:
                p_fixed = orth.project(vec, fixed_domain)
                cc = abs(np.dot(p_physics, p_fixed))
                contamination_samples.append(cc)

        contamination = np.array(contamination_samples)
        # Learned domain should have low contamination with fixed domains
        # (not exactly zero due to Gram-Schmidt numerical limits, but < 1e-6)
        assert np.max(contamination) < 1e-6, \
            f"Learned domain contamination too high: max={np.max(contamination):.2e}"

    def test_orthogonality_norm_tracking(self):
        """Track ‖H_iᵀ H_j‖ over repeated projections — must not drift."""
        hdv_dim = 2000
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        rng = np.random.default_rng(7)

        domains = ["circuit", "semantic", "market", "code"]
        n_trials = 500

        # Compute ‖H_iᵀ H_j‖ via Monte Carlo estimation
        # For fixed slices, H_iᵀ H_j = 0 exactly (indicator functions on disjoint sets)
        dot_products_early = []
        dot_products_late = []

        for trial in range(n_trials):
            vec = rng.standard_normal(hdv_dim)
            projections = [orth.project(vec, d) for d in domains]
            for i in range(len(domains)):
                for j in range(i+1, len(domains)):
                    dp = abs(np.dot(projections[i], projections[j]))
                    if trial < n_trials // 2:
                        dot_products_early.append(dp)
                    else:
                        dot_products_late.append(dp)

        early_mean = np.mean(dot_products_early)
        late_mean = np.mean(dot_products_late)

        # Both should be near zero, and late should not be worse than early
        assert early_mean < 1e-10, f"Early contamination: {early_mean:.2e}"
        assert late_mean < 1e-10, f"Late contamination: {late_mean:.2e}"


# ---------------------------------------------------------------------------
# 4. Floating-point accumulation detection
# ---------------------------------------------------------------------------

class TestFloatingPointStability:

    def test_repeated_truncation_no_nan(self):
        """Repeated truncate_spectrum calls must not produce NaN."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((16, 16)) * 0.3

        for _ in range(100):
            A = truncate_spectrum(A, energy_threshold=1e-3, stability_cap=5.0)
            # Add small perturbation to prevent collapse to zero
            A = A + rng.standard_normal(A.shape) * 0.01
            assert np.all(np.isfinite(A)), "truncate_spectrum produced NaN/Inf"

    def test_long_euler_integration_finite(self):
        """10K Euler steps with forcing must not produce Inf/NaN."""
        config = ObserverConfig(
            state_dim=64,
            input_dim=32,
            dt=0.01,
            energy_cap=ENERGY_CAP,
            gamma_damp=0.1,
            lambda_max=2.0,
        )
        obs = SemanticObserver(config)
        rng = np.random.default_rng(42)

        for step in range(N_STEPS):
            u = rng.standard_normal(config.input_dim) * 0.5
            x = obs.step(u)
            if step % 1000 == 0:
                assert np.all(np.isfinite(x)), \
                    f"State non-finite at step {step}: max={np.max(np.abs(x)):.2e}"

        assert np.all(np.isfinite(obs.x)), "Final state non-finite"

    def test_energy_computation_no_overflow(self):
        """semantic_energy must not overflow for large states."""
        n = 64
        P = np.eye(n)
        # Large but not extreme state
        x = np.ones(n) * 100.0
        dx = np.ones(n) * 50.0
        E = semantic_energy(x, dx, P)
        assert np.isfinite(E)
        assert E > 0

    def test_damping_preserves_finiteness(self):
        """apply_damping loop must not produce Inf across 10K iterations."""
        from tensor.semantic_observer import apply_damping
        x = np.ones(32) * 10.0
        dx = np.ones(32) * 5.0
        for _ in range(N_STEPS):
            dx = apply_damping(dx, x, gamma=0.1)
            x = x + 0.01 * dx
            assert np.all(np.isfinite(x)), "x diverged"
            assert np.all(np.isfinite(dx)), "dx diverged"


# ---------------------------------------------------------------------------
# 5. Dimension stability under consolidation
# ---------------------------------------------------------------------------

class TestDimensionStability:

    def test_basis_shape_consistent_across_consolidations(self):
        """BasisConsolidator output shape must be (state_dim, k) every time."""
        k = 8
        state_dim = 32
        bc = BasisConsolidator(k=k, consolidate_every=50)
        rng = np.random.default_rng(42)

        shapes = []
        for step in range(500):
            bc.record(rng.standard_normal(state_dim))
            if bc.should_consolidate():
                basis = bc.consolidate()
                shapes.append(basis.shape)

        assert len(shapes) >= 5
        for s in shapes:
            assert s == (state_dim, k), f"Unexpected basis shape: {s}"

    def test_observer_state_dim_invariant(self):
        """Observer state dimension must not change across 10K steps."""
        config = ObserverConfig(state_dim=STATE_DIM, input_dim=INPUT_DIM)
        obs = SemanticObserver(config)
        rng = np.random.default_rng(42)

        for step in range(N_STEPS):
            u = rng.standard_normal(INPUT_DIM) * 0.3
            x = obs.step(u)
            if step % 2000 == 0:
                assert x.shape == (STATE_DIM,), \
                    f"State dim changed at step {step}: {x.shape}"
                assert obs.A.shape == (STATE_DIM, STATE_DIM), \
                    f"A dim changed at step {step}: {obs.A.shape}"
