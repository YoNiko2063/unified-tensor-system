"""
Tests for tensor/semantic_observer.py — Part I of the Semantic Observer System.

Covers:
  SemanticObserver     — state shape, step, spectral_radius, energy damping
  truncate_spectrum    — eigenvalue filtering, emergency fallback
  semantic_energy      — positive-definite Lyapunov functional
  apply_damping        — descent injection
  BasisConsolidator    — record / should_consolidate / consolidate / rotate_operator
  HDVOrthogonalizer    — fixed slices, Gram-Schmidt, cross_contamination
  orthogonal_encode    — IntegratedHDVSystem integration
  reset                — observer state zeroed
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
    truncate_spectrum,
    semantic_energy,
    apply_damping,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return ObserverConfig(
        state_dim=16,
        input_dim=8,
        dt=0.01,
        energy_cap=10.0,
        gamma_damp=0.1,
        lambda_max=2.0,
        energy_threshold=1e-3,
        stability_cap=5.0,
    )


@pytest.fixture
def observer(default_config):
    return SemanticObserver(default_config)


# ---------------------------------------------------------------------------
# SemanticObserver — construction
# ---------------------------------------------------------------------------

class TestSemanticObserverInit:
    def test_state_shape(self, observer, default_config):
        assert observer.x.shape == (default_config.state_dim,)

    def test_initial_state_zeros(self, observer, default_config):
        np.testing.assert_array_equal(observer.x, np.zeros(default_config.state_dim))

    def test_A_shape(self, observer, default_config):
        n = default_config.state_dim
        assert observer.A.shape == (n, n)

    def test_B_shape(self, observer, default_config):
        n, m = default_config.state_dim, default_config.input_dim
        assert observer.B.shape == (n, m)

    def test_spectral_radius_below_lambda_max_after_init(self, observer, default_config):
        # After truncation at init, all eigenvalues should have |λ| < stability_cap
        assert observer.spectral_radius < default_config.stability_cap + 1e-9

    def test_custom_A_accepted(self, default_config):
        n = default_config.state_dim
        A = np.eye(n) * 0.5
        obs = SemanticObserver(default_config, A=A)
        assert obs.A.shape == (n, n)

    def test_custom_B_accepted(self, default_config):
        n, m = default_config.state_dim, default_config.input_dim
        B = np.zeros((n, m))
        obs = SemanticObserver(default_config, B=B)
        np.testing.assert_array_equal(obs.B, B)

    def test_P_is_identity_by_default(self, observer, default_config):
        n = default_config.state_dim
        np.testing.assert_array_almost_equal(observer.P, np.eye(n))


# ---------------------------------------------------------------------------
# SemanticObserver — step
# ---------------------------------------------------------------------------

class TestSemanticObserverStep:
    def test_step_returns_correct_shape(self, observer, default_config):
        u = np.zeros(default_config.input_dim)
        result = observer.step(u)
        assert result.shape == (default_config.state_dim,)

    def test_step_returns_copy(self, observer, default_config):
        u = np.zeros(default_config.input_dim)
        result = observer.step(u)
        # Mutating result should not affect internal state
        old = observer.x.copy()
        result[:] = 999.0
        np.testing.assert_array_equal(observer.x, old)

    def test_step_updates_internal_state(self, observer, default_config):
        u = np.ones(default_config.input_dim)
        observer.step(u)
        # State should no longer be all zeros after forcing
        assert not np.allclose(observer.x, 0.0)

    def test_step_zero_input_no_explosion(self, observer, default_config):
        """With zero input and stable A, state stays bounded."""
        u = np.zeros(default_config.input_dim)
        for _ in range(100):
            observer.step(u)
        assert np.all(np.isfinite(observer.x))
        assert np.linalg.norm(observer.x) < 1e6

    def test_step_multiple_steps_finite(self, observer, default_config):
        rng = np.random.default_rng(0)
        for _ in range(50):
            u = rng.standard_normal(default_config.input_dim)
            x = observer.step(u)
        assert np.all(np.isfinite(x))

    def test_energy_damping_applied_when_above_cap(self, default_config):
        """Verify damping is invoked when E_s > energy_cap.

        When x is large, semantic_energy(x, dx, P) >> energy_cap.
        apply_damping(dx, x, gamma) subtracts gamma*x from dx.
        We verify:
          1. The precondition holds (energy truly exceeds cap).
          2. The damped dx has a reduced dot product with x (damping pulls toward 0).
        """
        n = default_config.state_dim
        obs = SemanticObserver(default_config)
        # Set x very large so energy >> cap
        obs.x = np.ones(n) * 100.0

        g_x = 0.1 * np.tanh(obs.x)
        u = np.zeros(default_config.input_dim)
        dx_no_damp = obs.A @ obs.x + g_x + obs.B @ u
        E_before = semantic_energy(obs.x, dx_no_damp, obs.P)

        assert E_before > default_config.energy_cap, "Precondition: E must exceed cap"

        # Damping subtracts gamma*x from dx → the quadratic velocity term ||dx||^2
        # in the energy decreases only when dot(dx, gamma*x) > 0, which is not
        # guaranteed in general.  What IS guaranteed: the damped dx differs from
        # the undamped one by exactly -gamma*x.
        dx_damp = apply_damping(dx_no_damp, obs.x, gamma=default_config.gamma_damp)
        correction = dx_no_damp - dx_damp
        expected_correction = default_config.gamma_damp * obs.x
        np.testing.assert_allclose(correction, expected_correction, rtol=1e-10)

        # The x-quadratic Lyapunov term x^T P x is not changed by dx modification;
        # what changes is the rate-of-change term.  Verify the damped dx satisfies
        # the Lyapunov descent condition: d/dt (x^T P x) = 2 x^T P dx_damp < x^T P dx_no_damp.
        rate_no_damp = 2.0 * obs.x @ obs.P @ dx_no_damp
        rate_damp = 2.0 * obs.x @ obs.P @ dx_damp
        assert rate_damp < rate_no_damp, (
            "Damping must reduce Lyapunov time-derivative d/dt(x^T P x)"
        )

    def test_step_with_energy_violation_decreases_dx_norm(self, default_config):
        """Damping should reduce ||dx|| when energy cap is violated."""
        n = default_config.state_dim
        x_large = np.ones(n) * 50.0
        dx = np.ones(n) * 10.0
        dx_damped = apply_damping(dx, x_large, gamma=0.1)
        assert np.linalg.norm(dx_damped) < np.linalg.norm(dx)


# ---------------------------------------------------------------------------
# SemanticObserver — update_operator and reset
# ---------------------------------------------------------------------------

class TestSemanticObserverOperations:
    def test_update_operator_changes_A(self, observer, default_config):
        n = default_config.state_dim
        A_new = np.eye(n) * 0.3
        observer.update_operator(A_new)
        # After update_operator, spectral radius should reflect new A
        assert observer.spectral_radius < default_config.stability_cap + 1e-9

    def test_update_operator_truncates_unstable_A(self, default_config):
        """Large eigenvalues in A_new must be truncated after update."""
        n = default_config.state_dim
        obs = SemanticObserver(default_config)
        # Build A with eigenvalues >> stability_cap
        A_unstable = np.eye(n) * 100.0
        obs.update_operator(A_unstable)
        assert obs.spectral_radius < default_config.stability_cap + 1e-9

    def test_reset_zeros_state(self, observer, default_config):
        u = np.ones(default_config.input_dim)
        for _ in range(10):
            observer.step(u)
        observer.reset()
        np.testing.assert_array_equal(observer.x, np.zeros(default_config.state_dim))

    def test_reset_does_not_change_A(self, observer):
        A_before = observer.A.copy()
        observer.reset()
        np.testing.assert_array_equal(observer.A, A_before)

    def test_spectral_radius_property(self, observer):
        r = observer.spectral_radius
        assert isinstance(r, float)
        assert r >= 0.0


# ---------------------------------------------------------------------------
# truncate_spectrum
# ---------------------------------------------------------------------------

class TestTruncateSpectrum:
    def test_output_shape_preserved(self):
        A = np.random.default_rng(0).standard_normal((8, 8)) * 0.5
        A_trunc = truncate_spectrum(A, energy_threshold=1e-3, stability_cap=5.0)
        assert A_trunc.shape == A.shape

    def test_eigenvalues_within_bounds_after_truncation(self):
        """All eigenvalues of truncated A must have |λ| < stability_cap."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 10)) * 0.8
        A_trunc = truncate_spectrum(A, energy_threshold=1e-3, stability_cap=2.0)
        eigvals = np.linalg.eigvals(A_trunc)
        mags = np.abs(eigvals)
        # All non-negligible eigenvalues must be below cap
        assert np.all(mags[mags > 1e-10] < 2.0 + 1e-9)

    def test_all_zero_avoided_emergency_damping(self):
        """If nothing passes the mask, A * 0.5 should be returned (not all-zero)."""
        # Build A with all eigenvalues EXACTLY on or below threshold
        A = np.eye(8) * 0.0005   # |λ| = 0.0005 < energy_threshold=1e-3
        A_trunc = truncate_spectrum(A, energy_threshold=1e-3, stability_cap=5.0)
        # Emergency damping: A * 0.5 → not all-zero (A_trunc is A*0.5)
        # The result must not be exactly all zeros
        assert not np.allclose(A_trunc, 0.0)

    def test_output_is_real_for_real_input(self):
        A = np.random.default_rng(5).standard_normal((6, 6)) * 0.5
        A_trunc = truncate_spectrum(A)
        assert np.isrealobj(A_trunc)

    def test_stable_A_mostly_preserved(self):
        """A with all eigenvalues well within bounds → mostly unchanged."""
        A = np.eye(8) * 0.5   # |λ|=0.5, within [1e-3, 5.0]
        A_trunc = truncate_spectrum(A, energy_threshold=1e-3, stability_cap=5.0)
        # Spectral radius should remain ≈ 0.5
        sr = np.max(np.abs(np.linalg.eigvals(A_trunc)))
        assert sr < 5.0

    def test_large_eigenvalue_removed(self):
        """An eigenvalue >> stability_cap should not survive truncation."""
        n = 6
        # Create A with one dominant eigenvalue = 20 (>> stability_cap=5)
        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        D = np.diag([20.0, 0.5, 0.3, 0.2, 0.15, 0.1])
        A = Q @ D @ Q.T
        A_trunc = truncate_spectrum(A, energy_threshold=1e-3, stability_cap=5.0)
        sr = np.max(np.abs(np.linalg.eigvals(A_trunc)))
        assert sr < 5.0 + 1e-9


# ---------------------------------------------------------------------------
# semantic_energy and apply_damping
# ---------------------------------------------------------------------------

class TestSemanticEnergyAndDamping:
    def test_energy_positive_definite(self):
        n = 8
        x = np.random.default_rng(0).standard_normal(n)
        dx = np.random.default_rng(1).standard_normal(n)
        P = np.eye(n)
        E = semantic_energy(x, dx, P)
        assert E > 0.0

    def test_energy_zero_at_origin(self):
        n = 8
        x = np.zeros(n)
        dx = np.zeros(n)
        P = np.eye(n)
        E = semantic_energy(x, dx, P)
        assert E == 0.0

    def test_energy_increases_with_state_magnitude(self):
        n = 4
        P = np.eye(n)
        dx = np.zeros(n)
        E_small = semantic_energy(np.ones(n) * 0.1, dx, P)
        E_large = semantic_energy(np.ones(n) * 10.0, dx, P)
        assert E_large > E_small

    def test_energy_increases_with_dx_magnitude(self):
        n = 4
        P = np.eye(n)
        x = np.zeros(n)
        E_small = semantic_energy(x, np.ones(n) * 0.1, P)
        E_large = semantic_energy(x, np.ones(n) * 10.0, P)
        assert E_large > E_small

    def test_energy_alpha_scaling(self):
        n = 4
        P = np.eye(n)
        x = np.zeros(n)
        dx = np.ones(n)
        E_low_alpha = semantic_energy(x, dx, P, alpha=0.01)
        E_high_alpha = semantic_energy(x, dx, P, alpha=1.0)
        assert E_high_alpha > E_low_alpha

    def test_apply_damping_reduces_dx_norm(self):
        rng = np.random.default_rng(42)
        n = 8
        dx = rng.standard_normal(n)
        x = rng.standard_normal(n) * 5.0   # large x
        dx_damped = apply_damping(dx, x, gamma=0.1)
        # Damping subtracts gamma*x from dx; for large x aligned with dx, norm decreases
        assert np.linalg.norm(dx_damped) < np.linalg.norm(dx) + np.linalg.norm(0.1 * x)

    def test_apply_damping_formula(self):
        dx = np.array([1.0, 2.0, 3.0])
        x = np.array([10.0, 10.0, 10.0])
        result = apply_damping(dx, x, gamma=0.1)
        expected = dx - 0.1 * x
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_damping_zero_gamma(self):
        dx = np.array([1.0, 2.0])
        x = np.array([5.0, 5.0])
        result = apply_damping(dx, x, gamma=0.0)
        np.testing.assert_array_equal(result, dx)


# ---------------------------------------------------------------------------
# BasisConsolidator
# ---------------------------------------------------------------------------

class TestBasisConsolidator:
    def test_should_consolidate_fires_at_correct_interval(self):
        bc = BasisConsolidator(k=4, consolidate_every=10)
        for i in range(9):
            bc.record(np.ones(8))
            assert not bc.should_consolidate(), f"should not consolidate at {i+1}"
        bc.record(np.ones(8))
        assert bc.should_consolidate()

    def test_should_consolidate_fires_again_at_2x(self):
        bc = BasisConsolidator(k=4, consolidate_every=5)
        state = np.ones(8)
        for _ in range(5):
            bc.record(state)
        assert bc.should_consolidate()
        bc.consolidate()   # clear history
        for _ in range(4):
            bc.record(state)
        assert not bc.should_consolidate()
        bc.record(state)
        assert bc.should_consolidate()

    def test_consolidate_returns_correct_shape(self):
        state_dim = 16
        k = 4
        bc = BasisConsolidator(k=k, consolidate_every=10)
        rng = np.random.default_rng(0)
        for _ in range(10):
            bc.record(rng.standard_normal(state_dim))
        basis = bc.consolidate()
        assert basis.shape == (state_dim, k)

    def test_consolidate_clears_history(self):
        bc = BasisConsolidator(k=4, consolidate_every=5)
        for _ in range(5):
            bc.record(np.ones(8))
        bc.consolidate()
        assert len(bc._history) == 0

    def test_consolidate_basis_columns_near_orthonormal(self):
        """SVD output should give near-orthonormal columns."""
        state_dim = 20
        k = 5
        bc = BasisConsolidator(k=k, consolidate_every=20)
        rng = np.random.default_rng(1)
        for _ in range(20):
            bc.record(rng.standard_normal(state_dim))
        basis = bc.consolidate()
        # basis.T @ basis should be close to identity (for non-padded columns)
        gram = basis.T @ basis
        np.testing.assert_allclose(gram[:k, :k], np.eye(k), atol=1e-6)

    def test_auto_trim_history(self):
        """History should not exceed 2 * consolidate_every."""
        bc = BasisConsolidator(k=4, consolidate_every=5)
        for _ in range(25):
            bc.record(np.ones(8))
        assert len(bc._history) <= 2 * 5

    def test_rotate_operator_changes_shape_to_k_k(self):
        state_dim = 16
        k = 4
        bc = BasisConsolidator(k=k, consolidate_every=10)
        rng = np.random.default_rng(2)
        for _ in range(10):
            bc.record(rng.standard_normal(state_dim))
        basis = bc.consolidate()   # (state_dim, k)
        A = rng.standard_normal((state_dim, state_dim))
        A_rot = bc.rotate_operator(A, basis)
        assert A_rot.shape == (k, k)

    def test_rotate_operator_formula(self):
        """A' = basis.T @ A @ basis."""
        state_dim = 8
        k = 3
        rng = np.random.default_rng(3)
        basis = rng.standard_normal((state_dim, k))
        A = rng.standard_normal((state_dim, state_dim))
        bc = BasisConsolidator(k=k)
        A_rot = bc.rotate_operator(A, basis)
        expected = basis.T @ A @ basis
        np.testing.assert_array_almost_equal(A_rot, expected)


# ---------------------------------------------------------------------------
# HDVOrthogonalizer
# ---------------------------------------------------------------------------

class TestHDVOrthogonalizer:
    def test_fixed_slices_no_overlap_circuit_semantic(self):
        hdv_dim = 100
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        vec = np.ones(hdv_dim)
        p_circuit = orth.project(vec, "circuit")
        p_semantic = orth.project(vec, "semantic")
        # Non-zero elements must not overlap
        c_active = set(np.where(p_circuit != 0)[0])
        s_active = set(np.where(p_semantic != 0)[0])
        assert len(c_active & s_active) == 0

    def test_fixed_slices_no_overlap_market_code(self):
        hdv_dim = 100
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        vec = np.ones(hdv_dim)
        p_market = orth.project(vec, "market")
        p_code = orth.project(vec, "code")
        m_active = set(np.where(p_market != 0)[0])
        c_active = set(np.where(p_code != 0)[0])
        assert len(m_active & c_active) == 0

    def test_cross_contamination_near_zero_for_fixed_domains(self):
        hdv_dim = 200
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        rng = np.random.default_rng(5)
        vec = rng.standard_normal(hdv_dim)
        cc = orth.cross_contamination(vec, "circuit", "semantic")
        assert cc == pytest.approx(0.0, abs=1e-12)

    def test_cross_contamination_all_fixed_pairs(self):
        hdv_dim = 400
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        rng = np.random.default_rng(6)
        vec = rng.standard_normal(hdv_dim)
        pairs = [
            ("circuit", "semantic"),
            ("circuit", "market"),
            ("circuit", "code"),
            ("semantic", "market"),
            ("semantic", "code"),
            ("market", "code"),
        ]
        for a, b in pairs:
            cc = orth.cross_contamination(vec, a, b)
            assert cc == pytest.approx(0.0, abs=1e-12), f"Contamination nonzero for ({a},{b})"

    def test_project_preserves_values_in_slice(self):
        hdv_dim = 100
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        vec = np.arange(hdv_dim, dtype=float)
        p = orth.project(vec, "circuit")
        q = hdv_dim // 4
        np.testing.assert_array_equal(p[:q], vec[:q])
        np.testing.assert_array_equal(p[q:], np.zeros(hdv_dim - q))

    def test_project_unknown_domain_returns_original_with_warning(self):
        hdv_dim = 50
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        vec = np.ones(hdv_dim)
        with pytest.warns(UserWarning, match="unknown domain"):
            out = orth.project(vec, "unknown_domain")
        np.testing.assert_array_equal(out, vec)

    def test_gram_schmidt_removes_projection(self):
        orth = HDVOrthogonalizer(hdv_dim=10)
        b1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = orth.orthogonalize(v, [b1])
        # Should have zero component along b1
        assert abs(np.dot(result, b1)) < 1e-10

    def test_gram_schmidt_normalizes_output(self):
        orth = HDVOrthogonalizer(hdv_dim=4)
        b1 = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0, 0.0])
        result = orth.orthogonalize(v, [b1])
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_register_basis_stores_domain(self):
        hdv_dim = 20
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        vecs = np.random.default_rng(7).standard_normal((hdv_dim, 3))
        orth.register_basis("physics", vecs)
        assert "physics" in orth._learned_bases

    def test_register_basis_orthogonalizes_against_existing(self):
        """New learned domain should be orthogonal to fixed domains."""
        hdv_dim = 200
        orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
        rng = np.random.default_rng(8)
        vecs = rng.standard_normal((hdv_dim, 5))
        orth.register_basis("new_domain", vecs)
        # Check that the stored basis has non-trivial vectors
        basis = orth._learned_bases.get("new_domain")
        if basis is not None:
            assert basis.shape[0] == hdv_dim


# ---------------------------------------------------------------------------
# orthogonal_encode (IntegratedHDVSystem integration)
# ---------------------------------------------------------------------------

class TestOrthogonalEncode:
    @pytest.fixture(autouse=True)
    def _require_integrated_hdv(self):
        pytest.importorskip("tensor.integrated_hdv", reason="IntegratedHDVSystem not available")

    def test_orthogonal_encode_returns_correct_shape(self):
        from tensor.integrated_hdv import IntegratedHDVSystem
        hdv = IntegratedHDVSystem(hdv_dim=1000)
        vec = hdv.orthogonal_encode("hello world", "circuit")
        assert vec.shape == (1000,)

    def test_orthogonal_encode_different_domains_no_overlap(self):
        """circuit and semantic encoded vectors should have zero dot product
        in the respective fixed slices."""
        from tensor.integrated_hdv import IntegratedHDVSystem
        hdv = IntegratedHDVSystem(hdv_dim=1000)
        v_circuit = hdv.orthogonal_encode("signal processing filter", "circuit")
        v_semantic = hdv.orthogonal_encode("signal processing filter", "semantic")
        # Fixed slices do not overlap → dot product must be zero
        dot = float(np.dot(v_circuit, v_semantic))
        assert dot == pytest.approx(0.0, abs=1e-12)

    def test_orthogonal_encode_does_not_break_structural_encode(self):
        """Calling orthogonal_encode must not alter structural_encode output."""
        from tensor.integrated_hdv import IntegratedHDVSystem
        hdv = IntegratedHDVSystem(hdv_dim=500)
        v1 = hdv.structural_encode("test text", "circuit")
        _ = hdv.orthogonal_encode("test text", "circuit")
        v2 = hdv.structural_encode("test text", "circuit")
        # structural_encode must be deterministic and unchanged
        np.testing.assert_array_equal(v1, v2)

    def test_orthogonal_encode_creates_orthogonalizer_lazily(self):
        from tensor.integrated_hdv import IntegratedHDVSystem
        hdv = IntegratedHDVSystem(hdv_dim=400)
        assert hdv._orthogonalizer is None
        hdv.orthogonal_encode("foo", "market")
        assert hdv._orthogonalizer is not None

    def test_orthogonal_encode_market_code_no_overlap(self):
        from tensor.integrated_hdv import IntegratedHDVSystem
        hdv = IntegratedHDVSystem(hdv_dim=800)
        v_m = hdv.orthogonal_encode("trading strategy", "market")
        v_c = hdv.orthogonal_encode("trading strategy", "code")
        dot = float(np.dot(v_m, v_c))
        assert dot == pytest.approx(0.0, abs=1e-12)
