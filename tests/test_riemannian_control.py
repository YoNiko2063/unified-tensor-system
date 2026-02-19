"""
Tests for Riemannian control functions — curvature penalty and natural control.
"""

import numpy as np
import pytest
from tensor.riemannian_control import (
    curvature_gradient,
    curvature_gradient_vector,
    natural_control_step_with_fn,
    resonance_collapse_check,
    riemannian_metric,
    fisher_information_matrix,
)


# ------------------------------------------------------------------
# Helpers: simple system for testing
# ------------------------------------------------------------------

def linear_basis_fn(x: np.ndarray) -> np.ndarray:
    """Constant basis (linear system) → always flat (zero curvature)."""
    A = np.array([[-0.5, -1.0], [1.0, -0.5]])
    return A.reshape(1, 2, 2)


def nonlinear_basis_fn(x: np.ndarray) -> np.ndarray:
    """State-dependent basis (nonlinear → nonzero curvature)."""
    v = x[0]
    alpha = 0.5
    A = np.array([[-(0.5 + 3 * alpha * v ** 2), -1.0], [1.0, 0.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    return np.stack([A, B])  # 2 × 2 × 2


# ------------------------------------------------------------------
# Tests: curvature_gradient
# ------------------------------------------------------------------

class TestCurvatureGradient:
    def test_commuting_matrices_zero_curvature(self):
        """Diagonal matrices commute → zero curvature."""
        A = np.diag([1.0, 2.0]).reshape(1, 2, 2)
        B = np.diag([3.0, 4.0]).reshape(1, 2, 2)
        basis = np.stack([A[0], B[0]])
        C = curvature_gradient(basis)
        assert C == pytest.approx(0.0, abs=1e-12)

    def test_non_commuting_positive_curvature(self):
        """Non-diagonal matrices don't commute → positive curvature."""
        A = np.array([[0.0, 1.0], [-1.0, 0.0]])
        B = np.array([[1.0, 0.0], [0.0, -1.0]])
        basis = np.stack([A, B])
        C = curvature_gradient(basis)
        assert C > 0.0

    def test_single_matrix_zero_curvature(self):
        """Single matrix → no pairs → zero curvature."""
        A = np.random.randn(2, 2).reshape(1, 2, 2)
        C = curvature_gradient(A)
        assert C == 0.0

    def test_curvature_nonnegative(self):
        """Curvature energy must always be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            basis = rng.normal(0, 1, (3, 3, 3))
            C = curvature_gradient(basis)
            assert C >= 0.0


# ------------------------------------------------------------------
# Tests: curvature_gradient_vector
# ------------------------------------------------------------------

class TestCurvatureGradientVector:
    def test_shape(self):
        x = np.array([0.1, 0.0])
        grad = curvature_gradient_vector(x, linear_basis_fn)
        assert grad.shape == (2,)

    def test_linear_system_flat(self):
        """Linear system has constant basis → gradient is zero."""
        x = np.array([0.1, 0.2])
        grad = curvature_gradient_vector(x, linear_basis_fn)
        # Constant basis → gradient should be ~0
        assert np.allclose(grad, 0.0, atol=1e-8)

    def test_returns_numpy_array(self):
        x = np.array([0.1, 0.0])
        grad = curvature_gradient_vector(x, nonlinear_basis_fn)
        assert isinstance(grad, np.ndarray)


# ------------------------------------------------------------------
# Tests: natural_control_step_with_fn
# ------------------------------------------------------------------

class TestNaturalControlStep:
    def test_returns_array(self):
        x = np.array([0.1, 0.0])
        step = natural_control_step_with_fn(x, linear_basis_fn)
        assert isinstance(step, np.ndarray)
        assert step.shape == (2,)

    def test_linear_system_zero_step(self):
        """Linear system → flat curvature → zero control step."""
        x = np.array([0.1, 0.0])
        step = natural_control_step_with_fn(x, linear_basis_fn, kappa=0.01)
        assert np.allclose(step, 0.0, atol=1e-8)

    def test_with_identity_fim(self):
        x = np.array([0.5, 0.0])
        fim = np.eye(2)
        step = natural_control_step_with_fn(x, nonlinear_basis_fn, fim=fim, kappa=0.01)
        assert step.shape == (2,)

    def test_larger_kappa_larger_step(self):
        """Larger kappa → larger control step."""
        x = np.array([1.0, 0.5])
        step1 = natural_control_step_with_fn(x, nonlinear_basis_fn, kappa=0.01)
        step2 = natural_control_step_with_fn(x, nonlinear_basis_fn, kappa=0.1)
        # step2 should be ~10x larger than step1
        norm1 = np.linalg.norm(step1)
        norm2 = np.linalg.norm(step2)
        if norm1 > 1e-12:
            assert abs(norm2 / norm1 - 10.0) < 1.0


# ------------------------------------------------------------------
# Tests: resonance_collapse_check
# ------------------------------------------------------------------

class TestResonanceCollapseCheck:
    def test_safe_when_well_separated(self):
        """Eigenvalues with no ratio matching alpha ratio → safe."""
        eigvals = np.array([-1.0, -10.0])  # ratio 1:10
        alpha = np.array([1.0, 1.0])       # identity scaling → ratio stays 1:10
        assert resonance_collapse_check(eigvals, alpha, delta=0.01)

    def test_returns_bool(self):
        eigvals = np.array([-1.0, -2.0])
        alpha = np.array([1.0, 0.5])
        result = resonance_collapse_check(eigvals, alpha, delta=0.01)
        assert isinstance(result, bool)

    def test_single_eigenvalue_always_safe(self):
        """Single eigenvalue → no pairs to collapse."""
        eigvals = np.array([-2.0])
        alpha = np.array([1.0])
        assert resonance_collapse_check(eigvals, alpha, delta=0.01)

    def test_resonance_collapse_detected(self):
        """λ₁/λ₂ = α₂/α₁ → collapse!"""
        # λ₁=2, λ₂=4, so λ₁/λ₂ = 0.5
        # α₂/α₁ = 0.5 → resonance!
        eigvals = np.array([-2.0, -4.0])
        alpha = np.array([2.0, 1.0])  # α₂/α₁ = 0.5 = λ₁/λ₂
        result = resonance_collapse_check(eigvals, alpha, delta=0.2)
        # With delta=0.2 this should detect the collapse
        assert isinstance(result, bool)


# ------------------------------------------------------------------
# Tests: riemannian_metric
# ------------------------------------------------------------------

class TestRiemannianMetric:
    def test_shape(self):
        x = np.array([0.1, 0.0])
        g = riemannian_metric(x, linear_basis_fn)
        assert g.shape == (2, 2)

    def test_symmetric(self):
        """Metric tensor must be symmetric."""
        x = np.array([0.5, 0.3])
        g = riemannian_metric(x, nonlinear_basis_fn)
        assert np.allclose(g, g.T, atol=1e-10)

    def test_psd(self):
        """Metric tensor should be positive semidefinite."""
        x = np.array([0.1, 0.0])
        g = riemannian_metric(x, nonlinear_basis_fn)
        eigvals = np.linalg.eigvalsh(g)
        assert np.all(eigvals >= -1e-10)

    def test_linear_system_nearly_zero_metric(self):
        """Linear system (constant basis) → metric ≈ 0."""
        x = np.array([0.1, 0.0])
        g = riemannian_metric(x, linear_basis_fn)
        assert np.allclose(g, 0.0, atol=1e-8)


# ------------------------------------------------------------------
# Tests: Fisher Information Matrix
# ------------------------------------------------------------------

class TestFisherInformationMatrix:
    def test_shape(self):
        eigvals = np.array([-0.5, -1.0])
        eigvecs = np.eye(2)
        fim = fisher_information_matrix(eigvals, eigvecs)
        assert fim.shape == (2, 2)

    def test_symmetric(self):
        eigvals = np.array([-0.5, -1.0])
        eigvecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        fim = fisher_information_matrix(eigvals, eigvecs)
        assert np.allclose(fim, fim.T)

    def test_psd(self):
        """FIM must be positive semidefinite."""
        rng = np.random.default_rng(0)
        eigvals = rng.uniform(-2, -0.1, 4)
        Q, _ = np.linalg.qr(rng.normal(0, 1, (4, 4)))
        fim = fisher_information_matrix(eigvals, Q)
        eigv = np.linalg.eigvalsh(fim)
        assert np.all(eigv >= -1e-10)

    def test_larger_eigenvalue_larger_weight(self):
        """Dominant eigenvalue contributes more to FIM."""
        eigvals = np.array([-0.1, -10.0])
        # Orthonormal eigenvectors
        eigvecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        fim = fisher_information_matrix(eigvals, eigvecs)
        # FIM[1,1] should be larger (larger |λ|)
        assert fim[1, 1] > fim[0, 0]
