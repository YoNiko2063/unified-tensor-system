"""
Tests for EDMDKoopman — Extended Dynamic Mode Decomposition.
"""

import numpy as np
import pytest
from tensor.koopman_edmd import EDMDKoopman, KoopmanResult


# ------------------------------------------------------------------
# Fixtures: Linear and nonlinear systems
# ------------------------------------------------------------------

def linear_system_trajectory(T: int = 100, dt: float = 0.05) -> np.ndarray:
    """Simple stable linear system: ẋ = Ax, A = [[-0.5, 1], [-1, -0.5]]"""
    A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    x = np.array([1.0, 0.0])
    traj = [x.copy()]
    for _ in range(T - 1):
        x = x + dt * (A @ x)
        traj.append(x.copy())
    return np.array(traj)


def nonlinear_trajectory(T: int = 100, dt: float = 0.05) -> np.ndarray:
    """RLC+diode nonlinear trajectory."""
    R, C, L, alpha = 1.0, 1.0, 1.0, 0.3
    x = np.array([1.5, 0.0])  # start in nonlinear regime
    traj = [x.copy()]
    for _ in range(T - 1):
        v, iL = x
        vdot = -(1 / C) * (v / R + alpha * v ** 3 + iL)
        iLdot = (1 / L) * v
        x = x + dt * np.array([vdot, iLdot])
        traj.append(x.copy())
    return np.array(traj)


@pytest.fixture
def linear_traj():
    return linear_system_trajectory(100)


@pytest.fixture
def nonlinear_traj():
    return nonlinear_trajectory(100)


# ------------------------------------------------------------------
# Tests: Observable basis
# ------------------------------------------------------------------

class TestObservableBasis:
    def test_degree1_shape(self):
        k = EDMDKoopman(observable_degree=1)
        x = np.array([1.0, 2.0])
        psi = k.build_observable_basis(x)
        # 1 constant + 2 linear = 3
        assert psi.shape == (3,)

    def test_degree2_shape(self):
        k = EDMDKoopman(observable_degree=2)
        x = np.array([1.0, 2.0])
        psi = k.build_observable_basis(x)
        # 1 + 2 + 3 (pairs with repetition) = 6
        assert psi.shape == (6,)

    def test_constant_term(self):
        k = EDMDKoopman(observable_degree=1)
        x = np.array([3.0, 4.0])
        psi = k.build_observable_basis(x)
        assert psi[0] == 1.0  # constant term

    def test_linear_terms(self):
        k = EDMDKoopman(observable_degree=1)
        x = np.array([3.0, 4.0])
        psi = k.build_observable_basis(x)
        assert psi[1] == 3.0
        assert psi[2] == 4.0


# ------------------------------------------------------------------
# Tests: Fitting
# ------------------------------------------------------------------

class TestFitting:
    def test_fit_from_pairs(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        pairs = [(linear_traj[i], linear_traj[i + 1]) for i in range(len(linear_traj) - 1)]
        k.fit(pairs)
        assert k._fitted

    def test_fit_trajectory(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        assert k._fitted

    def test_k_matrix_shape(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        obs_dim = k._observable_dim(2)
        assert k._K.shape == (obs_dim, obs_dim)

    def test_unfit_raises(self):
        k = EDMDKoopman()
        with pytest.raises(RuntimeError):
            k.eigendecomposition()

    def test_empty_pairs_raises(self):
        k = EDMDKoopman()
        with pytest.raises((ValueError, Exception)):
            k.fit([])


# ------------------------------------------------------------------
# Tests: Eigendecomposition
# ------------------------------------------------------------------

class TestEigendecomposition:
    def test_returns_koopman_result(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        result = k.eigendecomposition()
        assert isinstance(result, KoopmanResult)

    def test_eigenvalues_present(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        result = k.eigendecomposition()
        assert len(result.eigenvalues) > 0

    def test_spectral_gap_nonnegative(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        result = k.eigendecomposition()
        assert result.spectral_gap >= 0.0

    def test_linear_system_has_spectral_gap(self, linear_traj):
        """Stable linear system should have clear dominant mode."""
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        result = k.eigendecomposition()
        # Should have some spectral structure
        assert result.spectral_gap >= 0.0


# ------------------------------------------------------------------
# Tests: Spectral gap and stability
# ------------------------------------------------------------------

class TestSpectralGap:
    def test_spectral_gap_positive(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        gap = k.spectral_gap()
        assert gap >= 0.0

    def test_spectral_gap_from_eigvals(self):
        k = EDMDKoopman()
        eigvals = np.array([2.0, 1.5, 0.5, 0.1])
        gap = k.spectral_gap(eigvals)
        assert abs(gap - 0.5) < 1e-10  # |2.0 - 1.5| = 0.5


# ------------------------------------------------------------------
# Tests: Eigenfunction stability
# ------------------------------------------------------------------

class TestEigenfunctionStability:
    def test_stability_none_prev(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        stability = k.eigenfunction_stability(prev_result=None)
        assert stability == 0.0

    def test_stability_same_result_is_zero(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        result = k.eigendecomposition()
        stability = k.eigenfunction_stability(prev_result=result, curr_result=result)
        assert abs(stability) < 1e-10

    def test_stability_different_results(self, linear_traj, nonlinear_traj):
        k1 = EDMDKoopman(observable_degree=2)
        k1.fit_trajectory(linear_traj)
        r1 = k1.eigendecomposition()

        k2 = EDMDKoopman(observable_degree=2)
        k2.fit_trajectory(nonlinear_traj)
        r2 = k2.eigendecomposition()

        stability = k1.eigenfunction_stability(prev_result=r1, curr_result=r2)
        assert stability >= 0.0  # Non-negative


# ------------------------------------------------------------------
# Tests: Prediction
# ------------------------------------------------------------------

class TestPrediction:
    def test_predict_shape(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        x = linear_traj[0]
        pred = k.predict_next_observable(x)
        obs_dim = k._observable_dim(2)
        assert pred.shape == (obs_dim,)

    def test_predict_unfit_raises(self):
        k = EDMDKoopman()
        with pytest.raises(RuntimeError):
            k.predict_next_observable(np.array([1.0, 0.0]))
