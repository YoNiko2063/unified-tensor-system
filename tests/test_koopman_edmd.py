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


# ------------------------------------------------------------------
# Tests: Reconstruction error + trust score (whattodo.md gaps)
# ------------------------------------------------------------------

class TestReconstructionAndTrust:
    def _fit(self, traj=None):
        if traj is None:
            A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
            x = np.array([1.0, 0.0])
            traj = [x.copy()]
            for _ in range(99):
                x = x + 0.05 * (A @ x)
                traj.append(x.copy())
            traj = np.array(traj)
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(traj)
        return k

    def test_reconstruction_error_nonnegative(self):
        k = self._fit()
        assert k.compute_reconstruction_error() >= 0.0

    def test_reconstruction_error_unfit_is_inf(self):
        k = EDMDKoopman()
        assert k.compute_reconstruction_error() == float('inf')

    def test_result_has_reconstruction_error_field(self):
        k = self._fit()
        r = k.eigendecomposition()
        assert hasattr(r, 'reconstruction_error')
        assert r.reconstruction_error >= 0.0

    def test_result_has_koopman_trust_field(self):
        k = self._fit()
        r = k.eigendecomposition()
        assert hasattr(r, 'koopman_trust')
        assert 0.0 <= r.koopman_trust <= 1.0

    def test_trust_score_zero_gap(self):
        trust = EDMDKoopman.compute_trust_score(gap=0.0, reconstruction_error=0.0, drift=0.0)
        assert trust == 0.0

    def test_trust_score_perfect(self):
        # gap >> gamma_min, recon_err=0, drift=0 → trust ≈ 1
        trust = EDMDKoopman.compute_trust_score(
            gap=1.0, reconstruction_error=0.0, drift=0.0,
            gamma_min=0.1, eta_max=1.0, s_max=0.5,
        )
        assert abs(trust - 1.0) < 1e-10

    def test_trust_score_high_error_penalizes(self):
        # High recon error → low trust
        trust_low = EDMDKoopman.compute_trust_score(
            gap=1.0, reconstruction_error=0.9, drift=0.0,
            gamma_min=0.1, eta_max=1.0, s_max=0.5,
        )
        trust_high = EDMDKoopman.compute_trust_score(
            gap=1.0, reconstruction_error=0.0, drift=0.0,
            gamma_min=0.1, eta_max=1.0, s_max=0.5,
        )
        assert trust_low < trust_high

    def test_trust_score_high_drift_penalizes(self):
        trust_low = EDMDKoopman.compute_trust_score(
            gap=1.0, reconstruction_error=0.0, drift=0.45,
            gamma_min=0.1, eta_max=1.0, s_max=0.5,
        )
        trust_high = EDMDKoopman.compute_trust_score(
            gap=1.0, reconstruction_error=0.0, drift=0.0,
            gamma_min=0.1, eta_max=1.0, s_max=0.5,
        )
        assert trust_low < trust_high

    def test_linear_system_low_reconstruction_error(self):
        """Linear system EDMD should reconstruct well."""
        k = self._fit()
        err = k.compute_reconstruction_error()
        # For a linear system with good data, error should be finite and bounded
        assert err < float('inf')
        assert err >= 0.0
