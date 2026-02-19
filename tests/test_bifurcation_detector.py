"""
Tests for BifurcationDetector — eigenvalue-crossing boundary detection.
"""

import numpy as np
import pytest
from tensor.bifurcation_detector import BifurcationDetector, BifurcationStatus


@pytest.fixture
def detector():
    return BifurcationDetector(zero_tol=1e-3)


# ------------------------------------------------------------------
# Tests: Basic status detection
# ------------------------------------------------------------------

class TestBasicDetection:
    def test_stable_eigenvalues(self, detector):
        """All eigenvalues well into left half-plane → stable."""
        eigvals = np.array([-1.0, -2.0])
        result = detector.check(eigvals)
        assert result.status == 'stable'
        assert isinstance(result, BifurcationStatus)

    def test_critical_eigenvalue(self, detector):
        """Eigenvalue near zero → critical."""
        eigvals = np.array([-1.0, 0.0005])  # just outside zero_tol
        detector.reset()
        result = detector.check(eigvals)
        assert result.status == 'critical'

    def test_bifurcation_detected(self, detector):
        """Sign change from prev to curr → bifurcation."""
        detector.reset()
        eigvals_before = np.array([-0.5, -0.2])
        eigvals_after = np.array([-0.5, 0.2])  # second eigenvalue crossed zero

        detector.check(eigvals_before)
        result = detector.check(eigvals_after)
        assert result.status == 'bifurcation'

    def test_returns_correct_type(self, detector):
        eigvals = np.array([-1.0 + 0.5j, -1.0 - 0.5j])
        result = detector.check(eigvals)
        assert isinstance(result, BifurcationStatus)


# ------------------------------------------------------------------
# Tests: Feature vector
# ------------------------------------------------------------------

class TestFeatureVector:
    def test_shape(self, detector):
        eigvals = np.array([-0.5, -1.0])
        fv = detector.feature_vector(eigvals)
        assert fv.shape == (4,)

    def test_dtype(self, detector):
        eigvals = np.array([-0.5, -1.0])
        fv = detector.feature_vector(eigvals)
        assert fv.dtype == np.float32

    def test_min_real_is_first(self, detector):
        eigvals = np.array([-0.3, -1.0])
        fv = detector.feature_vector(eigvals)
        # First element = min real part = -1.0
        assert abs(fv[0] - (-1.0)) < 1e-5

    def test_derivative_with_prev(self, detector):
        prev_eigvals = np.array([-1.0, -0.5])
        curr_eigvals = np.array([-0.8, -0.3])
        fv = detector.feature_vector(curr_eigvals, prev_eigvals=prev_eigvals)
        # Derivative should be positive (min_real moved from -1.0 to -0.8)
        assert fv[2] > 0  # derivative is positive

    def test_imag_magnitude(self, detector):
        eigvals = np.array([-0.5 + 2.0j, -0.5 - 2.0j])
        fv = detector.feature_vector(eigvals)
        assert abs(fv[3] - 2.0) < 1e-5


# ------------------------------------------------------------------
# Tests: Distance to boundary (neural training target)
# ------------------------------------------------------------------

class TestDistanceToBoundary:
    def test_stable_system_has_distance(self, detector):
        eigvals = np.array([-0.5, -1.0])
        d = detector.distance_to_boundary(eigvals)
        assert abs(d - 0.5) < 1e-10  # min |Re(λᵢ)| = 0.5

    def test_near_zero_has_small_distance(self, detector):
        eigvals = np.array([-0.001, -1.0])
        d = detector.distance_to_boundary(eigvals)
        assert d < 0.01

    def test_distance_nonnegative(self, detector):
        for _ in range(10):
            eigvals = np.random.randn(4)
            d = detector.distance_to_boundary(eigvals)
            assert d >= 0.0


# ------------------------------------------------------------------
# Tests: Bifurcation type classification
# ------------------------------------------------------------------

class TestBifurcationTypes:
    def test_hopf_detected(self):
        """Complex pair crossing imaginary axis → Hopf."""
        det = BifurcationDetector(zero_tol=0.1)
        # Before: stable complex pair
        det.check(np.array([-0.1 + 1.0j, -0.1 - 1.0j]))
        # After: unstable complex pair
        result = det.check(np.array([0.1 + 1.0j, 0.1 - 1.0j]))
        assert result.bifurcation_type == 'hopf'

    def test_saddle_node_detected(self):
        """Real eigenvalue crossing zero → saddle-node."""
        det = BifurcationDetector(zero_tol=0.05)
        det.check(np.array([-0.5, -0.1]))
        result = det.check(np.array([-0.5, 0.1]))
        assert result.bifurcation_type == 'saddle_node'

    def test_no_bifurcation_type_when_stable(self, detector):
        detector.reset()
        result = detector.check(np.array([-1.0, -2.0]))
        assert result.bifurcation_type == 'none'


# ------------------------------------------------------------------
# Tests: History and reset
# ------------------------------------------------------------------

class TestHistoryAndReset:
    def test_history_accumulates(self, detector):
        detector.reset()
        for _ in range(5):
            detector.check(np.array([-0.5, -1.0]))
        assert len(detector.history()) == 5

    def test_reset_clears_history(self, detector):
        detector.check(np.array([-0.5, -1.0]))
        detector.reset()
        assert len(detector.history()) == 0

    def test_reset_clears_prev_state(self):
        det = BifurcationDetector()
        det.check(np.array([-0.5, -1.0]))
        det.reset()
        # After reset, no bifurcation can be detected (no previous state)
        result = det.check(np.array([0.5, 1.0]))  # sign change from "nothing"
        assert result.status != 'bifurcation'


# ------------------------------------------------------------------
# Tests: Spectral gap field
# ------------------------------------------------------------------

class TestSpectralGapField:
    def test_spectral_gap_nonnegative(self, detector):
        eigvals = np.array([-2.0, -1.0, -0.5])
        result = detector.check(eigvals)
        assert result.spectral_gap >= 0.0

    def test_spectral_gap_from_result(self, detector):
        detector.reset()
        eigvals = np.array([-2.0, -0.5])
        result = detector.check(eigvals)
        # |Re(-2.0)| = 2.0, |Re(-0.5)| = 0.5, gap = 2.0 - 0.5 = 1.5
        assert abs(result.spectral_gap - 1.5) < 1e-10
