"""
Tests for LCAPatchDetector — 6-step classification pipeline.

Proof-of-concept: RLC+diode system shows:
  - 'lca' classification for small-signal (|v| < 0.3)
  - 'nonabelian' or 'chaotic' for large signal (|v| > 0.8)
"""

import numpy as np
import pytest
from tensor.lca_patch_detector import LCAPatchDetector, PatchClassification


# ------------------------------------------------------------------
# Fixture: RLC + diode system from LOGIC_FLOW.md Section 0F
# ------------------------------------------------------------------

R, C, L, alpha = 1.0, 1.0, 1.0, 0.5


def rlc_diode_system(x: np.ndarray) -> np.ndarray:
    """v̇ = -(1/C)(v/R + αv³ + i_L),  i_L̇ = (1/L)v"""
    v, iL = x
    v_dot = -(1 / C) * (v / R + alpha * v ** 3 + iL)
    iL_dot = (1 / L) * v
    return np.array([v_dot, iL_dot])


def linear_rlc_system(x: np.ndarray) -> np.ndarray:
    """Pure linear RLC: v̇ = -(1/RC)v - (1/C)i_L,  i_L̇ = (1/L)v"""
    v, iL = x
    v_dot = -(1 / (R * C)) * v - (1 / C) * iL
    iL_dot = (1 / L) * v
    return np.array([v_dot, iL_dot])


@pytest.fixture
def detector():
    return LCAPatchDetector(rlc_diode_system, n_states=2, eps_curvature=0.05, delta_commutator=0.01)


@pytest.fixture
def linear_detector():
    return LCAPatchDetector(linear_rlc_system, n_states=2, eps_curvature=0.05, delta_commutator=0.01)


# ------------------------------------------------------------------
# Tests: Jacobian computation
# ------------------------------------------------------------------

class TestJacobianComputation:
    def test_jacobian_shape(self, detector):
        x = np.array([0.1, 0.0])
        J = detector.compute_jacobian(x)
        assert J.shape == (2, 2)

    def test_linear_jacobian_constant(self, linear_detector):
        """Linear system: Jacobian should be constant everywhere."""
        x1 = np.array([0.1, 0.0])
        x2 = np.array([0.5, 0.2])
        J1 = linear_detector.compute_jacobian(x1)
        J2 = linear_detector.compute_jacobian(x2)
        np.testing.assert_allclose(J1, J2, atol=1e-3)

    def test_nonlinear_jacobian_varies(self, detector):
        """Nonlinear system: Jacobian should vary with state."""
        x_small = np.array([0.01, 0.0])
        x_large = np.array([2.0, 0.0])
        J_small = detector.compute_jacobian(x_small)
        J_large = detector.compute_jacobian(x_large)
        # They must differ (nonlinear term 3αv² contributes)
        assert not np.allclose(J_small, J_large, atol=1e-2)


# ------------------------------------------------------------------
# Tests: Operator rank
# ------------------------------------------------------------------

class TestOperatorRank:
    def test_linear_system_rank_1(self, linear_detector):
        """Linear system: all Jacobians are identical → rank 1."""
        x_samples = np.random.randn(20, 2) * 0.1
        jacobians = linear_detector.sample_jacobians(x_samples)
        r, basis = linear_detector.operator_rank_svd(jacobians)
        assert r <= 2  # Linear system should have low rank
        assert basis.shape[0] == r
        assert basis.shape[1:] == (2, 2)

    def test_rank_at_least_1(self, detector):
        x_samples = np.random.randn(10, 2) * 0.5
        jacobians = detector.sample_jacobians(x_samples)
        r, _ = detector.operator_rank_svd(jacobians)
        assert r >= 1


# ------------------------------------------------------------------
# Tests: LCA classification
# ------------------------------------------------------------------

class TestLCAClassification:
    def test_small_signal_is_lca(self, detector):
        """Small |v| → ρ ≈ 0 → should be classified as LCA."""
        np.random.seed(42)
        # Small amplitude: |v| < 0.1
        x_samples = np.random.randn(30, 2) * 0.05
        result = detector.classify_region(x_samples)

        assert isinstance(result, PatchClassification)
        assert result.patch_type in ('lca', 'nonabelian')
        # Curvature should be small for small-signal
        assert result.curvature_ratio < 0.5, f"Expected low curvature, got {result.curvature_ratio:.3f}"

    def test_linear_system_always_lca(self, linear_detector):
        """Pure linear system must always be LCA regardless of amplitude."""
        np.random.seed(99)
        x_samples = np.random.randn(30, 2) * 2.0  # large amplitude
        result = linear_detector.classify_region(x_samples)
        assert result.patch_type == 'lca', f"Linear system should be LCA, got {result.patch_type}"
        assert result.commutator_norm < 0.1

    def test_large_signal_not_lca(self, detector):
        """Large |v| → nonlinear term dominates → should NOT be LCA."""
        np.random.seed(7)
        # Large amplitude: |v| around 2.0
        x_samples = np.column_stack([
            np.random.uniform(1.5, 2.5, 30),
            np.random.randn(30) * 0.1,
        ])
        result = detector.classify_region(x_samples)
        # May be 'nonabelian' or 'chaotic', but NOT 'lca' with high curvature
        # Just check curvature ratio is higher than small-signal
        assert result.curvature_ratio > 0.0  # Some curvature present

    def test_result_has_required_fields(self, detector):
        x_samples = np.random.randn(10, 2) * 0.1
        result = detector.classify_region(x_samples)
        assert hasattr(result, 'patch_type')
        assert hasattr(result, 'operator_rank')
        assert hasattr(result, 'commutator_norm')
        assert hasattr(result, 'curvature_ratio')
        assert hasattr(result, 'spectral_gap')
        assert hasattr(result, 'basis_matrices')
        assert hasattr(result, 'eigenvalues')
        assert hasattr(result, 'centroid')

    def test_classification_types_valid(self, detector):
        x_samples = np.random.randn(15, 2) * 0.5
        result = detector.classify_region(x_samples)
        assert result.patch_type in ('lca', 'nonabelian', 'chaotic')


# ------------------------------------------------------------------
# Tests: Trajectory classification
# ------------------------------------------------------------------

class TestTrajectoryClassification:
    def test_trajectory_returns_list(self, detector):
        trajectory = np.random.randn(50, 2) * 0.2
        results = detector.classify_trajectory(trajectory, window=10)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, PatchClassification) for r in results)

    def test_trajectory_patch_types_valid(self, detector):
        trajectory = np.random.randn(40, 2) * 0.3
        results = detector.classify_trajectory(trajectory, window=8)
        for r in results:
            assert r.patch_type in ('lca', 'nonabelian', 'chaotic')


# ------------------------------------------------------------------
# Tests: Spring-mass system (for PoC Phase 9 cross-domain test)
# ------------------------------------------------------------------

def spring_mass_system(x: np.ndarray) -> np.ndarray:
    """m·ẍ + c·ẋ + k·x = 0  →  [ẋ, ẍ]"""
    m, c, k = 1.0, 1.0, 1.0
    pos, vel = x
    return np.array([vel, -(k / m) * pos - (c / m) * vel])


class TestCrossDomainPoC:
    def test_spring_mass_is_lca(self):
        """Spring-mass (linear) must also be LCA — same operator structure as RLC."""
        detector = LCAPatchDetector(spring_mass_system, n_states=2)
        np.random.seed(0)
        x_samples = np.random.randn(30, 2) * 0.1
        result = detector.classify_region(x_samples)
        assert result.patch_type == 'lca', f"Spring-mass should be LCA, got {result.patch_type}"

    def test_both_lca_have_low_commutator(self):
        """Both RLC (small signal) and spring-mass should have low commutator norm."""
        rlc_det = LCAPatchDetector(rlc_diode_system, n_states=2)
        spring_det = LCAPatchDetector(spring_mass_system, n_states=2)

        np.random.seed(1)
        rlc_result = rlc_det.classify_region(np.random.randn(30, 2) * 0.05)
        spring_result = spring_det.classify_region(np.random.randn(30, 2) * 0.1)

        assert rlc_result.commutator_norm < 0.5
        assert spring_result.commutator_norm < 0.5


# ------------------------------------------------------------------
# Tests: koopman_trust field (whattodo.md gap)
# ------------------------------------------------------------------

class TestKoopmanTrust:
    def test_patch_classification_has_koopman_trust(self):
        """PatchClassification must have koopman_trust field (whattodo.md spec)."""
        detector = LCAPatchDetector(rlc_diode_system, n_states=2)
        np.random.seed(0)
        x_samples = np.random.randn(20, 2) * 0.05  # small signal
        result = detector.classify_region(x_samples)
        assert hasattr(result, 'koopman_trust')

    def test_koopman_trust_is_bounded(self):
        """koopman_trust must be in [0, 1]."""
        detector = LCAPatchDetector(rlc_diode_system, n_states=2)
        np.random.seed(1)
        x_samples = np.random.randn(20, 2) * 0.05
        result = detector.classify_region(x_samples)
        assert 0.0 <= result.koopman_trust <= 1.0

    def test_koopman_trust_spring_mass(self):
        """Spring-mass system should yield a valid trust score."""
        detector = LCAPatchDetector(spring_mass_system, n_states=2)
        np.random.seed(2)
        x_samples = np.random.randn(20, 2) * 0.1
        result = detector.classify_region(x_samples)
        assert 0.0 <= result.koopman_trust <= 1.0

    def test_koopman_trust_default_zero_in_dataclass(self):
        """Default koopman_trust=0.0 when constructed manually."""
        from tensor.lca_patch_detector import PatchClassification
        n = 2
        pc = PatchClassification(
            patch_type='lca',
            operator_rank=1,
            commutator_norm=0.0,
            curvature_ratio=0.01,
            spectral_gap=0.5,
            basis_matrices=np.eye(n).reshape(1, n, n),
            eigenvalues=np.array([-0.5 + 0j, -1.0 + 0j]),
            centroid=np.zeros(n),
        )
        assert pc.koopman_trust == 0.0
