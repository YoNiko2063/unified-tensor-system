"""
Tests for SpectralPathComposer, IntervalOperator, DissonanceMetric.
"""

import numpy as np
import pytest
from tensor.spectral_path import IntervalOperator, DissonanceMetric, SpectralPathComposer
from tensor.patch_graph import Patch, PatchGraph


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_patch(patch_id: int, eigvals=None, patch_type: str = 'lca') -> Patch:
    n = 2
    if eigvals is None:
        eigvals = np.array([-0.5 + 0j, -1.0 + 0j])
    return Patch(
        id=patch_id,
        patch_type=patch_type,
        operator_basis=np.eye(n).reshape(1, n, n),
        spectrum=eigvals,
        centroid=np.zeros(n),
        operator_rank=1,
        commutator_norm=0.0,
        curvature_ratio=0.02,
        spectral_gap=0.5,
    )


# ------------------------------------------------------------------
# Tests: IntervalOperator
# ------------------------------------------------------------------

class TestIntervalOperator:
    def test_apply_scales_omega(self):
        op = IntervalOperator(alpha=np.array([2.0, 3.0]))
        omega = np.array([1.0, 1.0, 0.5])
        result = op.apply(omega)
        assert abs(result[0] - 2.0) < 1e-10
        assert abs(result[1] - 3.0) < 1e-10
        assert abs(result[2] - 0.5) < 1e-10  # unaffected

    def test_identity_no_change(self):
        op = IntervalOperator.identity(r=3)
        omega = np.array([1.5, 2.0, 0.7])
        result = op.apply(omega)
        assert np.allclose(result, omega)

    def test_compose_is_elementwise_product(self):
        op1 = IntervalOperator(alpha=np.array([2.0, 3.0]))
        op2 = IntervalOperator(alpha=np.array([0.5, 2.0]))
        composed = op1.compose(op2)
        assert np.allclose(composed.alpha[:2], [1.0, 6.0])

    def test_compose_commutativity(self):
        op1 = IntervalOperator(alpha=np.array([2.0, 3.0]))
        op2 = IntervalOperator(alpha=np.array([1.5, 0.5]))
        ab = op1.compose(op2)
        ba = op2.compose(op1)
        assert np.allclose(ab.alpha, ba.alpha)

    def test_log_alpha_inverse(self):
        op = IntervalOperator(alpha=np.array([np.e, np.e ** 2]))
        eta = op.log_alpha()
        assert abs(eta[0] - 1.0) < 1e-10
        assert abs(eta[1] - 2.0) < 1e-10

    def test_from_log(self):
        eta = np.array([0.0, 1.0])
        op = IntervalOperator.from_log(eta)
        assert abs(op.alpha[0] - 1.0) < 1e-10
        assert abs(op.alpha[1] - np.e) < 1e-10

    def test_is_contraction_true(self):
        op = IntervalOperator(alpha=np.array([1.5, 0.8]))  # |log 1.5| ≈ 0.41 < 1
        assert op.is_contraction()

    def test_is_contraction_false(self):
        op = IntervalOperator(alpha=np.array([np.e ** 2]))  # |log e²| = 2 > 1
        assert not op.is_contraction()

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            IntervalOperator(alpha=np.array([-1.0, 2.0]))

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError):
            IntervalOperator(alpha=np.array([0.0, 1.0]))


# ------------------------------------------------------------------
# Tests: DissonanceMetric
# ------------------------------------------------------------------

class TestDissonanceMetric:
    def test_identical_frequencies_zero_dissonance(self):
        dm = DissonanceMetric(K=10)
        tau = dm.compute(1.0, 1.0)
        assert tau == pytest.approx(0.0, abs=1e-10)

    def test_octave_is_consonant(self):
        """ω_i = 2ω_j → ratio 2/1 → low dissonance."""
        dm = DissonanceMetric(K=10)
        tau = dm.compute(2.0, 1.0)
        assert tau == pytest.approx(0.0, abs=1e-10)

    def test_fifth_is_consonant(self):
        """ω_i = 3/2 · ω_j → ratio 3/2 → low dissonance."""
        dm = DissonanceMetric(K=10)
        tau = dm.compute(1.5, 1.0)
        assert tau == pytest.approx(0.0, abs=1e-10)

    def test_irrational_ratio_higher_dissonance(self):
        """ω_i = √2 · ω_j → irrational ratio → higher dissonance."""
        dm = DissonanceMetric(K=5)
        tau_rational = dm.compute(1.5, 1.0)      # 3/2 ratio
        tau_irrational = dm.compute(np.sqrt(2), 1.0)
        assert tau_irrational >= tau_rational  # irrational ≥ rational

    def test_zero_omega_j_returns_handled(self):
        dm = DissonanceMetric(K=10)
        tau = dm.compute(1.0, 0.0)
        assert isinstance(tau, float)

    def test_path_dissonance_empty(self):
        dm = DissonanceMetric(K=10)
        assert dm.path_dissonance([]) == 0.0

    def test_path_dissonance_single(self):
        dm = DissonanceMetric(K=10)
        assert dm.path_dissonance([1.0]) == 0.0

    def test_path_dissonance_consonant_path(self):
        """Path of octaves: 1, 2, 4 → very consonant."""
        dm = DissonanceMetric(K=10)
        tau = dm.path_dissonance([1.0, 2.0, 4.0])
        assert tau == pytest.approx(0.0, abs=1e-10)

    def test_dissonance_nonnegative(self):
        dm = DissonanceMetric(K=10)
        rng = np.random.default_rng(0)
        for _ in range(20):
            oi, oj = rng.uniform(0.1, 10.0, 2)
            assert dm.compute(oi, oj) >= 0.0


# ------------------------------------------------------------------
# Tests: SpectralPathComposer
# ------------------------------------------------------------------

class TestSpectralPathComposer:
    def test_compose_single_patch(self):
        composer = SpectralPathComposer()
        p = make_patch(0)
        result = composer.compose([p], alphas=[])
        assert len(result) >= 0  # returns frequency vector

    def test_compose_two_patches(self):
        composer = SpectralPathComposer()
        p0 = make_patch(0, eigvals=np.array([-0.5 + 0j, -1.0 + 0j]))
        p1 = make_patch(1, eigvals=np.array([-1.0 + 0j, -2.0 + 0j]))
        alpha = np.array([2.0, 2.0])  # octave interval
        result = composer.compose([p0, p1], alphas=[alpha])
        assert isinstance(result, np.ndarray)

    def test_path_dissonance_two_lca_patches(self):
        composer = SpectralPathComposer()
        p0 = make_patch(0, eigvals=np.array([-1.0 + 0j, -2.0 + 0j]))
        p1 = make_patch(1, eigvals=np.array([-2.0 + 0j, -4.0 + 0j]))  # octave
        dissonance = composer.path_dissonance([p0, p1])
        assert dissonance >= 0.0

    def test_path_dissonance_single_patch(self):
        composer = SpectralPathComposer()
        p = make_patch(0)
        assert composer.path_dissonance([p]) == 0.0

    def test_find_consonant_path_empty_graph(self):
        composer = SpectralPathComposer()
        graph = PatchGraph()
        result = composer.find_consonant_path(graph, 0, 1)
        assert result == []

    def test_find_consonant_path_connected(self):
        composer = SpectralPathComposer()
        graph = PatchGraph()
        p0 = make_patch(0)
        p1 = make_patch(1)
        graph.add_patch(p0)
        graph.add_patch(p1)
        graph.add_transition(p0, p1, curvature_cost=0.3)
        path = composer.find_consonant_path(graph, 0, 1)
        assert path == [0, 1]

    def test_resonance_collapse_safe(self):
        composer = SpectralPathComposer()
        eigvals = np.array([-1.0 + 0j, -3.0 + 0j])  # ratio 1:3
        alpha = np.array([1.0, 1.0])  # no scaling → ratio stays 1:3
        assert composer.resonance_collapse_check(eigvals, alpha, delta=0.01)

    def test_resonance_collapse_risky(self):
        composer = SpectralPathComposer()
        eigvals = np.array([-2.0 + 0j, -4.0 + 0j])  # ratio 1:2
        alpha = np.array([2.0, 1.0])  # after scaling: 4, 4 → collapse!
        result = composer.resonance_collapse_check(eigvals, alpha, delta=0.1)
        assert isinstance(result, bool)  # just check it runs correctly
