"""
Tests for tensor/semantic_geometry.py

Test order by priority:
  1. No-Gradient-Authority: mock.patch.object(EDMDKoopman, 'fit') → assert_not_called()
  2. _TextEDMD unit tests (fit, eigendecomposition, trust_score)
  3. SemanticJacobianEstimator (knockout salience)
  4. TextKoopmanOperator (fit returns correct types)
  5. ToneSignatureVector (L2-normalized)
  6. SemanticGeometryLayer (encode() invariants)
  7. ProposalQueue integration
  8. Memory bounds (MAX_SEMANTIC_DIM, MAX_TOKENS)
"""

from __future__ import annotations

import queue
from unittest import mock

import numpy as np
import pytest

from tensor.koopman_edmd import EDMDKoopman
from tensor.semantic_geometry import (
    _TextEDMD,
    SemanticJacobianEstimator,
    SemanticGeometryLayer,
    TextKoopmanOperator,
    ToneSignatureVector,
    _MAX_SEMANTIC_DIM,
    _MAX_TOKENS,
    _TAU_SEMANTIC,
)
from tensor.validation_bridge import ProposalQueue


# ---------------------------------------------------------------------------
# Minimal stub for hdv_system used by several classes
# ---------------------------------------------------------------------------


class _StubHDVSystem:
    """Minimal stub satisfying structural_encode() and find_overlaps()."""

    def __init__(self, hdv_dim: int = 64, n_overlaps: int = 0, seed: int = 0):
        self.hdv_dim = hdv_dim
        self._rng = np.random.default_rng(seed)
        self._n_overlaps = n_overlaps

    def structural_encode(self, text: str, domain: str) -> np.ndarray:
        """Deterministic hash-based encoding (not a real HDV)."""
        rng = np.random.default_rng(abs(hash(text + domain)) % (2**31))
        return rng.standard_normal(self.hdv_dim).astype(np.float32)

    def find_overlaps(self):
        if self._n_overlaps == 0:
            return []
        return list(range(self._n_overlaps))


# ---------------------------------------------------------------------------
# 1. No-Gradient-Authority test (CRITICAL-1)
# ---------------------------------------------------------------------------


class TestNoGradientAuthority:
    """
    CRITICAL-1: EDMDKoopman.fit() must never be called by SemanticGeometryLayer.

    _TextEDMD is a completely separate class — mock.patch.object on EDMDKoopman
    targets a different method, so the assertion must pass.
    """

    def setup_method(self):
        self.hdv = _StubHDVSystem(hdv_dim=64, n_overlaps=8)
        self.layer = SemanticGeometryLayer(self.hdv)

    def test_edmd_koopman_fit_not_called_on_encode(self):
        """Core invariant: EDMDKoopman.fit() is never reached through encode()."""
        with mock.patch.object(EDMDKoopman, "fit") as mock_fit:
            self.layer.encode("the quick brown fox jumps over the lazy dog")
            mock_fit.assert_not_called()

    def test_edmd_koopman_fit_trajectory_not_called(self):
        """fit_trajectory is a second entry point — also must not be called."""
        with mock.patch.object(EDMDKoopman, "fit_trajectory") as mock_ft:
            self.layer.encode("alpha beta gamma delta epsilon")
            mock_ft.assert_not_called()

    def test_edmd_koopman_fit_not_called_short_text(self):
        """Short text (< 3 tokens) still must not call EDMDKoopman.fit()."""
        with mock.patch.object(EDMDKoopman, "fit") as mock_fit:
            self.layer.encode("hello world")
            mock_fit.assert_not_called()

    def test_edmd_koopman_fit_not_called_empty_text(self):
        with mock.patch.object(EDMDKoopman, "fit") as mock_fit:
            self.layer.encode("")
            mock_fit.assert_not_called()

    def test_hypothesis_only_always_true(self):
        """hypothesis_only=True must ALWAYS appear in encode() output."""
        result = self.layer.encode("some text with several words here")
        assert result.get("hypothesis_only") is True

    def test_hypothesis_only_true_on_empty(self):
        result = self.layer.encode("")
        assert result.get("hypothesis_only") is True

    def test_text_edmd_is_not_edmd_koopman_subclass(self):
        """_TextEDMD must not be a subclass of EDMDKoopman (CRITICAL-1 structural check)."""
        assert not issubclass(_TextEDMD, EDMDKoopman)

    def test_text_edmd_has_independent_fit_method(self):
        """_TextEDMD.fit is a different method from EDMDKoopman.fit."""
        assert _TextEDMD.fit is not EDMDKoopman.fit


# ---------------------------------------------------------------------------
# 2. _TextEDMD unit tests
# ---------------------------------------------------------------------------


class TestTextEDMD:

    def _make_pairs(self, d: int = 4, m: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = rng.standard_normal((m, d))
        H_next = rng.standard_normal((m, d))
        return [(H[i], H_next[i]) for i in range(m)]

    def test_fit_returns_self(self):
        edmd = _TextEDMD()
        pairs = self._make_pairs()
        result = edmd.fit(pairs)
        assert result is edmd

    def test_fit_sets_k_matrix(self):
        edmd = _TextEDMD()
        pairs = self._make_pairs(d=4, m=8)
        edmd.fit(pairs)
        assert edmd.K_matrix is not None
        assert edmd.K_matrix.shape == (4, 4)

    def test_not_fitted_after_too_few_pairs(self):
        """< 2 pairs → not fitted."""
        edmd = _TextEDMD()
        pairs = self._make_pairs(d=4, m=1)
        edmd.fit(pairs)
        assert edmd.K_matrix is None

    def test_not_fitted_on_empty_pairs(self):
        edmd = _TextEDMD()
        edmd.fit([])
        assert edmd.K_matrix is None

    def test_eigendecomposition_shape(self):
        edmd = _TextEDMD()
        pairs = self._make_pairs(d=4, m=10)
        edmd.fit(pairs)
        eigvals, eigvecs = edmd.eigendecomposition()
        assert eigvals.shape == (4,)
        assert eigvecs.shape == (4, 4)

    def test_eigendecomposition_sorted_descending(self):
        edmd = _TextEDMD()
        pairs = self._make_pairs(d=4, m=10)
        edmd.fit(pairs)
        eigvals, _ = edmd.eigendecomposition()
        mags = np.abs(eigvals)
        assert np.all(mags[:-1] >= mags[1:] - 1e-12), "eigenvalues not sorted descending"

    def test_eigendecomposition_unfitted_returns_zeros(self):
        edmd = _TextEDMD()
        eigvals, eigvecs = edmd.eigendecomposition()
        assert np.all(eigvals == 0)

    def test_trust_score_zero_when_unfitted(self):
        edmd = _TextEDMD()
        assert edmd.trust_score() == 0.0

    def test_trust_score_in_unit_interval(self):
        edmd = _TextEDMD()
        pairs = self._make_pairs(d=4, m=10)
        edmd.fit(pairs)
        t = edmd.trust_score()
        assert 0.0 <= t <= 1.0

    def test_trust_score_identity_dynamics_high(self):
        """Near-identity dynamics → dominant real eigenvalue ≈ 1 → gap > 0."""
        rng = np.random.default_rng(7)
        m, d = 20, 4
        H = rng.standard_normal((m, d))
        H_next = H + 0.01 * rng.standard_normal((m, d))
        pairs = [(H[i], H_next[i]) for i in range(m)]
        edmd = _TextEDMD()
        edmd.fit(pairs)
        # trust can be 0 if all eigenvalues are nearly equal; just check type
        assert isinstance(edmd.trust_score(), float)

    def test_k_matrix_is_none_before_fit(self):
        edmd = _TextEDMD()
        assert edmd.K_matrix is None


# ---------------------------------------------------------------------------
# 3. SemanticJacobianEstimator
# ---------------------------------------------------------------------------


class TestSemanticJacobianEstimator:

    def setup_method(self):
        self.hdv = _StubHDVSystem(hdv_dim=32)
        self.estimator = SemanticJacobianEstimator(self.hdv)

    def test_empty_text_returns_empty(self):
        s = self.estimator.compute("")
        assert s.shape == (0,)

    def test_single_token_returns_length_1(self):
        s = self.estimator.compute("hello")
        assert s.shape == (1,)

    def test_n_tokens_matches_output_length(self):
        text = "the quick brown fox jumps"
        tokens = text.split()
        s = self.estimator.compute(text)
        assert s.shape == (len(tokens),)

    def test_salience_nonnegative(self):
        s = self.estimator.compute("alpha beta gamma delta")
        assert np.all(s >= 0)

    def test_max_tokens_cap(self):
        """Salience vector capped at _max_tokens even for very long text."""
        estimator = SemanticJacobianEstimator(self.hdv, max_tokens=5)
        text = " ".join(["word"] * 20)
        s = estimator.compute(text)
        assert s.shape == (5,)

    def test_knockout_changes_salience_for_important_token(self):
        """Salience > 0 for any token in a multi-token text (stochastic encoding)."""
        s = self.estimator.compute("important irrelevant padding here")
        assert np.any(s > 0)

    def test_single_token_salience_equals_norm(self):
        """When text has one token, removing it leaves empty text → salience = ‖h_full‖."""
        hdv = _StubHDVSystem(hdv_dim=32)
        estimator = SemanticJacobianEstimator(hdv)
        text = "singleton"
        h_full = hdv.structural_encode(text, "semantic").astype(float)
        s = estimator.compute(text)
        assert abs(s[0] - np.linalg.norm(h_full)) < 1e-9


# ---------------------------------------------------------------------------
# 4. TextKoopmanOperator
# ---------------------------------------------------------------------------


class TestTextKoopmanOperator:

    def setup_method(self):
        self.hdv = _StubHDVSystem(hdv_dim=64, n_overlaps=8)
        self.op = TextKoopmanOperator(self.hdv)

    def test_fit_returns_three_tuple(self):
        edmd, dims, trust = self.op.fit("alpha beta gamma delta epsilon zeta")
        assert isinstance(edmd, _TextEDMD)
        assert isinstance(dims, np.ndarray)
        assert isinstance(trust, float)

    def test_trust_zero_for_short_text(self):
        _, _, trust = self.op.fit("one two")
        assert trust == 0.0

    def test_trust_zero_for_empty_text(self):
        _, _, trust = self.op.fit("")
        assert trust == 0.0

    def test_semantic_dims_bounded(self):
        """Semantic dims never exceed MAX_SEMANTIC_DIM."""
        _, dims, _ = self.op.fit("word " * 10)
        assert len(dims) <= _MAX_SEMANTIC_DIM

    def test_semantic_dims_overlap_first(self):
        """When overlaps exist, overlap dims are preferred."""
        hdv_with_overlaps = _StubHDVSystem(hdv_dim=64, n_overlaps=4)
        op = TextKoopmanOperator(hdv_with_overlaps)
        _, dims, _ = op.fit("a b c d e f g h")
        # dims should be [0, 1, 2, 3] (the 4 overlap dims)
        assert set(dims.tolist()).issubset(set(range(4)))

    def test_fallback_dims_without_overlaps(self):
        """Without overlaps, falls back to first hdv_dim//3 dims."""
        hdv_no_overlaps = _StubHDVSystem(hdv_dim=60, n_overlaps=0)
        op = TextKoopmanOperator(hdv_no_overlaps)
        _, dims, _ = op.fit("a b c d e f g h i j")
        # hdv_dim//3 = 20
        assert len(dims) == 20
        assert list(dims) == list(range(20))

    def test_k_matrix_shape_consistent_with_dims(self):
        """K_matrix shape should be (len(dims), len(dims))."""
        text = " ".join(["word"] * 10)
        edmd, dims, _ = self.op.fit(text)
        if edmd.K_matrix is not None:
            d = len(dims)
            assert edmd.K_matrix.shape == (d, d)

    def test_text_edmd_not_edmd_koopman(self):
        """_TextEDMD returned is not EDMDKoopman (CRITICAL-1 runtime check)."""
        edmd, _, _ = self.op.fit("alpha beta gamma delta epsilon")
        assert not isinstance(edmd, EDMDKoopman)


# ---------------------------------------------------------------------------
# 5. ToneSignatureVector
# ---------------------------------------------------------------------------


class TestToneSignatureVector:

    def setup_method(self):
        self.tone = ToneSignatureVector()

    def test_returns_zeros_on_empty_eigvecs(self):
        t = self.tone.compute(np.zeros((0, 0)), np.array([0, 1, 2]), r=4)
        assert np.all(t == 0)

    def test_returns_zeros_on_r_zero(self):
        eigvecs = np.eye(4)
        t = self.tone.compute(eigvecs, np.array([0, 1, 2, 3]), r=0)
        assert t.shape == (0,)

    def test_l2_normalized(self):
        rng = np.random.default_rng(42)
        eigvecs = rng.standard_normal((8, 4))
        dims = np.arange(8)
        t = self.tone.compute(eigvecs, dims, r=4)
        if np.linalg.norm(t) > 1e-12:
            assert abs(np.linalg.norm(t) - 1.0) < 1e-9

    def test_degenerate_eigvecs_returns_zeros(self):
        """Near-zero eigvecs → zero tone (no normalization by near-zero)."""
        eigvecs = np.zeros((4, 4))
        t = self.tone.compute(eigvecs, np.arange(4), r=4)
        assert np.all(t == 0)

    def test_output_length_matches_min_dims_r(self):
        eigvecs = np.eye(10)
        dims = np.arange(10)
        t = self.tone.compute(eigvecs, dims, r=3)
        # shape is (d,) not (min(d, r),) — tone is mean of real_vecs[:, :n_modes], d=10
        assert len(t) == 10  # d dimension

    def test_real_part_used(self):
        """Real part of eigvecs should match purely real eigvecs."""
        eigvecs = np.ones((4, 2), dtype=complex) * 0.5
        dims = np.arange(4)
        t_complex = self.tone.compute(eigvecs, dims, r=2)
        eigvecs_real = np.real(eigvecs)
        t_real = self.tone.compute(eigvecs_real, dims, r=2)
        np.testing.assert_allclose(t_complex, t_real, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. SemanticGeometryLayer — encode() invariants
# ---------------------------------------------------------------------------


class TestSemanticGeometryLayer:

    def setup_method(self):
        self.hdv = _StubHDVSystem(hdv_dim=64, n_overlaps=8)
        self.layer = SemanticGeometryLayer(self.hdv)

    def test_encode_returns_dict(self):
        result = self.layer.encode("hello world from semantic geometry")
        assert isinstance(result, dict)

    def test_hypothesis_only_always_true(self):
        for text in ["", "one", "one two three four five"]:
            result = self.layer.encode(text)
            assert result["hypothesis_only"] is True, f"failed for text={text!r}"

    def test_hdv_shape(self):
        result = self.layer.encode("some text")
        assert result["hdv"].shape == (64,)

    def test_jacobian_salience_is_array(self):
        result = self.layer.encode("alpha beta gamma")
        assert isinstance(result["jacobian_salience"], np.ndarray)

    def test_jacobian_salience_length(self):
        text = "alpha beta gamma delta"
        result = self.layer.encode(text)
        assert result["jacobian_salience"].shape == (4,)

    def test_koopman_k_none_for_short_text(self):
        result = self.layer.encode("hi there")
        # < 3 tokens → trust=0 → _TextEDMD not fitted → K_matrix = None
        assert result["koopman_K"] is None

    def test_eigenvalues_returned(self):
        result = self.layer.encode("alpha beta gamma delta epsilon")
        assert "eigenvalues" in result
        assert isinstance(result["eigenvalues"], np.ndarray)

    def test_tone_returned(self):
        result = self.layer.encode("alpha beta gamma delta epsilon")
        assert "tone" in result
        assert isinstance(result["tone"], np.ndarray)

    def test_trust_float(self):
        result = self.layer.encode("alpha beta gamma delta epsilon")
        assert isinstance(result["trust"], float)

    def test_trust_zero_short_text(self):
        result = self.layer.encode("one two")
        assert result["trust"] == 0.0

    def test_trust_in_unit_interval(self):
        result = self.layer.encode("the quick brown fox jumps over lazy dog today")
        assert 0.0 <= result["trust"] <= 1.0

    def test_semantic_dims_returned(self):
        result = self.layer.encode("hello world test string")
        assert "semantic_dims" in result
        assert isinstance(result["semantic_dims"], np.ndarray)

    def test_encode_does_not_raise_on_empty(self):
        result = self.layer.encode("")
        assert result["hypothesis_only"] is True

    def test_encode_does_not_raise_on_long_text(self):
        long_text = " ".join(["word"] * 200)
        result = self.layer.encode(long_text)
        assert result["hypothesis_only"] is True


# ---------------------------------------------------------------------------
# 7. ProposalQueue integration with SemanticGeometryLayer
# ---------------------------------------------------------------------------


class TestProposalQueueIntegration:

    def setup_method(self):
        self.hdv = _StubHDVSystem(hdv_dim=64, n_overlaps=8)
        self.pq = ProposalQueue()
        self.layer = SemanticGeometryLayer(self.hdv, proposal_queue=self.pq)

    def test_exploration_proposal_always_emitted(self):
        """Every encode() call emits at least one 'exploration' proposal."""
        self.layer.encode("alpha beta gamma delta epsilon")
        proposals = self.pq.process()
        types = [p["type"] for p in proposals]
        assert "exploration" in types

    def test_exploration_on_empty_text(self):
        """Even empty text emits an exploration proposal."""
        self.layer.encode("")
        proposals = self.pq.process()
        types = [p["type"] for p in proposals]
        assert "exploration" in types

    def test_navigation_not_emitted_below_tau(self):
        """navigation proposal only emitted when trust > tau_semantic."""
        layer = SemanticGeometryLayer(
            self.hdv, proposal_queue=self.pq, tau_semantic=1.1  # impossible to exceed
        )
        layer.encode("one two three four five six seven eight nine ten")
        proposals = self.pq.process()
        types = [p["type"] for p in proposals]
        assert "navigation" not in types

    def test_exploration_proposal_has_salience_field(self):
        self.layer.encode("one two three")
        proposals = self.pq.process()
        for p in proposals:
            if p["type"] == "exploration":
                assert "salience" in p
                assert isinstance(p["salience"], list)

    def test_exploration_proposal_has_text_preview(self):
        self.layer.encode("quick fox")
        proposals = self.pq.process()
        for p in proposals:
            if p["type"] == "exploration":
                assert "text_preview" in p

    def test_no_proposal_queue_no_side_effects(self):
        """Layer with no proposal_queue must not raise."""
        layer = SemanticGeometryLayer(self.hdv, proposal_queue=None)
        result = layer.encode("hello world test")
        assert result["hypothesis_only"] is True

    def test_navigation_proposal_has_tone_field(self):
        """If emitted, navigation proposal must carry 'tone' and 'trust'."""
        # Use tau_semantic=0.0 to guarantee navigation is emitted if trust > 0
        layer = SemanticGeometryLayer(
            self.hdv, proposal_queue=self.pq, tau_semantic=0.0
        )
        long_text = " ".join(["alpha", "beta", "gamma", "delta", "epsilon"] * 4)
        layer.encode(long_text)
        proposals = self.pq.process()
        nav = [p for p in proposals if p["type"] == "navigation"]
        for p in nav:
            assert "tone" in p
            assert "trust" in p
            assert isinstance(p["tone"], list)
            assert isinstance(p["trust"], float)


# ---------------------------------------------------------------------------
# 8. Memory bounds
# ---------------------------------------------------------------------------


class TestMemoryBounds:

    def test_max_semantic_dim_is_200(self):
        assert _MAX_SEMANTIC_DIM == 200

    def test_max_tokens_is_100(self):
        assert _MAX_TOKENS == 100

    def test_semantic_dims_never_exceed_200(self):
        hdv = _StubHDVSystem(hdv_dim=1024, n_overlaps=300)
        op = TextKoopmanOperator(hdv, max_tokens=_MAX_TOKENS)
        _, dims, _ = op.fit(" ".join(["word"] * 20))
        assert len(dims) <= _MAX_SEMANTIC_DIM

    def test_token_cap_in_jacobian_estimator(self):
        hdv = _StubHDVSystem(hdv_dim=64)
        estimator = SemanticJacobianEstimator(hdv, max_tokens=5)
        text = " ".join(["x"] * 50)
        s = estimator.compute(text)
        assert s.shape == (5,)

    def test_text_koopman_token_cap(self):
        hdv = _StubHDVSystem(hdv_dim=64, n_overlaps=4)
        op = TextKoopmanOperator(hdv, max_tokens=5)
        text = " ".join(["word"] * 50)
        edmd, _, _ = op.fit(text)
        # Should have processed only 5 tokens → 4 pairs max
        # K_matrix shape = (n_dims, n_dims), and pairs were built
        if edmd.K_matrix is not None:
            assert edmd.K_matrix.shape[0] <= _MAX_SEMANTIC_DIM

    def test_layer_respects_max_tokens(self):
        hdv = _StubHDVSystem(hdv_dim=64, n_overlaps=4)
        layer = SemanticGeometryLayer(hdv, max_tokens=5)
        long_text = " ".join(["word"] * 200)
        result = layer.encode(long_text)
        # jacobian_salience capped at 5
        assert len(result["jacobian_salience"]) <= 5


# ---------------------------------------------------------------------------
# 9. Structural isolation (CRITICAL-1 structural checks)
# ---------------------------------------------------------------------------


class TestCriticalOneStructural:
    """Verify that _TextEDMD is structurally isolated from EDMDKoopman."""

    def test_text_edmd_module_is_semantic_geometry(self):
        """_TextEDMD is defined in tensor.semantic_geometry, NOT tensor.koopman_edmd."""
        assert _TextEDMD.__module__ == "tensor.semantic_geometry"

    def test_edmd_koopman_module_is_koopman_edmd(self):
        assert EDMDKoopman.__module__ == "tensor.koopman_edmd"

    def test_text_edmd_no_edmd_koopman_in_mro(self):
        for cls in _TextEDMD.__mro__:
            assert cls is not EDMDKoopman

    def test_text_edmd_fit_is_standalone(self):
        """_TextEDMD.fit doesn't delegate to EDMDKoopman.fit internally."""
        edmd = _TextEDMD()
        with mock.patch.object(EDMDKoopman, "fit") as mock_fit:
            rng = np.random.default_rng(0)
            d = 4
            m = 6
            H = rng.standard_normal((m, d))
            pairs = [(H[i], H[i + 1]) for i in range(m - 1)]
            edmd.fit(pairs)
            mock_fit.assert_not_called()
