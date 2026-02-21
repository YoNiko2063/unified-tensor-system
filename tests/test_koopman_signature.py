"""
Tests for optimization/koopman_signature.py

Covers:
  1. compute_invariants() — eigenvalue ordering, zero-padding, histogram
  2. KoopmanInvariantDescriptor.to_retrieval_vector() — shape and content
  3. Edge cases — empty arrays, fewer than k eigenvalues, real-only eigenvalues
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from optimization.koopman_signature import (
    KoopmanInvariantDescriptor,
    compute_invariants,
)

_OPS = ["cutoff_eval", "Q_eval", "constraint_check", "objective_eval", "accepted"]


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_eigvecs(n: int) -> np.ndarray:
    """Return random real orthonormal (n, n) matrix."""
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((n, n)))
    return Q.astype(complex)


# ── 1. compute_invariants() ───────────────────────────────────────────────────


class TestComputeInvariants:

    def test_returns_dataclass(self):
        eigvals = np.array([0.95 + 0j, 0.5 + 0.3j, 0.2 - 0.1j])
        eigvecs = _make_eigvecs(3)
        inv = compute_invariants(eigvals, eigvecs, _OPS[:3], k=5)
        assert isinstance(inv, KoopmanInvariantDescriptor)

    def test_spectral_radius_is_max_magnitude(self):
        eigvals = np.array([0.9 + 0j, 0.5 + 0.5j, 0.1 + 0j])
        inv = compute_invariants(eigvals, _make_eigvecs(3), _OPS[:3])
        assert abs(inv.spectral_radius - max(abs(e) for e in eigvals)) < 1e-12

    def test_slow_mode_count_above_0p9(self):
        eigvals = np.array([0.95 + 0j, 0.91 + 0j, 0.85 + 0j, 0.5 + 0j])
        inv = compute_invariants(eigvals, _make_eigvecs(4), _OPS[:4])
        assert inv.slow_mode_count == 2

    def test_oscillatory_mode_count_im_above_0p1(self):
        eigvals = np.array([0.8 + 0.5j, 0.8 - 0.5j, 0.6 + 0j, 0.3 + 0.05j])
        inv = compute_invariants(eigvals, _make_eigvecs(4), _OPS[:4])
        # |Im| > 0.1: first two (0.5), fourth is 0.05 < 0.1
        assert inv.oscillatory_mode_count == 2

    def test_top_k_sorted_by_magnitude(self):
        eigvals = np.array([0.3 + 0j, 0.95 + 0j, 0.6 + 0j])
        inv = compute_invariants(eigvals, _make_eigvecs(3), _OPS[:3], k=3)
        # Top-1 should be 0.95
        assert abs(inv.top_k_real[0] - 0.95) < 1e-12

    def test_top_k_real_and_imag_length_equals_k(self):
        eigvals = np.array([0.9 + 0.1j, 0.5 - 0.3j])
        inv = compute_invariants(eigvals, _make_eigvecs(2), _OPS[:2], k=5)
        assert len(inv.top_k_real) == 5
        assert len(inv.top_k_imag) == 5

    def test_zero_padding_when_fewer_than_k_eigenvalues(self):
        eigvals = np.array([0.9 + 0j])
        inv = compute_invariants(eigvals, _make_eigvecs(1), _OPS[:1], k=5)
        assert inv.top_k_real[0] == pytest.approx(0.9)
        assert all(inv.top_k_real[1:] == 0.0)
        assert all(inv.top_k_imag == 0.0)

    def test_imag_part_preserved_not_discarded(self):
        eigvals = np.array([0.8 + 0.4j])
        inv = compute_invariants(eigvals, _make_eigvecs(1), _OPS[:1], k=5)
        assert abs(inv.top_k_imag[0] - 0.4) < 1e-12

    def test_dominant_operator_histogram_sums_to_1(self):
        n = 4
        eigvals = np.array([0.9 + 0j] * n)
        inv = compute_invariants(eigvals, _make_eigvecs(n), _OPS[:n])
        assert abs(inv.dominant_operator_histogram.sum() - 1.0) < 1e-10

    def test_dominant_operator_histogram_length_matches_operator_types(self):
        ops = _OPS[:3]
        eigvals = np.array([0.9, 0.5, 0.1])
        inv = compute_invariants(eigvals + 0j, _make_eigvecs(3), ops)
        assert len(inv.dominant_operator_histogram) == len(ops)

    def test_operator_basis_order_stored(self):
        ops = ["alpha", "beta", "gamma"]
        eigvals = np.array([0.9 + 0j, 0.5 + 0j, 0.1 + 0j])
        inv = compute_invariants(eigvals, _make_eigvecs(3), ops)
        assert inv.operator_basis_order == ops

    def test_empty_eigenvalues_returns_zeros(self):
        inv = compute_invariants(np.array([]), np.zeros((0, 0)), _OPS, k=5)
        assert inv.spectral_radius == 0.0
        assert inv.slow_mode_count == 0
        assert all(inv.top_k_real == 0.0)


# ── 2. to_retrieval_vector() ──────────────────────────────────────────────────


class TestToRetrievalVector:

    def test_length_is_2k_plus_dynamical_quantities(self):
        # shape = 2*k + 3 (3 domain-invariant dynamical quantities)
        eigvals = np.array([0.9 + 0.1j, 0.5 - 0.2j, 0.3 + 0j])
        inv = compute_invariants(eigvals, _make_eigvecs(3), _OPS[:3], k=5)
        rv = inv.to_retrieval_vector()
        assert rv.shape == (13,)    # 2*5 + 3

    def test_length_with_custom_dynamical_quantities(self):
        eigvals = np.array([0.9 + 0.1j, 0.5 - 0.2j, 0.3 + 0j])
        inv = compute_invariants(eigvals, _make_eigvecs(3), _OPS[:3], k=5,
                                 log_omega0_norm=0.3, log_Q_norm=-0.5,
                                 damping_ratio=0.1)
        rv = inv.to_retrieval_vector()
        assert rv.shape == (13,)
        assert rv[10] == pytest.approx(0.3)
        assert rv[11] == pytest.approx(-0.5)
        assert rv[12] == pytest.approx(0.1)

    def test_retrieval_vector_structure(self):
        """
        Retrieval vector layout: [top_k_real × 0.1, top_k_imag × 0.1,
        log_omega0_norm, log_Q_norm, damping_ratio].
        Eigenvalue components are dampened so dynamical quantities dominate.
        """
        eigvals = np.array([0.9 + 0.4j])
        inv = compute_invariants(eigvals, _make_eigvecs(1), _OPS[:1], k=3,
                                 log_omega0_norm=0.2, log_Q_norm=-0.1,
                                 damping_ratio=0.25)
        rv = inv.to_retrieval_vector()
        np.testing.assert_array_almost_equal(rv[:3], inv.top_k_real * 0.1)
        np.testing.assert_array_almost_equal(rv[3:6], inv.top_k_imag * 0.1)
        np.testing.assert_array_almost_equal(rv[6:], inv.to_query_vector())

    def test_to_query_vector_length_and_content(self):
        """to_query_vector() is the 3-D domain-invariant key."""
        inv = compute_invariants(
            np.array([0.9 + 0.1j]), _make_eigvecs(1), _OPS[:1], k=3,
            log_omega0_norm=0.3, log_Q_norm=-0.2, damping_ratio=0.15,
        )
        qv = inv.to_query_vector()
        assert qv.shape == (3,)
        assert qv[0] == pytest.approx(0.3)
        assert qv[1] == pytest.approx(-0.2)
        assert qv[2] == pytest.approx(0.15)

    def test_dynamical_omega0_roundtrip(self):
        """dynamical_omega0() should recover ω₀ from log_omega0_norm."""
        import math
        omega0_target = 2.0 * math.pi * 800.0
        log_omega0_ref = math.log(2.0 * math.pi * 1000.0)
        log_omega0_scale = math.log(10.0)
        lnorm = (math.log(omega0_target) - log_omega0_ref) / log_omega0_scale
        inv = compute_invariants(
            np.array([0.9 + 0j]), _make_eigvecs(1), _OPS[:1], k=3,
            log_omega0_norm=lnorm,
        )
        assert abs(inv.dynamical_omega0() - omega0_target) / omega0_target < 1e-10

    def test_different_imag_produces_different_retrieval_vector(self):
        """Systems with same |λ| but different Im(λ) should have different vectors."""
        e1 = np.array([0.8 + 0.5j])
        e2 = np.array([0.8 + 0.0j])   # same magnitude (approx), different Im

        ops1 = _OPS[:1]
        inv1 = compute_invariants(e1, _make_eigvecs(1), ops1, k=3)
        inv2 = compute_invariants(e2, _make_eigvecs(1), ops1, k=3)

        rv1 = inv1.to_retrieval_vector()
        rv2 = inv2.to_retrieval_vector()
        assert not np.allclose(rv1, rv2), (
            "Retrieval vectors should differ when Im(λ) differs"
        )
