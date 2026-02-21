"""
Tests for tensor/validation_bridge.py — ValidationBridge, ProposalQueue, make_merged_patch.

Coverage:
  - ProposalQueue: put/process, unbounded, thread-safety, qsize
  - make_merged_patch: algebraic construction, trust in metadata, no EDMDKoopman.fit()
  - ValidationBridge.validate_equivalence: pass, fail spectral, fail compression, fail both
  - ValidationBridge.validate_navigation: always True
  - ValidationBridge.process_queue: all proposal types, missing fields, stats tracking
  - CRITICAL-2: no EDMDKoopman.fit() called anywhere in this module
"""

from __future__ import annotations

import threading
import unittest.mock as mock

import numpy as np
import pytest

from tensor.patch_graph import Patch
from tensor.validation_bridge import (
    ProposalQueue,
    ValidationBridge,
    make_merged_patch,
    _spectral_gap_from_eigvals,
)
from tensor.koopman_edmd import EDMDKoopman


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_patch(pid: int, eigvals, centroid=None, trust: float = 0.8) -> Patch:
    """Build a minimal Patch for testing."""
    n = 2
    if centroid is None:
        centroid = np.zeros(n)
    basis = np.zeros((1, n, n))
    return Patch(
        id=pid,
        patch_type="lca",
        operator_basis=basis,
        spectrum=np.array(eigvals, dtype=complex),
        centroid=centroid,
        operator_rank=1,
        commutator_norm=0.01,
        curvature_ratio=0.03,
        spectral_gap=0.1,
        metadata={"trust": trust},
    )


def _simple_K(a: float, b: float) -> np.ndarray:
    """2×2 diagonal Koopman matrix with eigenvalues a, b."""
    return np.diag([a, b]).astype(float)


# ── ProposalQueue ─────────────────────────────────────────────────────────────


class TestProposalQueue:
    def test_put_and_process_roundtrip(self):
        q = ProposalQueue()
        q.put({"type": "navigation", "x": 1})
        q.put({"type": "exploration", "x": 2})
        items = q.process()
        assert len(items) == 2
        assert items[0]["type"] == "navigation"
        assert items[1]["type"] == "exploration"

    def test_process_empty_queue_returns_empty_list(self):
        q = ProposalQueue()
        assert q.process() == []

    def test_process_drains_queue(self):
        q = ProposalQueue()
        q.put({"type": "navigation"})
        q.process()
        assert q.process() == []  # second drain: empty

    def test_qsize_reflects_depth(self):
        q = ProposalQueue()
        assert q.qsize() == 0
        q.put({"type": "navigation"})
        q.put({"type": "navigation"})
        assert q.qsize() == 2
        q.process()
        assert q.qsize() == 0

    def test_put_never_blocks_high_volume(self):
        """Queue is unbounded — 10,000 puts complete without blocking."""
        q = ProposalQueue()
        for i in range(10_000):
            q.put({"type": "navigation", "i": i})
        assert q.qsize() == 10_000

    def test_thread_safe_concurrent_puts(self):
        """Multiple producer threads can put() concurrently without data loss."""
        q = ProposalQueue()
        n_threads, n_per_thread = 10, 100

        def producer():
            for _ in range(n_per_thread):
                q.put({"type": "navigation"})

        threads = [threading.Thread(target=producer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        items = q.process()
        assert len(items) == n_threads * n_per_thread

    def test_put_does_not_call_edmd_fit(self):
        """put() is pure queue operation — no EDMD access."""
        q = ProposalQueue()
        with mock.patch.object(EDMDKoopman, "fit") as mock_fit:
            q.put({"type": "navigation"})
            mock_fit.assert_not_called()


# ── make_merged_patch ─────────────────────────────────────────────────────────


class TestMakeMergedPatch:
    def test_returns_patch_with_trust_in_metadata(self):
        pa = _make_patch(0, [-0.5, -0.5], trust=0.8)
        pb = _make_patch(1, [-0.5, -0.5], trust=0.7)
        K_a = _simple_K(-0.5, -0.5)
        K_b = _simple_K(-0.5, -0.5)
        merged = make_merged_patch(pa, pb, 0.8, 0.7, K_a, K_b)
        assert "trust" in merged.metadata
        assert 0.0 <= merged.metadata["trust"] <= 1.0

    def test_K_matrix_stored_in_metadata(self):
        pa = _make_patch(0, [-0.5, -0.3])
        pb = _make_patch(1, [-0.4, -0.6])
        K_a = _simple_K(-0.5, -0.3)
        K_b = _simple_K(-0.4, -0.6)
        merged = make_merged_patch(pa, pb, 0.8, 0.6, K_a, K_b)
        assert "K_matrix" in merged.metadata
        assert merged.metadata["K_matrix"].shape == (2, 2)

    def test_merged_flag_set(self):
        pa = _make_patch(0, [-0.5, -0.5])
        pb = _make_patch(1, [-0.5, -0.5])
        K = _simple_K(-0.5, -0.5)
        merged = make_merged_patch(pa, pb, 0.7, 0.7, K, K)
        assert merged.metadata["merged"] is True

    def test_source_ids_stored(self):
        pa = _make_patch(0, [-0.5, -0.5])
        pb = _make_patch(7, [-0.5, -0.5])
        K = _simple_K(-0.5, -0.5)
        merged = make_merged_patch(pa, pb, 0.7, 0.7, K, K, merged_id=99)
        assert merged.metadata["source_ids"] == (0, 7)
        assert merged.id == 99

    def test_centroid_is_arithmetic_mean(self):
        pa = _make_patch(0, [-0.5], centroid=np.array([1.0, 0.0]))
        pb = _make_patch(1, [-0.5], centroid=np.array([3.0, 2.0]))
        K = _simple_K(-0.5, -0.5)
        merged = make_merged_patch(pa, pb, 0.5, 0.5, K, K)
        np.testing.assert_allclose(merged.centroid, [2.0, 1.0])

    def test_K_merged_is_weighted_average(self):
        K_a = _simple_K(-1.0, -2.0)
        K_b = _simple_K(-3.0, -4.0)
        pa = _make_patch(0, [-1.0, -2.0])
        pb = _make_patch(1, [-3.0, -4.0])
        # Equal trust → simple average
        merged = make_merged_patch(pa, pb, 0.5, 0.5, K_a, K_b)
        expected_K = (-2.0 + -4.0) / 2  # (-1-3)/2 for [0,0], (-2-4)/2 for [1,1]
        K_stored = merged.metadata["K_matrix"]
        np.testing.assert_allclose(K_stored[0, 0], -2.0)
        np.testing.assert_allclose(K_stored[1, 1], -3.0)

    def test_zero_trust_fallback_uses_equal_weights(self):
        """Both trusts=0 → equal weights rather than divide-by-zero."""
        pa = _make_patch(0, [-0.5, -0.5])
        pb = _make_patch(1, [-0.3, -0.3])
        K_a = _simple_K(-0.5, -0.5)
        K_b = _simple_K(-0.3, -0.3)
        merged = make_merged_patch(pa, pb, 0.0, 0.0, K_a, K_b)
        K_stored = merged.metadata["K_matrix"]
        np.testing.assert_allclose(K_stored[0, 0], -0.4, atol=1e-9)

    def test_does_not_call_edmd_fit(self):
        """CRITICAL-2: make_merged_patch must never call EDMDKoopman.fit()."""
        pa = _make_patch(0, [-0.5, -0.5])
        pb = _make_patch(1, [-0.5, -0.5])
        K = _simple_K(-0.5, -0.5)
        with mock.patch.object(EDMDKoopman, "fit") as mock_fit:
            make_merged_patch(pa, pb, 0.8, 0.7, K, K)
            mock_fit.assert_not_called()

    def test_spectrum_from_K_merged(self):
        """Spectrum of merged_patch = eigvals of K_merged, not of K_a or K_b."""
        K_a = _simple_K(-1.0, -2.0)
        K_b = _simple_K(-3.0, -4.0)
        pa = _make_patch(0, np.linalg.eigvals(K_a))
        pb = _make_patch(1, np.linalg.eigvals(K_b))
        merged = make_merged_patch(pa, pb, 1.0, 0.0, K_a, K_b)
        # trust_a=1, trust_b=0 → K_merged = K_a exactly
        np.testing.assert_allclose(
            np.sort(np.abs(np.real(merged.spectrum))),
            np.sort(np.abs(np.real(np.linalg.eigvals(K_a)))),
        )


# ── _spectral_gap_from_eigvals helper ────────────────────────────────────────


class TestSpectralGapHelper:
    def test_gap_two_values(self):
        eigvals = np.array([-0.8, -0.3])
        assert _spectral_gap_from_eigvals(eigvals) == pytest.approx(0.5, abs=1e-9)

    def test_gap_one_value(self):
        eigvals = np.array([-0.5])
        assert _spectral_gap_from_eigvals(eigvals) == pytest.approx(0.5, abs=1e-9)

    def test_gap_empty(self):
        assert _spectral_gap_from_eigvals(np.array([])) == 0.0

    def test_gap_complex_uses_real_part(self):
        eigvals = np.array([-0.5 + 0.8j, -0.5 - 0.8j])
        # |Re(λ₁)| = |Re(λ₂)| = 0.5 → gap = 0
        assert _spectral_gap_from_eigvals(eigvals) == pytest.approx(0.0, abs=1e-9)


# ── ValidationBridge.validate_equivalence ────────────────────────────────────


class TestValidateEquivalence:
    def setup_method(self):
        self.bridge = ValidationBridge(
            spectral_preservation_eps=0.2,
            compression_delta=0.05,
        )

    def _make_identical_patches(self):
        """Two patches with identical spectra → merge should preserve spectrum exactly."""
        eigvals = np.array([-0.5, -0.3], dtype=complex)
        pa = _make_patch(0, eigvals, trust=0.8)
        pb = _make_patch(1, eigvals, trust=0.7)
        K_a = _simple_K(-0.5, -0.3)
        K_b = _simple_K(-0.5, -0.3)
        merged = make_merged_patch(pa, pb, 0.8, 0.7, K_a, K_b)
        return pa, pb, merged

    def test_passes_when_both_conditions_met(self):
        pa, pb, merged = self._make_identical_patches()
        assert self.bridge.validate_equivalence(pa, pb, merged, 0.8, 0.7) is True

    def test_fails_when_spectral_preservation_violated(self):
        """Merged patch with very different spectrum → spectral distance > eps."""
        eigvals_a = np.array([-0.5, -0.3], dtype=complex)
        eigvals_b = np.array([-5.0, -4.0], dtype=complex)  # far from a
        pa = _make_patch(0, eigvals_a, trust=0.8)
        pb = _make_patch(1, eigvals_b, trust=0.7)
        K_a = _simple_K(-0.5, -0.3)
        K_b = _simple_K(-5.0, -4.0)
        merged = make_merged_patch(pa, pb, 0.8, 0.7, K_a, K_b)
        # dist(merged, K_b) will be large
        assert self.bridge.validate_equivalence(pa, pb, merged, 0.8, 0.7) is False

    def test_fails_when_trust_too_low(self):
        """Merged patch with trust below threshold → compression fails."""
        eigvals = np.array([-0.5, -0.3], dtype=complex)
        pa = _make_patch(0, eigvals, trust=0.9)
        pb = _make_patch(1, eigvals, trust=0.9)
        K_a = _simple_K(-0.5, -0.3)
        K_b = _simple_K(-0.5, -0.3)
        merged = make_merged_patch(pa, pb, 0.9, 0.9, K_a, K_b)
        # Manually override trust to be below threshold (0.9 - 0.05 = 0.85)
        merged.metadata["trust"] = 0.5  # well below threshold
        assert self.bridge.validate_equivalence(pa, pb, merged, 0.9, 0.9) is False

    def test_fails_when_metadata_trust_missing(self):
        """Missing trust in metadata → trust=0.0 → compression fails for high-trust sources."""
        eigvals = np.array([-0.5, -0.3], dtype=complex)
        pa = _make_patch(0, eigvals)
        pb = _make_patch(1, eigvals)
        K = _simple_K(-0.5, -0.3)
        merged = make_merged_patch(pa, pb, 0.8, 0.8, K, K)
        merged.metadata.pop("trust")  # simulate missing key
        # trust_threshold = max(0.8, 0.8) - 0.05 = 0.75; trust_merged=0.0 < 0.75
        assert self.bridge.validate_equivalence(pa, pb, merged, 0.8, 0.8) is False

    def test_stats_validated_increments_on_pass(self):
        pa, pb, merged = self._make_identical_patches()
        before = self.bridge.stats()["n_validated"]
        self.bridge.validate_equivalence(pa, pb, merged, 0.8, 0.7)
        assert self.bridge.stats()["n_validated"] == before + 1

    def test_stats_rejected_increments_on_fail(self):
        eigvals = np.array([-0.5], dtype=complex)
        pa = _make_patch(0, eigvals, trust=0.9)
        pb = _make_patch(1, eigvals, trust=0.9)
        K = _simple_K(-0.5, -0.3)
        merged = make_merged_patch(pa, pb, 0.9, 0.9, K, K)
        merged.metadata["trust"] = 0.0
        before = self.bridge.stats()["n_rejected"]
        self.bridge.validate_equivalence(pa, pb, merged, 0.9, 0.9)
        assert self.bridge.stats()["n_rejected"] == before + 1

    def test_does_not_call_edmd_fit(self):
        """CRITICAL-2: validate_equivalence must never call EDMDKoopman.fit()."""
        pa, pb, merged = self._make_identical_patches()
        with mock.patch.object(EDMDKoopman, "fit") as mock_fit:
            self.bridge.validate_equivalence(pa, pb, merged, 0.8, 0.7)
            mock_fit.assert_not_called()

    def test_zero_trust_sources_always_pass_compression(self):
        """trust_threshold = max(0,0) - delta = -delta < 0; any trust ≥ 0 passes."""
        eigvals = np.array([-0.5, -0.3], dtype=complex)
        pa = _make_patch(0, eigvals, trust=0.0)
        pb = _make_patch(1, eigvals, trust=0.0)
        K = _simple_K(-0.5, -0.3)
        merged = make_merged_patch(pa, pb, 0.0, 0.0, K, K)
        # Spectral preservation should also pass (identical spectra)
        assert self.bridge.validate_equivalence(pa, pb, merged, 0.0, 0.0) is True


# ── ValidationBridge.validate_navigation ────────────────────────────────────


class TestValidateNavigation:
    def test_always_true_for_navigation(self):
        bridge = ValidationBridge()
        assert bridge.validate_navigation({"type": "navigation"}) is True

    def test_always_true_for_exploration(self):
        bridge = ValidationBridge()
        assert bridge.validate_navigation({"type": "exploration"}) is True

    def test_always_true_for_empty_proposal(self):
        bridge = ValidationBridge()
        assert bridge.validate_navigation({}) is True


# ── ValidationBridge.process_queue ──────────────────────────────────────────


class TestProcessQueue:
    def setup_method(self):
        self.bridge = ValidationBridge(
            spectral_preservation_eps=0.2,
            compression_delta=0.05,
        )

    def _build_valid_equivalence_proposal(self) -> dict:
        eigvals = np.array([-0.5, -0.3], dtype=complex)
        pa = _make_patch(0, eigvals, trust=0.8)
        pb = _make_patch(1, eigvals, trust=0.7)
        K = _simple_K(-0.5, -0.3)
        merged = make_merged_patch(pa, pb, 0.8, 0.7, K, K)
        return {
            "type": "equivalence",
            "patch_a": pa,
            "patch_b": pb,
            "merged_patch": merged,
            "trust_a": 0.8,
            "trust_b": 0.7,
        }

    def test_empty_queue_returns_empty_list(self):
        q = ProposalQueue()
        assert self.bridge.process_queue(q) == []

    def test_navigation_proposal_passes_unconditionally(self):
        q = ProposalQueue()
        q.put({"type": "navigation", "hint": "patch_3"})
        accepted = self.bridge.process_queue(q)
        assert len(accepted) == 1
        assert accepted[0]["hint"] == "patch_3"

    def test_exploration_proposal_passes_unconditionally(self):
        q = ProposalQueue()
        q.put({"type": "exploration", "region": [1.0, 2.0]})
        accepted = self.bridge.process_queue(q)
        assert len(accepted) == 1

    def test_valid_equivalence_proposal_passes(self):
        q = ProposalQueue()
        q.put(self._build_valid_equivalence_proposal())
        accepted = self.bridge.process_queue(q)
        assert len(accepted) == 1
        assert accepted[0]["type"] == "equivalence"

    def test_equivalence_missing_patch_a_discarded(self):
        q = ProposalQueue()
        prop = self._build_valid_equivalence_proposal()
        prop.pop("patch_a")
        q.put(prop)
        accepted = self.bridge.process_queue(q)
        assert accepted == []

    def test_equivalence_missing_merged_patch_discarded(self):
        q = ProposalQueue()
        prop = self._build_valid_equivalence_proposal()
        prop.pop("merged_patch")
        q.put(prop)
        accepted = self.bridge.process_queue(q)
        assert accepted == []

    def test_equivalence_failing_validation_discarded(self):
        q = ProposalQueue()
        prop = self._build_valid_equivalence_proposal()
        prop["merged_patch"].metadata["trust"] = 0.0  # force trust failure
        q.put(prop)
        accepted = self.bridge.process_queue(q)
        assert accepted == []

    def test_unknown_proposal_type_discarded(self):
        q = ProposalQueue()
        q.put({"type": "gradient_update", "data": "malicious"})  # unknown
        accepted = self.bridge.process_queue(q)
        assert accepted == []

    def test_mixed_queue_correct_filtering(self):
        """Navigation passes, bad equivalence rejected, good equivalence passes."""
        q = ProposalQueue()
        q.put({"type": "navigation", "x": 1})

        bad_prop = self._build_valid_equivalence_proposal()
        bad_prop["merged_patch"].metadata["trust"] = 0.0
        q.put(bad_prop)

        q.put(self._build_valid_equivalence_proposal())

        accepted = self.bridge.process_queue(q)
        assert len(accepted) == 2  # navigation + valid equivalence
        types = [p["type"] for p in accepted]
        assert "navigation" in types
        assert "equivalence" in types

    def test_stats_after_mixed_queue(self):
        q = ProposalQueue()
        q.put({"type": "navigation"})
        q.put(self._build_valid_equivalence_proposal())
        bad = self._build_valid_equivalence_proposal()
        bad["merged_patch"].metadata["trust"] = 0.0
        q.put(bad)

        self.bridge.process_queue(q)
        s = self.bridge.stats()
        assert s["n_validated"] == 1
        assert s["n_rejected"] >= 1

    def test_process_queue_does_not_call_edmd_fit(self):
        """CRITICAL-2: process_queue must never call EDMDKoopman.fit()."""
        q = ProposalQueue()
        q.put({"type": "navigation"})
        q.put(self._build_valid_equivalence_proposal())
        with mock.patch.object(EDMDKoopman, "fit") as mock_fit:
            self.bridge.process_queue(q)
            mock_fit.assert_not_called()
