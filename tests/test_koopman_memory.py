"""
Tests for optimization/koopman_memory.py

Covers:
  1. OptimizationExperience — dataclass fields
  2. KoopmanExperienceMemory.add() — append and merge behaviour
  3. KoopmanExperienceMemory.retrieve_candidates() — ranking and top_n
  4. KoopmanExperienceMemory.confirm_match() — threshold, magnitude sort
  5. summary() and __len__()
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from tensor.koopman_edmd import EDMDKoopman, KoopmanResult
from optimization.koopman_signature import KoopmanInvariantDescriptor, compute_invariants
from optimization.koopman_memory import (
    EIGENVALUE_MATCH_THRESHOLD,
    KoopmanExperienceMemory,
    OptimizationExperience,
)

_OPS = ["cutoff_eval", "Q_eval", "constraint_check", "objective_eval", "accepted"]


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_signature(eigvals: np.ndarray) -> KoopmanResult:
    """Construct a minimal KoopmanResult with given eigenvalues."""
    d = len(eigvals)
    vecs = np.eye(d, dtype=complex)
    return KoopmanResult(
        eigenvalues=eigvals,
        eigenvectors=vecs,
        K_matrix=np.diag(eigvals.real),
        spectral_gap=float(abs(eigvals[0]) - abs(eigvals[1])) if d >= 2 else 0.0,
        is_stable=True,
    )


def _make_invariant(eigvals: np.ndarray, ops=None) -> KoopmanInvariantDescriptor:
    ops = ops or _OPS[:len(eigvals)]
    vecs = np.eye(len(eigvals), dtype=complex)
    return compute_invariants(eigvals, vecs, ops)


def _experience(bottleneck="cutoff_eval", improvement=0.5, params=None) -> OptimizationExperience:
    return OptimizationExperience(
        bottleneck_operator=bottleneck,
        replacement_applied="analytic_correction",
        runtime_improvement=improvement,
        n_observations=1,
        hardware_target="cpu",
        best_params=params or {"R": 100.0, "L": 0.01, "C": 1e-6},
    )


# ── 1. OptimizationExperience ─────────────────────────────────────────────────


class TestOptimizationExperience:

    def test_fields_accessible(self):
        exp = _experience()
        assert exp.bottleneck_operator == "cutoff_eval"
        assert exp.replacement_applied == "analytic_correction"
        assert exp.runtime_improvement == 0.5
        assert exp.n_observations == 1
        assert exp.hardware_target == "cpu"
        assert isinstance(exp.best_params, dict)

    def test_best_params_default_empty(self):
        exp = OptimizationExperience(
            bottleneck_operator="x", replacement_applied="y",
            runtime_improvement=0.1, n_observations=1, hardware_target="cpu",
        )
        assert exp.best_params == {}


# ── 2. add() ─────────────────────────────────────────────────────────────────


class TestMemoryAdd:

    def setup_method(self):
        self.mem = KoopmanExperienceMemory()
        self.eigvals_a = np.array([0.95 + 0j, 0.7 + 0.1j, 0.3 + 0j])
        self.sig_a = _make_signature(self.eigvals_a)
        self.inv_a = _make_invariant(self.eigvals_a)
        self.exp_a = _experience(improvement=0.5, params={"R": 100.0, "L": 0.01, "C": 1e-6})

    def test_add_increases_len(self):
        assert len(self.mem) == 0
        self.mem.add(self.inv_a, self.sig_a, self.exp_a)
        assert len(self.mem) == 1

    def test_add_non_matching_increases_len(self):
        self.mem.add(self.inv_a, self.sig_a, self.exp_a)
        # Clearly different eigenvalues
        eigvals_b = np.array([0.1 + 0j, 0.05 + 0j, 0.02 + 0j])
        sig_b = _make_signature(eigvals_b)
        inv_b = _make_invariant(eigvals_b)
        exp_b = _experience(improvement=0.3)
        self.mem.add(inv_b, sig_b, exp_b)
        assert len(self.mem) == 2

    def test_add_matching_signature_merges_not_appends(self):
        self.mem.add(self.inv_a, self.sig_a, self.exp_a)
        # Same eigenvalues → should merge
        sig_same = _make_signature(self.eigvals_a.copy())
        inv_same = _make_invariant(self.eigvals_a.copy())
        exp_same = _experience(improvement=0.3)  # lower improvement
        self.mem.add(inv_same, sig_same, exp_same)
        assert len(self.mem) == 1

    def test_merge_increments_n_observations(self):
        self.mem.add(self.inv_a, self.sig_a, self.exp_a)
        assert self.mem._entries[0].experience.n_observations == 1
        sig_same = _make_signature(self.eigvals_a.copy())
        inv_same = _make_invariant(self.eigvals_a.copy())
        self.mem.add(inv_same, sig_same, _experience(improvement=0.3))
        assert self.mem._entries[0].experience.n_observations == 2

    def test_merge_updates_best_params_when_improvement_higher(self):
        self.mem.add(self.inv_a, self.sig_a, self.exp_a)  # improvement=0.5
        sig_same = _make_signature(self.eigvals_a.copy())
        inv_same = _make_invariant(self.eigvals_a.copy())
        better_exp = _experience(
            improvement=0.9, params={"R": 50.0, "L": 0.02, "C": 2e-6}
        )
        self.mem.add(inv_same, sig_same, better_exp)
        assert self.mem._entries[0].experience.runtime_improvement == pytest.approx(0.9)
        assert self.mem._entries[0].experience.best_params["R"] == 50.0

    def test_merge_keeps_best_params_when_improvement_lower(self):
        self.mem.add(self.inv_a, self.sig_a, self.exp_a)  # improvement=0.5
        sig_same = _make_signature(self.eigvals_a.copy())
        inv_same = _make_invariant(self.eigvals_a.copy())
        worse_exp = _experience(improvement=0.2, params={"R": 999.0, "L": 1.0, "C": 1.0})
        self.mem.add(inv_same, sig_same, worse_exp)
        assert self.mem._entries[0].experience.runtime_improvement == pytest.approx(0.5)
        assert self.mem._entries[0].experience.best_params["R"] == pytest.approx(100.0)


# ── 3. retrieve_candidates() ─────────────────────────────────────────────────


class TestRetrieveCandidates:

    def setup_method(self):
        self.mem = KoopmanExperienceMemory()

    def test_returns_empty_when_memory_empty(self):
        inv = _make_invariant(np.array([0.9 + 0j]))
        assert self.mem.retrieve_candidates(inv) == []

    def test_returns_at_most_top_n(self):
        for i in range(5):
            eig = np.array([0.9 - i * 0.1 + 0j, 0.5 + 0j, 0.1 + 0j])
            self.mem.add(
                _make_invariant(eig), _make_signature(eig), _experience()
            )
        inv_q = _make_invariant(np.array([0.85 + 0j, 0.5 + 0j, 0.1 + 0j]))
        candidates = self.mem.retrieve_candidates(inv_q, top_n=3)
        assert len(candidates) <= 3

    def test_nearest_entry_is_first(self):
        eig_far = np.array([0.1 + 0j, 0.05 + 0j, 0.02 + 0j])
        eig_near = np.array([0.88 + 0j, 0.5 + 0j, 0.1 + 0j])
        self.mem.add(_make_invariant(eig_far), _make_signature(eig_far), _experience())
        self.mem.add(_make_invariant(eig_near), _make_signature(eig_near), _experience())

        inv_q = _make_invariant(np.array([0.9 + 0j, 0.5 + 0j, 0.1 + 0j]))
        candidates = self.mem.retrieve_candidates(inv_q, top_n=2)
        # First should be the near one
        assert candidates[0].invariant.spectral_radius > candidates[1].invariant.spectral_radius


# ── 4. confirm_match() ────────────────────────────────────────────────────────


class TestConfirmMatch:

    def setup_method(self):
        self.mem = KoopmanExperienceMemory()

    def test_identical_eigenvalues_match(self):
        eigs = np.array([0.9 + 0.1j, 0.5 - 0.2j, 0.3 + 0j])
        s1 = _make_signature(eigs)
        s2 = _make_signature(eigs.copy())
        assert self.mem.confirm_match(s1, s2)

    def test_distant_eigenvalues_do_not_match(self):
        s1 = _make_signature(np.array([0.9 + 0j, 0.8 + 0j, 0.7 + 0j]))
        s2 = _make_signature(np.array([0.1 + 0j, 0.05 + 0j, 0.02 + 0j]))
        assert not self.mem.confirm_match(s1, s2)

    def test_threshold_boundary(self):
        eigs1 = np.array([1.0 + 0j])
        # Distance = EIGENVALUE_MATCH_THRESHOLD exactly → should NOT match (strict <)
        eigs2 = np.array([(1.0 - EIGENVALUE_MATCH_THRESHOLD) + 0j])
        s1 = _make_signature(eigs1)
        s2 = _make_signature(eigs2)
        # L2([1.0], [1.0 - threshold]) = threshold → not < threshold
        assert not self.mem.confirm_match(s1, s2)

    def test_permutation_invariant(self):
        """Eigenvalues in different order should still match (sorted before compare)."""
        eigs1 = np.array([0.9 + 0j, 0.5 + 0j, 0.1 + 0j])
        eigs2 = np.array([0.1 + 0j, 0.9 + 0j, 0.5 + 0j])   # reordered
        s1 = _make_signature(eigs1)
        s2 = _make_signature(eigs2)
        assert self.mem.confirm_match(s1, s2)

    def test_different_lengths_compares_shorter(self):
        s1 = _make_signature(np.array([0.9 + 0j, 0.8 + 0j, 0.7 + 0j]))
        s2 = _make_signature(np.array([0.9 + 0j]))  # shorter
        # Should compare [0.9] vs [0.9] → distance = 0 → match
        assert self.mem.confirm_match(s1, s2)


# ── 5. summary() and __len__() ───────────────────────────────────────────────


class TestMemoryIntrospection:

    def test_len_empty(self):
        assert len(KoopmanExperienceMemory()) == 0

    def test_len_after_adds(self):
        mem = KoopmanExperienceMemory()
        for i in range(3):
            eig = np.array([0.5 * (i + 1) + 0j])
            mem.add(_make_invariant(eig), _make_signature(eig), _experience())
        assert len(mem) == 3

    def test_summary_fields(self):
        mem = KoopmanExperienceMemory()
        eig = np.array([0.9 + 0j])
        mem.add(_make_invariant(eig), _make_signature(eig), _experience("cutoff_eval"))
        s = mem.summary()
        assert "n_entries" in s
        assert "total_observations" in s
        assert "operators_seen" in s
        assert s["n_entries"] == 1
        assert "cutoff_eval" in s["operators_seen"]
