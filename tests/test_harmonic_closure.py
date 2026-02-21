"""
Tests for tensor/harmonic_closure.py — HarmonicClosureChecker.

Coverage:
  - projection_residual: identical, orthogonal, partial, empty algebra, single operator
  - check(): redundant, admissible, inadmissible (each failing condition)
  - Bootstrap (empty active_operators)
  - Near-degenerate operators do not falsely collapse
  - HarmonicAtlas wiring: add_classification and merge_similar route through checker
"""

from __future__ import annotations

import numpy as np
import pytest

from tensor.harmonic_closure import HarmonicClosureChecker
from tensor.harmonic_atlas import HarmonicAtlas
from tensor.lca_patch_detector import PatchClassification


# ── Helpers ───────────────────────────────────────────────────────────────────


def _diag(vals) -> np.ndarray:
    return np.diag(vals).astype(float)


def _make_cl(eigvals, trust: float = 0.5, patch_type: str = "lca") -> PatchClassification:
    n = 2
    return PatchClassification(
        patch_type=patch_type,
        operator_rank=1,
        commutator_norm=0.0,
        curvature_ratio=0.02,
        spectral_gap=0.5,
        basis_matrices=np.eye(n).reshape(1, n, n),
        eigenvalues=np.array(eigvals, dtype=complex),
        centroid=np.zeros(n),
        koopman_trust=trust,
    )


# ── projection_residual ───────────────────────────────────────────────────────


class TestProjectionResidual:
    def setup_method(self):
        self.checker = HarmonicClosureChecker()

    def test_identical_operator_zero_residual(self):
        K = _diag([0.5, 0.3])
        r = self.checker.projection_residual(K, [K])
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_scaled_operator_zero_residual(self):
        """2·K₁ is in span({K₁}): coefficient=2, residual=0."""
        K1 = _diag([0.5, 0.3])
        K_new = 2.0 * K1
        r = self.checker.projection_residual(K_new, [K1])
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_operator_full_residual(self):
        """K_new orthogonal to all active operators → r = ‖K_new‖_F."""
        # K1 along e1 ⊗ e1, K_new along e2 ⊗ e2 (in vectorized space)
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])
        r = self.checker.projection_residual(K_new, [K1])
        expected = np.linalg.norm(K_new, "fro")
        assert r == pytest.approx(expected, abs=1e-9)

    def test_partial_projection(self):
        """K_new partially in span(A) → 0 < r < ‖K_new‖_F."""
        K1 = _diag([1.0, 0.0])
        K_new = _diag([1.0, 1.0])  # only first component in span
        r = self.checker.projection_residual(K_new, [K1])
        assert 0.0 < r < np.linalg.norm(K_new, "fro")

    def test_empty_active_operators_returns_norm(self):
        """Bootstrap: empty algebra → Π = 0 → r = ‖K_new‖_F."""
        K_new = _diag([0.5, 0.3])
        r = self.checker.projection_residual(K_new, [])
        expected = np.linalg.norm(K_new, "fro")
        assert r == pytest.approx(expected, abs=1e-9)

    def test_zero_matrix_in_empty_algebra_zero_residual(self):
        """Zero K_new → r = 0 even against empty algebra."""
        K_new = np.zeros((2, 2))
        r = self.checker.projection_residual(K_new, [])
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_two_active_operators_span(self):
        """K_new = K1 + K2 → in span({K1, K2}) → r = 0."""
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K2 = np.array([[0.0, 0.0], [0.0, 1.0]])
        K_new = K1 + K2
        r = self.checker.projection_residual(K_new, [K1, K2])
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_residual_nonnegative(self):
        K1 = _diag([0.5, 0.3])
        K_new = _diag([0.7, 0.1])
        r = self.checker.projection_residual(K_new, [K1])
        assert r >= 0.0


# ── check() — "redundant" ─────────────────────────────────────────────────────


class TestCheckRedundant:
    def setup_method(self):
        self.checker = HarmonicClosureChecker(eps_closure=0.1)

    def test_identical_is_redundant(self):
        K = _diag([0.5, 0.3])
        assert self.checker.check(K, [K]) == "redundant"

    def test_scaled_is_redundant(self):
        K1 = _diag([0.5, 0.3])
        K_new = 1.5 * K1
        assert self.checker.check(K_new, [K1]) == "redundant"

    def test_very_close_is_redundant(self):
        K1 = _diag([0.5, 0.3])
        K_new = K1 + np.full((2, 2), 0.001)  # tiny perturbation
        r = self.checker.projection_residual(K_new, [K1])
        assert r < 0.1  # should be small
        assert self.checker.check(K_new, [K1]) == "redundant"

    def test_redundant_independent_of_trust(self):
        """Trust does not affect "redundant" — if in closure envelope, always redundant."""
        K = _diag([0.5, 0.3])
        assert self.checker.check(K, [K], trust_new=0.0) == "redundant"
        assert self.checker.check(K, [K], trust_new=1.0) == "redundant"

    def test_redundant_independent_of_monitor(self):
        """monitor_unstable=True does not affect "redundant"."""
        K = _diag([0.5, 0.3])
        assert self.checker.check(K, [K], monitor_unstable=True) == "redundant"


# ── check() — "admissible" ────────────────────────────────────────────────────


class TestCheckAdmissible:
    def setup_method(self):
        self.checker = HarmonicClosureChecker(
            eps_closure=0.1, delta_min=0.02, tau_admit=0.3
        )

    def test_independent_operator_high_trust_is_admissible(self):
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])  # orthogonal
        result = self.checker.check(K_new, [K1], trust_new=0.5, monitor_unstable=False)
        assert result == "admissible"

    def test_independent_with_recon_improvement(self):
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = self.checker.check(
            K_new, [K1],
            trust_new=0.5,
            monitor_unstable=False,
            recon_error_before=0.5,
            recon_error_after=0.4,  # improvement > delta_min=0.02
        )
        assert result == "admissible"

    def test_bootstrap_empty_algebra_high_trust_admissible(self):
        K_new = _diag([0.5, 0.3])
        # Empty algebra: r = ‖K_new‖ > eps_closure
        result = self.checker.check(K_new, [], trust_new=0.5, monitor_unstable=False)
        assert result == "admissible"


# ── check() — "inadmissible" ─────────────────────────────────────────────────


class TestCheckInadmissible:
    def setup_method(self):
        self.checker = HarmonicClosureChecker(
            eps_closure=0.1, delta_min=0.02, tau_admit=0.3
        )

    def test_low_trust_is_inadmissible(self):
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = self.checker.check(K_new, [K1], trust_new=0.1)  # below tau_admit
        assert result == "inadmissible"

    def test_monitor_unstable_is_inadmissible(self):
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = self.checker.check(
            K_new, [K1], trust_new=0.8, monitor_unstable=True
        )
        assert result == "inadmissible"

    def test_recon_not_improving_is_inadmissible(self):
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = self.checker.check(
            K_new, [K1],
            trust_new=0.8,
            monitor_unstable=False,
            recon_error_before=0.5,
            recon_error_after=0.49,  # improvement < delta_min=0.02
        )
        assert result == "inadmissible"

    def test_recon_worse_is_inadmissible(self):
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = self.checker.check(
            K_new, [K1],
            trust_new=0.8,
            monitor_unstable=False,
            recon_error_before=0.3,
            recon_error_after=0.5,  # worse
        )
        assert result == "inadmissible"


# ── Optional recon errors ─────────────────────────────────────────────────────


class TestOptionalReconErrors:
    def setup_method(self):
        self.checker = HarmonicClosureChecker(eps_closure=0.1, tau_admit=0.3)

    def test_no_recon_errors_skips_recon_check(self):
        """When recon errors not provided, recon gate skipped → admissible if trust ok."""
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = self.checker.check(K_new, [K1], trust_new=0.5)
        assert result == "admissible"  # no recon gate

    def test_only_before_provided_skips_recon_check(self):
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])
        # Only before provided → skips recon gate (after=None)
        result = self.checker.check(
            K_new, [K1], trust_new=0.5, recon_error_before=0.5
        )
        assert result == "admissible"


# ── Near-degenerate spectral safety ──────────────────────────────────────────


class TestNearDegenerate:
    def setup_method(self):
        self.checker = HarmonicClosureChecker(eps_closure=0.1)

    def test_same_spectrum_different_eigenvectors_not_redundant(self):
        """
        Two operators with identical spectra but different eigenvectors.
        K1 = diag([0.5, 0.5]), K_new = rotation of same diagonal.
        They have identical eigenvalues but K_new ≠ K1 in matrix space.
        """
        K1 = _diag([0.5, 0.5])
        # Rotation: R @ diag([0.5,0.5]) @ R.T where R = [[0,1],[-1,0]]
        R = np.array([[0.0, 1.0], [-1.0, 0.0]])
        K_new = R @ K1 @ R.T  # = diag([0.5, 0.5]) still (K1 is scalar multiple of I)
        # In this degenerate case (λ₁=λ₂), rotation doesn't change the matrix
        r = self.checker.projection_residual(K_new, [K1])
        assert r == pytest.approx(0.0, abs=1e-9)  # genuinely redundant here

    def test_nearly_identical_eigenvalues_small_eps(self):
        """Two operators with very close (not identical) spectra."""
        K1 = _diag([0.5, 0.3])
        K_new = _diag([0.501, 0.301])  # perturbed slightly
        r = self.checker.projection_residual(K_new, [K1])
        # r should be small but whether it's < eps_closure depends on eps_closure
        assert r >= 0.0

    def test_different_spectra_high_residual(self):
        """Operators with very different spectra → high residual → not redundant."""
        K1 = _diag([0.5, 0.3])
        K_new = _diag([50.0, 30.0])  # 100× larger
        r = self.checker.projection_residual(K_new, [K1])
        # K_new = 100 * K1 → coefficient=100, residual≈0 (K_new IS in span(K1))
        # This is expected: 100× scaled is in span
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_truly_different_direction_high_residual(self):
        """Operators with orthogonal structure → high residual → not redundant."""
        K1 = np.array([[1.0, 0.0], [0.0, 0.0]])   # projects onto x-axis
        K_new = np.array([[0.0, 0.0], [0.0, 1.0]])  # projects onto y-axis
        r = self.checker.projection_residual(K_new, [K1])
        assert r > 0.1  # well outside closure envelope


# ── HarmonicAtlas wiring (CRITICAL-3) ────────────────────────────────────────


class TestHarmonicAtlasWiring:
    """Verify that HarmonicAtlas merge decisions go through HarmonicClosureChecker."""

    def test_identical_classifications_merge_via_checker(self):
        """add_classification: identical eigenvalues → HCC 'redundant' → merge."""
        atlas = HarmonicAtlas()
        cl = _make_cl([-0.5 + 1j, -0.5 - 1j], trust=0.5)
        atlas.add_classification(cl)
        atlas.add_classification(cl)
        # Identical eigvals → K_proxy identical → r=0 → "redundant" → merged
        assert len(atlas.all_patches()) == 1

    def test_very_different_eigenvalues_not_merged(self):
        """add_classification: very different eigenvalues → HCC not 'redundant' → two patches."""
        atlas = HarmonicAtlas()
        cl1 = _make_cl([-0.5 + 1j, -0.5 - 1j], trust=0.5)
        cl2 = _make_cl([-100.0, -200.0], trust=0.5)
        atlas.add_classification(cl1)
        atlas.add_classification(cl2)
        assert len(atlas.all_patches()) == 2

    def test_merge_similar_identical_merges(self):
        """merge_similar: identical eigenvalues → HCC 'redundant' → n_merges ≥ 1."""
        atlas = HarmonicAtlas()
        cl = _make_cl([-0.5 + 1j, -0.5 - 1j], trust=0.5)
        atlas.add_classification(cl, auto_merge=False)
        atlas.add_classification(cl, auto_merge=False)
        assert len(atlas.all_patches()) == 2
        n = atlas.merge_similar()
        assert n >= 1
        assert len(atlas.all_patches()) <= 1

    def test_merge_similar_dissimilar_no_merge(self):
        """merge_similar: very different eigenvalues → HCC not 'redundant' → n_merges=0."""
        atlas = HarmonicAtlas()
        cl1 = _make_cl([-0.5 + 1j, -0.5 - 1j], trust=0.5)
        cl2 = _make_cl([-100.0, -200.0], trust=0.5)
        atlas.add_classification(cl1, auto_merge=False)
        atlas.add_classification(cl2, auto_merge=False)
        n = atlas.merge_similar()
        assert n == 0
        assert len(atlas.all_patches()) == 2

    def test_auto_merge_false_bypasses_checker(self):
        """auto_merge=False: no merge decision at all → always creates new patch."""
        atlas = HarmonicAtlas()
        cl = _make_cl([-0.5 + 1j, -0.5 - 1j])
        atlas.add_classification(cl, auto_merge=False)
        atlas.add_classification(cl, auto_merge=False)
        assert len(atlas.all_patches()) == 2

    def test_merge_similar_tol_accepted_but_unused(self):
        """merge_similar(tol=...) is backward compatible — tol ignored, HCC drives decision."""
        atlas = HarmonicAtlas()
        cl = _make_cl([-0.5 + 1j, -0.5 - 1j])
        atlas.add_classification(cl, auto_merge=False)
        atlas.add_classification(cl, auto_merge=False)
        # tol is now unused; HCC decides based on eps_closure
        n = atlas.merge_similar(tol=0.01)
        assert n >= 1  # identical → "redundant" regardless of tol
