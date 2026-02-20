"""Tests for tensor/operator_equivalence.py — OperatorEquivalenceDetector."""

import numpy as np
import pytest

from tensor.patch_graph import Patch
from tensor.operator_equivalence import OperatorEquivalenceDetector


# ---------------------------------------------------------------------------
# Helpers — create Patches with controlled spectra
# ---------------------------------------------------------------------------

def _patch(pid: int, spectrum: list, patch_type: str = 'lca') -> Patch:
    n = 2
    basis = np.eye(n)[np.newaxis, :, :]
    centroid = np.zeros(n)
    return Patch(
        id=pid,
        patch_type=patch_type,
        operator_basis=basis,
        spectrum=np.array(spectrum, dtype=complex),
        centroid=centroid,
    )


def _rlc_patch() -> Patch:
    """Second-order RLC: poles near -0.5 ± 0.866j (ζ=0.5, ω_n=1)."""
    return _patch(0, [-0.5 + 0.866j, -0.5 - 0.866j])


def _spring_mass_patch() -> Patch:
    """
    Damped spring-mass with same natural frequency: poles near -0.5 ± 0.866j.
    Slightly perturbed to test near-equivalence.
    """
    return _patch(1, [-0.48 + 0.87j, -0.48 - 0.87j])


def _high_freq_patch() -> Patch:
    """High-frequency system: poles near -2 ± 4j — clearly different."""
    return _patch(2, [-2.0 + 4.0j, -2.0 - 4.0j])


def _single_pole_patch() -> Patch:
    """First-order system: single real pole at -1."""
    return _patch(3, [-1.0 + 0j])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        det = OperatorEquivalenceDetector()
        assert det.threshold == 0.3

    def test_custom_threshold(self):
        det = OperatorEquivalenceDetector(threshold=0.5)
        assert det.threshold == 0.5


# ---------------------------------------------------------------------------
# spectrum_distance
# ---------------------------------------------------------------------------

class TestSpectrumDistance:
    def setup_method(self):
        self.det = OperatorEquivalenceDetector(threshold=0.3)

    def test_identical_patches_have_zero_distance(self):
        p = _rlc_patch()
        d = self.det.spectrum_distance(p, p)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_same_structure_small_distance(self):
        p1 = _rlc_patch()         # poles -0.5 ± 0.866j
        p2 = _spring_mass_patch() # poles -0.48 ± 0.87j — very close
        d = self.det.spectrum_distance(p1, p2)
        assert d < 0.1, f"Expected small distance, got {d}"

    def test_different_systems_large_distance(self):
        p1 = _rlc_patch()
        p2 = _high_freq_patch()   # poles -2 ± 4j — dominant |Re| = 2 vs 0.5
        d = self.det.spectrum_distance(p1, p2)
        assert d > 0.5, f"Expected large distance, got {d}"

    def test_symmetric(self):
        p1 = _rlc_patch()
        p2 = _high_freq_patch()
        assert self.det.spectrum_distance(p1, p2) == pytest.approx(
            self.det.spectrum_distance(p2, p1)
        )

    def test_different_length_spectra_padded(self):
        """Spectrum of different lengths: shorter padded with zeros."""
        p1 = _rlc_patch()            # 2 poles
        p2 = _single_pole_patch()    # 1 pole
        d = self.det.spectrum_distance(p1, p2)
        # |Re(λ)| of RLC = [0.5, 0.5] sorted → [0.5, 0.5]
        # |Re(λ)| of single = [1.0] padded → [1.0, 0.0]
        # Wasserstein-1 = mean(|[0.5,0.5] - [1.0,0.0]|) = mean([0.5, 0.5]) = 0.5
        assert d == pytest.approx(0.5, abs=0.01)

    def test_nonnegative(self):
        patches = [_rlc_patch(), _spring_mass_patch(), _high_freq_patch()]
        for i in range(len(patches)):
            for j in range(len(patches)):
                d = self.det.spectrum_distance(patches[i], patches[j])
                assert d >= 0.0


# ---------------------------------------------------------------------------
# are_equivalent
# ---------------------------------------------------------------------------

class TestAreEquivalent:
    def test_rlc_spring_mass_equivalent(self):
        """RLC and spring-mass with same frequency are structurally equivalent."""
        det = OperatorEquivalenceDetector(threshold=0.3)
        p1 = _rlc_patch()
        p2 = _spring_mass_patch()
        assert det.are_equivalent(p1, p2) is True

    def test_different_systems_not_equivalent(self):
        det = OperatorEquivalenceDetector(threshold=0.3)
        p1 = _rlc_patch()
        p2 = _high_freq_patch()
        assert det.are_equivalent(p1, p2) is False

    def test_identical_is_equivalent(self):
        det = OperatorEquivalenceDetector(threshold=0.3)
        p = _rlc_patch()
        assert det.are_equivalent(p, p) is True


# ---------------------------------------------------------------------------
# find_equivalences
# ---------------------------------------------------------------------------

class TestFindEquivalences:
    def test_returns_all_pairs(self):
        det = OperatorEquivalenceDetector(threshold=0.3)
        patches = [_rlc_patch(), _spring_mass_patch(), _high_freq_patch()]
        results = det.find_equivalences(patches)
        # 3 patches → 3 pairs (3 choose 2)
        assert len(results) == 3

    def test_result_structure(self):
        det = OperatorEquivalenceDetector()
        patches = [_rlc_patch(), _spring_mass_patch()]
        results = det.find_equivalences(patches)
        assert len(results) == 1
        r = results[0]
        assert "patch_a" in r
        assert "patch_b" in r
        assert "distance" in r
        assert "equivalent" in r
        assert isinstance(r["distance"], float)
        assert isinstance(r["equivalent"], (bool, np.bool_))

    def test_rlc_spring_mass_pair_found_equivalent(self):
        det = OperatorEquivalenceDetector(threshold=0.3)
        p1 = _rlc_patch()
        p2 = _spring_mass_patch()
        p3 = _high_freq_patch()
        results = det.find_equivalences([p1, p2, p3])
        equiv_pairs = [r for r in results if r["equivalent"]]
        # RLC ≡ spring-mass; high_freq should NOT be equivalent to either
        assert len(equiv_pairs) >= 1
        pair_ids = {(r["patch_a"], r["patch_b"]) for r in equiv_pairs}
        assert (0, 1) in pair_ids  # RLC(0) ≡ spring-mass(1)

    def test_empty_list(self):
        det = OperatorEquivalenceDetector()
        results = det.find_equivalences([])
        assert results == []

    def test_single_patch(self):
        det = OperatorEquivalenceDetector()
        results = det.find_equivalences([_rlc_patch()])
        assert results == []  # no pairs


# ---------------------------------------------------------------------------
# equivalence_matrix
# ---------------------------------------------------------------------------

class TestEquivalenceMatrix:
    def test_shape(self):
        det = OperatorEquivalenceDetector()
        patches = [_rlc_patch(), _spring_mass_patch(), _high_freq_patch()]
        mat = det.equivalence_matrix(patches)
        assert mat.shape == (3, 3)

    def test_diagonal_zero(self):
        det = OperatorEquivalenceDetector()
        patches = [_rlc_patch(), _spring_mass_patch()]
        mat = det.equivalence_matrix(patches)
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[1, 1] == pytest.approx(0.0)

    def test_symmetric(self):
        det = OperatorEquivalenceDetector()
        patches = [_rlc_patch(), _spring_mass_patch(), _high_freq_patch()]
        mat = det.equivalence_matrix(patches)
        np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_matches_pairwise_distance(self):
        det = OperatorEquivalenceDetector()
        p1, p2 = _rlc_patch(), _high_freq_patch()
        mat = det.equivalence_matrix([p1, p2])
        expected = det.spectrum_distance(p1, p2)
        assert mat[0, 1] == pytest.approx(expected)
        assert mat[1, 0] == pytest.approx(expected)

    def test_empty(self):
        det = OperatorEquivalenceDetector()
        mat = det.equivalence_matrix([])
        assert mat.shape == (0, 0)
