"""
Operator Equivalence Detector — cross-system comparison via Koopman spectral distance.

Mathematical basis (LOGIC_FLOW.md, Section 0J):
  Two systems are Koopman-equivalent iff their spectra are close under Wasserstein-1.
  Wasserstein-1 on 1D distributions = L1 distance of sorted arrays (closed form).

  d(A, B) = (1/k) Σᵢ |sort(|Re(λ_A)|)ᵢ - sort(|Re(λ_B)|)ᵢ|

  This identifies cross-domain structural equivalences:
    RLC ≡ spring-mass  (both second-order linear, same operator structure)
    PID ≡ damped harmonic oscillator  (same pole structure)

Reference: LOGIC_FLOW.md Sections 0F, 0J
"""

from __future__ import annotations

from typing import List

import numpy as np

from tensor.patch_graph import Patch


class OperatorEquivalenceDetector:
    """
    Detects structural equivalences between dynamical system patches by comparing
    their Koopman operator spectra using Wasserstein-1 distance.

    Usage:
        detector = OperatorEquivalenceDetector(threshold=0.3)

        pairs = detector.find_equivalences([patch_a, patch_b, patch_c])
        # [{"patch_a": 0, "patch_b": 1, "distance": 0.12, "equivalent": True}, ...]

        mat = detector.equivalence_matrix([patch_a, patch_b])
        # 2×2 symmetric distance matrix
    """

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Wasserstein-1 distance below which two patches are declared equivalent.
                       0.3 works well for normalized spectra; increase to 0.5 for more
                       tolerance across different physical scales.
        """
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Core distance metric
    # ------------------------------------------------------------------

    def spectrum_distance(self, patch_a: Patch, patch_b: Patch) -> float:
        """
        Wasserstein-1 distance between two Koopman spectra.

        Uses |Re(λ)| (real parts of eigenvalues) as 1D distributions.
        Wasserstein-1 on 1D = L1 distance of sorted arrays (closed form, O(k log k)).

        Padding: shorter spectrum padded with zeros (zero eigenvalues = stable modes
        that don't contribute to dynamics — physically meaningful default).

        Returns:
            float ≥ 0.0; near 0 means spectrally equivalent systems.
        """
        sa = np.abs(np.real(patch_a.spectrum))
        sb = np.abs(np.real(patch_b.spectrum))

        # Sort descending (dominant modes first, padding doesn't affect sorted order)
        sa = np.sort(sa)[::-1]
        sb = np.sort(sb)[::-1]

        # Pad shorter to same length with zeros
        max_len = max(len(sa), len(sb))
        sa = np.pad(sa, (0, max_len - len(sa)))
        sb = np.pad(sb, (0, max_len - len(sb)))

        return float(np.mean(np.abs(sa - sb)))

    # ------------------------------------------------------------------
    # Equivalence checks
    # ------------------------------------------------------------------

    def are_equivalent(self, patch_a: Patch, patch_b: Patch) -> bool:
        """True if spectral distance < threshold."""
        return self.spectrum_distance(patch_a, patch_b) < self.threshold

    def find_equivalences(self, patches: List[Patch]) -> List[dict]:
        """
        Find all equivalent patch pairs in a list.

        Returns:
            List of dicts, one per pair (i < j) below threshold:
            [{"patch_a": id_a, "patch_b": id_b, "distance": d, "equivalent": True}, ...]
            Includes non-equivalent pairs with "equivalent": False for full audit.
        """
        results = []
        n = len(patches)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.spectrum_distance(patches[i], patches[j])
                results.append({
                    "patch_a": patches[i].id,
                    "patch_b": patches[j].id,
                    "distance": d,
                    "equivalent": d < self.threshold,
                })
        return results

    def equivalence_matrix(self, patches: List[Patch]) -> np.ndarray:
        """
        Compute n×n symmetric pairwise Wasserstein-1 distance matrix.

        Entry [i,j] = spectrum_distance(patches[i], patches[j]).
        Diagonal = 0.0 (each patch is equivalent to itself).

        Useful for visualization (heatmap) and clustering.
        """
        n = len(patches)
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.spectrum_distance(patches[i], patches[j])
                mat[i, j] = d
                mat[j, i] = d
        return mat
