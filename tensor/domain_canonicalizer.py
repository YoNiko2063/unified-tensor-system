"""
DomainCanonicalizer — maps ingested equation signatures to HarmonicAtlas patches.

Pipeline:
  1. Receive eigenvalue array extracted from a paper's linearized DEQ
  2. Build spectral signature: dominant frequency, damping, spectral gap
  3. Compute pairwise DissonanceMetric τ against every LCA patch in atlas
  4. Find nearest patch (minimum τ on dominant frequencies)
  5. Compute integer interval ratio p:q for the frequency relationship
  6. Return CanonicalMatch if τ < tau_threshold, else None

This is real spectral domain mapping — not embedding cosine similarity.
Two systems are "the same domain" when their characteristic frequencies
satisfy a simple rational ratio (τ small), not when their descriptions
happen to share vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tensor.harmonic_atlas import HarmonicAtlas
from tensor.patch_graph import Patch
from tensor.spectral_path import DissonanceMetric


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class CanonicalMatch:
    """Result of a domain recognition query."""
    patch_id: int
    domain: str            # from Patch.metadata['domain'], or '' if unset
    interval_ratio: str    # e.g. '3:2' — paper freq : patch freq
    spectral_distance: float   # raw τ value (lower = closer)
    dominant_freq_ratio: float # omega_paper / omega_patch
    confidence: float      # 1 - spectral_distance / tau_threshold, clamped [0,1]


# ── DomainCanonicalizer ───────────────────────────────────────────────────────

class DomainCanonicalizer:
    """
    Recognizes which atlas domain a new eigenvalue signature belongs to.

    Usage:
        canonicalizer = DomainCanonicalizer(atlas, tau_threshold=0.5)

        # After ingesting a paper and extracting its linearized eigenvalues:
        eigvals = np.array([-50.0 + 6283j, -50.0 - 6283j])  # ~1 kHz damped
        match = canonicalizer.recognize(eigvals)
        if match:
            print(match.domain, match.interval_ratio)  # 'solar_mppt', '1:1'
    """

    def __init__(
        self,
        atlas: HarmonicAtlas,
        K: int = 10,
        tau_threshold: float = 0.5,
    ):
        """
        Args:
            atlas:         HarmonicAtlas to query against
            K:             max numerator/denominator for rational approximation
            tau_threshold: maximum dissonance τ to accept as a match.
                           Tune based on domain: 0.5 is permissive, 0.05 strict.
        """
        self.atlas = atlas
        self.tau_threshold = tau_threshold
        self._dissonance = DissonanceMetric(K=K)
        self._K = K

    def recognize(
        self,
        eigenvalues: np.ndarray,
        domain_hint: Optional[str] = None,
    ) -> Optional[CanonicalMatch]:
        """
        Find the nearest LCA atlas patch for the given eigenvalue signature.

        Args:
            eigenvalues:  complex array from linearized DEQ at equilibrium
            domain_hint:  optional filter — only consider patches from this domain.
                          Pass None to search all LCA patches.

        Returns:
            CanonicalMatch if τ < tau_threshold, else None.
        """
        patches = self.atlas.lca_patches()
        if not patches:
            return None

        if domain_hint is not None:
            filtered = [
                p for p in patches
                if p.metadata.get('domain') == domain_hint
            ]
            patches = filtered if filtered else patches

        omega_new = self._dominant_freq(eigenvalues)

        best_patch: Optional[Patch] = None
        best_tau = float('inf')
        best_omega_patch = 1.0

        for patch in patches:
            omega_patch = self._dominant_freq(patch.spectrum)
            tau = self._dissonance.compute(omega_new, omega_patch)
            if tau < best_tau:
                best_tau = tau
                best_patch = patch
                best_omega_patch = omega_patch

        if best_patch is None or best_tau >= self.tau_threshold:
            return None

        freq_ratio = (
            omega_new / best_omega_patch
            if abs(best_omega_patch) > 1e-12
            else 0.0
        )
        confidence = float(
            np.clip(1.0 - best_tau / self.tau_threshold, 0.0, 1.0)
        )

        return CanonicalMatch(
            patch_id=best_patch.id,
            domain=best_patch.metadata.get('domain', ''),
            interval_ratio=self._ratio_str(omega_new, best_omega_patch),
            spectral_distance=best_tau,
            dominant_freq_ratio=freq_ratio,
            confidence=confidence,
        )

    def recognize_batch(
        self,
        eigenvalue_sets: List[np.ndarray],
        domain_hint: Optional[str] = None,
    ) -> List[Optional[CanonicalMatch]]:
        """Batch recognition for multiple eigenvalue arrays."""
        return [self.recognize(ev, domain_hint) for ev in eigenvalue_sets]

    def nearest_patch_spectrum(
        self, eigenvalues: np.ndarray
    ) -> Optional[Patch]:
        """
        Return the nearest LCA patch regardless of tau_threshold.
        Useful for inspecting the atlas neighbour even when no match is found.
        """
        patches = self.atlas.lca_patches()
        if not patches:
            return None
        omega_new = self._dominant_freq(eigenvalues)
        taus = [
            self._dissonance.compute(omega_new, self._dominant_freq(p.spectrum))
            for p in patches
        ]
        return patches[int(np.argmin(taus))]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _dominant_freq(self, eigvals: np.ndarray) -> float:
        """Max |Im(λ)| across all eigenvalues. Zero for pure-real spectra."""
        if len(eigvals) == 0:
            return 0.0
        return float(np.max(np.abs(np.imag(eigvals))))

    def _ratio_str(self, omega_i: float, omega_j: float) -> str:
        """
        Find best p:q rational approximation (1 ≤ p,q ≤ K) for omega_i/omega_j.

        Returns '0:1' if omega_j is near zero, 'p:q' otherwise.
        """
        if abs(omega_j) < 1e-12:
            return '0:1'
        target = omega_i / omega_j
        best_p, best_q = 1, 1
        best_err = abs(target - 1.0)
        for p in range(1, self._K + 1):
            for q in range(1, self._K + 1):
                err = abs(target - p / q)
                if err < best_err:
                    best_err = err
                    best_p, best_q = p, q
        return f'{best_p}:{best_q}'
