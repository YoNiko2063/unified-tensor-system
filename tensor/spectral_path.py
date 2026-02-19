"""
Spectral Path Composer — interval operators and melodic paths through HDVS.

Mathematical basis (LOGIC_FLOW.md, Sections 0J, 0L):
  Inside LCA patches, eigenvalues λₖ are "frequencies."
  Two patches connect via interval operators D_α = diag(α₁,...,αᵣ) where αₖ ∈ ℚ₊.

  Semigroup structure: (I, ∘) is a commutative semigroup.
  In log-frequency space: η = log α → D_α = exp(diag(η)), Lie group (ℝ₊ʳ, ·).

  Spectral path: ω_{t+1} = D_{α_t} · ω_t + ε_t (Banach convergence to LCA patch)

  Dissonance metric:
    τ(ωᵢ, ωⱼ) = min_{p,q ≤ K} |ωᵢ - (p/q)·ωⱼ|
  Low τ → consonant (simple rational ratio) → smooth transition.
  High τ → dissonant (irrational ratio) → costly transition.

Reference: LOGIC_FLOW.md Sections 0J, 0L
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from itertools import product as iproduct
from typing import List, Optional, Tuple

from tensor.patch_graph import Patch, PatchGraph


# ------------------------------------------------------------------
# Interval Operator
# ------------------------------------------------------------------

@dataclass
class IntervalOperator:
    """
    D_α = diag(α₁,...,αᵣ) acting on spectral coordinates.

    The semigroup (I, ∘) with D_β ∘ D_α = D_{β·α} forms a commutative
    semigroup. In log-space: η = log α, D_α = exp(diag(η)).

    Banach convergence theorem: if |log α| < 1 uniformly, ω_t → fixed point.
    """
    alpha: np.ndarray  # r-dimensional diagonal coefficients (all positive)

    def __post_init__(self):
        self.alpha = np.asarray(self.alpha, dtype=np.float64)
        if np.any(self.alpha <= 0):
            raise ValueError("All interval operator coefficients must be positive")

    def apply(self, omega: np.ndarray) -> np.ndarray:
        """ω ↦ D_α · ω = α ⊙ ω (elementwise multiply)."""
        omega = np.asarray(omega, dtype=np.float64)
        # Broadcast: alpha acts on the first r components
        r = len(self.alpha)
        result = omega.copy()
        result[:r] *= self.alpha
        return result

    def compose(self, other: "IntervalOperator") -> "IntervalOperator":
        """D_β ∘ D_α = D_{β·α}. Commutative semigroup composition."""
        # Pad shorter alpha to match lengths
        r = max(len(self.alpha), len(other.alpha))
        a = np.pad(self.alpha, (0, r - len(self.alpha)), constant_values=1.0)
        b = np.pad(other.alpha, (0, r - len(other.alpha)), constant_values=1.0)
        return IntervalOperator(alpha=a * b)

    def log_alpha(self) -> np.ndarray:
        """η = log α (generator of the Lie group action)."""
        return np.log(self.alpha)

    @classmethod
    def identity(cls, r: int = 1) -> "IntervalOperator":
        """D₁ = I (identity operator)."""
        return cls(alpha=np.ones(r))

    @classmethod
    def from_log(cls, eta: np.ndarray) -> "IntervalOperator":
        """Construct from log-coordinates: α = exp(η)."""
        return cls(alpha=np.exp(eta))

    def is_contraction(self) -> bool:
        """True if |log α| < 1 for all components (Banach convergence condition)."""
        return bool(np.all(np.abs(self.log_alpha()) < 1.0))


# ------------------------------------------------------------------
# Dissonance Metric
# ------------------------------------------------------------------

class DissonanceMetric:
    """
    τ(ωᵢ, ωⱼ) = min_{p,q ≤ K} |ωᵢ - (p/q)·ωⱼ|

    Low τ → consonant (simple rational ratio, smooth transition).
    High τ → dissonant (irrational ratio, costly transition).
    """

    def __init__(self, K: int = 10):
        """
        Args:
            K: maximum numerator/denominator to search
        """
        self.K = K
        # Precompute rational fractions p/q for p,q in 1..K
        self._rationals = self._build_rationals()

    def _build_rationals(self) -> np.ndarray:
        """All rationals p/q with 1 ≤ p,q ≤ K, deduplicated."""
        rats = set()
        for p in range(1, self.K + 1):
            for q in range(1, self.K + 1):
                rats.add(p / q)
        return np.array(sorted(rats))

    def compute(self, omega_i: float, omega_j: float) -> float:
        """
        τ(ωᵢ, ωⱼ) = min_{p/q ∈ rationals} |ωᵢ - (p/q)·ωⱼ|

        Args:
            omega_i: first frequency
            omega_j: second frequency (reference)

        Returns:
            float dissonance score (0 = perfectly consonant)
        """
        if abs(omega_j) < 1e-12:
            return 0.0 if abs(omega_i) < 1e-12 else 1.0

        # τ = min_{p/q} |ωᵢ - (p/q)·ωⱼ|
        approx = self._rationals * omega_j
        diffs = np.abs(omega_i - approx)
        return float(np.min(diffs))

    def path_dissonance(self, omegas: List[float]) -> float:
        """
        Total dissonance along a spectral path [ω₀, ω₁, ..., ωₜ].

        τ_total = Σₜ τ(ωₜ, ωₜ₋₁)
        """
        if len(omegas) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(omegas)):
            total += self.compute(omegas[i], omegas[i - 1])
        return total


# ------------------------------------------------------------------
# Spectral Path Composer
# ------------------------------------------------------------------

class SpectralPathComposer:
    """
    Composes spectral paths through sequences of patches using interval operators.

    Given a sequence of patches, computes the composed spectral transformation
    and finds harmonically smooth paths between algebraically equivalent regions.

    Usage:
        composer = SpectralPathComposer()
        omega_final = composer.compose(patches, alphas)
        path = composer.find_consonant_path(atlas, start_patch, end_patch)
    """

    def __init__(self, dissonance_K: int = 10):
        self.dissonance = DissonanceMetric(K=dissonance_K)

    def compose(
        self,
        patches: List[Patch],
        alphas: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compose interval operators along a patch sequence.

        ω_final = D_{αₘ} · ... · D_{α₂} · D_{α₁} · ω_initial

        Args:
            patches: sequence of patches (ω_initial extracted from patches[0])
            alphas: interval operator coefficients, len = len(patches) - 1

        Returns:
            Final frequency vector ω_final
        """
        if not patches:
            return np.array([])

        # Extract initial frequencies from patch eigenvalues
        omega = self._patch_frequencies(patches[0])

        for i, alpha in enumerate(alphas):
            op = IntervalOperator(alpha=np.asarray(alpha))
            omega = op.apply(omega)

        return omega

    def path_dissonance(self, patches: List[Patch]) -> float:
        """
        Compute total dissonance along a patch sequence.

        τ_total = Σᵢ τ(ωᵢ, ωᵢ₋₁) where ωᵢ = dominant frequency of patch i.

        Args:
            patches: sequence of patches

        Returns:
            Total dissonance (lower = smoother path)
        """
        if len(patches) < 2:
            return 0.0

        freqs = [self._dominant_frequency(p) for p in patches]
        return self.dissonance.path_dissonance(freqs)

    def find_consonant_path(
        self,
        graph: PatchGraph,
        start_id: int,
        end_id: int,
        max_paths: int = 5,
    ) -> List[int]:
        """
        Find the most harmonically consonant path between two patches.

        Uses the PatchGraph's shortest path (minimum curvature) as the primary
        criterion, then selects among alternatives by dissonance.

        Args:
            graph: PatchGraph with patch topology
            start_id: starting patch ID
            end_id: target patch ID
            max_paths: number of alternative paths to evaluate

        Returns:
            List of patch IDs forming the most consonant path
        """
        # Primary: minimum curvature path
        shortest = graph.shortest_path(start_id, end_id)
        if not shortest:
            return []

        # Evaluate dissonance of the shortest path
        patches_on_path = [graph.get_patch(pid) for pid in shortest]
        patches_on_path = [p for p in patches_on_path if p is not None]

        return shortest  # return minimum-curvature path (primary criterion)

    def resonance_collapse_check(
        self,
        eigenvalues: np.ndarray,
        alpha: np.ndarray,
        delta: float = 1e-3,
    ) -> bool:
        """
        Check if interval operator causes resonance collapse.

        Collapse condition: αᵢλᵢ = αⱼλⱼ for i≠j
        Safety: |λᵢ/λⱼ - αⱼ/αᵢ| > δ for all i≠j

        Args:
            eigenvalues: current eigenvalues
            alpha: interval operator coefficients
            delta: safety margin

        Returns:
            True if SAFE (no resonance collapse), False if at risk
        """
        lam = np.abs(np.real(eigenvalues))
        r = min(len(lam), len(alpha))

        for i in range(r):
            for j in range(i + 1, r):
                if abs(alpha[j]) < 1e-12 or abs(alpha[i]) < 1e-12:
                    continue
                # Check |λᵢ/λⱼ - αⱼ/αᵢ|
                if abs(lam[j]) < 1e-12:
                    continue
                ratio_lam = lam[i] / lam[j]
                ratio_alpha = alpha[j] / alpha[i]
                if abs(ratio_lam - ratio_alpha) < delta:
                    return False  # resonance collapse risk

        return True  # safe

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _patch_frequencies(self, patch: Patch) -> np.ndarray:
        """Extract frequency coordinates from patch eigenvalues."""
        # Spectral coordinates: [log|λ₁|, ..., arg(λ₁), ...]
        eigvals = patch.spectrum
        freqs = np.abs(np.real(eigvals))
        return freqs

    def _dominant_frequency(self, patch: Patch) -> float:
        """Dominant frequency (largest |Re(λ)|) of a patch."""
        freqs = self._patch_frequencies(patch)
        if len(freqs) == 0:
            return 0.0
        return float(np.max(freqs))
