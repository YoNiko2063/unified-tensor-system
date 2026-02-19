"""
Koopman EDMD — Extended Dynamic Mode Decomposition for Koopman operator approximation.

Mathematical basis (LOGIC_FLOW.md, Section 0D Step 5, Section 0F):
  Given data pairs xₖ → xₖ₊₁ and observable basis ψ:
    G = (1/m) Σ ψ(xₖ)ψ(xₖ)ᵀ    (auto-correlation)
    A = (1/m) Σ ψ(xₖ)ψ(xₖ₊₁)ᵀ  (cross-correlation)
    K = G⁺A                       (Koopman matrix, pseudoinverse)

Eigenvalues of K → Koopman spectrum.
Eigenfunctions evolve as: ϕ(x_t) = e^{Λ(t)} ϕ(x_0)

Reference: LOGIC_FLOW.md Sections 0A, 0D, 0F
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class KoopmanResult:
    eigenvalues: np.ndarray       # complex (k,) Koopman eigenvalues
    eigenvectors: np.ndarray      # (k, k) right eigenvectors of K
    K_matrix: np.ndarray          # (k, k) Koopman operator matrix
    spectral_gap: float           # |Re(λ₁) - Re(λ₂)| of dominant modes
    is_stable: bool               # True if spectral gap > threshold


class EDMDKoopman:
    """
    Extended Dynamic Mode Decomposition — approximate Koopman operator from data.

    Koopman linearizes nonlinear dynamics in observable space:
      x_{t+1} = Φ(x_t)  [nonlinear in state space]
      g(x_{t+1}) = K · g(x_t)  [linear in observable space]

    Usage:
        koopman = EDMDKoopman(observable_degree=2)

        # Build pairs from trajectory
        pairs = [(traj[i], traj[i+1]) for i in range(len(traj)-1)]
        koopman.fit(pairs)

        result = koopman.eigendecomposition()
        print(result.spectral_gap)  # large gap → coherent dominant mode
    """

    def __init__(
        self,
        observable_degree: int = 2,
        spectral_gap_threshold: float = 0.1,
    ):
        """
        Args:
            observable_degree: polynomial degree for observable basis ψ
                degree=1: just state x
                degree=2: x + x² (monomials up to degree 2)
            spectral_gap_threshold: minimum gap for stability classification
        """
        self.degree = observable_degree
        self.gap_threshold = spectral_gap_threshold
        self._K = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Observable basis ψ(x)
    # ------------------------------------------------------------------

    def build_observable_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Polynomial observable basis ψ(x) up to specified degree.

        Args:
            x: (n,) state vector

        Returns:
            ψ: (k,) observable vector
        """
        features = [np.ones(1), x.copy()]
        if self.degree >= 2:
            # All degree-2 monomials: xᵢxⱼ for i ≤ j
            n = len(x)
            for i in range(n):
                for j in range(i, n):
                    features.append(np.array([x[i] * x[j]]))
        if self.degree >= 3:
            n = len(x)
            for i in range(n):
                features.append(np.array([x[i] ** 3]))
        return np.concatenate(features)

    def _observable_dim(self, n_states: int) -> int:
        """Number of observable dimensions for given state dimension."""
        k = 1 + n_states  # constant + linear
        if self.degree >= 2:
            k += n_states * (n_states + 1) // 2
        if self.degree >= 3:
            k += n_states
        return k

    # ------------------------------------------------------------------
    # EDMD core: G = (1/m)Σψ(xₖ)ψ(xₖ)ᵀ, A = (1/m)Σψ(xₖ)ψ(xₖ₊₁)ᵀ, K = G⁺A
    # ------------------------------------------------------------------

    def fit(self, trajectory_pairs: list) -> "EDMDKoopman":
        """
        Fit Koopman matrix from data pairs.

        Args:
            trajectory_pairs: list of (x_k, x_{k+1}) tuples or (T-1, 2, n) array

        Returns:
            self (for chaining)
        """
        if isinstance(trajectory_pairs, np.ndarray):
            # Shape (T-1, 2, n)
            pairs = [(trajectory_pairs[i, 0], trajectory_pairs[i, 1])
                     for i in range(len(trajectory_pairs))]
        else:
            pairs = trajectory_pairs

        m = len(pairs)
        if m == 0:
            raise ValueError("Need at least one trajectory pair")

        # Build observable matrices
        psi_k = np.array([self.build_observable_basis(xk) for xk, _ in pairs])    # (m, k)
        psi_kp1 = np.array([self.build_observable_basis(xkp1) for _, xkp1 in pairs])  # (m, k)

        # EDMD matrices
        G = (psi_k.T @ psi_k) / m        # (k, k) auto-correlation
        A = (psi_k.T @ psi_kp1) / m      # (k, k) cross-correlation

        # Koopman matrix K = G⁺ A
        self._K = np.linalg.lstsq(G, A, rcond=None)[0]
        self._fitted = True
        return self

    def fit_trajectory(self, trajectory: np.ndarray) -> "EDMDKoopman":
        """Convenience wrapper: fit from (T, n) trajectory array."""
        pairs = [(trajectory[i], trajectory[i + 1]) for i in range(len(trajectory) - 1)]
        return self.fit(pairs)

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def eigendecomposition(self) -> KoopmanResult:
        """
        Compute Koopman eigenvalues and eigenfunctions.

        Returns:
            KoopmanResult with eigenvalues, eigenvectors, spectral_gap, is_stable
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before eigendecomposition()")

        eigvals, eigvecs = np.linalg.eig(self._K)

        # Sort by magnitude (dominant modes first)
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        gap = self.spectral_gap(eigvals)
        stable = gap > self.gap_threshold

        return KoopmanResult(
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            K_matrix=self._K.copy(),
            spectral_gap=gap,
            is_stable=stable,
        )

    def spectral_gap(self, eigenvalues: Optional[np.ndarray] = None) -> float:
        """
        Δ = |Re(λ₁) - Re(λ₂)| of the two dominant Koopman eigenvalues.

        Large gap → dominant coherent mode → patch is spectrally coherent.
        """
        if eigenvalues is None:
            if not self._fitted:
                return 0.0
            eigenvalues = np.linalg.eigvals(self._K)
            eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]

        if len(eigenvalues) < 2:
            return float(np.abs(np.real(eigenvalues[0])))

        r = np.abs(np.real(eigenvalues))
        r_sorted = np.sort(r)[::-1]
        return float(r_sorted[0] - r_sorted[1])

    def eigenfunction_stability(
        self,
        prev_result: Optional[KoopmanResult],
        curr_result: Optional[KoopmanResult] = None,
    ) -> float:
        """
        Measure stability of eigenfunctions: ‖ϕₜ - ϕₜ₋Δₜ‖.

        Small value → stable spectral structure → patch classification reliable.
        """
        if curr_result is None:
            if not self._fitted:
                return float('inf')
            curr_result = self.eigendecomposition()

        if prev_result is None:
            return 0.0

        # Compare dominant eigenvalues (magnitude)
        prev_eigs = np.sort(np.abs(prev_result.eigenvalues))[::-1]
        curr_eigs = np.sort(np.abs(curr_result.eigenvalues))[::-1]

        min_len = min(len(prev_eigs), len(curr_eigs))
        return float(np.linalg.norm(curr_eigs[:min_len] - prev_eigs[:min_len]))

    def predict_next_observable(self, x: np.ndarray) -> np.ndarray:
        """
        Predict next observable state: ψ(x_{t+1}) ≈ K · ψ(x_t)

        Returns observable-space prediction (not state-space).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        psi = self.build_observable_basis(x)
        return self._K @ psi
