"""
LCA Patch Detector — classifies operating regions of nonlinear systems as:
  - 'lca'       : Locally Compact Abelian (Pontryagin duality, Laplace valid)
  - 'nonabelian': Non-abelian Lie algebra (Tannaka-Krein regime)
  - 'chaotic'   : High-curvature, no tractable spectral decomposition

Mathematical basis (LOGIC_FLOW.md, Section 0D):
  6-step pipeline: SVD rank → commutator test → curvature ratio → Koopman → classify

Reference: LOGIC_FLOW.md Sections 0A–0F
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Optional


@dataclass
class PatchClassification:
    patch_type: str                    # 'lca' | 'nonabelian' | 'chaotic'
    operator_rank: int                 # intrinsic dimension r
    commutator_norm: float             # max ‖[Aᵢ,Aⱼ]‖_F
    curvature_ratio: float             # ρ = ‖∇J‖ / λ_max
    spectral_gap: float                # |λ₁ - λ₂| from Jacobian eigenvalues
    basis_matrices: np.ndarray         # r × n × n operator basis
    eigenvalues: np.ndarray            # dominant eigenvalues at region centroid
    centroid: np.ndarray               # mean state in region


class LCAPatchDetector:
    """
    Detects and classifies LCA patches in a nonlinear dynamical system.

    Usage:
        def f(x):  # RLC+diode system
            v, iL = x
            return np.array([-(1/RC)*v - (3*alpha/C)*v**2 - (1/C)*iL, (1/L)*v])

        detector = LCAPatchDetector(f, n_states=2)
        x_samples = np.random.randn(50, 2) * 0.1  # small-signal region
        result = detector.classify_region(x_samples)
        # result.patch_type == 'lca'
    """

    def __init__(
        self,
        system_fn: Callable[[np.ndarray], np.ndarray],
        n_states: int,
        eps_curvature: float = 0.05,
        delta_commutator: float = 0.01,
        rank_tol: float = 1e-2,
        h: float = 1e-5,
    ):
        """
        Args:
            system_fn: callable x → ẋ (vector field)
            n_states: dimension of state space
            eps_curvature: curvature ratio threshold for exponential-dominant patch
            delta_commutator: commutator norm threshold for abelian classification
            rank_tol: singular value tolerance for operator rank estimation
            h: finite difference step for Jacobian/curvature computation
        """
        self.f = system_fn
        self.n = n_states
        self.eps_curvature = eps_curvature
        self.delta_commutator = delta_commutator
        self.rank_tol = rank_tol
        self.h = h

    # ------------------------------------------------------------------
    # Step 1: Sample Jacobian field
    # ------------------------------------------------------------------

    def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Finite-difference Jacobian J = Df(x), shape (n, n)."""
        J = np.zeros((self.n, self.n))
        f0 = self.f(x)
        for j in range(self.n):
            xph = x.copy()
            xph[j] += self.h
            J[:, j] = (self.f(xph) - f0) / self.h
        return J

    def sample_jacobians(self, x_samples: np.ndarray) -> List[np.ndarray]:
        """Compute Jacobians at all sample points. Returns list of (n,n) arrays."""
        return [self.compute_jacobian(x) for x in x_samples]

    # ------------------------------------------------------------------
    # Step 2: Operator subspace rank (SVD)
    # ------------------------------------------------------------------

    def operator_rank_svd(self, jacobians: List[np.ndarray]) -> tuple[int, np.ndarray]:
        """
        Stack vec(Jᵢ) → SVD → intrinsic operator dimension r.

        Returns:
            r: intrinsic rank
            basis: r × n × n array of orthonormal basis matrices
        """
        n = self.n
        stacked = np.array([J.reshape(-1) for J in jacobians])  # (m, n²)
        _, s, Vt = np.linalg.svd(stacked, full_matrices=False)

        # Estimate rank from singular value gap
        s_norm = s / (s[0] + 1e-12)
        r = int(np.sum(s_norm > self.rank_tol))
        r = max(r, 1)

        basis = Vt[:r].reshape(r, n, n)  # r × n × n
        return r, basis

    # ------------------------------------------------------------------
    # Step 3: Commutator test
    # ------------------------------------------------------------------

    def commutator_norms(self, basis: np.ndarray) -> np.ndarray:
        """
        Compute pairwise commutator norms ‖[Aᵢ, Aⱼ]‖_F.

        Returns: array of norms for all pairs (i < j)
        """
        r = basis.shape[0]
        norms = []
        for i, j in combinations(range(r), 2):
            C = basis[i] @ basis[j] - basis[j] @ basis[i]
            norms.append(np.linalg.norm(C, 'fro'))
        return np.array(norms) if norms else np.array([0.0])

    # ------------------------------------------------------------------
    # Step 4: Curvature ratio ρ(x) = ‖∇J(x)‖ / λ_max(x)
    # ------------------------------------------------------------------

    def curvature_ratio(self, x: np.ndarray) -> float:
        """
        ρ(x) = ‖∇J(x)‖_F / λ_max(x)

        Low ρ → exponential-dominant (Laplace valid)
        High ρ → curvature-dominated (need Koopman)
        """
        J0 = self.compute_jacobian(x)
        eigvals = np.linalg.eigvals(J0)
        lam_max = max(np.max(np.abs(np.real(eigvals))), 1e-10)

        # Frobenius norm of Jacobian derivative (finite difference of J)
        grad_J_norms = []
        for k in range(self.n):
            xph = x.copy()
            xph[k] += self.h
            Jph = self.compute_jacobian(xph)
            grad_J_norms.append(np.linalg.norm(Jph - J0, 'fro') / self.h)

        kappa = np.mean(grad_J_norms)
        return kappa / lam_max

    def curvature_ratios(self, x_samples: np.ndarray) -> np.ndarray:
        """Compute ρ(x) at all sample points."""
        return np.array([self.curvature_ratio(x) for x in x_samples])

    # ------------------------------------------------------------------
    # Step 5: Spectral gap from Jacobian eigenvalues
    # ------------------------------------------------------------------

    def spectral_gap(self, x: np.ndarray) -> float:
        """
        Δ = |Re(λ₁) - Re(λ₂)| for dominant eigenvalues.

        Large gap → dominant coherent mode → spectrally coherent patch.
        """
        J = self.compute_jacobian(x)
        eigvals = np.linalg.eigvals(J)
        real_parts = np.sort(np.abs(np.real(eigvals)))[::-1]
        if len(real_parts) < 2:
            return float(real_parts[0])
        return float(real_parts[0] - real_parts[1])

    # ------------------------------------------------------------------
    # Step 6: Classification
    # ------------------------------------------------------------------

    def classify_region(
        self,
        x_samples: np.ndarray,
    ) -> PatchClassification:
        """
        Classify a region of state space using the 6-step detection pipeline.

        Args:
            x_samples: (m, n) array of state samples within the region

        Returns:
            PatchClassification with patch_type ∈ {'lca', 'nonabelian', 'chaotic'}
        """
        centroid = x_samples.mean(axis=0)

        # Step 1: Jacobians
        jacobians = self.sample_jacobians(x_samples)

        # Step 2: Rank
        r, basis = self.operator_rank_svd(jacobians)

        # Step 3: Commutator
        comm_norms = self.commutator_norms(basis)
        max_comm = float(np.max(comm_norms))

        # Step 4: Curvature ratios
        rho_vals = self.curvature_ratios(x_samples)
        max_rho = float(np.max(rho_vals))

        # Step 5: Spectral gap
        gap = self.spectral_gap(centroid)

        # Step 6: Classify
        J_centroid = self.compute_jacobian(centroid)
        eigvals = np.linalg.eigvals(J_centroid)

        # Use mean curvature for classification (max can be noisy at outlier points)
        mean_rho = float(np.mean(rho_vals))
        low_curvature = mean_rho < self.eps_curvature
        moderate_curvature = mean_rho < (self.eps_curvature * 5)  # LCA with slow variation
        low_commutator = max_comm < self.delta_commutator
        low_rank = r <= max(self.n - 1, 1)

        if low_rank and low_commutator:
            # Abelian algebra regardless of curvature — this is the primary LCA condition
            patch_type = 'lca'
        elif low_rank and moderate_curvature:
            # Low rank but non-abelian commutator → Tannaka-Krein regime
            patch_type = 'nonabelian'
        else:
            patch_type = 'chaotic'

        return PatchClassification(
            patch_type=patch_type,
            operator_rank=r,
            commutator_norm=max_comm,
            curvature_ratio=mean_rho,    # store mean for consistency with classification
            spectral_gap=gap,
            basis_matrices=basis,
            eigenvalues=eigvals,
            centroid=centroid,
        )

    def classify_trajectory(
        self,
        trajectory: np.ndarray,
        window: int = 10,
    ) -> List[PatchClassification]:
        """
        Classify successive windows along a trajectory.

        Args:
            trajectory: (T, n) array of states
            window: number of points per classification window

        Returns:
            List of PatchClassification, one per window
        """
        T = len(trajectory)
        results = []
        for start in range(0, T - window + 1, window // 2):
            window_samples = trajectory[start:start + window]
            results.append(self.classify_region(window_samples))
        return results
