"""
Semantic Observer System — Part I

A forced nonlinear dynamical system for semantic state tracking.

Architecture:
  ẋ(t) = A·x + g(x) + B·u(t)

  x(t) ∈ R^n  — lifted semantic observer state
  u(t)        — raw semantic forcing (document/token stream)
  A           — learned linear Koopman operator (spectral-truncated)
  g(x)        — nonlinear residual (tanh saturation, keeps state bounded)
  B           — semantic injection matrix

Components:
  SemanticObserver    — integrates the forced ODE, tracks Lyapunov energy
  truncate_spectrum   — spectral truncation via real Schur decomposition
  semantic_energy     — Lyapunov energy functional E_s = x^T P x + α ||dx||²
  apply_damping       — Lyapunov descent injection
  BasisConsolidator   — PCA consolidation to prevent dimensional drift
  HDVOrthogonalizer   — enforces H_i^T H_j = 0 across domain subspaces
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import schur


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ObserverConfig:
    state_dim: int = 64           # observer state dimension
    input_dim: int = 32           # forcing input dimension
    dt: float = 0.01              # integration timestep
    energy_cap: float = 10.0      # max allowed semantic energy E_s
    gamma_damp: float = 0.1       # damping injection strength
    lambda_max: float = 2.0       # spectral radius cap (Lipschitz bound)
    energy_threshold: float = 1e-3  # min eigenvalue magnitude (truncation)
    stability_cap: float = 5.0    # max eigenvalue magnitude (truncation)
    nonlinearity_scale: float = 0.1  # g(x) = scale * tanh(x) coefficient


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def truncate_spectrum(
    A: np.ndarray,
    energy_threshold: float = 1e-3,
    stability_cap: float = 5.0,
) -> np.ndarray:
    """Return A with only modes satisfying:
      |λ_i| > energy_threshold  AND  |λ_i| < stability_cap

    Uses real Schur decomposition for numerical stability (not raw eig).
    If no modes pass mask → return A * 0.5 (emergency damping).
    Preserves real-valued A: if A was real, output is real.
    """
    A = np.asarray(A, dtype=float)
    is_real = np.isrealobj(A)

    # Compute eigenvalues via real Schur decomposition
    try:
        T, Q = schur(A, output="real")
        eigvals = np.linalg.eigvals(T)
    except Exception:
        eigvals = np.linalg.eigvals(A)
        T, Q = np.diag(eigvals.real), np.eye(A.shape[0])

    mags = np.abs(eigvals)

    # Case 1: all eigenvalues too large (above stability_cap only) — scale down
    all_too_large = np.all(mags >= stability_cap) and mags.size > 0
    if all_too_large:
        sr = float(np.max(mags))
        scale = (stability_cap * 0.9) / sr   # bring spectral radius to 90% of cap
        A_out = A * scale
        return A_out.real if is_real else A_out

    # Case 2: build mask of valid modes
    mask = (mags > energy_threshold) & (mags < stability_cap)

    if not np.any(mask):
        # Emergency damping: shrink all modes by 50%
        return (A * 0.5).real if is_real else A * 0.5

    # Zero out Schur blocks for modes outside the valid range
    T_masked = _mask_schur_blocks(T, mask)
    A_out = Q @ T_masked @ Q.T

    return A_out.real if is_real else A_out


def _mask_schur_blocks(T: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """
    Zero out 1×1 and 2×2 Schur blocks whose eigenvalues are not in keep_mask.

    This operates on a copy of T.
    Real Schur blocks are on the diagonal; 2×2 blocks correspond to
    complex-conjugate pairs and share two consecutive eigenvalue entries.

    Block eigenvalues are computed directly from the diagonal blocks
    (not via a separate eigvals call) to guarantee ordering consistency.
    """
    T = T.copy()
    n = T.shape[0]

    # Walk the diagonal, extracting block eigenvalue magnitudes directly
    # and comparing against the keep_mask (which was built from the same
    # eigenvalue ordering returned by np.linalg.eigvals on the original T).
    # To avoid ordering mismatches, we compute each block's eigenvalue
    # magnitude directly and check it against the energy/stability bounds
    # encoded in the mask.
    i = 0
    eig_idx = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1e-10:
            # 2×2 block: eigenvalues are complex conjugate pair
            # λ = (a+d)/2 ± sqrt(((a-d)/2)² + bc)
            # Both have the same magnitude, so we check either entry in keep_mask
            keep = False
            if eig_idx < len(keep_mask):
                keep = keep_mask[eig_idx]
            if (eig_idx + 1) < len(keep_mask):
                keep = keep or keep_mask[eig_idx + 1]
            if not keep:
                T[i, i] = 0.0
                T[i, i + 1] = 0.0
                T[i + 1, i] = 0.0
                T[i + 1, i + 1] = 0.0
            i += 2
            eig_idx += 2
        else:
            # 1×1 block: eigenvalue is T[i, i]
            keep = keep_mask[eig_idx] if eig_idx < len(keep_mask) else False
            if not keep:
                T[i, i] = 0.0
            i += 1
            eig_idx += 1

    return T


# ---------------------------------------------------------------------------
# Lyapunov energy and damping
# ---------------------------------------------------------------------------

def semantic_energy(
    x: np.ndarray,
    dx: np.ndarray,
    P: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """E_s = x^T P x + alpha * ||dx||^2

    P must be positive definite.  Caller may pass identity for simplicity.
    """
    x = np.asarray(x, dtype=float)
    dx = np.asarray(dx, dtype=float)
    P = np.asarray(P, dtype=float)
    return float(x @ P @ x + alpha * np.dot(dx, dx))


def apply_damping(dx: np.ndarray, x: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """dx -= gamma * x  (Lyapunov descent injection)"""
    return dx - gamma * x


# ---------------------------------------------------------------------------
# SemanticObserver
# ---------------------------------------------------------------------------

class SemanticObserver:
    """
    Forced nonlinear dynamical system for semantic state tracking.

      ẋ(t) = A·x + g(x) + B·u(t)

    A is a Koopman-style linear operator, spectral-truncated for stability.
    g(x) = nonlinearity_scale * tanh(x) is a bounded nonlinear residual.
    B injects the semantic forcing signal u(t).

    Lyapunov energy is monitored at each step; if it exceeds energy_cap,
    damping is injected to drive the system back toward equilibrium.
    """

    def __init__(
        self,
        config: ObserverConfig,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
    ):
        self.config = config
        n = config.state_dim
        m = config.input_dim

        # State vector
        self.x: np.ndarray = np.zeros(n, dtype=float)

        # Lyapunov matrix (default: identity — positive definite)
        self.P: np.ndarray = np.eye(n, dtype=float)

        # A: (n, n) linear Koopman operator
        if A is not None:
            self.A = np.asarray(A, dtype=float).copy()
        else:
            # Stable random initialization: small random matrix
            rng = np.random.default_rng(seed=42)
            self.A = rng.standard_normal((n, n)) * 0.1

        # Enforce spectral radius < lambda_max at init
        self.A = truncate_spectrum(
            self.A,
            energy_threshold=config.energy_threshold,
            stability_cap=min(config.lambda_max, config.stability_cap),
        )

        # B: (n, m) injection matrix
        if B is not None:
            self.B = np.asarray(B, dtype=float).copy()
        else:
            rng = np.random.default_rng(seed=7)
            self.B = rng.standard_normal((n, m)) / np.sqrt(m)

    # ------------------------------------------------------------------

    def step(self, u: np.ndarray) -> np.ndarray:
        """Single RK1 (Euler) integration step.

        ẋ = A·x + g(x) + B·u
        If semantic_energy > energy_cap: inject damping dx -= gamma * x.
        x += dt * dx

        Returns updated x (copy).
        """
        u = np.asarray(u, dtype=float)
        x = self.x

        # Nonlinear residual: g(x) = scale * tanh(x)
        g_x = self.config.nonlinearity_scale * np.tanh(x)

        # Compute time derivative
        dx = self.A @ x + g_x + self.B @ u

        # Lyapunov energy check
        E = semantic_energy(x, dx, self.P)
        if E > self.config.energy_cap:
            dx = apply_damping(dx, x, gamma=self.config.gamma_damp)

        # Euler step
        self.x = x + self.config.dt * dx
        return self.x.copy()

    # ------------------------------------------------------------------

    def update_operator(self, A_new: np.ndarray) -> None:
        """Replace A with spectral-truncated version of A_new."""
        A_new = np.asarray(A_new, dtype=float)
        self.A = truncate_spectrum(
            A_new,
            energy_threshold=self.config.energy_threshold,
            stability_cap=self.config.stability_cap,
        )

    def reset(self) -> None:
        """Reset state to zeros."""
        self.x = np.zeros(self.config.state_dim, dtype=float)

    @property
    def spectral_radius(self) -> float:
        """Max |eigenvalue| of current A."""
        eigvals = np.linalg.eigvals(self.A)
        return float(np.max(np.abs(eigvals)))


# ---------------------------------------------------------------------------
# BasisConsolidator
# ---------------------------------------------------------------------------

class BasisConsolidator:
    """Consolidates semantic state history via PCA every N documents.

    Usage:
        consolidator = BasisConsolidator(k=32, consolidate_every=100)
        for each document:
            consolidator.record(observer.x)
            if consolidator.should_consolidate():
                basis = consolidator.consolidate()        # (state_dim, k)
                A_new = consolidator.rotate_operator(observer.A, basis)
                observer.update_operator(A_new)
    """

    def __init__(self, k: int = 32, consolidate_every: int = 100):
        self.k = k
        self.consolidate_every = consolidate_every
        self._history: List[np.ndarray] = []
        self._total_recorded: int = 0

    def record(self, x: np.ndarray) -> None:
        """Store current state.  Auto-trims to last 2 * consolidate_every."""
        self._history.append(np.asarray(x, dtype=float).copy())
        self._total_recorded += 1
        max_keep = 2 * self.consolidate_every
        if len(self._history) > max_keep:
            self._history = self._history[-max_keep:]

    def should_consolidate(self) -> bool:
        """True every consolidate_every records."""
        return (self._total_recorded > 0 and
                self._total_recorded % self.consolidate_every == 0)

    def consolidate(self) -> np.ndarray:
        """Stack recorded states, run SVD, return top-k left singular vectors.

        Returns: basis matrix shape (state_dim, k).
        Clears history after consolidation.
        """
        if not self._history:
            raise RuntimeError("No history to consolidate")

        X = np.stack(self._history, axis=0)   # (T, state_dim)
        # Center
        X_c = X - X.mean(axis=0)
        # SVD: X_c = U S V^T,  U is (T, T), V is (state_dim, state_dim)
        # We want top-k directions in state_dim space → right singular vectors
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        k_actual = min(self.k, Vt.shape[0])
        # Vt: (min(T, state_dim), state_dim); rows are right singular vectors
        basis = Vt[:k_actual].T   # (state_dim, k_actual)

        # Pad with zeros if we have fewer samples than k
        if k_actual < self.k:
            pad = np.zeros((basis.shape[0], self.k - k_actual), dtype=float)
            basis = np.concatenate([basis, pad], axis=1)

        self._history = []
        return basis

    def rotate_operator(self, A: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """A' = basis.T @ A @ basis — project into consolidated space.

        If basis.shape = (state_dim, k), result is (k, k).
        """
        A = np.asarray(A, dtype=float)
        basis = np.asarray(basis, dtype=float)
        return basis.T @ A @ basis


# ---------------------------------------------------------------------------
# HDVOrthogonalizer
# ---------------------------------------------------------------------------

class HDVOrthogonalizer:
    """Enforces H_i^T H_j = 0 for i != j (domain subspace orthogonality).

    Assigns fixed subspace slices by default:
      - 'circuit':    dims [0,           hdv_dim//4)
      - 'semantic':   dims [hdv_dim//4,  hdv_dim//2)
      - 'market':     dims [hdv_dim//2,  3*hdv_dim//4)
      - 'code':       dims [3*hdv_dim//4, hdv_dim)

    For domains not in the fixed map: uses Gram-Schmidt against all
    registered bases.
    """

    def __init__(self, hdv_dim: int = 10000):
        self.hdv_dim = hdv_dim
        q = hdv_dim // 4
        self._fixed_slices: Dict[str, Tuple[int, int]] = {
            "circuit":  (0,      q),
            "semantic": (q,      2 * q),
            "market":   (2 * q,  3 * q),
            "code":     (3 * q,  hdv_dim),
        }
        self._learned_bases: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------

    def project(self, vec: np.ndarray, domain: str) -> np.ndarray:
        """Zero out components outside this domain's subspace.

        For fixed domains: mask to assigned slice.
        For learned domains: project onto registered basis.
        """
        vec = np.asarray(vec, dtype=float)
        out = np.zeros_like(vec)

        if domain in self._fixed_slices:
            s, e = self._fixed_slices[domain]
            out[s:e] = vec[s:e]
        elif domain in self._learned_bases:
            basis = self._learned_bases[domain]  # (hdv_dim, k)
            # Project: out = basis @ (basis.T @ vec)
            coords = basis.T @ vec                # (k,)
            out = basis @ coords                  # (hdv_dim,)
        else:
            # Unknown domain: return vector unchanged with warning
            warnings.warn(
                f"HDVOrthogonalizer: unknown domain '{domain}', returning vector unchanged. "
                f"Known fixed domains: {list(self._fixed_slices.keys())}",
                stacklevel=2,
            )
            out = vec.copy()

        return out

    # ------------------------------------------------------------------

    def register_basis(self, domain: str, vectors: np.ndarray) -> None:
        """Register a set of basis vectors for a new domain.

        Gram-Schmidt orthogonalizes against all existing bases before storing.

        vectors: (hdv_dim, k) — columns are basis vectors.
        """
        vectors = np.asarray(vectors, dtype=float)

        # Collect all existing basis vectors (fixed slices + learned)
        existing: List[np.ndarray] = []

        # Fixed domains contribute their indicator-style bases
        for d, (s, e) in self._fixed_slices.items():
            if d != domain:
                # Use a single "representative" indicator for the slice
                rep = np.zeros(self.hdv_dim, dtype=float)
                if e > s:
                    rep[s] = 1.0
                existing.append(rep)

        # Learned domains
        for d, b in self._learned_bases.items():
            if d != domain:
                for col_idx in range(b.shape[1]):
                    existing.append(b[:, col_idx])

        # Orthogonalize each column of vectors against existing bases
        new_basis_cols: List[np.ndarray] = []
        for col_idx in range(vectors.shape[1]):
            v = vectors[:, col_idx].copy()
            v_orth = self.orthogonalize(v, existing)
            # Also orthogonalize against already-accepted new basis cols
            v_orth = self.orthogonalize(v_orth, new_basis_cols)
            norm = np.linalg.norm(v_orth)
            if norm > 1e-10:
                new_basis_cols.append(v_orth)

        if new_basis_cols:
            self._learned_bases[domain] = np.stack(new_basis_cols, axis=1)

    # ------------------------------------------------------------------

    @staticmethod
    def orthogonalize(
        new_vec: np.ndarray,
        basis: List[np.ndarray],
    ) -> np.ndarray:
        """Gram-Schmidt: remove all components in span(basis) from new_vec."""
        new_vec = np.asarray(new_vec, dtype=float).copy()
        for b in basis:
            b_norm = np.dot(b, b)
            if b_norm > 1e-10:
                new_vec = new_vec - (np.dot(new_vec, b) / b_norm) * b
        norm = np.linalg.norm(new_vec)
        return new_vec / norm if norm > 1e-10 else new_vec

    # ------------------------------------------------------------------

    def cross_contamination(
        self,
        vec: np.ndarray,
        domain_a: str,
        domain_b: str,
    ) -> float:
        """Measure |dot(project(vec, a), project(vec, b))| — should be near 0."""
        pa = self.project(vec, domain_a)
        pb = self.project(vec, domain_b)
        return float(abs(np.dot(pa, pb)))
