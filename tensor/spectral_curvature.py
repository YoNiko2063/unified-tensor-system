"""
tensor/spectral_curvature.py — Curvature and bifurcation detection on the spectral manifold.

Three detection mechanisms:

  1. Stability margin     — min_i α_i → 0  (Hopf / saddle-node)
  2. Gap collapse         — |λ_i − λ_j| → 0 (mode interaction / coupling)
  3. Hessian signature    — λ_min(H_E) sign change (basin / saddle / unstable)

Together these define the boundary structure of the spectral manifold:

  Harmonic basin:  all α_i < 0, τ_ij < ε, spectral_gap > δ, H_E ≻ 0
  Saddle boundary: H_E has mixed eigenvalue signs
  Unstable region: ∃ α_i > 0

The curvature in the (ω_i, ω_j) plane is concentrated at rational-ratio
hypersurfaces (where the minimising rational (p/q) changes) — these are
the natural "chord boundaries" of the spectral landscape.

Usage:
    detector = BifurcationDetector()
    energy   = SpectralEnergy(config)
    report   = detector.classify(state, energy)

    print(report.region)           # 'harmonic_basin' | 'saddle' | 'unstable' | ...
    print(report.stability_margin) # min α_i
    print(report.hessian_min_eigval)
    print(report.approaching_bifurcation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spectral_energy import SpectralState, SpectralEnergy, SpectralEnergyConfig


# ── Bifurcation report ─────────────────────────────────────────────────────────

@dataclass
class BifurcationReport:
    """Result from BifurcationDetector.classify()."""

    # Stability crossing (B1)
    stability_margin:   float    # min_i Re(λ_i); < 0 → stable
    approaching_hopf:   bool     # margin < threshold → near boundary

    # Gap collapse (B2)
    min_eigenvalue_gap: float    # min |λ_i − λ_j| across all pairs
    gap_collapsed:      bool     # gap < threshold → mode interaction risk

    # Hessian signature (B3)
    hessian_eigvals:    List[float]
    hessian_min_eigval: float
    hessian_max_eigval: float
    hessian_signature:  str     # 'local_minimum' | 'saddle' | 'local_maximum'

    # Overall classification
    region: str    # 'harmonic_basin' | 'saddle_boundary' | 'mode_collision' |
                   # 'stability_boundary' | 'unstable'

    # Summary
    approaching_bifurcation: bool

    @property
    def is_harmonic_basin(self) -> bool:
        return self.region == 'harmonic_basin'

    @property
    def is_boundary(self) -> bool:
        return self.region not in ('harmonic_basin', 'unstable')


# ── Sectional curvature ────────────────────────────────────────────────────────

def sectional_curvature_ij(
    state:  SpectralState,
    i:      int,
    j:      int,
    energy: SpectralEnergy,
    eps:    float = 0.01,
) -> float:
    """
    Approximate sectional curvature of E in the (log₁₀ω₀_i, log₁₀ω₀_j) plane.

    K_ij = ∂²E_harm/∂u_i² + ∂²E_harm/∂u_j²   (u_k = log₁₀ω₀_k)

    High curvature near rational surfaces (where nearest p:q changes).
    Low curvature inside rational tubes Z_{p:q}.
    """
    if i >= state.n_modes() or j >= state.n_modes() or i == j:
        return 0.0

    modes = state.modes
    u_i = modes[i].log10_omega0
    u_j = modes[j].log10_omega0

    def _harm_ij(du_i: float, du_j: float) -> float:
        w0_i = 10.0 ** (u_i + du_i)
        w0_j = 10.0 ** (u_j + du_j)
        if w0_j < 1e-12:
            return 0.0
        r = w0_i / w0_j
        K = energy.config.K_ratios
        return min(abs(r - p / q) for p in range(1, K + 1) for q in range(1, K + 1))

    E0 = _harm_ij(0, 0)
    # Second partial w.r.t. u_i
    d2_ui = (_harm_ij(eps, 0) - 2 * E0 + _harm_ij(-eps, 0)) / eps ** 2
    # Second partial w.r.t. u_j
    d2_uj = (_harm_ij(0, eps) - 2 * E0 + _harm_ij(0, -eps)) / eps ** 2

    return float(d2_ui + d2_uj)


def curvature_matrix(
    state:  SpectralState,
    energy: SpectralEnergy,
) -> np.ndarray:
    """
    r×r matrix of sectional curvatures K_{ij} for all mode pairs.

    K[i,j] = sectional_curvature_ij(state, i, j, energy).
    Diagonal is zero (self-curvature not defined here).
    """
    r = state.n_modes()
    K = np.zeros((r, r))
    for i in range(r):
        for j in range(i + 1, r):
            k_ij = sectional_curvature_ij(state, i, j, energy)
            K[i, j] = K[j, i] = k_ij
    return K


# ── Eigenvalue gap ─────────────────────────────────────────────────────────────

def eigenvalue_gap_matrix(state: SpectralState) -> np.ndarray:
    """
    Matrix of |λ_i − λ_j| for all eigenvalue pairs (including conjugates).

    We reconstruct the full eigenvalue set (both conjugates for complex modes),
    then compute the n×n gap matrix.

    Shape: (n_eigvals, n_eigvals) where n_eigvals = 2*n_complex + n_real.
    """
    lambdas = []
    for m in state.modes:
        lambdas.append(m.alpha + 1j * m.omega)
        if m.omega > 1e-9:
            lambdas.append(m.alpha - 1j * m.omega)

    n = len(lambdas)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            g = abs(lambdas[i] - lambdas[j])
            G[i, j] = G[j, i] = g
    return G


def min_eigenvalue_gap(state: SpectralState) -> float:
    """
    min_{i≠j} |λ_i − λ_j| over the full eigenvalue set.

    Approaches 0 at mode collisions (potential bifurcation).
    """
    G = eigenvalue_gap_matrix(state)
    if G.size == 0:
        return float('inf')
    # Zero diagonal, find minimum off-diagonal
    np.fill_diagonal(G, np.inf)
    return float(np.min(G))


# ── Main detector ──────────────────────────────────────────────────────────────

class BifurcationDetector:
    """
    Classifies a spectral state using three independent detection mechanisms:

      B1 — Stability margin:  min_i α_i (Hurwitz crossing)
      B2 — Gap collapse:      min |λ_i − λ_j| (mode collision)
      B3 — Hessian signature: smallest eigenvalue of H_E

    Classification hierarchy:
      1. If ∃ α_i > 0 → unstable
      2. If stability_margin > -threshold → stability_boundary (near Hopf)
      3. If gap < gap_threshold → mode_collision
      4. If H_E has negative eigenvalues → saddle_boundary
      5. Otherwise → harmonic_basin

    Args:
        stability_threshold: |α_i| < this → near Hopf boundary
        gap_threshold:       |λ_i - λ_j| < this → mode collision risk
        hessian_eps:         step size for numerical Hessian
    """

    def __init__(
        self,
        stability_threshold: float = 0.01,
        gap_threshold:       float = 1.0,
        hessian_eps_zeta:    float = 0.005,
        hessian_eps_logw:    float = 0.01,
    ):
        self.stability_threshold = stability_threshold
        self.gap_threshold       = gap_threshold
        self.hessian_eps_zeta    = hessian_eps_zeta
        self.hessian_eps_logw    = hessian_eps_logw

    # ── B1: Stability margin ──────────────────────────────────────────────────

    def stability_margin(self, state: SpectralState) -> float:
        """min_i Re(λ_i). Negative = stable."""
        return state.stability_margin()

    # ── B2: Gap collapse ──────────────────────────────────────────────────────

    def gap(self, state: SpectralState) -> float:
        """min |λ_i − λ_j|. Small → mode collision approaching."""
        return min_eigenvalue_gap(state)

    # ── B3: Hessian signature ─────────────────────────────────────────────────

    def hessian_signature(
        self,
        state:  SpectralState,
        energy: SpectralEnergy,
    ) -> Tuple[str, np.ndarray]:
        """
        Classify via eigenvalues of H_E = ∇²E.

          All positive → local minimum (harmonic basin)
          Mixed signs  → saddle point
          All negative → local maximum

        Returns (signature_string, eigval_array).
        """
        H = energy.hessian_numerical(
            state,
            eps_zeta=self.hessian_eps_zeta,
            eps_logw=self.hessian_eps_logw,
        )
        eigvals = np.linalg.eigvalsh(H)
        tol = max(0.01 * abs(eigvals).max(), 1e-6) if len(eigvals) > 0 else 1e-6
        n_pos  = int(np.sum(eigvals >  tol))
        n_neg  = int(np.sum(eigvals < -tol))

        if n_neg == 0:
            sig = 'local_minimum'
        elif n_pos == 0:
            sig = 'local_maximum'
        else:
            sig = 'saddle'

        return sig, eigvals

    # ── Full classification ────────────────────────────────────────────────────

    def classify(
        self,
        state:  SpectralState,
        energy: SpectralEnergy,
    ) -> BifurcationReport:
        """
        Full bifurcation classification.  Runs B1, B2, B3 and determines region.
        """
        # B1
        margin = self.stability_margin(state)
        approaching_hopf = (margin > -self.stability_threshold)

        # B2
        g = self.gap(state)
        gap_collapsed = (g < self.gap_threshold) and (state.n_modes() > 1)

        # B3
        sig, hess_eigvals = self.hessian_signature(state, energy)

        # Classification hierarchy
        if margin > 0.0:
            region = 'unstable'
        elif approaching_hopf:
            region = 'stability_boundary'
        elif gap_collapsed:
            region = 'mode_collision'
        elif sig == 'saddle':
            region = 'saddle_boundary'
        else:
            region = 'harmonic_basin'

        approaching = (region != 'harmonic_basin')

        return BifurcationReport(
            stability_margin=float(margin),
            approaching_hopf=bool(approaching_hopf),
            min_eigenvalue_gap=float(g),
            gap_collapsed=bool(gap_collapsed),
            hessian_eigvals=[float(v) for v in hess_eigvals],
            hessian_min_eigval=float(hess_eigvals.min()) if len(hess_eigvals) > 0 else 0.0,
            hessian_max_eigval=float(hess_eigvals.max()) if len(hess_eigvals) > 0 else 0.0,
            hessian_signature=sig,
            region=region,
            approaching_bifurcation=bool(approaching),
        )

    def classify_fast(self, state: SpectralState) -> str:
        """
        Quick classification using only B1 and B2 (no Hessian).

        Returns: 'unstable' | 'stability_boundary' | 'mode_collision' | 'candidate_basin'
        """
        margin = self.stability_margin(state)
        if margin > 0.0:
            return 'unstable'
        if margin > -self.stability_threshold:
            return 'stability_boundary'
        g = self.gap(state)
        if g < self.gap_threshold and state.n_modes() > 1:
            return 'mode_collision'
        return 'candidate_basin'


# ── Basin classifier ───────────────────────────────────────────────────────────

class BasinClassifier:
    """
    Classify a spectral state as inside / outside a harmonic basin.

    Harmonic basin definition:
      B = { S : all α_i < 0, τ_ij < ε ∀ i<j, Δ_gap > δ }

    Three independent gate conditions:
      1. Hurwitz gate:  all α_i < 0
      2. Harmony gate:  all pairwise τ_ij < ε_harmony (inside a rational tube)
      3. Gap gate:      spectral gap > δ_gap

    Args:
        epsilon_harmony: harmonic zone radius (default from config)
        delta_gap:       minimum spectral gap for basin membership
    """

    def __init__(
        self,
        epsilon_harmony: float = 0.05,
        delta_gap:       float = 0.01,
        K_ratios:        int   = 8,
    ):
        self.epsilon_harmony = epsilon_harmony
        self.delta_gap       = delta_gap
        self.K_ratios        = K_ratios

    def hurwitz_gate(self, state: SpectralState) -> bool:
        return state.is_hurwitz()

    def harmony_gate(self, state: SpectralState) -> Tuple[bool, float]:
        """True iff all pairwise ω₀ ratios are within ε of some p/q."""
        from .spectral_energy import rational_proximity
        w0s = state.omega0s()
        max_dissonance = 0.0
        for i in range(len(w0s)):
            for j in range(i + 1, len(w0s)):
                if w0s[j] > 1e-12:
                    r = w0s[i] / w0s[j]
                    d = rational_proximity(r, self.K_ratios)
                    max_dissonance = max(max_dissonance, d)
        in_harmony = (max_dissonance < self.epsilon_harmony)
        return in_harmony, float(max_dissonance)

    def gap_gate(self, state: SpectralState) -> bool:
        return state.spectral_gap() > self.delta_gap

    def classify(self, state: SpectralState) -> Dict:
        """Full basin classification with gate diagnostics."""
        hurwitz  = self.hurwitz_gate(state)
        harmony, dissonance = self.harmony_gate(state)
        gap_ok   = self.gap_gate(state)

        in_basin = hurwitz and harmony and gap_ok

        return {
            'in_harmonic_basin': in_basin,
            'hurwitz_gate':  hurwitz,
            'harmony_gate':  harmony,
            'gap_gate':      gap_ok,
            'max_dissonance': round(float(dissonance), 6),
            'stability_margin': round(state.stability_margin(), 6),
            'spectral_gap':   round(state.spectral_gap(), 6),
        }

    def is_in_basin(self, state: SpectralState) -> bool:
        return self.classify(state)['in_harmonic_basin']


__all__ = [
    'BifurcationReport',
    'BifurcationDetector',
    'BasinClassifier',
    'sectional_curvature_ij',
    'curvature_matrix',
    'eigenvalue_gap_matrix',
    'min_eigenvalue_gap',
]
