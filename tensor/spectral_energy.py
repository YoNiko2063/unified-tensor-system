"""
tensor/spectral_energy.py — Spectral energy functional over eigenvalue manifold.

Formalizes the spectral landscape as a navigable scalar field:

  E(S) = w_stab·E_stab + w_damp·E_damp + w_freq·E_freq + w_harm·E_harm

Where S = {(α_i, ω_i)} is the spectral coordinate vector.

All quantities are domain-invariant — they depend only on eigenvalues,
not on the physical parameterization θ.  This is the invariant base layer
over which EigenWalker, BifurcationDetector, and DomainInverter operate.

n-mode formalization:
  SpectralMode:  one (α, ω) pair from one complex eigenvalue λ = α + iω
  SpectralState: r-mode collection S = {(α_1,ω_1), ..., (α_r,ω_r)}
  SpectralEnergy: E(S) + ∇_S E(S) in (ζ, log₁₀ω₀) coordinates

Training implication:
  Energy gradient ΔS* = −∇_S E(S) can be computed analytically for any S
  without a warmup scan.  Synthetic training samples cover the full spectral
  space uniformly, so the EigenWalker generalizes to any domain.

Reharmonization zones:
  Z_{p:q} = {S : |ω_i/ω_j − p/q| < ε}   for all i<j
  These are tubular neighborhoods around rational-ratio hypersurfaces
  inside the Hurwitz-stable half-space {α_i < 0 ∀i}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Spectral state representation ─────────────────────────────────────────────

@dataclass
class SpectralMode:
    """One oscillatory mode: λ = α + iω extracted from a complex eigenvalue."""
    alpha: float   # Re(λ) — decay rate;  < 0 ⟺ stable
    omega: float   # |Im(λ)| — angular frequency [rad/s]; 0 for real λ

    @property
    def omega0(self) -> float:
        """Undamped natural frequency ω₀ = |λ| = √(α² + ω²) [rad/s]."""
        return float(np.sqrt(self.alpha ** 2 + self.omega ** 2))

    @property
    def zeta(self) -> float:
        """Damping ratio ζ = −α/ω₀."""
        w0 = self.omega0
        return float(-self.alpha / w0) if w0 > 1e-15 else 0.0

    @property
    def freq_hz(self) -> float:
        """Damped frequency f_d = ω / (2π) [Hz]."""
        return self.omega / (2.0 * np.pi)

    @property
    def natural_freq_hz(self) -> float:
        """Undamped natural frequency f₀ = ω₀ / (2π) [Hz]."""
        return self.omega0 / (2.0 * np.pi)

    @property
    def log10_omega0(self) -> float:
        w0 = self.omega0
        return float(np.log10(w0)) if w0 > 1e-15 else -15.0

    def is_oscillatory(self) -> bool:
        return self.omega > 1e-9

    def is_stable(self) -> bool:
        return self.alpha < 0.0


@dataclass
class SpectralState:
    """
    n-mode spectral coordinate S = {(α_i, ω_i)}_{i=1..r}.

    Modes are ordered by descending |Im(λ)| (dominant oscillatory first).
    Complex conjugate pairs are deduplicated — only one representative stored.
    """
    modes: List[SpectralMode]

    @classmethod
    def from_eigvals(cls, eigvals: np.ndarray) -> 'SpectralState':
        """Extract unique modes from eigenvalue array (dedup conjugates)."""
        if len(eigvals) == 0:
            return cls(modes=[])

        used = [False] * len(eigvals)
        modes: List[SpectralMode] = []

        # Sort by descending |Im| to process oscillatory modes first
        order = np.argsort(np.abs(np.imag(eigvals)))[::-1]

        for i in order:
            if used[i]:
                continue
            lam = eigvals[i]
            alpha = float(np.real(lam))
            omega = float(abs(np.imag(lam)))

            if omega > 1e-9:
                # Find conjugate and mark it used
                for j in order:
                    if not used[j] and j != i:
                        if abs(eigvals[j] - np.conj(lam)) < 1e-8 * (abs(lam) + 1.0):
                            used[j] = True
                            break

            used[i] = True
            modes.append(SpectralMode(alpha=alpha, omega=omega))

        return cls(modes=modes)

    def n_modes(self) -> int:
        return len(self.modes)

    def dominant(self) -> Optional[SpectralMode]:
        """Return mode with largest |Im(λ)| (or first real mode if all real)."""
        if not self.modes:
            return None
        return self.modes[0]

    def alphas(self) -> np.ndarray:
        return np.array([m.alpha for m in self.modes])

    def omegas(self) -> np.ndarray:
        return np.array([m.omega for m in self.modes])

    def omega0s(self) -> np.ndarray:
        return np.array([m.omega0 for m in self.modes])

    def zetas(self) -> np.ndarray:
        return np.array([m.zeta for m in self.modes])

    def is_hurwitz(self) -> bool:
        """True iff all α_i < 0."""
        return all(m.alpha < 0.0 for m in self.modes)

    def stability_margin(self) -> float:
        """min_i α_i — negative means stable."""
        if not self.modes:
            return 0.0
        return float(np.min(self.alphas()))

    def spectral_gap(self) -> float:
        """|Re(λ_dom)| − |Re(λ_sub)| — gap between first two modes."""
        if len(self.modes) < 2:
            return 0.0
        re_abs = np.sort(np.abs(self.alphas()))[::-1]
        return float(re_abs[0] - re_abs[1])

    def pairwise_ratio_matrix(self) -> np.ndarray:
        """R_{ij} = ω₀_i / ω₀_j for oscillatory modes i,j with ω₀_j > 0."""
        w0s = self.omega0s()
        r = len(w0s)
        R = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                if w0s[j] > 1e-12:
                    R[i, j] = w0s[i] / w0s[j]
        return R

    def to_vector(self) -> np.ndarray:
        """Flatten to [α_1, ω_1, α_2, ω_2, ...] (2r-dim)."""
        out = []
        for m in self.modes:
            out.extend([m.alpha, m.omega])
        return np.array(out)


# ── Energy configuration ───────────────────────────────────────────────────────

@dataclass
class SpectralEnergyConfig:
    """
    Target spectral region and energy weights.

    Defines what "harmonic" means for a given design task.
    E(S) is zero inside the target region and positive outside.

    Per-mode targets can be set via target_zetas / target_omega0s_hz;
    if unset, scalar defaults are broadcast across all modes.
    """
    # Scalar targets (broadcast to all modes)
    target_zeta:      float = 0.20     # ζ*  desired damping ratio
    target_omega0_hz: float = 10.0     # f₀* desired natural frequency [Hz]

    # Per-mode targets (override scalar when set)
    target_zetas:     Optional[List[float]] = None
    target_omega0s_hz: Optional[List[float]] = None

    # Rational ratio search depth
    K_ratios: int = 8

    # Energy zone tolerance
    epsilon_harmony: float = 0.05   # |ω_i/ω_j − p/q| < ε → in harmony zone

    # Energy weights
    w_stab: float = 10.0   # large: stability is hard constraint
    w_damp: float = 1.0
    w_freq: float = 0.1
    w_harm: float = 1.0

    def get_target_zeta(self, mode_idx: int) -> float:
        if self.target_zetas and mode_idx < len(self.target_zetas):
            return self.target_zetas[mode_idx]
        return self.target_zeta

    def get_target_omega0(self, mode_idx: int) -> float:
        if self.target_omega0s_hz and mode_idx < len(self.target_omega0s_hz):
            return 2.0 * np.pi * self.target_omega0s_hz[mode_idx]
        return 2.0 * np.pi * self.target_omega0_hz


# ── Rational proximity ─────────────────────────────────────────────────────────

def _nearest_rational(r: float, K: int) -> Tuple[float, float, int, int]:
    """
    Find (p*, q*) in [1,K]² minimising |r − p/q|.

    Returns (min_dist, nearest_value, p*, q*).
    """
    best_dist  = float('inf')
    best_val   = 1.0
    best_p, best_q = 1, 1
    for p in range(1, K + 1):
        for q in range(1, K + 1):
            val  = p / q
            dist = abs(r - val)
            if dist < best_dist:
                best_dist, best_val, best_p, best_q = dist, val, p, q
    return best_dist, best_val, best_p, best_q


def rational_proximity(r: float, K: int) -> float:
    """min_{p,q ∈ [1,K]} |r − p/q| — dissonance of frequency ratio r."""
    return _nearest_rational(r, K)[0]


def is_in_harmony_zone(r: float, K: int, epsilon: float) -> bool:
    """True iff r is within ε of some p/q with p,q ∈ [1,K]."""
    return rational_proximity(r, K) < epsilon


# ── Spectral energy functional ─────────────────────────────────────────────────

class SpectralEnergy:
    """
    Scalar energy field E(S) over the spectral manifold.

    E(S) = w_stab·E_stab(S) + w_damp·E_damp(S)
         + w_freq·E_freq(S) + w_harm·E_harm(S)

    All four terms and their analytical gradients ∂E/∂(ζ_i, log₁₀ω₀_i)
    are provided for each mode.

    Gradient is expressed in (ζ, log₁₀ω₀) coordinates — the same space
    that EigenWalker outputs — so training targets are directly usable.
    """

    def __init__(self, config: Optional[SpectralEnergyConfig] = None):
        self.config = config or SpectralEnergyConfig()

    # ── Individual energy components ─────────────────────────────────────────

    def E_stab(self, state: SpectralState) -> float:
        """Stability energy: Σ max(0, α_i). Zero inside Hurwitz region."""
        return float(np.sum(np.maximum(0.0, state.alphas())))

    def E_damp(self, state: SpectralState) -> float:
        """Damping energy: Σ |ζ_i − ζ_i*|."""
        total = 0.0
        for i, m in enumerate(state.modes):
            total += abs(m.zeta - self.config.get_target_zeta(i))
        return total

    def E_freq(self, state: SpectralState) -> float:
        """Frequency energy: Σ |log₁₀(ω₀_i) − log₁₀(ω₀_i*)|."""
        total = 0.0
        for i, m in enumerate(state.modes):
            target_w0 = self.config.get_target_omega0(i)
            if m.omega0 > 1e-15 and target_w0 > 1e-15:
                total += abs(np.log10(m.omega0) - np.log10(target_w0))
        return total

    def E_harm(self, state: SpectralState) -> float:
        """Harmonic energy: Σ_{i<j} min_{p,q} |ω₀_i/ω₀_j − p/q|."""
        w0s = state.omega0s()
        total = 0.0
        K = self.config.K_ratios
        for i in range(len(w0s)):
            for j in range(i + 1, len(w0s)):
                if w0s[j] > 1e-12:
                    r = w0s[i] / w0s[j]
                    total += rational_proximity(r, K)
        return total

    # ── Total energy ─────────────────────────────────────────────────────────

    def compute(self, state: SpectralState) -> float:
        c = self.config
        return (c.w_stab * self.E_stab(state) +
                c.w_damp * self.E_damp(state) +
                c.w_freq * self.E_freq(state) +
                c.w_harm * self.E_harm(state))

    def components(self, state: SpectralState) -> dict:
        c = self.config
        E_s = self.E_stab(state)
        E_d = self.E_damp(state)
        E_f = self.E_freq(state)
        E_h = self.E_harm(state)
        return {
            'E_stab': round(E_s, 6),
            'E_damp': round(E_d, 6),
            'E_freq': round(E_f, 6),
            'E_harm': round(E_h, 6),
            'E_total': round(c.w_stab*E_s + c.w_damp*E_d + c.w_freq*E_f + c.w_harm*E_h, 6),
        }

    # ── Analytical gradient ∂E/∂(ζ_i, log₁₀ω₀_i) ─────────────────────────

    def gradient(self, state: SpectralState) -> np.ndarray:
        """
        Compute ∂E/∂(ζ_1, log₁₀ω₀_1, ζ_2, log₁₀ω₀_2, ...) analytically.

        Returns a 2r-dim vector.  Gradient for each mode i:

          ∂E/∂ζ_i = w_damp · sign(ζ_i − ζ_i*)
                  + w_stab · ∂E_stab/∂ζ_i  (if mode is unstable)

          ∂E/∂log₁₀ω₀_i = w_freq · sign(log₁₀ω₀_i − log₁₀ω₀_i*)
                          + w_harm · ∂E_harm/∂log₁₀ω₀_i

        For E_stab: α_i = −ζ_i·ω₀_i, so ∂E_stab/∂ζ_i = −ω₀_i if α_i > 0 else 0.

        For E_harm (in log₁₀ω₀ space):
          r_ij = ω₀_i/ω₀_j = 10^(u_i − u_j)  where u_k = log₁₀ω₀_k
          τ_ij = |r_ij − p*/q*|
          ∂τ_ij/∂u_i = sign(r_ij − p*/q*) · r_ij · ln(10)
          ∂τ_ij/∂u_j = −sign(r_ij − p*/q*) · r_ij · ln(10)
        """
        c = self.config
        r = state.n_modes()
        grad = np.zeros(2 * r)    # [∂/∂ζ_0, ∂/∂log_w₀_0, ∂/∂ζ_1, ∂/∂log_w₁_0, ...]

        for i, mode in enumerate(state.modes):
            gi_zeta = 0.0    # ∂E/∂ζ_i
            gi_logw = 0.0    # ∂E/∂log₁₀ω₀_i

            # ── E_stab ──────────────────────────────────────────────────────
            if mode.alpha > 0.0:
                # ∂max(0,α_i)/∂ζ_i = ∂α_i/∂ζ_i = -ω₀_i
                gi_zeta += c.w_stab * (-mode.omega0)
                # ∂α_i/∂log₁₀ω₀_i = -ζ_i · ω₀_i · ln(10)
                gi_logw += c.w_stab * (-mode.zeta * mode.omega0 * np.log(10))

            # ── E_damp ──────────────────────────────────────────────────────
            zeta_err = mode.zeta - c.get_target_zeta(i)
            gi_zeta += c.w_damp * float(np.sign(zeta_err))
            # E_damp has no ω₀ dependence

            # ── E_freq ──────────────────────────────────────────────────────
            if mode.omega0 > 1e-15:
                target_w0 = c.get_target_omega0(i)
                if target_w0 > 1e-15:
                    log_err = np.log10(mode.omega0) - np.log10(target_w0)
                    gi_logw += c.w_freq * float(np.sign(log_err))

            # ── E_harm ──────────────────────────────────────────────────────
            # Pairwise rational distance contributions for mode i
            if mode.omega0 > 1e-12:
                u_i = np.log10(mode.omega0)
                for j, mode_j in enumerate(state.modes):
                    if i == j or mode_j.omega0 < 1e-12:
                        continue
                    u_j = np.log10(mode_j.omega0)
                    r_ij = mode.omega0 / mode_j.omega0
                    dist, pq_val, _, _ = _nearest_rational(r_ij, c.K_ratios)
                    # ∂τ_ij/∂u_i in log₁₀ space
                    sign_ij = float(np.sign(r_ij - pq_val))
                    # Avoid double-counting: only add when i < j
                    if i < j:
                        dτ_dui = sign_ij * r_ij * np.log(10)
                        gi_logw += c.w_harm * dτ_dui

            grad[2 * i]     = gi_zeta
            grad[2 * i + 1] = gi_logw

        return grad

    def descent_direction(
        self,
        state: SpectralState,
        scale: float = 0.5,
    ) -> np.ndarray:
        """
        Returns −∇E(S), clipped per component and scaled.

        Output shape: (2r,) = [Δζ_1, Δlog₁₀ω₀_1, Δζ_2, Δlog₁₀ω₀_2, ...]

        This is the training target for EigenWalker (energy-gradient mode).
        The scale controls step magnitude — corresponds to step_scale in walker.
        """
        g = self.gradient(state)
        # Clip each component independently: Δζ ≤ 0.5, Δlog₁₀ω₀ ≤ 1.0
        r = state.n_modes()
        for i in range(r):
            g[2 * i]     = np.clip(g[2 * i],     -0.5, 0.5)    # ζ component
            g[2 * i + 1] = np.clip(g[2 * i + 1], -1.0, 1.0)    # log₁₀ω₀ component
        return -g * scale

    # ── Numerical Hessian ────────────────────────────────────────────────────

    def hessian_numerical(
        self,
        state: SpectralState,
        eps_zeta: float = 0.005,
        eps_logw: float = 0.01,
    ) -> np.ndarray:
        """
        Numerical Hessian of E w.r.t. (ζ_1, log₁₀ω₀_1, ...) via central differences.

        Shape: (2r, 2r).  Useful for curvature and basin classification.
        Diagonal entries are approximate sectional curvatures.
        """
        r = state.n_modes()
        n = 2 * r
        eps = np.array([eps_zeta if k % 2 == 0 else eps_logw for k in range(n)])

        def _state_from_vec(vec: np.ndarray) -> SpectralState:
            new_modes = []
            for i in range(r):
                zeta  = float(vec[2 * i])
                log_w = float(vec[2 * i + 1])
                w0 = float(10.0 ** log_w)
                alpha = -zeta * w0
                omega = w0 * float(np.sqrt(max(0.0, 1.0 - zeta ** 2)))
                new_modes.append(SpectralMode(alpha=alpha, omega=omega))
            return SpectralState(modes=new_modes)

        # Reference vector in (ζ, log₁₀ω₀) space
        x0 = np.array([
            val
            for m in state.modes
            for val in [m.zeta, m.log10_omega0]
        ])

        E0 = self.compute(state)
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Second derivative: (E(x+h) - 2E(x) + E(x-h)) / h²
                    dx = np.zeros(n); dx[i] = eps[i]
                    Ep = self.compute(_state_from_vec(x0 + dx))
                    Em = self.compute(_state_from_vec(x0 - dx))
                    H[i, i] = (Ep - 2 * E0 + Em) / (eps[i] ** 2)
                else:
                    # Cross term: (E++ - E+- - E-+ + E--) / (4·h_i·h_j)
                    di = np.zeros(n); di[i] = eps[i]
                    dj = np.zeros(n); dj[j] = eps[j]
                    Epp = self.compute(_state_from_vec(x0 + di + dj))
                    Epm = self.compute(_state_from_vec(x0 + di - dj))
                    Emp = self.compute(_state_from_vec(x0 - di + dj))
                    Emm = self.compute(_state_from_vec(x0 - di - dj))
                    H[i, j] = H[j, i] = (Epp - Epm - Emp + Emm) / (4 * eps[i] * eps[j])

        return H

    # ── Synthetic training data generation ───────────────────────────────────

    def generate_training_samples(
        self,
        n_samples: int = 5000,
        n_modes:   int = 1,
        zeta_range:  Tuple[float, float] = (0.01, 1.0),
        logw_range:  Tuple[float, float] = (-1.0, 6.0),
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate (state, ΔS*) pairs for energy-gradient training.

        Samples (ζ, log₁₀ω₀) uniformly; ΔS* = descent_direction(state).
        No warmup scan required — training data is fully synthetic.

        Returns:
            states: List[SpectralState]  (for feature extraction)
            targets: np.ndarray, shape (n_samples, 2*n_modes)
        """
        rng = np.random.default_rng(seed)
        states: List[SpectralState] = []
        targets: List[np.ndarray]  = []

        for _ in range(n_samples):
            modes = []
            for _ in range(n_modes):
                zeta  = rng.uniform(*zeta_range)
                log_w = rng.uniform(*logw_range)
                w0    = float(10.0 ** log_w)
                alpha = -zeta * w0
                omega = w0 * float(np.sqrt(max(0.0, 1.0 - zeta ** 2)))
                modes.append(SpectralMode(alpha=alpha, omega=omega))

            state = SpectralState(modes=modes)
            target = self.descent_direction(state, scale=1.0)

            states.append(state)
            targets.append(target)

        return states, np.array(targets)


__all__ = [
    'SpectralMode',
    'SpectralState',
    'SpectralEnergyConfig',
    'SpectralEnergy',
    'rational_proximity',
    'is_in_harmony_zone',
]
