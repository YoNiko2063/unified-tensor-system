"""
tensor/eigen_walker.py — Domain-invariant spectral navigator.

Architecture separation (corrected from ParameterSpaceWalker):

  Layer 1 (Invariant)    EigenWalker:   Φ(λ) → [Δζ, Δlog₁₀ω₀]
  Layer 2 (Domain)       DomainInverter: (ζ, ω₀, anchor) → θ
  Layer 3 (Invariant)    DiscreteHarmonicJumper: ω' = (p/q)·ω₀  in λ-space

Why ParameterSpaceWalker transfer failed:
  It learned (λ, θ_RLC) → Δθ_RLC, which encodes ∂λ/∂θ_RLC.
  Those Jacobians differ from ∂λ/∂θ_MSD → transfer breaks.

Why EigenWalker transfers:
  It learns Φ(λ) → [Δζ, Δlog₁₀ω₀] with NO θ features.
  For the same (ζ, ω₀), Φ(λ) is identical across all 2nd-order domains.
  Δ(ζ, log₁₀ω₀) is dimensionless and scale-invariant.
  DomainInverter maps the spectral step to domain-specific θ.

Input features (12-dim, no θ):
  Re(λ₁..₄)/scale  — normalized real parts      (4-dim)
  Im(λ₁..₄)/scale  — normalized imaginary parts  (4-dim)
  spectral_gap      — |Re(λ₁)|-|Re(λ₂)| normalized (1-dim)
  regime_onehot     — [lca, nonabelian, chaotic]   (3-dim)

Output (2-dim):
  Δζ              — change in damping ratio
  Δlog₁₀(ω₀)     — change in log₁₀ of natural frequency (log-decades)

Training signal: imitation on lowest-dissonance transitions.
Same dissonance metric τ as ParameterSpaceWalker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_N_SPECTRAL = 4    # eigenvalues retained for features
_IN_DIM     = 2 * _N_SPECTRAL + 1 + 3   # 12 — no θ features
_OUT_DIM    = 2    # [Δζ, Δlog₁₀(ω₀)]


# ── Spectral coordinates ──────────────────────────────────────────────────────

@dataclass
class SpectralCoords:
    """(ζ, ω₀) representation of a 2nd-order (or dominant) oscillatory mode."""
    zeta:   float   # damping ratio
    omega0: float   # undamped natural frequency [rad/s]

    @property
    def freq_hz(self) -> float:
        return self.omega0 / (2.0 * np.pi)

    @property
    def log10_omega0(self) -> float:
        if self.omega0 <= 0.0:
            return -9.0
        return float(np.log10(self.omega0))

    @classmethod
    def from_eigvals(cls, eigvals: np.ndarray) -> 'SpectralCoords':
        """Extract dominant-mode (ζ, ω₀) from an eigenvalue array.

        For complex conjugate pair λ = a ± ib:
            ω₀ = |λ|  (magnitude),  ζ = −Re(λ)/ω₀

        For real roots λ₁, λ₂ (overdamped):
            ω₀ = √(λ₁·λ₂)  (product formula),  ζ = −(λ₁+λ₂)/(2·ω₀)
        """
        if len(eigvals) == 0:
            return cls(zeta=0.0, omega0=0.0)

        im_abs = np.abs(np.imag(eigvals))

        if np.max(im_abs) > 1e-12:
            # Underdamped: find dominant imaginary part
            idx = np.argmax(im_abs)
            lam = eigvals[idx]
            omega0 = float(np.abs(lam))
            zeta   = float(-np.real(lam) / omega0) if omega0 > 1e-12 else 0.0
        else:
            # Overdamped or critically damped: all real
            re = np.sort(np.real(eigvals))[::-1]  # descending
            if len(re) >= 2:
                a1, a2 = float(re[0]), float(re[1])
                prod = abs(a1 * a2)
                omega0 = float(np.sqrt(prod)) if prod > 0 else abs(a1)
                zeta   = (abs(a1) + abs(a2)) / (2.0 * omega0) if omega0 > 1e-12 else 1.0
            else:
                omega0 = abs(float(re[0]))
                zeta   = 1.0

        return cls(zeta=float(np.clip(zeta, 0.0, 10.0)), omega0=float(max(omega0, 1e-12)))


# ── Experience ────────────────────────────────────────────────────────────────

@dataclass
class EigenWalkerExperience:
    """One observed spectral transition."""
    eigvals_before: np.ndarray
    eigvals_after:  np.ndarray
    dissonance:     float    # τ(ω_before, ω_after); lower = more harmonic


# ── Feature engineering (mirrors parameter_space_walker.py, no θ) ─────────────

def _eigval_features(eigvals: np.ndarray, n: int = _N_SPECTRAL) -> np.ndarray:
    if len(eigvals) == 0:
        return np.zeros(2 * n)
    idx = np.argsort(np.abs(eigvals))[::-1]
    ev = eigvals[idx]
    scale = float(np.max(np.abs(ev)))
    scale = scale if scale > 1e-12 else 1.0
    re = np.real(ev) / scale
    im = np.imag(ev) / scale
    re = np.pad(re[:n], (0, max(0, n - len(re))))
    im = np.pad(im[:n], (0, max(0, n - len(im))))
    return np.concatenate([re, im])


def _spectral_gap(eigvals: np.ndarray) -> float:
    if len(eigvals) < 2:
        return 0.0
    real_abs = np.sort(np.abs(np.real(eigvals)))[::-1]
    return float(real_abs[0] - real_abs[1])


def _regime_onehot(regime: str) -> np.ndarray:
    return {
        'lca':        np.array([1.0, 0.0, 0.0]),
        'nonabelian': np.array([0.0, 1.0, 0.0]),
        'chaotic':    np.array([0.0, 0.0, 1.0]),
    }.get(regime, np.zeros(3))


def _build_feature(eigvals: np.ndarray, regime: str) -> np.ndarray:
    """12-dim input feature — eigenvalues only, no θ."""
    if len(eigvals) > 0:
        scale = float(np.max(np.abs(eigvals)))
        scale = scale if scale > 1e-12 else 1.0
        ev_n = eigvals / scale
    else:
        ev_n = eigvals
    spec  = _eigval_features(ev_n, _N_SPECTRAL)
    gap   = np.array([_spectral_gap(ev_n)])
    regime_enc = _regime_onehot(regime)
    return np.concatenate([spec, gap, regime_enc])    # 12-dim


# ── MLP ───────────────────────────────────────────────────────────────────────

class _MLP:
    """2-hidden-layer ReLU MLP with Adam, pure NumPy."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / in_dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.standard_normal((hidden, in_dim))  * s1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * s2
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((out_dim, hidden)) * s2
        self.b3 = np.zeros(out_dim)
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)
        h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def mse_loss_and_grads(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, list]:
        N = X.shape[0]
        H1_pre = X @ self.W1.T + self.b1
        H1 = np.maximum(0.0, H1_pre)
        H2_pre = H1 @ self.W2.T + self.b2
        H2 = np.maximum(0.0, H2_pre)
        pred = H2 @ self.W3.T + self.b3
        diff = pred - Y
        loss = float(np.mean(diff ** 2))
        dL = 2.0 * diff / N
        dW3 = dL.T @ H2
        db3 = dL.sum(0)
        dH2 = dL @ self.W3
        dH2p = dH2 * (H2_pre > 0)
        dW2 = dH2p.T @ H1
        db2 = dH2p.sum(0)
        dH1 = dH2p @ self.W2
        dH1p = dH1 * (H1_pre > 0)
        dW1 = dH1p.T @ X
        db1 = dH1p.sum(0)
        return loss, [dW1, db1, dW2, db2, dW3, db3]

    def adam_step(self, grads: list, lr: float = 1e-3,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        for i, (g, p) in enumerate(zip(grads, self._params())):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** self._t)
            v_hat = self._v[i] / (1 - beta2 ** self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ── EigenWalker ───────────────────────────────────────────────────────────────

class EigenWalker:
    """
    Domain-invariant spectral navigator.

    Learns: Φ(λ) → [Δζ₁, Δlog₁₀ω₁, ..., Δζₙ, Δlog₁₀ωₙ]   (2·n_modes outputs)

    No θ features. Transfer across domains is automatic because:
    - Φ(λ) depends only on eigenvalues (same structure for equivalent (ζ,ω₀))
    - Output is dimensionless and scale-invariant in (ζ, log₁₀ω₀) coordinates

    n_modes=1 (default): [Δζ, Δlog₁₀ω₀] — single dominant mode (original behavior)
    n_modes=2:           [Δζ₁, Δlog₁₀ω₁, Δζ₂, Δlog₁₀ω₂] — two-mode (r=2 extension)

    Output ordering matches SpectralState.modes ordering (descending |Im(λ)|):
      mode 0 = dominant (highest ω)
      mode 1 = secondary

    Usage (n_modes=2):
        walker = EigenWalker(n_modes=2)
        # train via build_energy_training_data(n_modes=2) + _train_mlp_on_arrays()

        delta = walker.predict_step(eigvals)  # shape (4,)
        # delta[0:2] → [Δζ_dom, Δlog₁₀ω_dom]
        # delta[2:4] → [Δζ_sec, Δlog₁₀ω_sec]
    """

    # Clamp outputs to avoid runaway steps
    _DELTA_ZETA_MAX  = 0.50    # max |Δζ| per step per mode
    _DELTA_LOGW_MAX  = 1.00    # max |Δlog₁₀ω₀| per step per mode (1 decade)

    def __init__(
        self,
        hidden:              int   = 128,
        n_modes:             int   = 1,
        dissonance_quantile: float = 0.25,
        seed:                int   = 0,
    ):
        self._n_modes = n_modes
        self._out_dim = 2 * n_modes
        self._mlp = _MLP(_IN_DIM, hidden, self._out_dim, seed=seed)
        self._buffer: List[EigenWalkerExperience] = []
        self.dissonance_quantile = dissonance_quantile

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, exp: EigenWalkerExperience) -> None:
        self._buffer.append(exp)

    def record_from_scan(self, results: list, dissonance_fn) -> None:
        """Build buffer from consecutive MapResult pairs."""
        for i in range(len(results) - 1):
            r0, r1 = results[i], results[i + 1]
            tau = dissonance_fn(
                r0.classification.eigenvalues,
                r1.classification.eigenvalues,
            )
            self._buffer.append(EigenWalkerExperience(
                eigvals_before=r0.classification.eigenvalues,
                eigvals_after=r1.classification.eigenvalues,
                dissonance=tau,
            ))

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_step(
        self,
        eigvals: np.ndarray,
        regime:  str = 'lca',
    ) -> np.ndarray:
        """
        Predict spectral descent step from current eigenvalue state.

        Returns:
            np.ndarray of shape (2·n_modes,):
              [Δζ₁, Δlog₁₀ω₁, Δζ₂, Δlog₁₀ω₂, ...]
            Ordering matches SpectralState.modes (descending |Im(λ)|).
        """
        x = _build_feature(eigvals, regime)
        raw = self._mlp.forward(x)
        delta = np.zeros(self._out_dim)
        for i in range(self._n_modes):
            delta[2 * i]     = float(np.clip(raw[2 * i],
                                              -self._DELTA_ZETA_MAX,
                                              self._DELTA_ZETA_MAX))
            delta[2 * i + 1] = float(np.clip(raw[2 * i + 1],
                                              -self._DELTA_LOGW_MAX,
                                              self._DELTA_LOGW_MAX))
        return delta

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        n_epochs:   int   = 100,
        lr:         float = 1e-3,
        batch_size: int   = 32,
        rng_seed:   Optional[int] = None,
    ) -> float:
        """Train on lowest-dissonance transitions."""
        buf = self._buffer
        if len(buf) < 4:
            return 0.0

        taus = np.array([e.dissonance for e in buf])
        threshold = np.quantile(taus, self.dissonance_quantile)
        positive = [e for e in buf if e.dissonance <= threshold] or buf

        X_list, Y_list = [], []
        for exp in positive:
            x = _build_feature(exp.eigvals_before, 'lca')
            # Target: spectral coordinate delta between before and after
            s_before = SpectralCoords.from_eigvals(exp.eigvals_before)
            s_after  = SpectralCoords.from_eigvals(exp.eigvals_after)
            delta_zeta    = s_after.zeta        - s_before.zeta
            delta_log_w   = s_after.log10_omega0 - s_before.log10_omega0
            # Clamp targets to training range
            delta_zeta  = float(np.clip(delta_zeta,  -self._DELTA_ZETA_MAX, self._DELTA_ZETA_MAX))
            delta_log_w = float(np.clip(delta_log_w, -self._DELTA_LOGW_MAX, self._DELTA_LOGW_MAX))
            X_list.append(x)
            Y_list.append([delta_zeta, delta_log_w])

        X = np.array(X_list, dtype=np.float64)
        Y = np.array(Y_list, dtype=np.float64)

        rng = np.random.default_rng(rng_seed)
        N = len(X)
        final_loss = 0.0

        for _ in range(n_epochs):
            idx = rng.permutation(N)
            X, Y = X[idx], Y[idx]
            ep_loss, n_b = 0.0, 0
            for start in range(0, N, batch_size):
                Xb, Yb = X[start:start + batch_size], Y[start:start + batch_size]
                loss, grads = self._mlp.mse_loss_and_grads(Xb, Yb)
                self._mlp.adam_step(grads, lr=lr)
                ep_loss += loss
                n_b += 1
            final_loss = ep_loss / max(n_b, 1)

        return final_loss

    def buffer_size(self) -> int:
        return len(self._buffer)


# ── Discrete harmonic jumper (eigenvalue-space, domain-agnostic) ──────────────

class DiscreteHarmonicJumper:
    """
    Rational-ratio harmonic jumps in spectral (ζ, ω₀) space.

    Operates BEFORE domain inversion — purely in eigenvalue coordinates.

    Strategy A: ω' = (p/q)·ω₀ for rational p:q  (harmonic pivoting)
    Strategy B: direct target-range (ζ*, ω₀*) anchors  (goal-directed)

    Returns List[(ζ', ω₀')] candidates; caller applies DomainInverter.
    """

    def __init__(
        self,
        K:              int   = 8,
        zeta_targets:   Tuple = (0.10, 0.15, 0.20, 0.25, 0.30),
        n_freq_anchors: int   = 5,
    ):
        self.zeta_targets = zeta_targets
        ratios: set = set()
        for p in range(1, K + 1):
            for q in range(1, K + 1):
                ratios.add(p / q)
        self._ratios = sorted(ratios)
        self.n_freq_anchors = n_freq_anchors

    def candidates(
        self,
        zeta:  float,
        omega0: float,
        target_zeta_lo:    float,
        target_zeta_hi:    float,
        target_freq_lo_hz: float,
        target_freq_hi_hz: float,
    ) -> List[Tuple[float, float]]:
        """All (ζ', ω₀') candidates from both strategies."""
        out: List[Tuple[float, float]] = []
        out.extend(self._strategy_a(omega0))
        out.extend(self._strategy_b(target_freq_lo_hz, target_freq_hi_hz))
        return out

    def best_candidate(
        self,
        zeta:  float,
        omega0: float,
        target_zeta_lo:    float,
        target_zeta_hi:    float,
        target_freq_lo_hz: float,
        target_freq_hi_hz: float,
    ) -> Optional[Tuple[float, float]]:
        """Return best (ζ', ω₀') candidate."""
        cands = self.candidates(
            zeta, omega0,
            target_zeta_lo, target_zeta_hi,
            target_freq_lo_hz, target_freq_hi_hz,
        )
        if not cands:
            return None

        target_omega_lo = 2.0 * np.pi * target_freq_lo_hz
        target_omega_hi = 2.0 * np.pi * target_freq_hi_hz

        def _in_target(s: Tuple[float, float]) -> bool:
            z, w = s
            return (target_zeta_lo <= z <= target_zeta_hi
                    and target_omega_lo <= w <= target_omega_hi)

        in_tgt = [c for c in cands if _in_target(c)]
        if in_tgt:
            return min(in_tgt, key=lambda s: abs(s[0] - (target_zeta_lo + target_zeta_hi) / 2))

        target_omega_mid = (target_omega_lo * target_omega_hi) ** 0.5  # geometric mean
        target_zeta_mid  = (target_zeta_lo + target_zeta_hi) / 2.0

        def _dist(s: Tuple[float, float]) -> float:
            z, w = s
            z_dist = max(0.0, target_zeta_lo - z) + max(0.0, z - target_zeta_hi)
            if w < target_omega_lo:
                f_dist = np.log10(target_omega_lo / max(w, 1e-12))
            elif w > target_omega_hi:
                f_dist = np.log10(w / target_omega_hi)
            else:
                f_dist = 0.0
            return z_dist / max(target_zeta_hi - target_zeta_lo, 1e-9) + f_dist

        return min(cands, key=_dist)

    def _strategy_a(self, omega0: float) -> List[Tuple[float, float]]:
        """Rational-ratio pivots around current ω₀."""
        if omega0 < 1e-9:
            return []
        out = []
        for ratio in self._ratios:
            omega_prime = ratio * omega0
            if omega_prime < 1e-9:
                continue
            for zeta in self.zeta_targets:
                out.append((zeta, omega_prime))
        return out

    def _strategy_b(
        self,
        target_freq_lo_hz: float,
        target_freq_hi_hz: float,
    ) -> List[Tuple[float, float]]:
        """Direct target-range anchors."""
        out = []
        for f0_hz in np.logspace(
            np.log10(target_freq_lo_hz),
            np.log10(target_freq_hi_hz),
            self.n_freq_anchors,
        ):
            omega = 2.0 * np.pi * f0_hz
            for zeta in self.zeta_targets:
                out.append((zeta, omega))
        return out


__all__ = [
    'SpectralCoords',
    'EigenWalkerExperience',
    'EigenWalker',
    'DiscreteHarmonicJumper',
]
