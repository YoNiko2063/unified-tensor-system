"""
RLCDesignMapper — deterministic HDV → (R, L, C) projection.

Mapping:
    R = R_center * exp(clip(a_R · z, ±max_exp))
    L = L_center * exp(clip(a_L · z, ±max_exp))
    C = C_center * exp(clip(a_C · z, ±max_exp))

where a_R, a_L, a_C are fixed orthonormal vectors drawn from a seeded RNG.

Properties:
  - Always positive (exponential mapping)
  - Smooth and differentiable (useful for gradient methods later)
  - Deterministic given seed (reproducible)
  - Orthonormal projection vectors: no two parameters share the same HDV direction

The inverse encode() uses the minimum-norm pseudoinverse solution, making it
useful for warm-starting the optimizer from a known good (R, L, C) design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RLCParams:
    R: float   # Ohms
    L: float   # Henrys
    C: float   # Farads

    def __str__(self) -> str:
        return f"R={self.R:.4g} Ω  L={self.L:.4g} H  C={self.C:.4g} F"

    def as_dict(self) -> dict:
        return {"R": self.R, "L": self.L, "C": self.C}


class RLCDesignMapper:
    """
    Deterministic projection from HDV vector z ∈ ℝ^hdv_dim → RLCParams.

    The three projection vectors a_R, a_L, a_C are orthonormal (via QR), so
    they span independent directions in HDV space — changes along one axis
    do not conflate two different parameters.

    Args:
        hdv_dim:    dimensionality of the HDV search space
        R_center:   nominal resistance  (Ω) — center of the exponential range
        L_center:   nominal inductance  (H)
        C_center:   nominal capacitance (F)
        seed:       RNG seed for reproducibility
        max_exp:    clip bound on a · z before exp(), prevents overflow
    """

    def __init__(
        self,
        hdv_dim: int = 64,
        R_center: float = 100.0,   # Ω
        L_center: float = 0.01,    # H  (nominal f₀ ≈ 1.6 kHz with C_center)
        C_center: float = 1e-6,    # F
        seed: int = 42,
        max_exp: float = 3.0,
    ) -> None:
        self.hdv_dim = hdv_dim
        self.R_center = R_center
        self.L_center = L_center
        self.C_center = C_center
        self._max_exp = max_exp

        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((hdv_dim, 3))   # (hdv_dim, 3)
        Q, _ = np.linalg.qr(raw)                  # orthonormal columns
        self._a_R = Q[:, 0]   # unit vector for R
        self._a_L = Q[:, 1]   # unit vector for L
        self._a_C = Q[:, 2]   # unit vector for C

        # Pre-stack for encode()
        self._A = np.stack([self._a_R, self._a_L, self._a_C])  # (3, hdv_dim)

    # ── Forward: z → params ────────────────────────────────────────────────────

    def decode(self, z: np.ndarray) -> RLCParams:
        """
        Project HDV vector z → RLCParams.

        z must have shape (hdv_dim,).
        The exponential mapping guarantees R, L, C > 0 for any z.
        """
        if z.shape != (self.hdv_dim,):
            raise ValueError(
                f"Expected z.shape == ({self.hdv_dim},), got {z.shape}"
            )
        e_R = float(np.clip(self._a_R @ z, -self._max_exp, self._max_exp))
        e_L = float(np.clip(self._a_L @ z, -self._max_exp, self._max_exp))
        e_C = float(np.clip(self._a_C @ z, -self._max_exp, self._max_exp))
        return RLCParams(
            R=self.R_center * float(np.exp(e_R)),
            L=self.L_center * float(np.exp(e_L)),
            C=self.C_center * float(np.exp(e_C)),
        )

    # ── Inverse: params → z (approximate) ────────────────────────────────────

    def encode(self, params: RLCParams) -> np.ndarray:
        """
        Minimum-norm HDV vector that decodes to approximately params.

        Uses A^+ (pseudoinverse) to find the shortest z satisfying A z ≈ log-ratios.
        Useful for warm-starting the optimizer from a known good design.

        Note: decode(encode(params)) ≈ params but is not exact in general
        because the three projection vectors span only a 3D subspace of ℝ^hdv_dim.
        """
        log_R = np.log(max(params.R, 1e-30) / self.R_center)
        log_L = np.log(max(params.L, 1e-30) / self.L_center)
        log_C = np.log(max(params.C, 1e-30) / self.C_center)

        target = np.array([log_R, log_L, log_C])
        # Minimum-norm solution: z = A^T (A A^T)^{-1} target
        z = self._A.T @ np.linalg.lstsq(self._A @ self._A.T, target, rcond=None)[0]
        return z

    def random_z(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a unit-normal HDV vector."""
        return rng.standard_normal(self.hdv_dim)
