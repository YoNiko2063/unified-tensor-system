"""
tensor/domain_inverter.py — Domain-specific (ζ, ω₀) → θ inversion layers.

Each domain has an analytic inverse map from spectral coordinates to physical
parameters.  This is the domain-specific adapter that makes EigenWalker
transfer automatic: same spectral navigator, different inversion.

Analogy table (2nd-order oscillators):
  Physical      Series RLC          Mass-Spring-Damper
  ─────────     ──────────────────  ────────────────────
  Inertia       L  [H]              m  [kg]
  Damping       R  [Ω]              c  [N·s/m]
  Restoring     1/C  [1/F]          k  [N/m]
  ω₀            1/√(LC)             √(k/m)
  ζ             (R/2)√(C/L)         c/(2√(km))

Both invert to:
  component1 = anchor · ω₀²          (C = 1/(L·ω₀²)    k = m·ω₀²)
  component2 = 2·ζ·anchor·ω₀         (R = 2·ζ·L·ω₀    c = 2·ζ·m·ω₀)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class DomainInverter(ABC):
    """Abstract base for spectral → physical parameter inversion."""

    @abstractmethod
    def invert(
        self,
        zeta:   float,
        omega0: float,
        theta_current: dict,
    ) -> dict:
        """
        Compute θ from (ζ, ω₀) using the anchor extracted from theta_current.

        Args:
            zeta:          target damping ratio
            omega0:        target natural frequency [rad/s]
            theta_current: current parameter dict (used to extract anchor)

        Returns:
            New parameter dict with (ζ, ω₀) as specified.
        """

    @abstractmethod
    def extract_anchor(self, theta: dict) -> float:
        """Extract the anchor parameter value from θ."""

    @abstractmethod
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for clipping."""

    @abstractmethod
    def theta_keys(self) -> list:
        """Ordered list of parameter names."""

    def invert_with_anchor(
        self,
        zeta:   float,
        omega0: float,
        anchor: float,
    ) -> dict:
        """Invert with an explicit anchor value (override current)."""
        # Subclasses can override; default just wraps invert with a fake theta.
        return self.invert(zeta, omega0, {self.anchor_key: anchor})

    @property
    @abstractmethod
    def anchor_key(self) -> str:
        """Name of the anchor parameter."""

    def clip(self, theta: dict) -> dict:
        bounds = self.param_bounds()
        return {k: float(np.clip(theta[k], bounds[k][0], bounds[k][1]))
                for k in self.theta_keys()}


# ── RLC inverter ──────────────────────────────────────────────────────────────

class RLCInverter(DomainInverter):
    """
    Series RLC: anchor = L (inductance).

    Forward:  ω₀ = 1/√(LC),  ζ = (R/2)√(C/L)
    Inverse:  C = 1/(L·ω₀²),  R = 2·ζ·L·ω₀
    """

    _BOUNDS: Dict[str, Tuple[float, float]] = {
        'R': (1.0,   1_000.0),
        'L': (1e-6,  1e-2),
        'C': (1e-9,  1e-6),
    }
    _KEYS = ['R', 'L', 'C']

    @property
    def anchor_key(self) -> str:
        return 'L'

    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._BOUNDS

    def theta_keys(self) -> list:
        return self._KEYS

    def extract_anchor(self, theta: dict) -> float:
        return float(theta.get('L', 1e-3))

    def invert(self, zeta: float, omega0: float, theta_current: dict) -> dict:
        L = self.extract_anchor(theta_current)
        if omega0 < 1e-12:
            return dict(theta_current)
        C = 1.0 / (L * omega0 ** 2)
        R = 2.0 * zeta * L * omega0
        return {'R': R, 'L': L, 'C': C}

    def invert_with_anchor(self, zeta: float, omega0: float, anchor: float) -> dict:
        L = anchor
        if omega0 < 1e-12:
            return {'R': 0.0, 'L': L, 'C': 0.0}
        C = 1.0 / (L * omega0 ** 2)
        R = 2.0 * zeta * L * omega0
        return {'R': R, 'L': L, 'C': C}


# ── MSD inverter ──────────────────────────────────────────────────────────────

class MSDInverter(DomainInverter):
    """
    Mass-spring-damper: anchor = m (mass).

    Forward:  ω₀ = √(k/m),  ζ = c/(2√(km))
    Inverse:  k = m·ω₀²,    c = 2·ζ·m·ω₀
    """

    _BOUNDS: Dict[str, Tuple[float, float]] = {
        'm': (0.01,  100.0),
        'c': (1e-4,  1e3),
        'k': (1e-2,  1e5),
    }
    _KEYS = ['m', 'c', 'k']

    @property
    def anchor_key(self) -> str:
        return 'm'

    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._BOUNDS

    def theta_keys(self) -> list:
        return self._KEYS

    def extract_anchor(self, theta: dict) -> float:
        return float(theta.get('m', 1.0))

    def invert(self, zeta: float, omega0: float, theta_current: dict) -> dict:
        m = self.extract_anchor(theta_current)
        if omega0 < 1e-12:
            return dict(theta_current)
        k = m * omega0 ** 2
        c = 2.0 * zeta * m * omega0
        return {'m': m, 'c': c, 'k': k}

    def invert_with_anchor(self, zeta: float, omega0: float, anchor: float) -> dict:
        m = anchor
        if omega0 < 1e-12:
            return {'m': m, 'c': 0.0, 'k': 0.0}
        k = m * omega0 ** 2
        c = 2.0 * zeta * m * omega0
        return {'m': m, 'c': c, 'k': k}


# ── TwoMassSpring inverter ─────────────────────────────────────────────────────

class TwoMassSpringInverter(DomainInverter):
    """
    Symmetric two-mass spring system with Rayleigh damping.

    Mode ordering follows SpectralState (descending |Im(λ)|):
      modes[0] = dominant (out-of-phase, ω₂, higher frequency)
      modes[1] = secondary (in-phase, ω₁, lower frequency)

    Analytic inverse given (ζ₂, ω₂, ζ₁, ω₁, m):
      k   = m · ω₁²
      k_c = m · (ω₂² − ω₁²) / 2           (requires ω₂ > ω₁)
      β   = 2(ζ₂ω₂ − ζ₁ω₁) / (ω₂² − ω₁²)
      α   = 2ζ₁ω₁ − β·ω₁²

    Forward (for reference):
      ω₁ = √(k/m),  ω₂ = √((k+2k_c)/m)
      ζₙ = α/(2ωₙ) + β·ωₙ/2
    """

    _BOUNDS: Dict[str, Tuple[float, float]] = {
        'm':     (0.01,  100.0),
        'k':     (0.001, 1e6),
        'k_c':   (0.001, 5e5),
        'alpha': (0.0,   200.0),
        'beta':  (0.0,   1.0),
    }
    _KEYS = ['m', 'k', 'k_c', 'alpha', 'beta']

    @property
    def anchor_key(self) -> str:
        return 'm'

    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._BOUNDS

    def theta_keys(self) -> list:
        return self._KEYS

    def extract_anchor(self, theta: dict) -> float:
        return float(theta.get('m', 1.0))

    def invert(self, zeta: float, omega0: float, theta_current: dict) -> dict:
        """Single-mode inversion: not meaningful for two-mass system.

        Uses single mode as in-phase frequency, assumes 2:1 ratio for out-of-phase.
        For full two-mode inversion, use invert_modes().
        """
        m = self.extract_anchor(theta_current)
        # Default: 2:1 ratio, equal damping for both modes
        omega1 = omega0
        omega2 = 2.0 * omega0
        return self.invert_modes([(zeta, omega2), (zeta, omega1)], theta_current)

    def invert_with_anchor(self, zeta: float, omega0: float, anchor: float) -> dict:
        m = anchor
        return self.invert(zeta, omega0, {'m': m})

    def invert_modes(
        self,
        modes: list,        # [(zeta_dom, omega_dom), (zeta_sec, omega_sec)]
                            # modes[0] = dominant (higher ω), modes[1] = secondary (lower ω)
        theta_current: dict,
    ) -> dict:
        """Full two-mode analytic inversion.

        Args:
            modes: [(ζ₂, ω₂), (ζ₁, ω₁)] where ω₂ > ω₁ (dominant first)
            theta_current: used only to extract anchor m

        Returns:
            dict with keys 'm', 'k', 'k_c', 'alpha', 'beta'
        """
        m = self.extract_anchor(theta_current)

        if len(modes) < 2:
            # Single mode: fall back
            zeta, omega = modes[0]
            return self.invert(zeta, omega, theta_current)

        zeta2, omega2 = float(modes[0][0]), float(modes[0][1])   # dominant (out-of-phase)
        zeta1, omega1 = float(modes[1][0]), float(modes[1][1])   # secondary (in-phase)

        # Enforce ordering: ω₂ > ω₁ required for k_c > 0
        if omega2 < omega1 + 1e-9:
            omega2 = omega1 * 1.02

        k   = m * omega1 ** 2
        k_c = m * (omega2 ** 2 - omega1 ** 2) / 2.0

        denom = omega2 ** 2 - omega1 ** 2
        if abs(denom) < 1e-12:
            beta  = 0.0
            alpha = 2.0 * zeta1 * omega1
        else:
            beta  = 2.0 * (zeta2 * omega2 - zeta1 * omega1) / denom
            alpha = 2.0 * zeta1 * omega1 - beta * omega1 ** 2

        return {
            'm':     m,
            'k':     k,
            'k_c':   k_c,
            'alpha': max(alpha, 0.0),
            'beta':  max(beta,  0.0),
        }


__all__ = ['DomainInverter', 'RLCInverter', 'MSDInverter', 'TwoMassSpringInverter']
