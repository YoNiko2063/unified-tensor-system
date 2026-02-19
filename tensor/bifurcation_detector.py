"""
Bifurcation Detector — detects regime transitions via eigenvalue-crossing analysis.

Mathematical basis (LOGIC_FLOW.md, Section 0G):
  A bifurcation occurs when Re(λᵢ(x)) = 0 and changes sign.
  This is a codimension-1 spectral boundary separating stable from unstable regimes.

  Four detectable types:
    Saddle-node:    one real eigenvalue crosses 0
    Hopf:           complex conjugate pair crosses imaginary axis
    Pitchfork:      simple eigenvalue crosses 0 with symmetry
    Transcritical:  eigenvalue zero with non-degenerate crossing

  Neural training signal: predict d_true = min_i |Re(λᵢ)| before crossing occurs.

Reference: LOGIC_FLOW.md Sections 0G, 0H
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BifurcationStatus:
    status: str                   # 'stable' | 'critical' | 'bifurcation'
    bifurcation_type: str         # 'none' | 'saddle_node' | 'hopf' | 'unknown'
    min_real_part: float          # min_i Re(λᵢ) — distance to stability boundary
    spectral_gap: float           # |Re(λ₁) - Re(λ₂)|
    real_part_derivative: float   # d(min_real)/dt — approach rate
    imag_magnitude: float         # max_i |Im(λᵢ)| — oscillatory magnitude
    feature_vector: np.ndarray    # [min_real, gap, derivative, imag_mag] for DNN


class BifurcationDetector:
    """
    Detects bifurcation boundaries by tracking eigenvalue sign changes.

    Provides:
      1. Real-time status: stable / critical / bifurcation
      2. Bifurcation type classification (saddle-node, Hopf, etc.)
      3. A 4-scalar feature vector suitable for DNN training

    The trainable signal: predict distance-to-bifurcation (min_i |Re(λᵢ)|)
    before the crossing happens. Loss: ‖d̂ - d_true‖².

    Usage:
        detector = BifurcationDetector(zero_tol=1e-3)

        for eigvals in trajectory_eigenvalues:
            result = detector.check(eigvals)
            if result.status == 'bifurcation':
                # trigger Koopman mode
                switch_to_koopman()
    """

    def __init__(
        self,
        zero_tol: float = 1e-3,
        dt: float = 1.0,
    ):
        """
        Args:
            zero_tol: tolerance for eigenvalue proximity to imaginary axis
            dt: timestep (used to compute derivative of min real part)
        """
        self.zero_tol = zero_tol
        self.dt = dt
        self._prev_eigvals: Optional[np.ndarray] = None
        self._prev_min_real: Optional[float] = None
        self._history: List[BifurcationStatus] = []

    def check(self, eigvals: np.ndarray) -> BifurcationStatus:
        """
        Check current eigenvalues for bifurcation proximity.

        Args:
            eigvals: (n,) array of complex eigenvalues

        Returns:
            BifurcationStatus with status, type, and feature vector
        """
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        min_real = float(np.min(real_parts))
        gap = self._spectral_gap(real_parts)
        imag_mag = float(np.max(np.abs(imag_parts)))

        # Derivative of min real part
        if self._prev_min_real is not None:
            real_deriv = (min_real - self._prev_min_real) / self.dt
        else:
            real_deriv = 0.0

        # Determine status — check if ANY eigenvalue is near imaginary axis
        if np.any(np.abs(real_parts) < self.zero_tol):
            status = 'critical'
            bif_type = self._classify_critical(eigvals)
        elif (self._prev_eigvals is not None and
              np.any(np.sign(np.real(self._prev_eigvals)) != np.sign(real_parts))):
            status = 'bifurcation'
            bif_type = self._classify_crossing(self._prev_eigvals, eigvals)
        else:
            status = 'stable'
            bif_type = 'none'

        feature_vec = np.array([min_real, gap, real_deriv, imag_mag], dtype=np.float32)

        result = BifurcationStatus(
            status=status,
            bifurcation_type=bif_type,
            min_real_part=min_real,
            spectral_gap=gap,
            real_part_derivative=real_deriv,
            imag_magnitude=imag_mag,
            feature_vector=feature_vec,
        )

        # Update history
        self._prev_eigvals = eigvals.copy()
        self._prev_min_real = min_real
        self._history.append(result)

        return result

    def distance_to_boundary(self, eigvals: np.ndarray) -> float:
        """
        d_true = min_i |Re(λᵢ)| — the neural training target.

        Small distance → system approaching bifurcation boundary.
        """
        return float(np.min(np.abs(np.real(eigvals))))

    def feature_vector(
        self,
        eigvals: np.ndarray,
        prev_eigvals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        4-scalar feature vector for DNN training: [min_real, gap, derivative, imag_mag]

        Suitable as input to Phase 1 spectral geometry learning (LOGIC_FLOW.md 0K).
        """
        real_parts = np.real(eigvals)
        min_real = float(np.min(real_parts))
        gap = self._spectral_gap(real_parts)
        imag_mag = float(np.max(np.abs(np.imag(eigvals))))

        if prev_eigvals is not None:
            prev_min = float(np.min(np.real(prev_eigvals)))
            deriv = (min_real - prev_min) / self.dt
        else:
            deriv = 0.0

        return np.array([min_real, gap, deriv, imag_mag], dtype=np.float32)

    def reset(self) -> None:
        """Reset detector state (use when starting new trajectory)."""
        self._prev_eigvals = None
        self._prev_min_real = None
        self._history.clear()

    def history(self) -> List[BifurcationStatus]:
        """Return list of all BifurcationStatus results since last reset."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spectral_gap(real_parts: np.ndarray) -> float:
        """|Re(λ₁) - Re(λ₂)| for top two dominant eigenvalues."""
        r = np.abs(real_parts)
        r_sorted = np.sort(r)[::-1]
        if len(r_sorted) < 2:
            return float(r_sorted[0])
        return float(r_sorted[0] - r_sorted[1])

    def _classify_critical(self, eigvals: np.ndarray) -> str:
        """Classify type of critical point based on eigenvalue structure."""
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        near_zero = np.abs(real_parts) < self.zero_tol

        # Hopf: complex conjugate pair near imaginary axis
        if np.any(near_zero & (np.abs(imag_parts) > self.zero_tol)):
            return 'hopf'

        # Saddle-node or pitchfork: real eigenvalue near zero
        if np.any(near_zero & (np.abs(imag_parts) < self.zero_tol)):
            return 'saddle_node'

        return 'unknown'

    def _classify_crossing(
        self,
        prev_eigvals: np.ndarray,
        curr_eigvals: np.ndarray,
    ) -> str:
        """Classify bifurcation type from eigenvalue sign change."""
        prev_real = np.real(prev_eigvals)
        curr_real = np.real(curr_eigvals)
        prev_imag = np.imag(prev_eigvals)

        crossed = np.sign(prev_real) != np.sign(curr_real)

        # Hopf: crossing pair has imaginary component
        if np.any(crossed & (np.abs(prev_imag) > self.zero_tol)):
            return 'hopf'

        # Saddle-node: real eigenvalue crosses zero
        if np.any(crossed & (np.abs(prev_imag) < self.zero_tol)):
            return 'saddle_node'

        return 'unknown'
