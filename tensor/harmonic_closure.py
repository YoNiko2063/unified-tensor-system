"""
HarmonicClosureChecker — algebraic closure test for Koopman operator admissibility.

Mathematical basis (Plan §III, INV-3):
  Replaces all dissonance/consonance comparisons with an algebraic projection test.

  Harmonic envelope of active algebra A = {K₁, ..., K_m}:
    H(A) = {K : ‖K - Π_{span(A)} K‖_F < ε_closure}

  Closure test for K_new:
    r = ‖K_new - Π_{span(A)} K_new‖_F   (Frobenius residual)
    r < ε_closure  → K_new ∈ H(A)  → REDUNDANT  (do not add, safe to merge)
    r ≥ ε_closure  → K_new ∉ H(A)  → check admissibility

  Admissibility (ADMITTED iff all hold):
    r ≥ ε_closure                        (spectrally independent of current algebra)
    recon_error_after < before - δ_min   (reduces reconstruction error, if provided)
    trust_new > τ_admit                  (well-fitted, not noise)
    NOT monitor_unstable                 (within success membrane S)

  Bootstrap (empty active algebra):
    Π_{∅} K_new = 0, so r = ‖K_new‖_F.
    If K_new is nonzero, r ≥ ε_closure; admissibility check applies normally.

MINOR-1 fix: empty active_operators handled explicitly (no lstsq with empty matrix).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class HarmonicClosureChecker:
    """
    Determines whether a candidate operator K_new is REDUNDANT, ADMISSIBLE,
    or INADMISSIBLE relative to the set of active Koopman operators A.

    Works on arbitrary numpy arrays treated as operators. Callers are responsible
    for providing a consistent operator representation:
      - Full Koopman K matrix (from EDMDKoopman._K): for full algebraic tests
      - Diagonal spectral proxy diag(sorted |Re(λ)|): for HarmonicAtlas integration
        where K matrices are not available but eigenvalues are (from PatchClassification)

    The projection test is the same in both cases: measure how well K_new lies
    within the span of the existing operators (treated as a flat vector space).

    Usage:
        checker = HarmonicClosureChecker(eps_closure=0.1, delta_min=0.02, tau_admit=0.3)
        result = checker.check(K_new, active_operators, trust_new=0.6)
        if result == "redundant": atlas.merge(...)
        elif result == "admissible": atlas.add(...)
        # "inadmissible": discard
    """

    def __init__(
        self,
        eps_closure: float = 0.1,
        delta_min: float = 0.02,
        tau_admit: float = 0.3,
    ) -> None:
        """
        Args:
            eps_closure:  Frobenius projection residual below which K_new is REDUNDANT
            delta_min:    minimum reconstruction error reduction required for ADMISSIBLE
            tau_admit:    minimum trust score required for ADMISSIBLE
        """
        self._eps_closure = eps_closure
        self._delta_min = delta_min
        self._tau_admit = tau_admit

    # ── Core projection ────────────────────────────────────────────────────────

    def projection_residual(
        self,
        K_new: np.ndarray,
        active_operators: List[np.ndarray],
    ) -> float:
        """
        Frobenius residual of projecting K_new onto span(active_operators).

        r = ‖K_new - Π_{span(A)} K_new‖_F
        Π_{span(A)} K_new = least-squares solution minimizing ‖K_new - Σᵢ αᵢ Kᵢ‖_F

        MINOR-1 fix: if active_operators is empty, Π_{∅} K_new = 0, r = ‖K_new‖_F.
        This is the bootstrap case; the first operator is always potentially admissible.

        Args:
            K_new:            candidate operator matrix (any shape, treated as flat vector)
            active_operators: list of existing operator matrices (same shape as K_new)

        Returns:
            float ≥ 0; near 0 → K_new ∈ H(A) → REDUNDANT
        """
        k_flat = K_new.ravel().astype(float)

        if not active_operators:
            # Bootstrap: empty algebra, Π = 0
            return float(np.linalg.norm(k_flat))

        # Build matrix whose columns are vectorized active operators: shape (d, m)
        A_cols = np.column_stack(
            [op.ravel().astype(float) for op in active_operators]
        )

        # lstsq: find α minimizing ‖k_flat - A_cols @ α‖²
        coeffs, _, _, _ = np.linalg.lstsq(A_cols, k_flat, rcond=None)
        projected = A_cols @ coeffs
        residual = k_flat - projected
        return float(np.linalg.norm(residual))

    # ── Main decision ─────────────────────────────────────────────────────────

    def check(
        self,
        K_new: np.ndarray,
        active_operators: List[np.ndarray],
        trust_new: float = 0.0,
        monitor_unstable: bool = False,
        recon_error_before: Optional[float] = None,
        recon_error_after: Optional[float] = None,
    ) -> str:
        """
        Classify K_new relative to the active operator algebra A.

        Returns:
            "redundant"    — K_new ∈ H(A): lies within closure envelope, safe to merge
            "admissible"   — K_new ∉ H(A) AND passes all extension conditions
            "inadmissible" — K_new ∉ H(A) but fails at least one extension condition

        Extension conditions (all required for "admissible"):
          1. r ≥ ε_closure          (spectrally independent — checked first)
          2. recon_ok                (if recon_errors provided: after < before - δ_min;
                                      if not provided: condition skipped, assumed True)
          3. trust_new > τ_admit    (K_new is well-fitted, not fitting noise)
          4. NOT monitor_unstable   (system is within success membrane S)

        Args:
            K_new:               candidate operator
            active_operators:    current algebra generators
            trust_new:           Koopman trust of K_new (from prior EDMD result)
            monitor_unstable:    GeometryMonitor.is_unstable() value
            recon_error_before:  reconstruction error of current algebra (optional)
            recon_error_after:   reconstruction error after adding K_new (optional)
        """
        r = self.projection_residual(K_new, active_operators)

        if r < self._eps_closure:
            return "redundant"

        # Check reconstruction error reduction (only if both values provided)
        if recon_error_before is not None and recon_error_after is not None:
            recon_ok = recon_error_after < recon_error_before - self._delta_min
        else:
            recon_ok = True  # not enough information; skip this gate

        if recon_ok and trust_new > self._tau_admit and not monitor_unstable:
            return "admissible"

        return "inadmissible"
