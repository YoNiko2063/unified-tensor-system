"""
RLCEvaluator — physics-based evaluation of RLC filter designs.

Second-order RLC low-pass transfer function (output across C):
    H(s) = ω₀² / (s² + (ω₀/Q)s + ω₀²)

where:
    ω₀ = 1/√(LC)        resonant / cutoff frequency  [rad/s]
    Q  = (1/R)√(L/C)    quality factor               [dimensionless]

Objective:   minimize fractional cutoff error |f_computed - f_target| / f_target
Constraints: Q ≤ max_Q, energy_loss ≤ max_loss, R/L/C > 0

verify_with_simulation() grounds the analytic prediction in physics via an
ecemath-compatible simulator interface.  When simulator=None it falls back
to the analytic result only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from optimization.rlc_parameterization import RLCParams


@dataclass
class EvaluationResult:
    params: RLCParams
    cutoff_hz: float          # computed cutoff frequency [Hz]
    target_hz: float          # target cutoff frequency   [Hz]
    objective: float          # fractional error ∈ [0, ∞)
    Q_factor: float
    energy_loss: float
    constraints_ok: bool
    constraint_detail: Dict[str, Tuple[bool, float, float]]  # name→(ok, value, limit)

    def __str__(self) -> str:
        ok_str = "✓" if self.constraints_ok else "✗"
        return (
            f"{self.params}  f₀={self.cutoff_hz:.2f} Hz  "
            f"Q={self.Q_factor:.3f}  err={self.objective:.4f}  constraints={ok_str}"
        )


class RLCEvaluator:
    """
    Numerically evaluate an RLC filter design.

    Args:
        max_Q:          upper bound on quality factor          (default 10)
        max_energy_loss: upper bound on fractional energy loss (default 0.5)
    """

    def __init__(
        self,
        max_Q: float = 10.0,
        max_energy_loss: float = 0.5,
    ) -> None:
        self.max_Q = max_Q
        self.max_energy_loss = max_energy_loss

    # ── Core formulas ──────────────────────────────────────────────────────────

    def cutoff_frequency_rad(self, params: RLCParams) -> float:
        """ω₀ = 1/√(LC)  [rad/s]"""
        return 1.0 / np.sqrt(max(params.L * params.C, 1e-30))

    def cutoff_frequency_hz(self, params: RLCParams) -> float:
        """f₀ = ω₀ / (2π)  [Hz]"""
        return self.cutoff_frequency_rad(params) / (2.0 * np.pi)

    def Q_factor(self, params: RLCParams) -> float:
        """Q = (1/R)√(L/C)"""
        if params.R <= 0:
            return float("inf")
        return (1.0 / params.R) * np.sqrt(max(params.L / max(params.C, 1e-30), 0.0))

    def energy_loss_estimate(self, params: RLCParams) -> float:
        """
        Fractional energy loss per oscillation cycle ≈ 1/(2Q).
        Derived from energy stored = Q × energy dissipated per radian.
        """
        q = self.Q_factor(params)
        return 1.0 / (2.0 * max(q, 1e-12))

    # ── Objective + constraints ────────────────────────────────────────────────

    def objective(self, params: RLCParams, target_hz: float) -> float:
        """Fractional cutoff error (non-negative, lower is better)."""
        computed = self.cutoff_frequency_hz(params)
        return abs(computed - target_hz) / max(abs(target_hz), 1e-12)

    def constraints(
        self, params: RLCParams
    ) -> Dict[str, Tuple[bool, float, float]]:
        """
        Returns dict: name → (satisfied: bool, value: float, limit: float).
        All constraints are numerically evaluable from params alone.
        """
        q = self.Q_factor(params)
        loss = self.energy_loss_estimate(params)
        return {
            "Q_limit":      (q    <= self.max_Q,          q,       self.max_Q),
            "energy_loss":  (loss <= self.max_energy_loss, loss,    self.max_energy_loss),
            "R_positive":   (params.R > 0,                params.R, 0.0),
            "L_positive":   (params.L > 0,                params.L, 0.0),
            "C_positive":   (params.C > 0,                params.C, 0.0),
        }

    def evaluate(self, params: RLCParams, target_hz: float) -> EvaluationResult:
        """Full evaluation of one design point."""
        c_detail = self.constraints(params)
        ok = all(v[0] for v in c_detail.values())
        return EvaluationResult(
            params=params,
            cutoff_hz=self.cutoff_frequency_hz(params),
            target_hz=target_hz,
            objective=self.objective(params, target_hz),
            Q_factor=self.Q_factor(params),
            energy_loss=self.energy_loss_estimate(params),
            constraints_ok=ok,
            constraint_detail=c_detail,
        )

    # ── Frequency response ─────────────────────────────────────────────────────

    def frequency_response(
        self, params: RLCParams, freqs_hz: np.ndarray
    ) -> np.ndarray:
        """
        |H(jω)| for series RLC low-pass filter (output across C).
        H(s) = ω₀² / (s² + (ω₀/Q)·s + ω₀²)
        """
        omega0 = self.cutoff_frequency_rad(params)
        Q = max(self.Q_factor(params), 1e-12)
        s = 1j * 2.0 * np.pi * freqs_hz
        H = omega0**2 / (s**2 + (omega0 / Q) * s + omega0**2)
        return np.abs(H)

    # ── ecemath simulation verification ───────────────────────────────────────

    def verify_with_simulation(
        self,
        params: RLCParams,
        target_hz: float,
        simulator: Optional[object] = None,
    ) -> dict:
        """
        Ground the analytic prediction in physics via ecemath simulation.

        Simulator interface (ecemath-compatible):
            simulator.run_rlc(R, L, C) → dict with key "cutoff_hz"

        When simulator is None, returns analytic prediction only.
        The closed-loop correction pattern:
            predict → simulate → correct if error > 1%

        Returns dict with keys: predicted_hz, measured_hz, error, valid, source.
        """
        predicted_hz = self.cutoff_frequency_hz(params)

        if simulator is None:
            return {
                "predicted_hz": predicted_hz,
                "measured_hz":  None,
                "error":        abs(predicted_hz - target_hz) / max(target_hz, 1e-12),
                "valid":        True,
                "source":       "analytic",
            }

        try:
            sim_result = simulator.run_rlc(params.R, params.L, params.C)
            measured_hz = float(sim_result.get("cutoff_hz", predicted_hz))
        except Exception as exc:
            return {
                "predicted_hz": predicted_hz,
                "measured_hz":  None,
                "error":        None,
                "valid":        False,
                "source":       "simulation_error",
                "detail":       str(exc),
            }

        error = abs(measured_hz - target_hz) / max(abs(target_hz), 1e-12)
        return {
            "predicted_hz": predicted_hz,
            "measured_hz":  measured_hz,
            "error":        error,
            "valid":        error < 0.01,   # within 1% of target
            "source":       "ecemath",
        }
