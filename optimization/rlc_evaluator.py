"""
RLCEvaluator — physics-based evaluation of RLC filter designs.

Second-order RLC low-pass transfer function (output across C):
    H(s) = ω₀² / (s² + (ω₀/Q)s + ω₀²)

where:
    ω₀ = 1/√(LC)        resonant / cutoff frequency  [rad/s]
    Q  = (1/R)√(L/C)    quality factor               [dimensionless]

Objective (multi-objective when Q_target is set):
    J = w_freq * |f_computed - f_target| / f_target
      + w_Q    * |Q_computed - Q_target| / Q_target

Constraints: Q ≤ max_Q, energy_loss ≤ max_loss, R/L/C > 0

When Q_target is None the evaluator reduces to the original single-objective
(frequency-only) mode, preserving full backward compatibility.

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
    objective: float          # combined objective J ∈ [0, ∞)
    Q_factor: float
    energy_loss: float
    constraints_ok: bool
    constraint_detail: Dict[str, Tuple[bool, float, float]]  # name→(ok, value, limit)
    Q_target: Optional[float] = None   # Q target (None = frequency-only mode)
    Q_error: float = 0.0              # |Q - Q_target| / Q_target  (0 if no Q_target)
    freq_error: float = 0.0           # |f - f_target| / f_target (always computed)

    def __str__(self) -> str:
        ok_str = "✓" if self.constraints_ok else "✗"
        q_str = f"  Q_target={self.Q_target:.2f}  Q_err={self.Q_error:.4f}" \
            if self.Q_target is not None else ""
        return (
            f"{self.params}  f₀={self.cutoff_hz:.2f} Hz  "
            f"Q={self.Q_factor:.3f}  J={self.objective:.4f}"
            f"{q_str}  constraints={ok_str}"
        )


class RLCEvaluator:
    """
    Numerically evaluate an RLC filter design.

    Args:
        max_Q:           upper bound on quality factor          (default 10)
        max_energy_loss: upper bound on fractional energy loss  (default 0.5)
        Q_target:        desired Q factor for multi-objective mode (None = freq-only)
        w_freq:          weight on frequency error term          (default 1.0)
        w_Q:             weight on Q error term                  (default 1.0)
    """

    def __init__(
        self,
        max_Q: float = 10.0,
        max_energy_loss: float = 0.5,
        Q_target: Optional[float] = None,
        w_freq: float = 1.0,
        w_Q: float = 1.0,
    ) -> None:
        self.max_Q = max_Q
        self.max_energy_loss = max_energy_loss
        self.Q_target = Q_target
        self.w_freq = w_freq
        self.w_Q = w_Q

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

    def freq_error(self, params: RLCParams, target_hz: float) -> float:
        """Fractional cutoff error |f - f_target| / f_target."""
        computed = self.cutoff_frequency_hz(params)
        return abs(computed - target_hz) / max(abs(target_hz), 1e-12)

    def Q_error_val(self, params: RLCParams) -> float:
        """Fractional Q error |Q - Q_target| / Q_target (0 if Q_target is None)."""
        if self.Q_target is None:
            return 0.0
        q = self.Q_factor(params)
        return abs(q - self.Q_target) / max(abs(self.Q_target), 1e-12)

    def objective(self, params: RLCParams, target_hz: float) -> float:
        """
        Combined objective J = w_freq * freq_error + w_Q * Q_error.

        In single-objective mode (Q_target=None): J = freq_error.
        In multi-objective mode: J penalises both frequency and Q deviation.
        Lower is better.
        """
        fe = self.freq_error(params, target_hz)
        qe = self.Q_error_val(params)
        return self.w_freq * fe + self.w_Q * qe

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
        fe = self.freq_error(params, target_hz)
        qe = self.Q_error_val(params)
        return EvaluationResult(
            params=params,
            cutoff_hz=self.cutoff_frequency_hz(params),
            target_hz=target_hz,
            objective=self.w_freq * fe + self.w_Q * qe,
            Q_factor=self.Q_factor(params),
            energy_loss=self.energy_loss_estimate(params),
            constraints_ok=ok,
            constraint_detail=c_detail,
            Q_target=self.Q_target,
            Q_error=qe,
            freq_error=fe,
        )

    # ── Domain-invariant dynamical quantities ─────────────────────────────────

    def dynamical_quantities(self, params: RLCParams) -> Tuple[float, float, float]:
        """
        Return (omega0, Q, zeta) — domain-invariant dynamical quantities.

        omega0 = 1/√(LC)        [rad/s]  resonant angular frequency
        Q      = (1/R)√(L/C)             quality factor
        zeta   = 1/(2Q)                  damping ratio
        """
        omega0 = self.cutoff_frequency_rad(params)
        Q = max(self.Q_factor(params), 1e-12)
        zeta = 1.0 / (2.0 * Q)
        return float(omega0), float(Q), float(zeta)

    def infer_params_from_dynamical(
        self,
        omega0: float,
        Q: float,
        R: Optional[float] = None,
    ) -> RLCParams:
        """
        Infer RLC params from domain-invariant (omega0, Q).

        Formula (with R given):
            L = Q·R / ω₀
            C = 1 / (ω₀·Q·R)
        Satisfies: 1/√(LC) = ω₀  and  (1/R)√(L/C) = Q.

        Args:
            omega0:  target resonant angular frequency [rad/s]
            Q:       target quality factor
            R:       resistance [Ω]; defaults to 100 Ω (nominal)
        """
        if R is None:
            R = 100.0
        omega0 = max(float(omega0), 1e-30)
        Q = max(float(Q), 1e-12)
        R = max(float(R), 1e-30)
        L = Q * R / omega0
        C = 1.0 / (omega0 * Q * R)
        return RLCParams(R=R, L=L, C=C)

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
