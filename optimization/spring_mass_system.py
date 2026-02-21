"""
Spring-mass-damper system: analogue to the RLC filter.

Physical equivalence:
  RLC series circuit  ←→  spring-mass-damper
  charge q            ←→  displacement x
  L (inductance)      ←→  m (mass)
  C⁻¹ (stiffness)     ←→  k (spring constant)
  R (resistance)      ←→  b (damping coefficient)

Domain-invariant quantities (same formula in both domains):
  ω₀ = 1/√(LC)  =  √(k/m)         natural angular frequency [rad/s]
  Q  = (1/R)√(L/C) = √(km)/b       quality factor
  ζ  = R/(2√(L/C)) = b/(2√(km))    damping ratio = 1/(2Q)

This lets a KoopmanExperienceMemory trained on spring-mass optimisations
provide warm starts for RLC optimisations via the shared (ω₀, Q) descriptor.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from optimization.koopman_signature import (
    KoopmanInvariantDescriptor,
    compute_invariants,
    _LOG_OMEGA0_REF,
    _LOG_OMEGA0_SCALE,
)
from optimization.koopman_memory import (
    KoopmanExperienceMemory,
    OptimizationExperience,
)
from tensor.koopman_edmd import EDMDKoopman


# ── Params + Mapper ────────────────────────────────────────────────────────────


@dataclass
class SpringMassParams:
    k: float   # spring constant   [N/m]
    m: float   # mass              [kg]
    b: float   # damping coefficient [N·s/m]

    def __str__(self) -> str:
        return f"k={self.k:.4g} N/m  m={self.m:.4g} kg  b={self.b:.4g} N·s/m"

    def as_dict(self) -> dict:
        return {"k": self.k, "m": self.m, "b": self.b}


class SpringMassDesignMapper:
    """
    Deterministic HDV → (k, m, b) projection, mirroring RLCDesignMapper.

    k = k_center * exp(clip(a_k · z, ±max_exp))
    m = m_center * exp(clip(a_m · z, ±max_exp))
    b = b_center * exp(clip(a_b · z, ±max_exp))

    Default centers give ω₀ = √(k/m) = 2π × 1000 rad/s (1 kHz) with Q = 1:
      k = 1000 N/m
      m = k / ω₀² = 1000 / (2π×1000)² ≈ 2.533e-5 kg
      b = √(k·m) / Q = √(1000 × 2.533e-5) / 1 ≈ 0.1591 N·s/m  → Q=1 → ζ=0.5
    """

    def __init__(
        self,
        hdv_dim: int = 64,
        k_center: float = 1_000.0,     # N/m
        m_center: float = 2.533e-5,    # kg   → ω₀ = √(k/m) ≈ 6283 rad/s ≈ 2π×1kHz
        b_center: float = 0.15915,     # N·s/m → Q = √(km)/b ≈ 1.0
        seed: int = 7,
        max_exp: float = 3.0,
    ) -> None:
        self.hdv_dim = hdv_dim
        self.k_center = k_center
        self.m_center = m_center
        self.b_center = b_center
        self._max_exp = max_exp

        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((hdv_dim, 3))
        Q_mat, _ = np.linalg.qr(raw)
        self._a_k = Q_mat[:, 0]
        self._a_m = Q_mat[:, 1]
        self._a_b = Q_mat[:, 2]
        self._A = np.stack([self._a_k, self._a_m, self._a_b])  # (3, hdv_dim)

    def decode(self, z: np.ndarray) -> SpringMassParams:
        e_k = float(np.clip(self._a_k @ z, -self._max_exp, self._max_exp))
        e_m = float(np.clip(self._a_m @ z, -self._max_exp, self._max_exp))
        e_b = float(np.clip(self._a_b @ z, -self._max_exp, self._max_exp))
        return SpringMassParams(
            k=self.k_center * float(np.exp(e_k)),
            m=self.m_center * float(np.exp(e_m)),
            b=self.b_center * float(np.exp(e_b)),
        )

    def encode(self, params: SpringMassParams) -> np.ndarray:
        log_k = np.log(max(params.k, 1e-30) / self.k_center)
        log_m = np.log(max(params.m, 1e-30) / self.m_center)
        log_b = np.log(max(params.b, 1e-30) / self.b_center)
        target = np.array([log_k, log_m, log_b])
        z = self._A.T @ np.linalg.lstsq(self._A @ self._A.T, target, rcond=None)[0]
        return z


# ── Evaluator ──────────────────────────────────────────────────────────────────


@dataclass
class SpringMassResult:
    params: SpringMassParams
    natural_freq_hz: float    # f₀ = ω₀/(2π) [Hz]
    target_hz: float
    objective: float          # combined J = w_freq·freq_err + w_Q·Q_err
    Q_factor: float
    energy_loss: float
    constraints_ok: bool
    constraint_detail: Dict[str, Tuple[bool, float, float]]
    Q_target: Optional[float] = None   # Q target (None = freq-only mode)
    Q_error: float = 0.0              # |Q - Q_target| / Q_target
    freq_error: float = 0.0           # |f - f_target| / f_target


class SpringMassEvaluator:
    """
    Evaluate a spring-mass-damper design.

    Objective (multi-objective when Q_target is set):
        J = w_freq * |f₀ - f_target| / f_target
          + w_Q    * |Q  - Q_target| / Q_target
    Constraints: Q ≤ max_Q, energy_loss ≤ max_loss, k/m/b > 0

    When Q_target is None: reduces to pure frequency-error objective.
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

    def natural_frequency_rad(self, p: SpringMassParams) -> float:
        """ω₀ = √(k/m)  [rad/s]"""
        return float(np.sqrt(max(p.k / max(p.m, 1e-30), 0.0)))

    def natural_frequency_hz(self, p: SpringMassParams) -> float:
        return self.natural_frequency_rad(p) / (2.0 * math.pi)

    def Q_factor(self, p: SpringMassParams) -> float:
        """Q = √(k·m) / b"""
        if p.b <= 0:
            return float("inf")
        return float(np.sqrt(max(p.k * p.m, 0.0)) / max(p.b, 1e-30))

    def energy_loss_estimate(self, p: SpringMassParams) -> float:
        """ζ = 1/(2Q) — fractional energy loss per cycle."""
        return 1.0 / (2.0 * max(self.Q_factor(p), 1e-12))

    def dynamical_quantities(self, p: SpringMassParams) -> Tuple[float, float, float]:
        """Return (omega0, Q, zeta) — domain-invariant dynamical quantities."""
        omega0 = self.natural_frequency_rad(p)
        Q = max(self.Q_factor(p), 1e-12)
        zeta = 1.0 / (2.0 * Q)
        return float(omega0), float(Q), float(zeta)

    def freq_error(self, p: SpringMassParams, target_hz: float) -> float:
        """Fractional frequency error |f₀ - f_target| / f_target."""
        return abs(self.natural_frequency_hz(p) - target_hz) / max(abs(target_hz), 1e-12)

    def Q_error_val(self, p: SpringMassParams) -> float:
        """Fractional Q error |Q - Q_target| / Q_target (0 if Q_target is None)."""
        if self.Q_target is None:
            return 0.0
        q = self.Q_factor(p)
        return abs(q - self.Q_target) / max(abs(self.Q_target), 1e-12)

    def objective(self, p: SpringMassParams, target_hz: float) -> float:
        """Combined J = w_freq * freq_error + w_Q * Q_error."""
        fe = self.freq_error(p, target_hz)
        qe = self.Q_error_val(p)
        return self.w_freq * fe + self.w_Q * qe

    def constraints(self, p: SpringMassParams) -> Dict[str, Tuple[bool, float, float]]:
        q = self.Q_factor(p)
        loss = self.energy_loss_estimate(p)
        return {
            "Q_limit":      (q    <= self.max_Q,            q,       self.max_Q),
            "energy_loss":  (loss <= self.max_energy_loss,  loss,    self.max_energy_loss),
            "k_positive":   (p.k > 0,                       p.k,     0.0),
            "m_positive":   (p.m > 0,                       p.m,     0.0),
            "b_positive":   (p.b > 0,                       p.b,     0.0),
        }

    def evaluate(self, p: SpringMassParams, target_hz: float) -> SpringMassResult:
        c_detail = self.constraints(p)
        ok = all(v[0] for v in c_detail.values())
        fe = self.freq_error(p, target_hz)
        qe = self.Q_error_val(p)
        return SpringMassResult(
            params=p,
            natural_freq_hz=self.natural_frequency_hz(p),
            target_hz=target_hz,
            objective=self.w_freq * fe + self.w_Q * qe,
            Q_factor=self.Q_factor(p),
            energy_loss=self.energy_loss_estimate(p),
            constraints_ok=ok,
            constraint_detail=c_detail,
            Q_target=self.Q_target,
            Q_error=qe,
            freq_error=fe,
        )


# ── Optimizer ─────────────────────────────────────────────────────────────────

_TRACE_OPS: List[str] = [
    "freq_eval", "Q_eval", "constraint_check", "objective_eval", "accepted",
]
_WINDOW_SIZE: int = 20
_MIN_TRACE_FOR_KOOPMAN: int = 10


class SpringMassOptimizer:
    """
    Gradient-free constrained search in HDV space for spring-mass-damper systems.

    Stores OptimizationExperience with domain="spring_mass" in a shared
    KoopmanExperienceMemory so that RLC optimisers can retrieve cross-domain
    warm starts via the shared (ω₀, Q) metric.

    Interface mirrors ConstrainedHDVOptimizer for easy substitution.
    """

    def __init__(
        self,
        mapper: SpringMassDesignMapper,
        evaluator: SpringMassEvaluator,
        memory: KoopmanExperienceMemory,
        n_iter: int = 500,
        seed: int = 0,
        tol: float = 1e-3,
        step_init: float = 0.5,
    ) -> None:
        self.mapper = mapper
        self.evaluator = evaluator
        self.memory = memory
        self.n_iter = n_iter
        self._rng = np.random.default_rng(seed)
        self.tol = tol
        self.step_init = step_init

    def optimize(self, target_hz: float) -> SpringMassResult:
        """
        Search for a spring-mass-damper design matching target_hz.

        Stores the optimisation experience in memory with domain="spring_mass".
        """
        z = self._rng.standard_normal(self.mapper.hdv_dim)
        event_buffer: deque = deque(maxlen=_WINDOW_SIZE * 20)
        param_trace: List[np.ndarray] = []

        best_result = self._eval_and_log(z, target_hz, event_buffer, force_log=True)
        if best_result.constraints_ok:
            self._log_param_step(best_result.params, param_trace)

        step = self.step_init

        for _ in range(self.n_iter):
            z_cand = z + self._rng.standard_normal(self.mapper.hdv_dim) * step
            result = self._eval_and_log(z_cand, target_hz, event_buffer)

            if result.constraints_ok and result.objective < best_result.objective:
                best_result = result
                z = z_cand
                step = min(step * 1.1, 3.0)
                self._log_param_step(result.params, param_trace)
                event_buffer.append("accepted")
            else:
                step = max(step * 0.95, 1e-3)

            if best_result.objective < self.tol:
                break

        self._store_experience(event_buffer, param_trace, best_result)
        return best_result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _eval_and_log(
        self,
        z: np.ndarray,
        target_hz: float,
        event_buffer: deque,
        force_log: bool = False,
    ) -> SpringMassResult:
        params = self.mapper.decode(z)
        result = self.evaluator.evaluate(params, target_hz)
        event_buffer.extend(["freq_eval", "Q_eval", "constraint_check"])
        if result.constraints_ok or force_log:
            event_buffer.append("objective_eval")
        return result

    @staticmethod
    def _log_param_step(params: SpringMassParams, trace: List[np.ndarray]) -> None:
        trace.append(np.array([
            np.log(max(params.k, 1e-30)),
            np.log(max(params.m, 1e-30)),
            np.log(max(params.b, 1e-30)),
        ]))

    def _dynamical_invariants(
        self, best_result: SpringMassResult
    ) -> Tuple[float, float, float]:
        omega0, Q, zeta = self.evaluator.dynamical_quantities(best_result.params)
        log_omega0_norm = float(np.clip(
            (np.log(max(omega0, 1e-30)) - _LOG_OMEGA0_REF) / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        log_Q_norm = float(np.clip(
            np.log(max(Q, 1e-30)) / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        return log_omega0_norm, log_Q_norm, float(np.clip(zeta, 0.0, 10.0))

    def _build_operator_trace(self, event_buffer: deque) -> Optional[np.ndarray]:
        events = list(event_buffer)
        if len(events) < _WINDOW_SIZE + 1:
            return None
        op_idx = {op: i for i, op in enumerate(_TRACE_OPS)}
        n_ops = len(_TRACE_OPS)
        rates = []
        for t in range(_WINDOW_SIZE, len(events) + 1):
            window = events[t - _WINDOW_SIZE: t]
            rate = np.zeros(n_ops)
            for ev in window:
                if ev in op_idx:
                    rate[op_idx[ev]] += 1
            rates.append(rate / _WINDOW_SIZE)
        arr = np.array(rates)
        return arr if len(arr) >= 2 else None

    def _fit_koopman(self, trace: Optional[np.ndarray]):
        if trace is None or len(trace) < _MIN_TRACE_FOR_KOOPMAN:
            return None
        pairs = [(trace[i], trace[i + 1]) for i in range(len(trace) - 1)]
        edmd = EDMDKoopman(observable_degree=1)
        try:
            edmd.fit(pairs)
            result = edmd.eigendecomposition()
        except Exception:
            return None
        return result

    def _store_experience(
        self,
        event_buffer: deque,
        param_trace: List[np.ndarray],
        best_result: SpringMassResult,
    ) -> None:
        op_trace = self._build_operator_trace(event_buffer)
        koop_result = self._fit_koopman(op_trace)

        if koop_result is None and len(param_trace) >= _MIN_TRACE_FOR_KOOPMAN:
            p_arr = np.array(param_trace)
            koop_result = self._fit_koopman(p_arr)

        if koop_result is None:
            return

        log_omega0_norm, log_Q_norm, zeta = self._dynamical_invariants(best_result)
        invariant = compute_invariants(
            koop_result.eigenvalues, koop_result.eigenvectors, _TRACE_OPS,
            log_omega0_norm=log_omega0_norm,
            log_Q_norm=log_Q_norm,
            damping_ratio=zeta,
        )

        if len(invariant.dominant_operator_histogram) > 0:
            dom_idx = int(np.argmax(invariant.dominant_operator_histogram))
            bottleneck_op = (
                _TRACE_OPS[dom_idx]
                if dom_idx < len(_TRACE_OPS)
                else "unknown"
            )
        else:
            bottleneck_op = "unknown"

        improvement = max(0.0, 1.0 - best_result.objective)
        experience = OptimizationExperience(
            bottleneck_operator=bottleneck_op,
            replacement_applied="analytic_correction",
            runtime_improvement=improvement,
            n_observations=1,
            hardware_target="cpu",
            best_params=best_result.params.as_dict(),
            domain="spring_mass",
        )
        self.memory.add(invariant, koop_result, experience)
