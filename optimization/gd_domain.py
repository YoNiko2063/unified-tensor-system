"""
Gradient Descent Domain — third domain in the dynamical invariant framework.

Maps momentum gradient descent on quadratic losses to the same (ω₀, Q)
invariant space used by RLC circuits and spring-mass systems.

Physical correspondence
-----------------------
GD with momentum on a quadratic loss f(x) = κ·x²/2 satisfies:

    v_{t+1} = β·v_t − α·κ·x_t      (velocity update)
    x_{t+1} = x_t + v_{t+1}         (position update)

which is equivalent to a damped harmonic oscillator:

    m·ẍ + b·ẋ + k·x = 0

with:
    ω₀ = √(α·κ)            natural angular frequency   [rad / step]
    Q  = √(α·κ) / (1 − β)  quality factor
    ζ  = (1 − β) / (2√(α·κ)) damping ratio = 1/(2Q)

So the "design parameters" (α, β) map to the same (ω₀, Q, ζ) triple
used for RLC and spring-mass — no memory code changes required.

Transfer semantics
------------------
A KoopmanExperienceMemory trained on RLC or spring-mass experiments can
warm-start GD hyperparameter search via the shared (ω₀, Q) metric.

Concretely: if a physics problem converged at Q=2, ω₀=1kHz, and a GD
problem targets the same Q-regime, the memory can suggest a (α, β) pair
that produces the right convergence rate — before any GD trial is run.

Design constraints
------------------
To keep the analogy exact, the curvature κ must be known (or estimated).
This module provides:
  - `QuadraticLoss(kappa)` — n-d quadratic with scalar curvature κ
  - `GDParams(lr, momentum)` — the optimizer's two hyperparameters
  - `GDDesignMapper` — HDV ↔ (lr, momentum)
  - `GDEvaluator` — runs GD, measures convergence speed, extracts (ω₀, Q)
  - `GDOptimizer` — same interface as SpringMassOptimizer / ConstrainedHDVOptimizer
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


# ── Loss function ─────────────────────────────────────────────────────────────


class QuadraticLoss:
    """
    n-dimensional quadratic loss  f(x) = (κ/2) ‖x‖².

    Gradient: ∇f(x) = κ·x.
    Hessian:  H = κ·I (constant — makes the GD↔oscillator analogy exact).
    Optimal point: x* = 0.

    Args:
        kappa:  curvature (Hessian eigenvalue)  [must be > 0]
        n_dim:  dimensionality of x (default 1 — scalar for clean analysis)
    """

    def __init__(self, kappa: float, n_dim: int = 1) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        self.kappa = float(kappa)
        self.n_dim = int(n_dim)

    def value(self, x: np.ndarray) -> float:
        """f(x) = (κ/2) ‖x‖²"""
        return float(0.5 * self.kappa * float(np.dot(x, x)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """∇f(x) = κ·x"""
        return self.kappa * x

    def initial_x(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """Draw a random starting point with ‖x₀‖ ≈ scale."""
        x = rng.standard_normal(self.n_dim)
        x *= scale / max(np.linalg.norm(x), 1e-30)
        return x


# ── Params + Mapper ───────────────────────────────────────────────────────────


@dataclass
class GDParams:
    """GD with momentum hyperparameters."""
    lr: float        # learning rate α  ∈ (0, ∞)
    momentum: float  # momentum β       ∈ [0, 1)

    def __str__(self) -> str:
        return f"lr={self.lr:.4g}  β={self.momentum:.4g}"

    def as_dict(self) -> dict:
        return {"lr": self.lr, "momentum": self.momentum}


class GDDesignMapper:
    """
    Deterministic HDV → (lr, momentum) projection, mirroring RLCDesignMapper.

        lr       = lr_center       * exp(clip(a_lr · z, ±max_exp))
        momentum = momentum_center * exp(clip(a_β  · z, ±max_exp))  [clipped to [0, 1)]

    Default centers target ω₀ ≈ 2π×1kHz with Q=1 on a unit curvature (κ=1) loss:
        ω₀ = √(lr·κ)  →  lr_center = (2π×1000)²/κ  at κ=1  ≈ 3.948e7
                         but we normalise to a slow regime: lr_center=0.01 (1 kHz in
                         units of κ, i.e. κ=ω₀²/lr ≈ 1e6 for f_ref=1 kHz)
        Q=1  →  β_center = 1 − ω₀/Q = 1 − √(lr·κ) = 1 − 0.1 = 0.9  at lr=0.01, κ=1

    In practice the mapper explores the log-space around (lr=0.01, β=0.9).
    """

    def __init__(
        self,
        hdv_dim: int = 64,
        lr_center: float = 0.01,
        momentum_center: float = 0.9,
        seed: int = 13,
        max_exp: float = 3.0,
    ) -> None:
        self.hdv_dim = hdv_dim
        self.lr_center = lr_center
        self.momentum_center = momentum_center
        self._max_exp = max_exp

        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((hdv_dim, 2))
        Q_mat, _ = np.linalg.qr(raw)
        self._a_lr  = Q_mat[:, 0]
        self._a_mom = Q_mat[:, 1]
        self._A = np.stack([self._a_lr, self._a_mom])  # (2, hdv_dim)

    def decode(self, z: np.ndarray) -> GDParams:
        e_lr  = float(np.clip(self._a_lr  @ z, -self._max_exp, self._max_exp))
        e_mom = float(np.clip(self._a_mom @ z, -self._max_exp, self._max_exp))
        lr  = self.lr_center       * float(np.exp(e_lr))
        mom = self.momentum_center * float(np.exp(e_mom))
        mom = float(np.clip(mom, 0.0, 1.0 - 1e-6))  # momentum ∈ [0, 1)
        return GDParams(lr=lr, momentum=mom)

    def encode(self, params: GDParams) -> np.ndarray:
        log_lr  = np.log(max(params.lr,                  1e-30) / self.lr_center)
        log_mom = np.log(max(params.momentum + 1e-9,     1e-30) / self.momentum_center)
        target  = np.array([log_lr, log_mom])
        z = self._A.T @ np.linalg.lstsq(self._A @ self._A.T, target, rcond=None)[0]
        return z


# ── Evaluator ─────────────────────────────────────────────────────────────────


@dataclass
class GDResult:
    """Result of one GD hyperparameter evaluation."""
    params: GDParams
    kappa: float              # loss curvature used
    convergence_rate: float   # steps to reach target_loss (lower = faster = better)
    target_steps: int         # budget
    actual_steps: int         # steps actually taken before convergence (or budget)
    final_loss: float
    objective: float          # 1 − (budget − actual_steps)/budget  [0=best, 1=worst]
    Q_factor: float           # extracted Q = ω₀/(1−β)
    omega0: float             # extracted ω₀ = √(lr·κ)  [rad/step]
    constraints_ok: bool
    constraint_detail: Dict[str, Tuple[bool, float, float]]
    Q_target: Optional[float] = None
    Q_error: float = 0.0
    freq_error: float = 0.0


class GDEvaluator:
    """
    Evaluate a (lr, momentum) pair by running GD on a QuadraticLoss.

    Objective: minimise convergence time (steps to reach loss < tol).
    Constraints: ω₀ ≤ ω₀_max (stability), momentum ∈ [0, 1).

    When Q_target is set:
        J = w_freq * freq_error + w_Q * Q_error

    where freq_error = |ω₀_achieved − ω₀_target| / ω₀_target.

    The "target frequency" for GD is supplied as an ω₀ in [rad/step],
    not as Hz — the _LOG_OMEGA0_REF normalisation handles the mapping.
    """

    # Stability condition for GD with momentum on quadratic:
    #   lr·κ < 2(1+β)  →  ω₀² = lr·κ < 2(1+β)
    # Keep a margin (95%) below this boundary.
    _STABILITY_MARGIN: float = 0.95

    def __init__(
        self,
        loss: QuadraticLoss,
        max_steps: int = 500,
        tol: float = 1e-4,            # convergence threshold on loss value
        max_omega0: Optional[float] = None,  # if None, derived from stability
        Q_target: Optional[float] = None,
        w_freq: float = 1.0,
        w_Q: float = 1.0,
    ) -> None:
        self.loss = loss
        self.max_steps = max_steps
        self.tol = tol
        self.max_omega0 = max_omega0
        self.Q_target = Q_target
        self.w_freq = w_freq
        self.w_Q = w_Q

    # ── Domain-invariant dynamical quantities ─────────────────────────────────

    def dynamical_quantities(self, params: GDParams) -> Tuple[float, float, float]:
        """
        Return (omega0, Q, zeta) for GD with momentum on this loss.

        omega0 = √(lr · κ)        [rad / step]
        Q      = sqrt(lr·κ) / (1 − β)
        zeta   = (1 − β) / (2·sqrt(lr·κ))
        """
        lr, beta = params.lr, params.momentum
        kappa = self.loss.kappa
        omega0 = float(np.sqrt(max(lr * kappa, 1e-30)))
        damp = 1.0 - beta
        Q = omega0 / max(damp, 1e-12)
        zeta = damp / (2.0 * max(omega0, 1e-30))
        return omega0, Q, float(zeta)

    def infer_params_from_dynamical(
        self,
        omega0: float,
        Q: float,
    ) -> GDParams:
        """
        Recover (lr, momentum) from target (ω₀, Q) for this loss's κ.

        lr       = ω₀² / κ
        momentum = 1 − ω₀/Q   (clamped to [0, 1))
        """
        kappa = self.loss.kappa
        lr = max(omega0 ** 2 / max(kappa, 1e-30), 1e-30)
        beta = float(np.clip(1.0 - omega0 / max(Q, 1e-12), 0.0, 1.0 - 1e-6))
        return GDParams(lr=lr, momentum=beta)

    # ── Constraints ───────────────────────────────────────────────────────────

    def _stability_bound(self, params: GDParams) -> float:
        """ω₀² upper bound for stability: 2(1+β) × margin.

        Full stability condition: lr·κ < 2(1+β)  ↔  ω₀² < 2(1+β).
        """
        return 2.0 * (1.0 + params.momentum) * self._STABILITY_MARGIN

    def constraints(self, params: GDParams) -> Dict[str, Tuple[bool, float, float]]:
        omega0, Q, _ = self.dynamical_quantities(params)
        bound = self._stability_bound(params)
        return {
            "stability":   (omega0 ** 2 < bound,    omega0 ** 2, bound),
            "momentum_lo": (params.momentum >= 0.0,  params.momentum, 0.0),
            "momentum_hi": (params.momentum < 1.0,   params.momentum, 1.0),
            "lr_positive": (params.lr > 0,            params.lr, 0.0),
        }

    # ── Run GD and measure convergence ────────────────────────────────────────

    def _run_gd(self, params: GDParams, seed: int = 0) -> Tuple[int, float]:
        """
        Run GD with momentum on the quadratic loss.

        Returns (steps_to_converge, final_loss).
        steps_to_converge = max_steps if convergence not reached.
        """
        rng = np.random.default_rng(seed)
        x = self.loss.initial_x(rng, scale=1.0)
        v = np.zeros_like(x)
        lr, beta = params.lr, params.momentum

        for t in range(self.max_steps):
            g = self.loss.gradient(x)
            v = beta * v - lr * g
            x = x + v
            if self.loss.value(x) < self.tol:
                return t + 1, float(self.loss.value(x))

        return self.max_steps, float(self.loss.value(x))

    def freq_error_val(self, params: GDParams, target_omega0: float) -> float:
        """Fractional ω₀ error |ω₀ − ω₀_target| / ω₀_target."""
        omega0, _, _ = self.dynamical_quantities(params)
        return abs(omega0 - target_omega0) / max(abs(target_omega0), 1e-30)

    def Q_error_val(self, params: GDParams) -> float:
        """Fractional Q error |Q − Q_target| / Q_target (0 if Q_target is None)."""
        if self.Q_target is None:
            return 0.0
        _, Q, _ = self.dynamical_quantities(params)
        return abs(Q - self.Q_target) / max(abs(self.Q_target), 1e-12)

    def evaluate(
        self,
        params: GDParams,
        target_omega0: float,
        seed: int = 0,
    ) -> GDResult:
        """
        Full evaluation: run GD, measure convergence, compute objective.

        target_omega0: desired ω₀ [rad/step].  For multi-objective mode,
        freq_error = |ω₀_achieved − target_omega0| / target_omega0.
        For convergence-only mode (Q_target=None), objective = convergence_cost.
        """
        c_detail = self.constraints(params)
        ok = all(v[0] for v in c_detail.values())

        omega0, Q, zeta = self.dynamical_quantities(params)
        fe = self.freq_error_val(params, target_omega0)
        qe = self.Q_error_val(params)

        if ok:
            steps, final_loss = self._run_gd(params, seed=seed)
            # Convergence cost: fraction of budget consumed [0=instant, 1=failed]
            conv_cost = steps / max(self.max_steps, 1)
        else:
            steps = self.max_steps
            final_loss = float("inf")
            conv_cost = 1.0

        # Combined objective
        if self.Q_target is None:
            # Pure convergence objective: how many steps did we need?
            objective = conv_cost
        else:
            objective = self.w_freq * fe + self.w_Q * qe

        return GDResult(
            params=params,
            kappa=self.loss.kappa,
            convergence_rate=1.0 / max(steps, 1),  # faster = higher rate
            target_steps=self.max_steps,
            actual_steps=steps,
            final_loss=final_loss,
            objective=float(objective),
            Q_factor=float(Q),
            omega0=float(omega0),
            constraints_ok=ok,
            constraint_detail=c_detail,
            Q_target=self.Q_target,
            Q_error=float(qe),
            freq_error=float(fe),
        )


# ── Optimizer ─────────────────────────────────────────────────────────────────

# GD ω₀ is in [rad/step], not [rad/s].  Use 1 rad/step as reference so that
# log_omega0_norm = 0 at ω₀=1 rad/step (a natural GD scale), rather than
# at 2π×1kHz (a physical-system scale that would push all GD values off-chart).
_GD_LOG_OMEGA0_REF: float = 0.0   # log(1 rad/step) = 0

_TRACE_OPS: List[str] = [
    "gd_eval", "constraint_check", "convergence_eval", "objective_eval", "accepted",
]
_WINDOW_SIZE: int = 20
_MIN_TRACE_FOR_KOOPMAN: int = 10


class GDOptimizer:
    """
    Gradient-free search over (lr, momentum) hyperparameter space.

    Stores OptimizationExperience with domain="gradient_descent" so that
    the shared KoopmanExperienceMemory can warm-start cross-domain from
    physics experiments (RLC / spring-mass) with matching (ω₀, Q).

    Interface mirrors SpringMassOptimizer / ConstrainedHDVOptimizer.
    """

    def __init__(
        self,
        mapper: GDDesignMapper,
        evaluator: GDEvaluator,
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

    def optimize(self, target_omega0: float) -> GDResult:
        """
        Search for (lr, momentum) that achieves ω₀ ≈ target_omega0.

        In convergence-only mode (Q_target=None): minimises convergence steps.
        In multi-objective mode: minimises w_freq*freq_err + w_Q*Q_err.

        Stores OptimizationExperience with domain="gradient_descent".
        """
        z = self._rng.standard_normal(self.mapper.hdv_dim)
        event_buffer: deque = deque(maxlen=_WINDOW_SIZE * 20)
        param_trace: List[np.ndarray] = []

        best_result = self._eval_and_log(z, target_omega0, event_buffer, force_log=True)
        if best_result.constraints_ok:
            self._log_param_step(best_result.params, param_trace)

        step = self.step_init

        for _ in range(self.n_iter):
            z_cand = z + self._rng.standard_normal(self.mapper.hdv_dim) * step
            result = self._eval_and_log(z_cand, target_omega0, event_buffer)

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
        target_omega0: float,
        event_buffer: deque,
        force_log: bool = False,
    ) -> GDResult:
        params = self.mapper.decode(z)
        result = self.evaluator.evaluate(params, target_omega0, seed=int(self._rng.integers(2**16)))
        event_buffer.extend(["gd_eval", "constraint_check"])
        if result.constraints_ok or force_log:
            event_buffer.extend(["convergence_eval", "objective_eval"])
        return result

    @staticmethod
    def _log_param_step(params: GDParams, trace: List[np.ndarray]) -> None:
        trace.append(np.array([
            np.log(max(params.lr, 1e-30)),
            np.log(max(1.0 - params.momentum, 1e-30)),  # log(1−β) = log(damping)
        ]))

    def _dynamical_invariants(
        self, best_result: GDResult
    ) -> Tuple[float, float, float]:
        omega0, Q, zeta = (best_result.omega0, best_result.Q_factor,
                           1.0 / (2.0 * max(best_result.Q_factor, 1e-12)))
        # Use GD-specific reference (1 rad/step) — NOT the physical 2π×1kHz ref.
        log_omega0_norm = float(np.clip(
            (math.log(max(omega0, 1e-30)) - _GD_LOG_OMEGA0_REF) / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        log_Q_norm = float(np.clip(
            math.log(max(Q, 1e-30)) / _LOG_OMEGA0_SCALE,
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
            return edmd.eigendecomposition()
        except Exception:
            return None

    def _store_experience(
        self,
        event_buffer: deque,
        param_trace: List[np.ndarray],
        best_result: GDResult,
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
                _TRACE_OPS[dom_idx] if dom_idx < len(_TRACE_OPS) else "unknown"
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
            domain="gradient_descent",
        )
        self.memory.add(invariant, koop_result, experience)


# ── Analytic helpers (for testing / warm-start generation) ────────────────────


def optimal_gd_params(kappa: float, target_Q: float = 1.0) -> GDParams:
    """
    Analytically compute optimal (lr, momentum) for a quadratic with given κ.

    Uses a conservative ω₀ = 0.5 [rad/step], always inside the stability
    boundary ω₀² < 2(1+β)·0.95 ≥ 1.9 for any β ≥ 0 (since 0.25 < 1.9).

        lr       = ω₀² / κ  =  0.25 / κ
        momentum = clip(1 − ω₀/Q, 0, 1−ε)
    """
    omega0 = 0.5   # ω₀² = 0.25 < 2(1+β)·0.95 for all β ≥ 0 → always stable
    lr = omega0 ** 2 / max(kappa, 1e-30)
    beta = float(np.clip(1.0 - omega0 / max(target_Q, 1e-12), 0.0, 1.0 - 1e-6))
    return GDParams(lr=lr, momentum=beta)
