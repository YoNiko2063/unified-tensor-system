"""
ConstrainedHDVOptimizer — constrained search in HDV space with Koopman memory.

Algorithm per call to optimize():
  1. Warm-start from KoopmanExperienceMemory if any prior experience exists
     (encodes stored best_params back to HDV z via RLCDesignMapper.encode())
  2. Adaptive random walk in HDV space:
       - Sample perturbation Δz ~ N(0, step²·I)
       - Accept if constraints satisfied AND objective improves
       - Expand step on acceptance, shrink on rejection
       - Terminate early when objective < tol
  3. Collect parameter trace [log R_t, log L_t, log C_t] at accepted steps
  4. Fit EDMDKoopman to trace (sliding-window activation rates for operator trace;
     log-parameter trajectory for parameter trace)
  5. Compute KoopmanInvariantDescriptor and store OptimizationExperience in memory

Operator trace (for Koopman):
  The sequence of evaluator calls is logged as a sliding-window activation rate
  over operator types: ["cutoff_eval", "Q_eval", "constraint_check",
  "objective_eval", "accepted"].
  Rates (not cumulative counts) give genuine dynamics where Koopman analysis
  has meaningful eigenvalue structure.

No coupling to: HarmonicClosureChecker, ValidationBridge, GeometryMonitor,
growth schedulers, or semantic layers.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from tensor.koopman_edmd import EDMDKoopman
from optimization.koopman_signature import (
    FullKoopmanSignature,
    KoopmanInvariantDescriptor,
    compute_invariants,
)
from optimization.koopman_memory import (
    KoopmanExperienceMemory,
    OptimizationExperience,
    _MemoryEntry,
)
from optimization.rlc_parameterization import RLCDesignMapper, RLCParams
from optimization.rlc_evaluator import EvaluationResult, RLCEvaluator


# Operator types logged during the optimization loop
_TRACE_OPS: List[str] = [
    "cutoff_eval",
    "Q_eval",
    "constraint_check",
    "objective_eval",
    "accepted",
]
_WINDOW_SIZE: int = 20          # sliding window for activation rate
_MIN_TRACE_FOR_KOOPMAN: int = 10  # minimum accepted steps before Koopman fit


class ConstrainedHDVOptimizer:
    """
    Gradient-free constrained search in HDV space.

    Args:
        mapper:     RLCDesignMapper  (HDV → R, L, C)
        evaluator:  RLCEvaluator     (physics evaluation + constraints)
        memory:     KoopmanExperienceMemory  (warm start + experience storage)
        n_iter:     maximum search iterations  (default 500)
        seed:       RNG seed for reproducibility
        tol:        fractional cutoff error below which search terminates early
        step_init:  initial HDV perturbation standard deviation
    """

    def __init__(
        self,
        mapper: RLCDesignMapper,
        evaluator: RLCEvaluator,
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

    # ── Public API ─────────────────────────────────────────────────────────────

    def optimize(self, target_hz: float, pilot_steps: int = 40) -> EvaluationResult:
        """
        Search for an RLC design whose cutoff frequency matches target_hz.

        Returns the best EvaluationResult found, with constraints satisfied.
        Also stores a new OptimizationExperience in memory for future warm starts.

        Args:
            pilot_steps: steps in the exploratory pilot used to compute a
                         Koopman fingerprint for memory retrieval.  Set to 0
                         to skip the pilot (always cold-starts from memory).
        """
        # 1. Invariant-based warm start — no frequency formula, purely spectral
        z = self._invariant_warm_start(target_hz, pilot_steps)
        if z is None:
            z = self._rng.standard_normal(self.mapper.hdv_dim)

        # 2. Initialise trace buffers
        event_buffer: deque = deque(maxlen=_WINDOW_SIZE * 20)
        param_trace: List[np.ndarray] = []   # [log_R, log_L, log_C] per accepted step

        # 3. Evaluate initial point
        best_result = self._eval_and_log(z, target_hz, event_buffer, force_log=True)
        best_z = z.copy()
        if best_result.constraints_ok:
            self._log_param_step(best_result.params, param_trace)

        step = self.step_init

        # 4. Adaptive random walk
        for _ in range(self.n_iter):
            z_cand = z + self._rng.standard_normal(self.mapper.hdv_dim) * step
            result = self._eval_and_log(z_cand, target_hz, event_buffer)

            if result.constraints_ok and result.objective < best_result.objective:
                best_result = result
                best_z = z_cand.copy()
                z = z_cand
                step = min(step * 1.1, 3.0)
                self._log_param_step(result.params, param_trace)
                event_buffer.append("accepted")
            else:
                step = max(step * 0.95, 1e-3)

            if best_result.objective < self.tol:
                break

        # 5. Fit Koopman to traces and store experience
        self._store_experience(event_buffer, param_trace, best_result)

        return best_result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _eval_and_log(
        self,
        z: np.ndarray,
        target_hz: float,
        event_buffer: deque,
        force_log: bool = False,
    ) -> EvaluationResult:
        params = self.mapper.decode(z)
        result = self.evaluator.evaluate(params, target_hz)
        event_buffer.extend(["cutoff_eval", "Q_eval", "constraint_check"])
        if result.constraints_ok or force_log:
            event_buffer.append("objective_eval")
        return result

    @staticmethod
    def _log_param_step(params: RLCParams, trace: List[np.ndarray]) -> None:
        trace.append(np.array([
            np.log(max(params.R, 1e-30)),
            np.log(max(params.L, 1e-30)),
            np.log(max(params.C, 1e-30)),
        ]))

    def _invariant_warm_start(
        self, target_hz: float, pilot_steps: int
    ) -> Optional[np.ndarray]:
        """
        Run a short pilot to compute a Koopman fingerprint for this problem,
        then retrieve the nearest stored experience by invariant distance.

        Purely spectral retrieval — no frequency formula heuristic.
        The pilot uses an independent RNG so it does not consume the main
        optimizer's RNG state.

        Returns an encoded HDV z ready to use as the main-loop start,
        or None if memory is empty or Koopman fit fails.
        """
        if not self.memory._entries or pilot_steps <= 0:
            return None

        # Independent RNG so the pilot does not shift the main-loop RNG stream
        pilot_rng = np.random.default_rng(self._rng.integers(2 ** 32))
        pilot_buffer: deque = deque(maxlen=_WINDOW_SIZE * 20)
        pilot_params: List[np.ndarray] = []

        z_p = pilot_rng.standard_normal(self.mapper.hdv_dim)
        step = self.step_init
        best = self._eval_and_log(z_p, target_hz, pilot_buffer, force_log=True)
        if best.constraints_ok:
            self._log_param_step(best.params, pilot_params)

        for _ in range(pilot_steps):
            z_cand = z_p + pilot_rng.standard_normal(self.mapper.hdv_dim) * step
            r = self._eval_and_log(z_cand, target_hz, pilot_buffer)
            if r.constraints_ok and r.objective < best.objective:
                best = r
                z_p = z_cand
                step = min(step * 1.1, 3.0)
                self._log_param_step(r.params, pilot_params)
                pilot_buffer.append("accepted")
            else:
                step = max(step * 0.95, 1e-3)

        # Fit Koopman to pilot traces (operator trace preferred, param fallback)
        op_trace = self._build_operator_trace(pilot_buffer)
        fit = self._fit_koopman(op_trace)
        if fit is None and len(pilot_params) >= _MIN_TRACE_FOR_KOOPMAN:
            fit = self._fit_koopman(np.array(pilot_params))
        if fit is None:
            return None

        _, query_invariant = fit

        # Retrieve nearest stored experience by invariant L2 distance
        candidates = self.memory.retrieve_candidates(query_invariant, top_n=1)
        if not candidates:
            return None

        bp = candidates[0].experience.best_params
        try:
            return self.mapper.encode(RLCParams(**bp))
        except Exception:
            return None

    def _build_operator_trace(self, event_buffer: deque) -> Optional[np.ndarray]:
        """
        Convert the event buffer into a sliding-window activation rate matrix.

        Each row x_t is the fraction of each operator type fired in the last
        _WINDOW_SIZE events — a genuine rate signal suitable for Koopman EDMD.
        """
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

    def _fit_koopman(
        self, trace: np.ndarray
    ) -> Optional[Tuple[FullKoopmanSignature, KoopmanInvariantDescriptor]]:
        """Fit EDMDKoopman to a trace matrix (T, d) and return full + invariant."""
        if trace is None or len(trace) < _MIN_TRACE_FOR_KOOPMAN:
            return None
        pairs = [(trace[i], trace[i + 1]) for i in range(len(trace) - 1)]
        edmd = EDMDKoopman(observable_degree=1)
        try:
            edmd.fit(pairs)
            result = edmd.eigendecomposition()
        except Exception:
            return None
        invariant = compute_invariants(
            result.eigenvalues, result.eigenvectors, _TRACE_OPS
        )
        return result, invariant

    def _param_centroid(self, param_trace: List[np.ndarray]) -> np.ndarray:
        """
        Normalized mean of accepted log-parameter steps.

        Returns a 3-vector in the same space as the mapper's log-ratio
        coordinates: (log_R - log_R_center) / max_exp, similarly for L, C.
        Values lie in [-1, 1] and encode WHERE in parameter space the
        optimization converged, anchoring the descriptor physically.
        """
        if not param_trace:
            return np.zeros(3)
        mean_log = np.mean(np.array(param_trace), axis=0)   # [log_R, log_L, log_C]
        log_centers = np.array([
            np.log(max(self.mapper.R_center, 1e-30)),
            np.log(max(self.mapper.L_center, 1e-30)),
            np.log(max(self.mapper.C_center, 1e-30)),
        ])
        return np.clip(
            (mean_log - log_centers) / max(self.mapper._max_exp, 1e-9),
            -1.0, 1.0,
        )

    def _store_experience(
        self,
        event_buffer: deque,
        param_trace: List[np.ndarray],
        best_result: EvaluationResult,
    ) -> None:
        """
        Fit Koopman to operator trace and store experience in memory.

        Uses operator trace (event buffer) for Koopman signature.
        Falls back to parameter trace if operator trace is too short.
        Attaches param_centroid to the invariant to anchor it physically.
        """
        op_trace = self._build_operator_trace(event_buffer)
        fit = self._fit_koopman(op_trace)

        if fit is None and len(param_trace) >= _MIN_TRACE_FOR_KOOPMAN:
            p_arr = np.array(param_trace)
            fit = self._fit_koopman(p_arr)

        if fit is None:
            return  # not enough data — skip storage

        # Compute physical-location anchor and attach to invariant
        centroid = self._param_centroid(param_trace)
        signature, raw_inv = fit
        invariant = compute_invariants(
            signature.eigenvalues, signature.eigenvectors, _TRACE_OPS,
            param_centroid=centroid,
        )

        # Identify dominant operator from invariant histogram
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
        )
        self.memory.add(invariant, signature, experience)


# ── Artifact export ────────────────────────────────────────────────────────────

def export_design(
    result: EvaluationResult,
    evaluator: RLCEvaluator,
    path: str = "optimization/rlc_design",
) -> None:
    """
    Save optimized design as JSON + frequency response plot.

    JSON: {R, L, C, cutoff_hz, target_hz, objective, Q_factor, energy_loss,
           constraints_ok, constraint_detail}
    Plot: magnitude response |H(jf)| from 0.01×f₀ to 100×f₀ (saved as PNG).
    """
    import json
    from pathlib import Path

    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    c_detail_serializable = {
        k: {"satisfied": bool(v[0]), "value": float(v[1]), "limit": float(v[2])}
        for k, v in result.constraint_detail.items()
    }
    payload = {
        "R_ohm":          float(result.params.R),
        "L_henry":        float(result.params.L),
        "C_farad":        float(result.params.C),
        "cutoff_hz":      float(result.cutoff_hz),
        "target_hz":      float(result.target_hz),
        "objective":      float(result.objective),
        "Q_factor":       float(result.Q_factor),
        "energy_loss":    float(result.energy_loss),
        "constraints_ok": bool(result.constraints_ok),
        "constraints":    c_detail_serializable,
    }
    json_path = Path(str(base) + ".json")
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  [artifact] saved → {json_path}")

    # Plot (optional — skips gracefully if matplotlib unavailable)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f0 = result.cutoff_hz
        freqs = np.logspace(
            np.log10(max(f0 * 0.01, 1.0)),
            np.log10(f0 * 100),
            500,
        )
        H = evaluator.frequency_response(result.params, freqs)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogx(freqs, 20 * np.log10(np.maximum(H, 1e-10)), color="steelblue")
        ax.axvline(f0, color="tomato", linestyle="--", label=f"f₀ = {f0:.1f} Hz")
        ax.axvline(
            result.target_hz, color="green", linestyle=":",
            label=f"target = {result.target_hz:.1f} Hz",
        )
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|H(jf)| [dB]")
        ax.set_title(f"RLC Filter — Optimized Design  (Q={result.Q_factor:.2f})")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        plot_path = Path(str(base) + "_response.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  [artifact] saved → {plot_path}")

    except ImportError:
        print("  [artifact] matplotlib not available — plot skipped")
