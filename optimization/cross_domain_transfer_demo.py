"""
Cross-domain transfer demo: spring-mass → RLC.

Demonstrates that KoopmanExperienceMemory with domain-invariant (ω₀, Q, ζ)
descriptors enables warm-starting an RLC optimiser from spring-mass priors.

Run:
    python -m optimization.cross_domain_transfer_demo
"""

from __future__ import annotations

import math
import time

import numpy as np

from optimization.koopman_memory import KoopmanExperienceMemory
from optimization.rlc_evaluator import RLCEvaluator
from optimization.rlc_parameterization import RLCDesignMapper
from optimization.hdv_optimizer import ConstrainedHDVOptimizer
from optimization.spring_mass_system import (
    SpringMassDesignMapper,
    SpringMassEvaluator,
    SpringMassOptimizer,
)


def _fmt_hz(hz: float) -> str:
    return f"{hz:,.1f} Hz"


def _separator(char: str = "─", width: int = 60) -> str:
    return char * width


def run_demo() -> None:
    print(_separator("═"))
    print("  Cross-Domain Transfer Demo: Spring-Mass → RLC")
    print(_separator("═"))

    # ── Phase A: Train spring-mass memory ────────────────────────────────────
    print("\n[A] Training spring-mass memory on [500, 1000, 1500] Hz …")
    sm_mapper    = SpringMassDesignMapper(hdv_dim=64, seed=7)
    sm_evaluator = SpringMassEvaluator(max_Q=10.0, max_energy_loss=0.5)
    shared_memory = KoopmanExperienceMemory()

    sm_targets = [500.0, 1000.0, 1500.0]
    t0 = time.perf_counter()
    for f in sm_targets:
        opt = SpringMassOptimizer(sm_mapper, sm_evaluator, shared_memory,
                                  n_iter=500, seed=0)
        result = opt.optimize(f)
        tag = "✓" if result.objective < 0.05 else "✗"
        print(f"    {tag} {_fmt_hz(f)} → {_fmt_hz(result.natural_freq_hz)}  "
              f"err={result.objective:.4f}  Q={result.Q_factor:.2f}")

    print(f"  Spring-mass memory: {shared_memory.summary()}")
    print(f"  Entries: {len(shared_memory)}")

    # Print stored invariant dynamical quantities
    print("\n  Stored invariant (ω₀, Q, ζ):")
    for e in shared_memory._entries:
        inv = e.invariant
        omega0 = inv.dynamical_omega0()
        Q_val  = inv.dynamical_Q()
        f0 = omega0 / (2.0 * math.pi)
        print(f"    domain={e.experience.domain}  "
              f"f₀={_fmt_hz(f0)}  Q={Q_val:.2f}  ζ={inv.damping_ratio:.3f}  "
              f"log_ω₀_norm={inv.log_omega0_norm:.4f}")

    # ── Phase B: RLC cold vs warm ─────────────────────────────────────────────
    print(f"\n[B] RLC optimisation at 750 Hz (cold vs warm from spring-mass) …")
    rlc_mapper    = RLCDesignMapper(hdv_dim=64, seed=42)
    rlc_evaluator = RLCEvaluator(max_Q=10.0, max_energy_loss=0.5)
    test_target   = 750.0

    # Cold
    cold_mem = KoopmanExperienceMemory()
    cold_opt = ConstrainedHDVOptimizer(rlc_mapper, rlc_evaluator, cold_mem,
                                       n_iter=200, seed=42)
    t_cold = time.perf_counter()
    cold_result = cold_opt.optimize(test_target, pilot_steps=0)
    dt_cold = time.perf_counter() - t_cold

    # Warm (using spring-mass prior)
    warm_opt = ConstrainedHDVOptimizer(rlc_mapper, rlc_evaluator, shared_memory,
                                       n_iter=200, seed=42)
    t_warm = time.perf_counter()
    warm_result = warm_opt.optimize(test_target, pilot_steps=40)
    dt_warm = time.perf_counter() - t_warm

    print(f"\n  {'':>30}  {'Cold':>10}  {'Warm (SM→RLC)':>14}")
    print(f"  {'Cutoff [Hz]':>30}  {cold_result.cutoff_hz:>10.2f}  {warm_result.cutoff_hz:>14.2f}")
    print(f"  {'Objective (err)':>30}  {cold_result.objective:>10.6f}  {warm_result.objective:>14.6f}")
    print(f"  {'Q factor':>30}  {cold_result.Q_factor:>10.3f}  {warm_result.Q_factor:>14.3f}")
    print(f"  {'Constraints OK':>30}  {'yes' if cold_result.constraints_ok else 'no':>10}  "
          f"{'yes' if warm_result.constraints_ok else 'no':>14}")
    print(f"  {'Wall time [s]':>30}  {dt_cold:>10.3f}  {dt_warm:>14.3f}")

    ratio = cold_result.objective / max(warm_result.objective, 1e-9)
    print(f"\n  Improvement ratio: {ratio:.1f}×  "
          f"({'warm beats cold' if warm_result.objective < cold_result.objective else 'cold comparable'})")

    # Show what was retrieved
    print(f"\n[C] What did the warm start retrieve?")
    entries = shared_memory._entries
    if entries:
        # Simulate retrieval from the demo warm start
        from optimization.koopman_signature import compute_invariants, _LOG_OMEGA0_REF, _LOG_OMEGA0_SCALE
        # Infer pilot retrieval: for 750 Hz, pilot best near 750 Hz
        omega0_q = 2.0 * math.pi * test_target
        log_norm = (math.log(omega0_q) - _LOG_OMEGA0_REF) / _LOG_OMEGA0_SCALE
        Q_q = 1.0   # neutral Q
        zeta_q = 0.5
        query = compute_invariants(np.array([]), np.zeros((0, 0)), [],
                                   log_omega0_norm=log_norm,
                                   log_Q_norm=0.0, damping_ratio=zeta_q)
        candidates = shared_memory.retrieve_candidates(query, top_n=1)
        if candidates:
            c = candidates[0]
            c_f0 = c.invariant.dynamical_omega0() / (2.0 * math.pi)
            print(f"  Nearest spring-mass entry: f₀≈{_fmt_hz(c_f0)}  "
                  f"domain={c.experience.domain}  "
                  f"orig_params={c.experience.best_params}")
            # Show inferred RLC params
            omega0_s = c.invariant.dynamical_omega0()
            Q_s = c.invariant.dynamical_Q()
            inferred = rlc_evaluator.infer_params_from_dynamical(omega0_s, Q_s)
            f_inferred = rlc_evaluator.cutoff_frequency_hz(inferred)
            print(f"  Inferred RLC params: {inferred}  → f₀={_fmt_hz(f_inferred)}")

    print(f"\n{_separator('═')}")
    print("  Demo complete.")
    print(_separator("═"))


if __name__ == "__main__":
    run_demo()
