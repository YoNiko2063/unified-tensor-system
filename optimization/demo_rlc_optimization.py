"""
FICUTS v3.0 — RLC Filter Optimization Demo

Usage:
    python optimization/demo_rlc_optimization.py --target 1000
    python optimization/demo_rlc_optimization.py --target 5000 --iter 1000

Demonstrates:
    - HDV-parameterized constrained optimization
    - Koopman signature stored after run 1, warm-started for run 2
    - Physics artifact: JSON + frequency response plot
    - All existing invariants untouched (no EDMDKoopman.fit() in tensor/ paths)
"""

from __future__ import annotations

import argparse
import time

from optimization.hdv_optimizer import ConstrainedHDVOptimizer, export_design
from optimization.koopman_memory import KoopmanExperienceMemory
from optimization.rlc_evaluator import RLCEvaluator
from optimization.rlc_parameterization import RLCDesignMapper

SEP = "─" * 60


def run(target_hz: float, n_iter: int = 500, hdv_dim: int = 64) -> None:
    print(SEP)
    print(f"  RLC Filter Optimization  |  target = {target_hz:.1f} Hz")
    print(SEP)

    mapper    = RLCDesignMapper(hdv_dim=hdv_dim)
    evaluator = RLCEvaluator(max_Q=10.0, max_energy_loss=0.5)
    memory    = KoopmanExperienceMemory()
    optimizer = ConstrainedHDVOptimizer(
        mapper, evaluator, memory, n_iter=n_iter, seed=0
    )

    # ── Run 1: cold start ────────────────────────────────────────────────────
    print("\n  [Run 1]  cold start (no prior experience)")
    t0 = time.perf_counter()
    result1 = optimizer.optimize(target_hz)
    dt1 = time.perf_counter() - t0

    _print_result(result1, dt1)
    print(f"\n  Memory after run 1:  {memory.summary()}")

    # ── Run 2: warm start from memory ────────────────────────────────────────
    print(f"\n  [Run 2]  warm start from memory")
    optimizer2 = ConstrainedHDVOptimizer(
        mapper, evaluator, memory, n_iter=n_iter, seed=1
    )
    t0 = time.perf_counter()
    result2 = optimizer2.optimize(target_hz)
    dt2 = time.perf_counter() - t0

    _print_result(result2, dt2)
    print(f"\n  Memory after run 2:  {memory.summary()}")

    # ── Compare ──────────────────────────────────────────────────────────────
    best = result1 if result1.objective <= result2.objective else result2
    print(f"\n{SEP}")
    print("  BEST DESIGN")
    print(SEP)
    print(f"  {best.params}")
    print(f"  cutoff frequency:  {best.cutoff_hz:.4f} Hz")
    print(f"  target frequency:  {best.target_hz:.4f} Hz")
    print(f"  fractional error:  {best.objective:.6f}")
    print(f"  Q factor:          {best.Q_factor:.4f}")
    print(f"  energy loss:       {best.energy_loss:.4f}")
    print(f"  constraints OK:    {best.constraints_ok}")

    # ── Constraint breakdown ──────────────────────────────────────────────────
    print(f"\n  Constraint detail:")
    for name, (ok, val, lim) in best.constraint_detail.items():
        tick = "✓" if ok else "✗"
        print(f"    {tick}  {name:<18} value={val:.4g}  limit={lim:.4g}")

    # ── Export artifact ───────────────────────────────────────────────────────
    print(f"\n  Exporting artifact...")
    export_design(best, evaluator, path="optimization/rlc_design")

    print(f"\n{SEP}")
    speedup = dt1 / max(dt2, 1e-9)
    print(f"  Run 1: {dt1*1e3:.0f} ms  |  Run 2: {dt2*1e3:.0f} ms  "
          f"|  speedup: {speedup:.2f}×  (warm start)")
    print(SEP)


def _print_result(result, elapsed_s: float) -> None:
    ok = "PASS" if result.constraints_ok else "FAIL"
    print(f"    {result.params}")
    print(f"    cutoff = {result.cutoff_hz:.2f} Hz  "
          f"err = {result.objective:.5f}  Q = {result.Q_factor:.3f}  "
          f"constraints = {ok}  ({elapsed_s*1e3:.0f} ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FICUTS RLC filter optimization demo"
    )
    parser.add_argument(
        "--target", type=float, default=1000.0,
        help="Target cutoff frequency in Hz (default: 1000)",
    )
    parser.add_argument(
        "--iter", type=int, default=500,
        help="Maximum search iterations per run (default: 500)",
    )
    parser.add_argument(
        "--hdv-dim", type=int, default=64,
        help="HDV search space dimensionality (default: 64)",
    )
    args = parser.parse_args()
    run(target_hz=args.target, n_iter=args.iter, hdv_dim=args.hdv_dim)
