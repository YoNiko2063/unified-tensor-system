"""
IEEE 39-Bus CCT Benchmark — Demonstration Script

Runs the regime-aware EAC spectral method against the RK4 reference
on all 10 generators of the IEEE 39-bus New England system.

Usage:
    python examples/ieee39_demo.py

Output:
    Per-generator CCT comparison, speedup factor, and aggregate summary.
"""

from __future__ import annotations

import os
import sys
import time

# Locate project root (one level above examples/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from optimization.ieee39_benchmark import (
    compute_summary,
    print_ieee39_table,
    run_ieee39_benchmark,
)


def main() -> None:
    print()
    print("Unified Tensor Systems — IEEE 39-Bus CCT Benchmark")
    print("=" * 60)
    print("Method A (Reference):  RK4 binary-search (dt=0.01s, tol=1e-3)")
    print("Method B (Fast):       Regime-aware EAC (analytic, spectral)")
    print()
    print("Running benchmark for all 10 generators...")
    print("(RK4 reference runs take ~30–60 s total)")
    print()

    t_start = time.perf_counter()
    results = run_ieee39_benchmark(ref_tol=1e-3, ref_dt=0.01)
    elapsed = time.perf_counter() - t_start

    print_ieee39_table(results)

    summary = compute_summary(results)
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mean CCT error :  {summary['mean_error_pct']:.2f}%")
    print(f"  Max  CCT error :  {summary['max_error_pct']:.2f}%")
    print(f"  Mean speedup   :  {summary['mean_speedup']:,.0f}×")
    print(f"  Total wall time:  {elapsed:.1f}s (dominated by RK4 reference)")
    print()
    print("Validated on:  IEEE 39-bus New England System, 10 generators")
    print("Patent status: Provisional USPTO filing")
    print()


if __name__ == "__main__":
    main()
