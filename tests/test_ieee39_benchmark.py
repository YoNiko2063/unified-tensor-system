"""
IEEE 39-Bus CCT Benchmark — Test Suite.

Proves:
 Group 1 — Generator data integrity (3 tests)
   1.  All 10 generators produce valid PowerGridParams (no ValueError)
   2.  All generators have δ_s ≈ 30° (sin(δ_s) ≈ 0.5, within 0.01)
   3.  All generators have E_sep > 0

 Group 2 — EAC formula correctness (3 tests)
   4.  eac_cct() returns positive value for all generators
   5.  eac_cct() < physical upper bound √(2M(δ_u−δ_s)/P_m) for all generators
   6.  G1 (H=500, large inertia) has larger CCT than G8 (H=24.3, small inertia)

 Group 3 — Benchmark accuracy gate (2 tests)
   7.  Mean CCT error (EAC vs. reference) < 5%
   8.  Max CCT error < 10%

 Group 4 — Speedup gate (1 test)
   9.  Mean speedup > 50× (EAC is pure arithmetic vs. RK4 binary search)

 Group 5 — Anchor sentence (1 test, always passes, prints table)
  10.  test_ieee39_anchor_table: runs full benchmark, prints table, asserts gates

Total: 10 tests
"""

from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pytest

from optimization.ieee39_benchmark import (
    IEEE39_GENERATORS,
    BenchmarkResult,
    compute_summary,
    eac_cct,
    print_ieee39_table,
    run_ieee39_benchmark,
    to_power_grid_params,
)


# ── Module-level benchmark cache (avoid re-running expensive RK4 reference) ──

_RESULTS: list[BenchmarkResult] | None = None


def _get_results() -> list[BenchmarkResult]:
    """Run benchmark once; cache results for all accuracy/speedup tests."""
    global _RESULTS
    if _RESULTS is None:
        _RESULTS = run_ieee39_benchmark(ref_tol=1e-3, ref_dt=0.01)
    return _RESULTS


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — Generator data integrity
# ═══════════════════════════════════════════════════════════════════════════════


def test_all_generators_valid_params():
    """All 10 IEEE39 generators produce valid PowerGridParams (no ValueError)."""
    assert len(IEEE39_GENERATORS) == 10, (
        f"Expected 10 generators, got {len(IEEE39_GENERATORS)}"
    )
    for gen in IEEE39_GENERATORS:
        params = to_power_grid_params(gen)
        assert params.M > 0,  f"{gen.name}: M={params.M:.4f} not > 0"
        assert params.D >= 0, f"{gen.name}: D={params.D} not >= 0"
        assert params.P_e > 0, f"{gen.name}: P_e={params.P_e} not > 0"
        assert 0 < params.P_m < params.P_e, (
            f"{gen.name}: need 0 < P_m < P_e; got P_m={params.P_m}, P_e={params.P_e}"
        )
        print(f"  {gen.name}: M={params.M:.4f}  δ_s={math.degrees(params.delta_s):.2f}°")


def test_all_generators_delta_s_approx_30deg():
    """All generators have δ_s ≈ 30° (P_e = 2×P_m → sin(δ_s) = 0.5)."""
    target_sin = 0.5   # sin(30°)
    for gen in IEEE39_GENERATORS:
        params   = to_power_grid_params(gen)
        sin_ds   = math.sin(params.delta_s)
        err      = abs(sin_ds - target_sin)
        assert err < 0.01, (
            f"{gen.name}: sin(δ_s)={sin_ds:.4f}, expected ≈ {target_sin} "
            f"(error={err:.4f} > 0.01).  P_m={gen.P_m}, P_e={gen.P_e}"
        )


def test_all_generators_esep_positive():
    """All generators have separatrix energy E_sep > 0."""
    for gen in IEEE39_GENERATORS:
        params = to_power_grid_params(gen)
        E_sep  = params.separatrix_energy
        assert E_sep > 0, (
            f"{gen.name}: E_sep={E_sep:.4f} ≤ 0 (P_m={gen.P_m}, P_e={gen.P_e})"
        )
        print(f"  {gen.name}: E_sep={E_sep:.4f} pu")


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — EAC formula correctness
# ═══════════════════════════════════════════════════════════════════════════════


def test_eac_cct_positive_for_all():
    """eac_cct() returns a strictly positive value for all 10 generators."""
    for gen in IEEE39_GENERATORS:
        params = to_power_grid_params(gen)
        cct    = eac_cct(params)
        assert cct > 0.0, (
            f"{gen.name}: eac_cct={cct:.4f} s is not positive"
        )
        assert math.isfinite(cct), (
            f"{gen.name}: eac_cct={cct} is not finite"
        )
        print(f"  {gen.name}: CCT_EAC={cct:.4f} s")


def test_eac_cct_below_physical_upper_bound():
    """
    eac_cct() < √(2M·(δ_u−δ_s)/P_m) for all generators.

    Physical argument: the clearing angle δ_c is always strictly less than
    the unstable equilibrium δ_u (otherwise the decelerating area would be
    zero and the system could not return to synchronism).  Therefore:

        CCT_EAC = √(2M(δ_c−δ_s)/P_m) < √(2M(δ_u−δ_s)/P_m)
    """
    for gen in IEEE39_GENERATORS:
        params      = to_power_grid_params(gen)
        cct_eac     = eac_cct(params)
        upper_bound = math.sqrt(
            2.0 * params.M * (params.delta_u - params.delta_s) / params.P_m
        )
        assert cct_eac < upper_bound, (
            f"{gen.name}: CCT_EAC={cct_eac:.4f} ≥ upper_bound={upper_bound:.4f}"
        )
        print(f"  {gen.name}: CCT_EAC={cct_eac:.4f} s  upper={upper_bound:.4f} s")


def test_g1_larger_cct_than_g8():
    """
    G1 (H=500 s, large inertia) has a larger EAC CCT than G8 (H=24.3 s).

    Physical argument: CCT_EAC = √(2M(δ_c−δ_s)/P_m).  Since all generators
    share the same δ_s = 30° and the same P_e/P_m = 2 ratio, δ_c is identical.
    CCT scales as √(M/P_m) = √(2H/(ω_s·P_m)).  G1's huge inertia (H=500)
    dominates G8 (H=24.3) even accounting for P_m differences.
    """
    g1 = next(g for g in IEEE39_GENERATORS if g.name == "G1")
    g8 = next(g for g in IEEE39_GENERATORS if g.name == "G8")

    cct_g1 = eac_cct(to_power_grid_params(g1))
    cct_g8 = eac_cct(to_power_grid_params(g8))

    print(f"\n  G1 (H=500): CCT_EAC={cct_g1:.4f} s")
    print(f"  G8 (H=24.3): CCT_EAC={cct_g8:.4f} s")

    assert cct_g1 > cct_g8, (
        f"Expected G1 CCT ({cct_g1:.4f} s) > G8 CCT ({cct_g8:.4f} s)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3 — Benchmark accuracy gate
# ═══════════════════════════════════════════════════════════════════════════════


def test_mean_cct_error_under_5pct():
    """Mean CCT error (EAC vs. RK4 reference) < 5% across all 10 generators."""
    results = _get_results()
    summary = compute_summary(results)
    mean_err = summary["mean_error_pct"]

    print(f"\n  Mean CCT error: {mean_err:.2f}%")
    for r in results:
        print(f"    {r.gen_name}: EAC={r.cct_eac:.4f} s  Ref={r.cct_ref:.4f} s  "
              f"err={r.cct_error_pct:.2f}%")

    assert mean_err < 5.0, (
        f"Mean CCT error {mean_err:.2f}% exceeds 5% gate"
    )


def test_max_cct_error_under_10pct():
    """Max CCT error (worst generator) < 10%."""
    results  = _get_results()
    summary  = compute_summary(results)
    max_err  = summary["max_error_pct"]
    worst    = max(results, key=lambda r: r.cct_error_pct)

    print(f"\n  Max CCT error: {max_err:.2f}% (generator {worst.gen_name})")

    assert max_err < 10.0, (
        f"Max CCT error {max_err:.2f}% (gen {worst.gen_name}) exceeds 10% gate"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4 — Speedup gate
# ═══════════════════════════════════════════════════════════════════════════════


def test_mean_speedup_over_50x():
    """
    Mean EAC speedup > 50× over the RK4 binary-search reference.

    Conservative lower bound: EAC is O(1) arithmetic; reference requires
    ~13 binary-search iterations × 3000 settle-steps × 4 RK4 evaluations.
    In practice the measured speedup is typically 100–10 000×.
    """
    results      = _get_results()
    summary      = compute_summary(results)
    mean_speedup = summary["mean_speedup"]

    print(f"\n  Mean speedup: {mean_speedup:.0f}×")
    for r in results:
        print(f"    {r.gen_name}: t_Ref={r.t_ref_ms:.1f} ms  "
              f"t_EAC={r.t_eac_us:.2f} μs  speedup={r.speedup:.0f}×")

    assert mean_speedup > 50.0, (
        f"Mean speedup {mean_speedup:.1f}× is below 50× gate"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 5 — Anchor sentence (always passes, prints table)
# ═══════════════════════════════════════════════════════════════════════════════


def test_ieee39_anchor_table():
    """
    Run full benchmark, print formatted table, emit anchor sentence.

    This test always passes as long as the accuracy and speedup gates hold.
    Its primary purpose is to produce the publishable anchor sentence:

        "EAC method: X× faster CCT screening with Y% error on IEEE 39-bus."
    """
    results = _get_results()
    print("\n")
    print_ieee39_table(results)

    summary      = compute_summary(results)
    mean_err     = summary["mean_error_pct"]
    mean_speedup = summary["mean_speedup"]

    # Re-assert both gates so the anchor sentence is only printed on a true PASS.
    assert mean_err < 5.0, (
        f"Anchor FAIL: mean CCT error {mean_err:.2f}% ≥ 5%"
    )
    assert mean_speedup > 50.0, (
        f"Anchor FAIL: mean speedup {mean_speedup:.1f}× < 50×"
    )
