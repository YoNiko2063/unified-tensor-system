"""
IEEE 39-Bus New England System — CCT Benchmark.

Compares two methods for Critical Clearing Time estimation:

  Method A (Reference):  RK4 binary-search via estimate_cct()
                         ~13 iterations × 3000 settle-steps × 4 RK4 evals per generator
  Method B (Fast):       Equal-Area Criterion (EAC) — pure analytic formula, zero ODE calls

EAC derivation (three-phase fault, D=0, exact)
-----------------------------------------------
During fault:  M·δ̈ = P_m                         [P_e → 0]
Post-fault:    M·δ̈ = P_m − P_e·sin(δ)

Equal-area condition: accelerating area == decelerating area gives

    cos(δ_c) = P_m·(π − 2δ_s)/P_e  −  cos(δ_s)        [critical clearing angle]
    CCT_EAC  = √(2M·(δ_c − δ_s)/P_m)                   [critical clearing time, s]

This is analytically exact for undamped (D=0) three-phase fault (P_e→0 during fault).

Generator data source
---------------------
Anderson & Fouad (2003), "Power Systems Control and Stability", 2nd ed., Table 2.7.
Ten-generator equivalent of the New England 39-bus system.

H values [s] are the inertia constants; M = 2H/ω_s with ω_s = 2π×60 rad/s.
P_e = 2×P_m → sin(δ_s)=0.5 → δ_s = 30° for all generators (standard textbook loading).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List

from optimization.power_grid_evaluator import PowerGridParams, estimate_cct


# ── Module constants ──────────────────────────────────────────────────────────

_OMEGA_S: float = 2.0 * math.pi * 60.0     # synchronous speed 376.991 rad/s (60 Hz)


# ── Generator data ─────────────────────────────────────────────────────────────


class IEEE39Generator:
    """
    IEEE 39-bus generator record.

    Args:
        name: generator label (G1 … G10)
        bus:  bus number in 39-bus topology
        H:    inertia constant [s]        (M = 2H/ω_s)
        P_m:  mechanical input power [pu]
        P_e:  peak electrical power [pu]  (= 2 × P_m → δ_s = 30°)
        D:    damping coefficient [pu]    (0.0 for EAC benchmark)
    """

    __slots__ = ("name", "bus", "H", "P_m", "P_e", "D")

    def __init__(
        self,
        name: str,
        bus: int,
        H: float,
        P_m: float,
        P_e: float,
        D: float = 0.0,
    ) -> None:
        self.name = name
        self.bus  = bus
        self.H    = H
        self.P_m  = P_m
        self.P_e  = P_e
        self.D    = D

    def __repr__(self) -> str:
        return (
            f"IEEE39Generator({self.name!r}, bus={self.bus}, "
            f"H={self.H}, P_m={self.P_m}, P_e={self.P_e})"
        )


# Anderson & Fouad (2003) Table 2.7 — 10-generator New England equivalent.
# M = 2H/ω_s (computed in to_power_grid_params); P_e = 2×P_m → δ_s = 30°.
IEEE39_GENERATORS: List[IEEE39Generator] = [
    IEEE39Generator("G1",  30, 500.0,  2.50,  5.00),
    IEEE39Generator("G2",  31,  30.3,  5.73, 11.46),
    IEEE39Generator("G3",  32,  35.8,  6.50, 13.00),
    IEEE39Generator("G4",  33,  28.6,  6.32, 12.64),
    IEEE39Generator("G5",  34,  26.0,  5.08, 10.16),
    IEEE39Generator("G6",  35,  34.8,  6.50, 13.00),
    IEEE39Generator("G7",  36,  26.4,  5.60, 11.20),
    IEEE39Generator("G8",  37,  24.3,  5.40, 10.80),
    IEEE39Generator("G9",  38,  34.5,  8.30, 16.60),
    IEEE39Generator("G10", 39,  42.0, 10.04, 20.08),
]


# ── Converter ─────────────────────────────────────────────────────────────────


def to_power_grid_params(gen: IEEE39Generator) -> PowerGridParams:
    """
    Convert an IEEE39Generator record to a PowerGridParams instance.

    Inertia: M = 2H/ω_s  where ω_s = 2π×60 rad/s (376.991 rad/s).
    Damping: D = gen.D (0.0 for EAC benchmark).
    """
    M = 2.0 * gen.H / _OMEGA_S
    return PowerGridParams(M=M, D=gen.D, P_m=gen.P_m, P_e=gen.P_e)


# ── EAC analytic formula ──────────────────────────────────────────────────────


def eac_cct(params: PowerGridParams) -> float:
    """
    Analytic Equal-Area Criterion CCT for three-phase fault (P_e→0, D=0).

    Derivation (EAC for three-phase fault):
        Accelerating area  = P_m·(δ_c − δ_s)
        Decelerating area  = P_e·cos(δ_s) − P_e·cos(δ_c) − P_m·(δ_u_eff − δ_c)

    Setting accelerating = decelerating and solving for δ_c:

        cos(δ_c) = P_m·(π − 2δ_s)/P_e  −  cos(δ_s)

    Then from the swing equation during fault (ω from zero):

        CCT_EAC = √(2M·(δ_c − δ_s) / P_m)

    This result is exact for the undamped (D=0) classical model with complete
    power loss during the fault (three-phase, fault_factor=0).

    Args:
        params: PowerGridParams with D=0 assumed (D is not used by this formula)

    Returns:
        Critical clearing time in seconds [s].
    """
    ds  = params.delta_s
    cos_dc = (params.P_m * (math.pi - 2.0 * ds) / params.P_e) - math.cos(ds)
    cos_dc = max(-1.0, min(1.0, cos_dc))   # numerical safety clamp
    dc  = math.acos(cos_dc)
    M   = params.M
    Pm  = params.P_m
    return float(math.sqrt(max(0.0, 2.0 * M * (dc - ds) / Pm)))


# ── Benchmark result ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Per-generator result from run_ieee39_benchmark()."""

    gen_name:       str     # "G1" … "G10"
    bus:            int     # bus number
    H:              float   # inertia constant [s]
    omega0_linear:  float   # small-signal ω₀ = √(P_e·cos(δ_s)/M)  [rad/s]

    cct_eac:        float   # analytic EAC CCT  [s]
    cct_ref:        float   # RK4 binary-search CCT  [s]
    cct_error_pct:  float   # |cct_eac − cct_ref| / cct_ref × 100  [%]

    t_eac_us:       float   # EAC wall-clock time  [μs]
    t_ref_ms:       float   # reference wall-clock time  [ms]
    speedup:        float   # t_ref_ms / (t_eac_us / 1000)


# ── Benchmark runner ──────────────────────────────────────────────────────────


def run_ieee39_benchmark(
    ref_tol: float = 1e-3,
    ref_dt:  float = 0.01,
) -> List[BenchmarkResult]:
    """
    Run the IEEE 39-bus CCT benchmark for all 10 generators.

    For each generator:
      1. Build PowerGridParams from IEEE39Generator data.
      2. Time EAC formula (analytic, no ODE calls).
      3. Time reference binary-search estimate_cct (RK4, dt=ref_dt, tol=ref_tol).
      4. Compute CCT error and speedup.

    Args:
        ref_tol: binary-search convergence tolerance [s]  (default 1e-3)
        ref_dt:  RK4 integration timestep [s]             (default 0.01)

    Returns:
        List of BenchmarkResult, one per generator, in IEEE39_GENERATORS order.
    """
    results: List[BenchmarkResult] = []

    for gen in IEEE39_GENERATORS:
        params = to_power_grid_params(gen)
        omega0 = params.omega0_linear

        # ── EAC (fast) ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        cct_eac = eac_cct(params)
        t_eac_us = (time.perf_counter() - t0) * 1.0e6

        # ── Reference (RK4 binary search) ─────────────────────────────────────
        t0 = time.perf_counter()
        cct_result = estimate_cct(
            params,
            fault_duration_range=(0.0, 5.0),
            fault_factor=0.0,
            dt=ref_dt,
            tol=ref_tol,
        )
        t_ref_ms = (time.perf_counter() - t0) * 1.0e3

        cct_ref       = cct_result.cct
        cct_error_pct = abs(cct_eac - cct_ref) / max(cct_ref, 1e-12) * 100.0
        speedup       = t_ref_ms / max(t_eac_us / 1000.0, 1e-9)

        results.append(BenchmarkResult(
            gen_name      = gen.name,
            bus           = gen.bus,
            H             = gen.H,
            omega0_linear = omega0,
            cct_eac       = cct_eac,
            cct_ref       = cct_ref,
            cct_error_pct = cct_error_pct,
            t_eac_us      = t_eac_us,
            t_ref_ms      = t_ref_ms,
            speedup       = speedup,
        ))

    return results


# ── Summary stats ─────────────────────────────────────────────────────────────


def compute_summary(results: List[BenchmarkResult]) -> dict:
    """
    Compute aggregate statistics across all generators.

    Returns:
        dict with keys:
          mean_error_pct  — mean CCT error [%]
          mean_speedup    — mean speedup factor
          max_error_pct   — maximum CCT error [%]
    """
    errors   = [r.cct_error_pct for r in results]
    speedups = [r.speedup        for r in results]
    n        = len(results)
    return {
        "mean_error_pct": sum(errors)   / n,
        "mean_speedup":   sum(speedups) / n,
        "max_error_pct":  max(errors),
    }


# ── Table printer ─────────────────────────────────────────────────────────────


def print_ieee39_table(results: List[BenchmarkResult]) -> None:
    """
    Print formatted benchmark table and anchor sentence to stdout.

    Example output::

        ================================================================================
          IEEE 39-Bus New England System — CCT Benchmark
          Method A (Ref):  RK4 binary-search  (dt=0.01, tol=1e-3)
          Method B (Fast): Equal-Area Criterion (analytic, D=0, three-phase fault)
        ================================================================================
          Gen   Bus    H[s]  ω₀[rad/s]  CCT_EAC[s]  CCT_Ref[s]   Err%  t_Ref[ms] t_EAC[μs]    Speedup
          ...
        ================================================================================
          SUMMARY: mean CCT error = X.X%   mean speedup = XXXXXX×
          ANCHOR: "EAC method: X× faster CCT screening with Y% error on IEEE 39-bus."
        ================================================================================
    """
    summary = compute_summary(results)
    W = 88

    print("=" * W)
    print("  IEEE 39-Bus New England System — CCT Benchmark")
    print("  Method A (Ref):  RK4 binary-search  (dt=0.01, tol=1e-3)")
    print("  Method B (Fast): Equal-Area Criterion (analytic, D=0, three-phase fault)")
    print("=" * W)
    print(
        f"  {'Gen':<5} {'Bus':>4} {'H[s]':>7}  {'ω₀[rad/s]':>9}  "
        f"{'CCT_EAC[s]':>10}  {'CCT_Ref[s]':>10}  {'Err%':>5}  "
        f"{'t_Ref[ms]':>9}  {'t_EAC[μs]':>9}  {'Speedup':>9}"
    )
    print("-" * W)

    for r in results:
        print(
            f"  {r.gen_name:<5} {r.bus:>4} {r.H:>7.1f}  {r.omega0_linear:>9.3f}  "
            f"{r.cct_eac:>10.4f}  {r.cct_ref:>10.4f}  {r.cct_error_pct:>4.1f}%  "
            f"{r.t_ref_ms:>9.1f}  {r.t_eac_us:>9.2f}  {r.speedup:>8.0f}x"
        )

    print("=" * W)
    print(
        f"  SUMMARY: mean CCT error = {summary['mean_error_pct']:.1f}%   "
        f"mean speedup = {summary['mean_speedup']:.0f}\u00d7"
    )
    print(
        f"  ANCHOR: \"EAC method: {summary['mean_speedup']:.0f}\u00d7 faster CCT screening "
        f"with {summary['mean_error_pct']:.1f}% error on IEEE 39-bus.\""
    )
    print("=" * W)
