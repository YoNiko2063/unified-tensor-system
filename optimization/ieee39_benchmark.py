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
from itertools import combinations
from typing import List

from optimization.power_grid_evaluator import PowerGridParams, estimate_cct
from optimization.koopman_signature import _LOG_OMEGA0_REF, _LOG_OMEGA0_SCALE


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


# ── Damping sweep ─────────────────────────────────────────────────────────────


@dataclass
class DampingSweepResult:
    """
    Per-(generator, ζ) result from run_damping_sweep().

    Invariant metrics are analytic (no EDMD) — fast and exact.
    CCT reference uses actual D in RK4 binary search.
    """

    gen_name:         str    # "G1" … "G10"
    bus:              int
    H:                float  # inertia constant [s]
    zeta:             float  # target damping ratio ζ = D/(2Mω₀)
    D:                float  # actual damping coefficient [pu·s/rad]

    # ── Analytic invariant metrics ────────────────────────────────────────────
    omega0_linear:    float  # undamped ω₀ = √(P_e·cos(δ_s)/M)  [rad/s]
    omega0_damped:    float  # ω₀·√(1−ζ²)  — effective resonance under damping [rad/s]
    omega0_drift_pct: float  # (ω₀_damped/ω₀_linear − 1) × 100  [%]  (≤ 0)
    Q_analytic:       float  # 1/(2ζ)
    embed_dist:       float  # ‖v(ζ) − v(ζ_ref)‖ in 3D invariant space, ζ_ref = min(sweep)

    # ── CCT accuracy ─────────────────────────────────────────────────────────
    cct_eac:          float  # EAC formula (D=0 assumption)  [s]
    cct_ref:          float  # RK4 binary search with actual D  [s]
    cct_error_pct:    float  # signed: (CCT_EAC − CCT_ref)/CCT_ref × 100
                             #   negative → EAC conservative (underestimates CCT)


def _invariant_embed_dist(
    omega0: float,
    zeta: float,
    zeta_ref: float,
) -> float:
    """
    Euclidean distance in 3D normalised invariant space between ζ and ζ_ref.

    v(ζ) = (log_ω₀_norm_damped, log_Q_norm, ζ)

        log_ω₀_norm_damped = (log(ω₀·√(1−ζ²)) − _LOG_OMEGA0_REF) / _LOG_OMEGA0_SCALE
        log_Q_norm          = log(1/(2ζ)) / _LOG_OMEGA0_SCALE

    Key property: Δlog_ω₀_norm is independent of ω₀ (cancels in the difference),
    so embed_dist is the same for all generators at a given (ζ, ζ_ref).
    """
    d_log_w0 = (
        0.5 * math.log(max((1.0 - zeta ** 2) / (1.0 - zeta_ref ** 2), 1e-30))
        / _LOG_OMEGA0_SCALE
    )
    d_log_Q  = math.log(zeta_ref / max(zeta, 1e-30)) / _LOG_OMEGA0_SCALE
    d_zeta   = zeta - zeta_ref
    return float(math.sqrt(d_log_w0 ** 2 + d_log_Q ** 2 + d_zeta ** 2))


def run_damping_sweep(
    zeta_values: tuple = (0.01, 0.03, 0.05, 0.10, 0.20),
    ref_tol: float = 1e-3,
    ref_dt:  float = 0.01,
) -> List[DampingSweepResult]:
    """
    Sweep damping ratio ζ across all 10 IEEE 39-bus generators.

    For each (generator, ζ):
      - Set D = 2·M·ω₀_linear·ζ  (exact ζ targeting)
      - Compute analytic invariant metrics: ω₀_drift, Q_analytic, embed_dist
      - Compute CCT via EAC (D=0 formula) and RK4 reference (actual D)
      - Record signed CCT error: negative = EAC is conservative (safe side)

    Args:
        zeta_values: tuple of damping ratios to sweep
        ref_tol:     binary-search tolerance [s]
        ref_dt:      RK4 timestep [s]

    Returns:
        List ordered by (ζ outer, generator inner).
    """
    zeta_ref = min(zeta_values)
    results: List[DampingSweepResult] = []

    for zeta in zeta_values:
        for gen in IEEE39_GENERATORS:
            params_d0  = to_power_grid_params(gen)          # D=0 reference
            omega0_lin = params_d0.omega0_linear
            M          = params_d0.M

            D_actual = 2.0 * M * omega0_lin * zeta
            params_damped = PowerGridParams(
                M=M, D=D_actual, P_m=gen.P_m, P_e=gen.P_e
            )

            omega0_damp   = omega0_lin * math.sqrt(max(1.0 - zeta ** 2, 0.0))
            drift_pct     = (omega0_damp / omega0_lin - 1.0) * 100.0
            Q_an          = 1.0 / (2.0 * zeta)
            embed         = _invariant_embed_dist(omega0_lin, zeta, zeta_ref)

            cct_eac = eac_cct(params_d0)
            cct_ref = estimate_cct(
                params_damped,
                fault_duration_range=(0.0, 5.0),
                fault_factor=0.0,
                dt=ref_dt,
                tol=ref_tol,
            ).cct

            signed_err = (cct_eac - cct_ref) / max(cct_ref, 1e-12) * 100.0

            results.append(DampingSweepResult(
                gen_name         = gen.name,
                bus              = gen.bus,
                H                = gen.H,
                zeta             = zeta,
                D                = D_actual,
                omega0_linear    = omega0_lin,
                omega0_damped    = omega0_damp,
                omega0_drift_pct = drift_pct,
                Q_analytic       = Q_an,
                embed_dist       = embed,
                cct_eac          = cct_eac,
                cct_ref          = cct_ref,
                cct_error_pct    = signed_err,
            ))

    return results


def compute_sweep_summary(results: List[DampingSweepResult]) -> dict:
    """
    Aggregate sweep results per ζ.

    Returns dict keyed by ζ with per-ζ stats, plus top-level 'zeta_star':
    largest ζ where max |CCT error| across all generators < 5%.
    """
    zeta_values = sorted(set(r.zeta for r in results))
    per_zeta: dict = {}

    for zeta in zeta_values:
        subset = [r for r in results if r.zeta == zeta]
        errs   = [r.cct_error_pct for r in subset]
        per_zeta[zeta] = {
            "mean_cct_error_pct": sum(errs) / len(errs),
            "max_abs_cct_error":  max(abs(e) for e in errs),
            "omega0_drift_pct":   subset[0].omega0_drift_pct,  # generator-independent
            "Q_analytic":         subset[0].Q_analytic,
            "embed_dist":         subset[0].embed_dist,         # generator-independent
        }

    zeta_star = None
    for zeta in sorted(zeta_values):
        if per_zeta[zeta]["max_abs_cct_error"] < 5.0:
            zeta_star = zeta
    per_zeta["zeta_star"] = zeta_star

    return per_zeta


def print_damping_table(results: List[DampingSweepResult]) -> None:
    """
    Print two tables to stdout:

    Table 1 — Per-ζ summary: mean/max CCT error + invariant geometry metrics.
    Table 2 — Per-generator signed CCT error [%] for each ζ.
    """
    summary     = compute_sweep_summary(results)
    zeta_values = sorted(set(r.zeta for r in results))
    zeta_star   = summary["zeta_star"]
    W = 88

    # ── Table 1: summary by ζ ─────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  IEEE 39-Bus — Damping Sweep: EAC formula vs. RK4 reference")
    print("  D = 2Mω₀ζ per generator.  Signed error < 0 → EAC conservative.")
    print("=" * W)
    print(
        f"  {'ζ':>5}  {'Q':>6}  {'ω₀ drift':>9}  {'embed dist':>10}  "
        f"{'mean err%':>9}  {'max|err|%':>9}  {'<5% gate':>8}"
    )
    print("-" * W)

    for zeta in zeta_values:
        s    = summary[zeta]
        gate = "PASS" if s["max_abs_cct_error"] < 5.0 else "FAIL"
        star = " ← ζ*" if zeta == zeta_star else ""
        print(
            f"  {zeta:>5.2f}  {s['Q_analytic']:>6.1f}  "
            f"{s['omega0_drift_pct']:>8.3f}%  "
            f"{s['embed_dist']:>10.4f}  "
            f"{s['mean_cct_error_pct']:>+9.2f}%  "
            f"{s['max_abs_cct_error']:>9.2f}%  "
            f"{gate:>8}{star}"
        )

    print("=" * W)
    if zeta_star is not None:
        print(
            f"  ζ* = {zeta_star:.2f}  "
            f"(EAC max |error| < 5% for all 10 generators at ζ ≤ ζ*)"
        )
        print(
            f"  Claim scope: EAC valid for lightly-damped grids  "
            f"ζ ≤ {zeta_star:.2f}  (Q ≥ {1.0/(2*zeta_star):.0f})"
        )
    else:
        print("  ζ* < min tested ζ  (EAC exceeds 5% gate at all tested values)")
    print("=" * W)

    # ── Table 2: per-generator error grid ─────────────────────────────────────
    print("\n" + "=" * W)
    print("  Per-generator signed CCT error [%]  (EAC formula − RK4 with actual D)")
    print("=" * W)
    header = f"  {'Gen':<5}  {'H[s]':>6}" + "".join(
        f"  {f'ζ={z:.2f}':>9}" for z in zeta_values
    )
    print(header)
    print("-" * W)

    for gen in IEEE39_GENERATORS:
        row = {r.zeta: r for r in results if r.gen_name == gen.name}
        errs = "".join(
            f"  {row[z].cct_error_pct:>+8.2f}%" for z in zeta_values
        )
        print(f"  {gen.name:<5}  {gen.H:>6.1f}{errs}")

    print("=" * W)


# ── Global damping correction ─────────────────────────────────────────────────


def fit_damping_correction(results: List[DampingSweepResult]) -> float:
    """
    Fit global scalar `a` for the first-order damping correction:

        CCT_corrected = CCT_EAC / (1 − a·ζ)

    OLS derivation
    --------------
    Rearranging: CCT_ref·(1 − a·ζ) ≈ CCT_EAC
    → (CCT_EAC − CCT_ref) + a·(ζ·CCT_ref) = 0
    → eᵢ + a·xᵢ = residual,   eᵢ = CCT_EAC − CCT_ref < 0,  xᵢ = ζ·CCT_ref > 0

    Minimise Σ(eᵢ + a·xᵢ)²  →  a_opt = −Σ(eᵢ·xᵢ) / Σ(xᵢ²)

    Why the correction is generator-independent
    -------------------------------------------
    For all IEEE-39 generators P_e = 2·P_m → δ_s = 30° for all.
    Under this constraint ω₀·CCT_EAC = √(2·√3·(δ_c−δ_s)) = constant ≈ 1.732.
    The first-order perturbation gives CCT_ref/CCT_EAC ≈ 1 + C·ζ/3 where
    C = ω₀·CCT_EAC is the same for every generator — hence a single global `a`.
    The invariant embedding distance being generator-independent is the
    geometric signature of this universal perturbation structure.

    Args:
        results: output of run_damping_sweep() (all ζ values included)

    Returns:
        Fitted scalar a > 0.
    """
    num = 0.0
    den = 0.0
    for r in results:
        e_i = r.cct_eac - r.cct_ref      # < 0 for D > 0
        x_i = r.zeta * r.cct_ref         # > 0
        num += e_i * x_i
        den += x_i * x_i
    return float(-num / max(den, 1e-30))


def compute_corrected_errors(
    results: List[DampingSweepResult],
    a: float,
) -> List[dict]:
    """
    Apply CCT_corrected = CCT_EAC / (1 − a·ζ) and return residual errors.

    Returns list of dicts with keys:
        gen_name, H, zeta, cct_eac, cct_corr, cct_ref,
        raw_err_pct, corr_err_pct
    """
    out = []
    for r in results:
        denom    = max(1.0 - a * r.zeta, 1e-6)
        cct_corr = r.cct_eac / denom
        corr_err = (cct_corr - r.cct_ref) / max(r.cct_ref, 1e-12) * 100.0
        out.append({
            "gen_name":    r.gen_name,
            "H":           r.H,
            "zeta":        r.zeta,
            "cct_eac":     r.cct_eac,
            "cct_corr":    cct_corr,
            "cct_ref":     r.cct_ref,
            "raw_err_pct":  r.cct_error_pct,
            "corr_err_pct": corr_err,
        })
    return out


def compute_corrected_summary(corrected: List[dict]) -> dict:
    """
    Per-ζ aggregates of raw vs corrected CCT errors, plus zeta_star_corrected.
    """
    zeta_values = sorted(set(c["zeta"] for c in corrected))
    per_zeta: dict = {}

    for zeta in zeta_values:
        subset     = [c for c in corrected if c["zeta"] == zeta]
        raw_errs   = [c["raw_err_pct"]  for c in subset]
        corr_errs  = [c["corr_err_pct"] for c in subset]
        per_zeta[zeta] = {
            "mean_raw_err":  sum(raw_errs)  / len(raw_errs),
            "max_abs_raw":   max(abs(e) for e in raw_errs),
            "mean_corr_err": sum(corr_errs) / len(corr_errs),
            "max_abs_corr":  max(abs(e) for e in corr_errs),
        }

    zeta_star_corr = None
    for zeta in sorted(zeta_values):
        if per_zeta[zeta]["max_abs_corr"] < 5.0:
            zeta_star_corr = zeta
    per_zeta["zeta_star_corrected"] = zeta_star_corr

    return per_zeta


def print_correction_table(results: List[DampingSweepResult]) -> None:
    """
    Fit a, apply correction, print raw-vs-corrected comparison tables.

    Table 1 — Per-ζ summary: mean/max error before and after correction.
    Table 2 — Per-generator corrected errors at each ζ.
    """
    a         = fit_damping_correction(results)
    corrected = compute_corrected_errors(results, a)
    summary   = compute_corrected_summary(corrected)
    zeta_vals = sorted(set(c["zeta"] for c in corrected))
    zeta_star = summary["zeta_star_corrected"]
    W = 92

    # ── Table 1: raw vs corrected summary ─────────────────────────────────────
    print("\n" + "=" * W)
    print(f"  IEEE 39-Bus — Damping Correction: CCT_corr = CCT_EAC / (1 − a·ζ)")
    print(f"  Fitted: a = {a:.4f}  (OLS, 10 generators × {len(zeta_vals)} ζ values)")
    print(f"  Analytic basis: ω₀·CCT_EAC ≈ √(2√3·(δ_c−δ_s)) ≈ 1.73 for all generators")
    print("=" * W)
    print(
        f"  {'ζ':>5}  {'mean raw%':>10}  {'max|raw|%':>10}  "
        f"{'mean corr%':>11}  {'max|corr|%':>11}  {'gate':>6}"
    )
    print("-" * W)

    for zeta in zeta_vals:
        s    = summary[zeta]
        gate = "PASS" if s["max_abs_corr"] < 5.0 else "FAIL"
        star = " ← ζ*" if zeta == zeta_star else ""
        print(
            f"  {zeta:>5.2f}  {s['mean_raw_err']:>+10.2f}%  "
            f"{s['max_abs_raw']:>10.2f}%  "
            f"{s['mean_corr_err']:>+11.2f}%  "
            f"{s['max_abs_corr']:>11.2f}%  "
            f"{gate:>6}{star}"
        )

    print("=" * W)
    if zeta_star is not None:
        print(
            f"  ζ*_corrected = {zeta_star:.2f}  "
            f"(corrected EAC: max |error| < 5% for all 10 generators)"
        )
        print(
            f"  Corrected anchor: "
            f"\"EAC+correction valid for ζ ≤ {zeta_star:.2f}  "
            f"(Q ≥ {1.0/(2*zeta_star):.0f}) on IEEE 39-bus.\""
        )
    else:
        print("  Corrected formula did not extend ζ* beyond uncorrected sweep.")
    print("=" * W)

    # ── Table 2: per-generator corrected errors ────────────────────────────────
    print("\n" + "=" * W)
    print("  Per-generator corrected CCT error [%]  (CCT_corr − CCT_ref) / CCT_ref")
    print("=" * W)
    header = f"  {'Gen':<5}  {'H[s]':>6}" + "".join(
        f"  {f'ζ={z:.2f}':>9}" for z in zeta_vals
    )
    print(header)
    print("-" * W)

    for gen in IEEE39_GENERATORS:
        row  = {c["zeta"]: c for c in corrected if c["gen_name"] == gen.name}
        errs = "".join(f"  {row[z]['corr_err_pct']:>+8.2f}%" for z in zeta_vals)
        print(f"  {gen.name:<5}  {gen.H:>6.1f}{errs}")

    print("=" * W)


# ── Leave-2-out cross-validation of scalar a ─────────────────────────────────


def cross_validate_correction(
    results: List[DampingSweepResult],
    n_test: int = 2,
) -> List[dict]:
    """
    Leave-n-out cross-validation of the global correction parameter `a`.

    All computation is algebraic on existing DampingSweepResult data.
    No new ODE evaluations — pure sum operations over subsets.

    For each of C(10, n_test) = 45 splits:
      1. Fit a on 8 training generators (all ζ values, 40 data points).
      2. Apply CCT_corrected = CCT_EAC / (1 − a_train·ζ) to 2 test generators.
      3. Record a_train, test corrected errors.

    Args:
        results: output of run_damping_sweep()
        n_test:  generators held out per split (default 2)

    Returns:
        List of dicts with keys:
            test_gens, a_train, mean_test_err_pct, max_test_err_pct
    """
    gen_names  = [g.name for g in IEEE39_GENERATORS]
    cv_results = []

    for test_indices in combinations(range(len(gen_names)), n_test):
        test_names  = {gen_names[i] for i in test_indices}
        train_names = {n for n in gen_names if n not in test_names}

        train = [r for r in results if r.gen_name in train_names]
        test  = [r for r in results if r.gen_name in test_names]

        a_train   = fit_damping_correction(train)
        test_corr = compute_corrected_errors(test, a_train)
        abs_errs  = [abs(c["corr_err_pct"]) for c in test_corr]

        cv_results.append({
            "test_gens":       sorted(test_names),
            "a_train":         a_train,
            "mean_test_err":   sum(abs_errs) / len(abs_errs),
            "max_test_err":    max(abs_errs),
        })

    return cv_results


def compute_cv_summary(cv_results: List[dict]) -> dict:
    """
    Aggregate cross-validation results.

    Returns dict with:
        a_mean, a_std, a_min, a_max, a_range_pct  (range / mean × 100)
        mean_test_err, max_test_err               (across all splits)
        is_stable                                  (a_range_pct < 10%)
    """
    a_vals      = [r["a_train"]       for r in cv_results]
    test_means  = [r["mean_test_err"] for r in cv_results]
    test_maxs   = [r["max_test_err"]  for r in cv_results]

    a_mean      = sum(a_vals) / len(a_vals)
    a_var       = sum((a - a_mean) ** 2 for a in a_vals) / len(a_vals)
    a_std       = math.sqrt(a_var)
    a_range_pct = (max(a_vals) - min(a_vals)) / a_mean * 100.0

    return {
        "a_mean":        a_mean,
        "a_std":         a_std,
        "a_min":         min(a_vals),
        "a_max":         max(a_vals),
        "a_range_pct":   a_range_pct,
        "mean_test_err": sum(test_means) / len(test_means),
        "max_test_err":  max(test_maxs),
        "n_splits":      len(cv_results),
        "is_stable":     a_range_pct < 10.0,   # ±5% criterion
    }


def print_cv_table(cv_results: List[dict]) -> None:
    """
    Print cross-validation summary: per-split a_train + aggregate statistics.
    """
    summary = compute_cv_summary(cv_results)
    W = 72

    print("\n" + "=" * W)
    print("  Leave-2-out Cross-Validation: stability of correction parameter a")
    print(f"  {summary['n_splits']} splits  (C(10,2))  —  no new ODE calls")
    print("=" * W)
    print(f"  {'Split':>5}  {'Test gens':<12}  {'a_train':>8}  "
          f"{'mean|err|%':>11}  {'max|err|%':>10}")
    print("-" * W)

    for i, r in enumerate(cv_results, 1):
        gens = ",".join(r["test_gens"])
        print(f"  {i:>5}  {gens:<12}  {r['a_train']:>8.4f}  "
              f"{r['mean_test_err']:>10.2f}%  {r['max_test_err']:>9.2f}%")

    print("=" * W)
    stable = "STABLE ✓" if summary["is_stable"] else "UNSTABLE ✗"
    print(f"  a:  mean={summary['a_mean']:.4f}  "
          f"std={summary['a_std']:.4f}  "
          f"min={summary['a_min']:.4f}  "
          f"max={summary['a_max']:.4f}  "
          f"range={summary['a_range_pct']:.1f}%  [{stable}]")
    print(f"  Test corrected errors:  "
          f"mean={summary['mean_test_err']:.2f}%  "
          f"max={summary['max_test_err']:.2f}%")
    if summary["is_stable"]:
        print(f"  VERDICT: a is robust.  "
              f"Range {summary['a_range_pct']:.1f}% < 10% (±5%) gate.")
    else:
        print(f"  VERDICT: a is not robust.  "
              f"Range {summary['a_range_pct']:.1f}% ≥ 10% gate — re-evaluate.")
    print("=" * W)
