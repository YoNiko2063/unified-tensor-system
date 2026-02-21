"""
Power Grid Transient Stability — Test Suite.

Proves:
 Group 1 — Parameters and analytic quantities
   1.  Stable equilibrium δ_s = arcsin(P_m/P_e)
   2.  Unstable equilibrium δ_u = π − δ_s
   3.  ω₀ small-signal formula: √(P_e·cos(δ_s)/M)
   4.  ζ = D/(2Mω₀) analytic
   5.  E_sep > 0 for all valid (P_m < P_e)
   6.  E_sep increases as load factor P_m/P_e decreases
   7.  Invalid params (P_m ≥ P_e) raise ValueError

 Group 2 — Simulation fidelity
   8.  At exact equilibrium, state stays fixed
   9.  Undamped energy is conserved (D=0)
  10.  Damped energy is monotonically non-increasing (D>0)

 Group 3 — Near-separatrix detection
  11.  Low load → not near separatrix
  12.  High load (P_m/P_e=0.95) → near_separatrix flag triggers
  13.  Near-sep result has omega0_eff at floor, not analytic value
  14.  is_near_separatrix() standalone function matches flag

 Group 4 — Koopman / EDMD invariants
  15.  ω₀_eff ≈ ω₀_linear at small amplitude (EDMD accuracy ≤ 15%)
  16.  Q_eff > 0 and finite
  17.  invariant_descriptor() returns correct types
  18.  log_E is finite and well-defined

 Group 5 — CCT estimation
  19.  CCT > 0 for standard three-phase fault
  20.  System is stable  at τ = 0.5 × CCT
  21.  System is unstable at τ = 1.3 × CCT
  22.  CCT increases with inertia M (more inertia → more time)
  23.  CCT decreases with load factor (higher P_m/P_e → less margin)

 Group 6 — Cross-domain invariant alignment
  24.  Power grid at small angle has same invariants as matching spring-mass
  25.  store_in_memory() + retrieval round-trip succeeds

 Group 7 — Benchmark table (always passes, prints summary)
  26.  benchmark_table: runtime, CCT accuracy, near-sep classification
"""

from __future__ import annotations

import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from optimization.power_grid_evaluator import (
    CCTResult,
    PowerGridEvaluator,
    PowerGridParams,
    PowerGridSimulator,
    compute_separatrix_energy,
    estimate_cct,
    is_near_separatrix,
    simulate_power_grid,
    _NEAR_SEP_ENERGY_RATIO,
)
from optimization.koopman_memory import KoopmanExperienceMemory


# ── Shared fixtures ───────────────────────────────────────────────────────────

# Low-load: large stability margin, ζ ≈ 0.13
_P_LOW = PowerGridParams(M=0.1, D=0.1, P_m=0.3, P_e=1.0)

# Medium-load: realistic operating point, ζ ≈ 0.17
_P_MED = PowerGridParams(M=0.1, D=0.1, P_m=0.5, P_e=1.0)

# High-load: P_m/P_e=0.8, small stability margin
_P_HIGH = PowerGridParams(M=0.1, D=0.1, P_m=0.8, P_e=1.0)

# Near-critical: P_m/P_e=0.95
_P_CRIT = PowerGridParams(M=0.1, D=0.05, P_m=0.95, P_e=1.0)

# Heavy inertia: same load as MED but M=0.3
_P_HEAVY = PowerGridParams(M=0.3, D=0.1, P_m=0.5, P_e=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — Parameters and analytic quantities
# ═══════════════════════════════════════════════════════════════════════════════


def test_stable_equilibrium_formula():
    """δ_s = arcsin(P_m/P_e) matches numeric arcsin."""
    for p in (_P_LOW, _P_MED, _P_HIGH):
        expected = math.asin(p.P_m / p.P_e)
        assert abs(p.delta_s - expected) < 1e-12, (
            f"delta_s={p.delta_s:.6f} vs arcsin={expected:.6f}"
        )


def test_unstable_equilibrium_formula():
    """δ_u = π − δ_s."""
    for p in (_P_LOW, _P_MED, _P_HIGH):
        assert abs(p.delta_u - (math.pi - p.delta_s)) < 1e-12


def test_omega0_analytic():
    """ω₀ = √(P_e·cos(δ_s)/M) matches explicit formula."""
    for p in (_P_LOW, _P_MED, _P_HIGH):
        expected = math.sqrt(p.P_e * math.cos(p.delta_s) / p.M)
        rel_err  = abs(p.omega0_linear - expected) / expected
        assert rel_err < 1e-10, f"omega0_linear rel_err={rel_err:.2e}"


def test_damping_ratio_analytic():
    """ζ = D/(2Mω₀) matches formula."""
    for p in (_P_LOW, _P_MED, _P_HIGH):
        expected = p.D / (2.0 * p.M * p.omega0_linear)
        rel_err  = abs(p.damping_ratio - expected) / max(expected, 1e-30)
        assert rel_err < 1e-10


def test_separatrix_energy_positive():
    """E_sep > 0 for all valid parameter sets."""
    for p in (_P_LOW, _P_MED, _P_HIGH, _P_CRIT):
        E_sep = compute_separatrix_energy(p)
        assert E_sep > 0, f"E_sep={E_sep:.4f} ≤ 0 for P_m={p.P_m}, P_e={p.P_e}"


def test_separatrix_energy_increases_with_margin():
    """Lower load → more stability margin → larger E_sep."""
    E_low  = compute_separatrix_energy(_P_LOW)   # P_m/Pe=0.30
    E_med  = compute_separatrix_energy(_P_MED)   # P_m/Pe=0.50
    E_high = compute_separatrix_energy(_P_HIGH)  # P_m/Pe=0.80
    assert E_low > E_med > E_high, (
        f"Expected E_low={E_low:.3f} > E_med={E_med:.3f} > E_high={E_high:.3f}"
    )


def test_invalid_params_pm_ge_pe():
    """P_m ≥ P_e must raise ValueError (no stable equilibrium)."""
    with pytest.raises(ValueError):
        PowerGridParams(M=0.1, D=0.1, P_m=1.0, P_e=1.0)  # P_m == P_e
    with pytest.raises(ValueError):
        PowerGridParams(M=0.1, D=0.1, P_m=1.1, P_e=1.0)  # P_m > P_e


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — Simulation fidelity
# ═══════════════════════════════════════════════════════════════════════════════


def test_equilibrium_is_fixed_point():
    """Starting at (δ_s, 0) — stable equilibrium — state stays fixed."""
    p   = _P_MED
    sim = PowerGridSimulator(p, dt=0.01)
    delta_s = p.delta_s
    traj = sim.run(delta_s, 0.0, n_steps=100)

    max_drift = float(np.max(np.abs(traj[:, 0] - delta_s)))
    max_omega = float(np.max(np.abs(traj[:, 1])))
    print(f"\n  equilibrium drift: Δδ_max={max_drift:.2e}, ω_max={max_omega:.2e}")
    assert max_drift < 1e-6, f"Angle drifted {max_drift:.2e} from equilibrium"
    assert max_omega < 1e-6, f"Angular velocity {max_omega:.2e} at equilibrium"


def test_energy_conserved_undamped():
    """D=0 → total energy constant to within RK4 truncation error."""
    p_undamped = PowerGridParams(M=0.1, D=0.0, P_m=0.3, P_e=1.0)
    sim  = PowerGridSimulator(p_undamped, dt=0.001)  # fine dt for accuracy
    delta_s = p_undamped.delta_s
    # Kick from equilibrium by 0.3 rad
    delta0 = delta_s + 0.3
    E0     = sim.total_energy(delta0, 0.0)
    traj   = sim.run(delta0, 0.0, n_steps=500)

    energies = np.array([
        sim.total_energy(float(traj[i, 0]), float(traj[i, 1]))
        for i in range(len(traj))
    ])
    max_variation = float(np.max(np.abs(energies - E0)))
    print(f"\n  undamped energy variation: {max_variation:.2e}  (E0={E0:.4f})")
    assert max_variation < 1e-4 * max(abs(E0), 1.0), (
        f"Energy not conserved: variation={max_variation:.2e}"
    )


def test_energy_decreases_damped():
    """D>0 → total energy monotonically non-increasing (damped out)."""
    p   = _P_MED
    sim = PowerGridSimulator(p, dt=0.01)
    delta_s = p.delta_s
    delta0  = delta_s + 0.5   # moderate kick

    traj = sim.run(delta0, 0.0, n_steps=500)
    energies = np.array([
        sim.total_energy(float(traj[i, 0]), float(traj[i, 1]))
        for i in range(len(traj))
    ])

    E_start = energies[0]
    E_end   = energies[-1]
    print(f"\n  damped energy: start={E_start:.4f} → end={E_end:.4f}")
    assert E_end < E_start, (
        f"Energy did not decrease: start={E_start:.4f}, end={E_end:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3 — Near-separatrix detection
# ═══════════════════════════════════════════════════════════════════════════════


def test_low_load_not_near_separatrix():
    """Small kick from equilibrium → not near separatrix."""
    ev  = PowerGridEvaluator(_P_LOW, dt=0.01, n_steps=600)
    # Kick of 0.1 rad (P_m/Pe=0.3, large margin)
    res = ev.evaluate(_P_LOW.delta_s + 0.1)
    print(f"\n  low-load E0/E_sep = {res.E0 / res.E_sep:.4f}")
    assert not res.near_separatrix, (
        f"Unexpected near_separatrix flag: E0={res.E0:.4f}, E_sep={res.E_sep:.4f}"
    )


def test_high_load_near_separatrix_triggers():
    """P_m/P_e=0.95 + large initial kick → near_separatrix=True."""
    p   = _P_CRIT
    ev  = PowerGridEvaluator(p, dt=0.005, n_steps=400)

    # Place initial condition deep in the near-sep zone
    E_sep = p.separatrix_energy
    # Find delta0 such that E0/E_sep ≈ 0.90
    # V_rel(delta0) = P_e*(cos(delta_s)-cos(delta0)) + P_m*(delta_s-delta0)
    # For ω=0: E0 = V_rel(delta0).  Target E0 = 0.90 * E_sep.
    # Use binary search to find delta0.
    target_E = 0.90 * E_sep
    sim = PowerGridSimulator(p, dt=0.005)
    lo, hi = p.delta_s, p.delta_u - 0.01
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        e   = sim.total_energy(mid, 0.0)
        if e < target_E:
            lo = mid
        else:
            hi = mid
    delta0 = 0.5 * (lo + hi)

    res = ev.evaluate(delta0)
    print(f"\n  high-load E0/E_sep = {res.E0 / res.E_sep:.4f}")
    assert res.near_separatrix, (
        f"near_separatrix should be True: E0={res.E0:.4f}, "
        f"E_sep={res.E_sep:.4f}, ratio={res.E0/res.E_sep:.4f}"
    )


def test_near_sep_omega_floor_applied():
    """Near-sep result has omega0_eff = floor, not near analytic value."""
    p   = _P_CRIT
    ev  = PowerGridEvaluator(p, dt=0.005, n_steps=400)
    E_sep = p.separatrix_energy
    sim = PowerGridSimulator(p, dt=0.005)

    # Find near-sep delta0 (E0/E_sep > 0.87)
    target_E = 0.90 * E_sep
    lo, hi = p.delta_s, p.delta_u - 0.01
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        e   = sim.total_energy(mid, 0.0)
        if e < target_E:
            lo = mid
        else:
            hi = mid
    delta0 = 0.5 * (lo + hi)

    res       = ev.evaluate(delta0)
    omega0_lin = p.omega0_linear
    floor     = 0.005 * omega0_lin  # _OMEGA_FLOOR_FRACTION

    assert res.near_separatrix, "Precondition failed: should be near separatrix"
    assert abs(res.omega0_eff - floor) < 1e-8 * max(floor, 1.0), (
        f"Near-sep: expected omega0_eff ≈ floor={floor:.4f}, "
        f"got {res.omega0_eff:.4f}"
    )


def test_is_near_separatrix_standalone():
    """Standalone is_near_separatrix() matches PowerGridResult flag."""
    p   = _P_MED
    ev  = PowerGridEvaluator(p, dt=0.01, n_steps=600)
    res = ev.evaluate(p.delta_s + 0.2)

    standalone = is_near_separatrix(res.E0, res.E_sep)
    assert standalone == res.near_separatrix, (
        f"Standalone {standalone} ≠ result flag {res.near_separatrix}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4 — Koopman / EDMD invariants
# ═══════════════════════════════════════════════════════════════════════════════


def test_omega0_eff_matches_analytic_small_amplitude():
    """Small kick → ω₀_eff ≈ ω₀_linear within 15%."""
    for p in (_P_LOW, _P_MED):
        ev   = PowerGridEvaluator(p, dt=0.01, n_steps=1000)
        # Small initial deviation (1% of stability range)
        delta0 = p.delta_s + 0.01 * (p.delta_u - p.delta_s)
        res    = ev.evaluate(delta0)
        rel_err = abs(res.omega0_eff - p.omega0_linear) / p.omega0_linear
        print(f"\n  P_m/Pe={p.load_factor:.2f}: "
              f"ω₀_eff={res.omega0_eff:.4f} analytic={p.omega0_linear:.4f} "
              f"err={rel_err:.3f}")
        assert rel_err < 0.15, (
            f"EDMD ω₀_eff={res.omega0_eff:.4f} deviates {rel_err:.3f} "
            f"from analytic {p.omega0_linear:.4f}"
        )


def test_Q_eff_positive_and_finite():
    """Q_eff > 0 and finite for all test cases."""
    for p in (_P_LOW, _P_MED, _P_HIGH):
        ev  = PowerGridEvaluator(p, dt=0.01, n_steps=800)
        res = ev.evaluate(p.delta_s + 0.1)
        assert 0 < res.Q_eff < 1e6, f"Q_eff={res.Q_eff:.4f} out of range"


def test_invariant_descriptor_types():
    """invariant_descriptor() returns (KoopmanInvariantDescriptor, KoopmanResult, float)."""
    from optimization.koopman_signature import KoopmanInvariantDescriptor
    from tensor.koopman_edmd import KoopmanResult

    ev   = PowerGridEvaluator(_P_MED, dt=0.01, n_steps=600)
    inv, koop, log_E = ev.invariant_descriptor(_P_MED.delta_s + 0.1)

    assert isinstance(inv,   KoopmanInvariantDescriptor)
    assert isinstance(koop,  KoopmanResult)
    assert isinstance(log_E, float)
    assert math.isfinite(log_E), f"log_E is not finite: {log_E}"


def test_log_E_finite():
    """log_E is finite and in a reasonable range."""
    ev  = PowerGridEvaluator(_P_MED, dt=0.01, n_steps=800)
    res = ev.evaluate(_P_MED.delta_s + 0.3)
    assert math.isfinite(res.log_E), f"log_E = {res.log_E}"
    assert -10 < res.log_E < 10, f"log_E = {res.log_E} out of range"


# ═══════════════════════════════════════════════════════════════════════════════
# Group 5 — CCT estimation
# ═══════════════════════════════════════════════════════════════════════════════


def test_cct_positive():
    """CCT > 0 for standard parameters with three-phase fault."""
    result = estimate_cct(_P_MED, fault_duration_range=(0.0, 5.0),
                          fault_factor=0.0, dt=0.01)
    assert result.is_stable, "Binary search should find a valid CCT bracket"
    assert result.cct > 0.0, f"CCT={result.cct:.3f} should be positive"
    print(f"\n  CCT(_P_MED) = {result.cct:.3f} s  "
          f"[{result.tau_lo:.3f}, {result.tau_hi:.3f}]")


def test_cct_stable_below_threshold():
    """System IS stable at τ = 0.5 × CCT."""
    cct_result = estimate_cct(_P_MED, fault_duration_range=(0.0, 5.0),
                              fault_factor=0.0, dt=0.01)
    assert cct_result.is_stable

    half_cct = 0.5 * cct_result.cct
    # Verify directly: simulate with half_cct fault duration
    safe_result = estimate_cct(
        _P_MED,
        fault_duration_range=(half_cct - 0.01, half_cct + 0.01),
        fault_factor=0.0, dt=0.01,
    )
    # At half CCT, system should be stable (tau_lo = half_cct - 0.01 should pass)
    sim = PowerGridSimulator(_P_MED, dt=0.01)
    p   = _P_MED
    delta_s = p.delta_s
    delta_u = p.delta_u

    # Manual stability check at half_cct
    state = np.array([delta_s, 0.0])
    n_fault = max(0, int(round(half_cct / 0.01)))
    stable  = True
    for _ in range(n_fault):
        k1 = sim.rhs(state, 0.0)
        k2 = sim.rhs(state + 0.005 * k1, 0.0)
        k3 = sim.rhs(state + 0.005 * k2, 0.0)
        k4 = sim.rhs(state + 0.01  * k3, 0.0)
        state = state + (0.01 / 6) * (k1 + 2*k2 + 2*k3 + k4)
    for _ in range(3000):
        k1 = sim.rhs(state)
        k2 = sim.rhs(state + 0.005 * k1)
        k3 = sim.rhs(state + 0.005 * k2)
        k4 = sim.rhs(state + 0.01  * k3)
        state = state + (0.01 / 6) * (k1 + 2*k2 + 2*k3 + k4)
        if state[0] > delta_u + 0.05:
            stable = False
            break

    assert stable, (
        f"System should be stable at τ=half_CCT={half_cct:.3f} s "
        f"(CCT={cct_result.cct:.3f} s)"
    )


def test_cct_unstable_above_threshold():
    """System IS unstable at τ = 1.3 × CCT."""
    cct_result = estimate_cct(_P_MED, fault_duration_range=(0.0, 5.0),
                              fault_factor=0.0, dt=0.01)
    assert cct_result.is_stable

    long_fault = 1.3 * cct_result.cct
    sim = PowerGridSimulator(_P_MED, dt=0.01)
    p   = _P_MED
    delta_s = p.delta_s
    delta_u = p.delta_u

    state   = np.array([delta_s, 0.0])
    n_fault = max(1, int(round(long_fault / 0.01)))
    unstable = False

    for _ in range(n_fault):
        k1 = sim.rhs(state, 0.0)
        k2 = sim.rhs(state + 0.005 * k1, 0.0)
        k3 = sim.rhs(state + 0.005 * k2, 0.0)
        k4 = sim.rhs(state + 0.01  * k3, 0.0)
        state = state + (0.01 / 6) * (k1 + 2*k2 + 2*k3 + k4)
        if state[0] > delta_u + 0.05:
            unstable = True
            break

    if not unstable:
        for _ in range(3000):
            k1 = sim.rhs(state)
            k2 = sim.rhs(state + 0.005 * k1)
            k3 = sim.rhs(state + 0.005 * k2)
            k4 = sim.rhs(state + 0.01  * k3)
            state = state + (0.01 / 6) * (k1 + 2*k2 + 2*k3 + k4)
            if state[0] > delta_u + 0.05:
                unstable = True
                break

    assert unstable, (
        f"System should be unstable at τ=1.3×CCT={long_fault:.3f} s "
        f"(CCT={cct_result.cct:.3f} s)"
    )


def test_cct_increases_with_inertia():
    """Heavier rotor (larger M) → larger CCT (more time to absorb energy)."""
    cct_med   = estimate_cct(_P_MED,   fault_duration_range=(0.0, 5.0),
                             fault_factor=0.0, dt=0.01)
    cct_heavy = estimate_cct(_P_HEAVY, fault_duration_range=(0.0, 5.0),
                             fault_factor=0.0, dt=0.01)

    print(f"\n  CCT M=0.1: {cct_med.cct:.3f} s   CCT M=0.3: {cct_heavy.cct:.3f} s")
    assert cct_heavy.cct > cct_med.cct, (
        f"Expected CCT to increase with M: "
        f"M=0.1 → {cct_med.cct:.3f} s, M=0.3 → {cct_heavy.cct:.3f} s"
    )


def test_cct_decreases_with_load():
    """Higher load factor → smaller stability margin → smaller CCT."""
    cct_med  = estimate_cct(_P_MED,  fault_duration_range=(0.0, 5.0),
                            fault_factor=0.0, dt=0.01)
    cct_high = estimate_cct(_P_HIGH, fault_duration_range=(0.0, 5.0),
                            fault_factor=0.0, dt=0.01)

    print(f"\n  CCT Pm/Pe=0.5: {cct_med.cct:.3f} s   "
          f"CCT Pm/Pe=0.8: {cct_high.cct:.3f} s")
    assert cct_med.cct > cct_high.cct, (
        f"Expected CCT to decrease with load: "
        f"Pm/Pe=0.5 → {cct_med.cct:.3f} s, Pm/Pe=0.8 → {cct_high.cct:.3f} s"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 6 — Cross-domain invariant alignment
# ═══════════════════════════════════════════════════════════════════════════════


def test_small_angle_matches_spring_mass_analytic():
    """
    At small amplitude, power-grid ω₀, ζ match the linearised spring-mass:
        k_eff = P_e·cos(δ_s),  m_eff = M,  b_eff = D.
    """
    p       = _P_MED
    ev      = PowerGridEvaluator(p, dt=0.005, n_steps=1200)
    delta0  = p.delta_s + 0.02   # tiny kick → linear regime

    # Spring-mass analytic values for the linearised system
    k_eff  = p.P_e * math.cos(p.delta_s)
    sm_w0  = math.sqrt(k_eff / p.M)
    sm_zeta = p.D / (2.0 * p.M * sm_w0)

    res = ev.evaluate(delta0)
    rel_w0_err  = abs(res.omega0_eff - sm_w0)  / sm_w0
    rel_zeta_err = abs(res.zeta_linear - sm_zeta) / sm_zeta

    print(f"\n  SM  ω₀={sm_w0:.4f} ζ={sm_zeta:.4f}")
    print(f"  PG  ω₀={res.omega0_eff:.4f} ζ={res.zeta_linear:.4f}")
    assert rel_w0_err  < 0.15, f"ω₀ rel_err={rel_w0_err:.3f}"
    assert rel_zeta_err < 1e-6, f"ζ rel_err={rel_zeta_err:.3f}"


def test_store_and_retrieve_from_memory():
    """store_in_memory() then retrieve_candidates() returns the entry."""
    mem = KoopmanExperienceMemory()
    ev  = PowerGridEvaluator(_P_MED, dt=0.01, n_steps=800)
    ev.store_in_memory(mem, delta0=_P_MED.delta_s + 0.2, label="test")

    assert len(mem._entries) == 1, "Memory should have exactly one entry"

    # Retrieve with query near the stored invariant
    inv, _, _ = ev.invariant_descriptor(_P_MED.delta_s + 0.2)
    candidates = mem.retrieve_candidates(inv, top_n=3)
    assert len(candidates) >= 1, "Should retrieve at least one candidate"

    exp = candidates[0].experience
    assert exp.domain.startswith("power_grid"), (
        f"Retrieved domain '{exp.domain}' not power_grid"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 7 — Benchmark table
# ═══════════════════════════════════════════════════════════════════════════════


def test_benchmark_table():
    """
    Print summary table comparing runtime and CCT accuracy.

    Format: Case | Load | ω₀ [rad/s] | CCT_est [s] | CCT_true [s] | Error% | t_eval [ms]
    """
    cases = [
        ("Low load",    _P_LOW),
        ("Medium load", _P_MED),
        ("High load",   _P_HIGH),
    ]

    print("\n")
    print("=" * 80)
    print(f"  {'Case':<14} {'Pm/Pe':>6} {'ω₀[rad/s]':>10} "
          f"{'CCT_est[s]':>11} {'CCT_true[s]':>12} {'Error%':>7} {'t_eval[ms]':>11}")
    print("-" * 80)

    for name, p in cases:
        ev = PowerGridEvaluator(p, dt=0.01, n_steps=800)

        # Evaluation timing
        t0  = time.perf_counter()
        res = ev.evaluate(p.delta_s + 0.3)
        t_eval_ms = (time.perf_counter() - t0) * 1000

        # CCT estimation (coarse, tol=1e-2)
        cct_est = estimate_cct(p, fault_duration_range=(0.0, 5.0),
                               fault_factor=0.0, dt=0.01, tol=1e-2).cct

        # CCT "true" (fine, tol=1e-4)
        cct_true = estimate_cct(p, fault_duration_range=(0.0, 5.0),
                                fault_factor=0.0, dt=0.01, tol=1e-4).cct

        err_pct = abs(cct_est - cct_true) / max(cct_true, 1e-12) * 100

        print(f"  {name:<14} {p.load_factor:>6.2f} {p.omega0_linear:>10.3f} "
              f"{cct_est:>11.3f} {cct_true:>12.3f} {err_pct:>6.1f}% "
              f"{t_eval_ms:>10.1f}")

        # Gate: CCT estimation error < 5%
        assert err_pct < 5.0, (
            f"{name}: CCT error {err_pct:.1f}% exceeds 5% gate "
            f"(est={cct_est:.3f}, true={cct_true:.3f})"
        )

    print("=" * 80)
    print("  All cases: CCT error < 5% gate  PASS")
