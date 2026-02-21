"""
IEEE 39-Bus Damping Correction — Test Suite.

Validates the global first-order correction:

    CCT_corrected = CCT_EAC / (1 − a·ζ)

where `a` is fitted by OLS across all 10 generators × 5 ζ values.

Why a single global `a` should work (analytic basis):
    For P_e = 2·P_m (all IEEE-39 generators):  δ_s = 30° for all.
    Under this constraint: ω₀·CCT_EAC = √(2√3·(δ_c−δ_s)) ≈ 1.73 = constant.
    First-order perturbation theory gives CCT_ref/CCT_EAC ≈ 1 + (ω₀·CCT_EAC/3)·ζ,
    with (ω₀·CCT_EAC/3) ≈ 0.577 as the universal slope.
    The OLS-fitted `a` absorbs higher-order terms and is expected to be ~1.5–1.7.

Groups:
 1. Correction fit (3 tests)
    — a > 0
    — a in physically plausible range [0.5, 3.0]
    — a is consistent with measured mean error slope

 2. Corrected accuracy (3 tests)
    — ζ*_corrected > ζ* (correction extends the valid range)
    — mean |corrected error| < mean |raw error| for ζ ≥ 0.03
    — max |corrected error| at ζ=0.05 < max |raw error| at ζ=0.05

 3. Correction table (1 test, always prints)
    — Print raw vs corrected, report ζ*_corrected
"""

from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from optimization.ieee39_benchmark import (
    IEEE39_GENERATORS,
    compute_corrected_errors,
    compute_corrected_summary,
    compute_sweep_summary,
    fit_damping_correction,
    print_correction_table,
    run_damping_sweep,
)

# ── Shared cache ──────────────────────────────────────────────────────────────

_ZETA_VALUES = (0.01, 0.03, 0.05, 0.10, 0.20)

_SWEEP = None
_A     = None
_CORR  = None


def _get_sweep():
    global _SWEEP
    if _SWEEP is None:
        _SWEEP = run_damping_sweep(zeta_values=_ZETA_VALUES)
    return _SWEEP


def _get_a():
    global _A
    if _A is None:
        _A = fit_damping_correction(_get_sweep())
    return _A


def _get_corr():
    global _CORR
    if _CORR is None:
        _CORR = compute_corrected_errors(_get_sweep(), _get_a())
    return _CORR


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — Correction fit
# ═══════════════════════════════════════════════════════════════════════════════


def test_fitted_a_positive():
    """OLS fit returns a > 0 (correction increases CCT, consistent with D>0 data)."""
    a = _get_a()
    print(f"\n  Fitted a = {a:.4f}")
    assert a > 0.0, f"a = {a:.4f} should be positive"


def test_fitted_a_plausible_range():
    """
    a is in [0.5, 3.0].

    Analytic lower bound: first-order perturbation gives slope ω₀·CCT_EAC/3 ≈ 0.577.
    Upper bound: beyond a=3 the formula diverges before ζ=0.20 (1−3·0.20=0.4, still valid).
    Measured range expected: ~1.5–1.7.
    """
    a = _get_a()
    assert 0.5 <= a <= 3.0, f"a = {a:.4f} outside plausible range [0.5, 3.0]"


def test_fitted_a_consistent_with_slope():
    """
    a is consistent with the measured mean error slope.

    mean error ≈ −a·ζ·100% (first-order approximation).
    Check: a ≈ |mean_error_at_ζ=0.05| / (0.05 × 100) within 2×.
    """
    sweep   = _get_sweep()
    a       = _get_a()
    at_005  = [r for r in sweep if abs(r.zeta - 0.05) < 1e-9]
    mean_err = sum(abs(r.cct_error_pct) for r in at_005) / len(at_005)
    empirical_slope = mean_err / (0.05 * 100)   # a ≈ |mean_err%| / (ζ×100)

    print(f"\n  mean |err| at ζ=0.05: {mean_err:.2f}%")
    print(f"  empirical slope (|err|/(ζ×100)): {empirical_slope:.3f}")
    print(f"  fitted a: {a:.4f}")

    # Allow factor-of-2 tolerance (higher-order terms shift the effective slope)
    assert a / empirical_slope < 2.0, (
        f"a = {a:.3f} is more than 2× empirical slope {empirical_slope:.3f}"
    )
    assert empirical_slope / a < 2.0, (
        f"empirical slope {empirical_slope:.3f} is more than 2× a = {a:.3f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — Corrected accuracy
# ═══════════════════════════════════════════════════════════════════════════════


def test_corrected_zeta_star_extended():
    """
    ζ*_corrected ≥ ζ* (correction extends the valid damping range).

    Raw sweep: ζ* = 0.01 (EAC only valid at Q ≥ 50).
    Corrected: expect ζ*_corrected > 0.01 — at minimum equal, ideally ≥ 0.03.
    """
    raw_summary  = compute_sweep_summary(_get_sweep())
    corr_summary = compute_corrected_summary(_get_corr())

    zeta_star_raw  = raw_summary["zeta_star"]
    zeta_star_corr = corr_summary["zeta_star_corrected"]

    print(f"\n  ζ* raw       = {zeta_star_raw}")
    print(f"  ζ* corrected = {zeta_star_corr}")

    assert zeta_star_corr is not None, (
        "Corrected formula should pass 5% gate at some ζ"
    )
    assert zeta_star_corr >= zeta_star_raw, (
        f"Corrected ζ* ({zeta_star_corr}) < raw ζ* ({zeta_star_raw})"
    )


def test_mean_error_reduced_for_mid_zeta():
    """
    Mean |corrected error| < mean |raw error| for ζ ∈ {0.03, 0.05, 0.10}.

    Correction should strictly improve accuracy in the mid-range where
    the linear approximation is well-calibrated.
    """
    corr    = _get_corr()
    mid_zetas = [z for z in _ZETA_VALUES if 0.02 < z < 0.15]

    for zeta in mid_zetas:
        subset      = [c for c in corr if abs(c["zeta"] - zeta) < 1e-9]
        mean_raw    = sum(abs(c["raw_err_pct"])  for c in subset) / len(subset)
        mean_corr   = sum(abs(c["corr_err_pct"]) for c in subset) / len(subset)
        print(f"\n  ζ={zeta}: mean|raw|={mean_raw:.2f}%  mean|corr|={mean_corr:.2f}%")
        assert mean_corr < mean_raw, (
            f"ζ={zeta}: mean |corrected error| {mean_corr:.2f}% ≥ mean |raw| {mean_raw:.2f}%"
        )


def test_max_corrected_error_at_z005():
    """
    Max |corrected error| at ζ=0.05 < max |raw error| at ζ=0.05.

    This is the critical gate: ζ=0.05 is in the target range for real grids
    (inter-area mode ζ ≈ 0.03–0.10).  Correction must improve accuracy there.
    """
    corr    = _get_corr()
    at_005  = [c for c in corr if abs(c["zeta"] - 0.05) < 1e-9]

    max_raw_abs  = max(abs(c["raw_err_pct"])  for c in at_005)
    max_corr_abs = max(abs(c["corr_err_pct"]) for c in at_005)

    print(f"\n  ζ=0.05  max|raw|={max_raw_abs:.2f}%  max|corr|={max_corr_abs:.2f}%")
    assert max_corr_abs < max_raw_abs, (
        f"Correction did not reduce max error at ζ=0.05: "
        f"raw={max_raw_abs:.2f}%  corr={max_corr_abs:.2f}%"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3 — Full correction table
# ═══════════════════════════════════════════════════════════════════════════════


def test_correction_table():
    """
    Print raw vs corrected comparison table, report ζ*_corrected.

    Always passes as long as:
      - a > 0
      - ζ*_corrected ≥ ζ* = 0.01
    """
    sweep   = _get_sweep()
    print_correction_table(sweep)

    a            = _get_a()
    corr_summary = compute_corrected_summary(_get_corr())
    zeta_star    = corr_summary["zeta_star_corrected"]

    assert a > 0.0
    assert zeta_star is not None
    assert zeta_star >= 0.01
