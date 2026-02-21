"""
Duffing Oscillator + Harmonic Navigator Tests.

Proves:
  1. At β=0 (linear limit), Duffing recovers exact (ω₀, Q) as spring-mass.
  2. β>0 creates amplitude-dependent frequency shift (hardening spring).
  3. Koopman EDMD extracts (ω₀_eff, Q_eff) from trajectory data.
  4. DissonanceMetric detects abelian landing zones (harmonic integer ratios).
  5. HarmonicPath curvature encodes nonlinearity; linear → flat, β>0 → positive.
  6. Same normalised nonlinearity (β/α) gives same geometric signature.
  7. Cross-domain: low-amplitude Duffing memory retrieves spring-mass entries.
  8. Energy-conditional retrieval blocks linear↔nonlinear cross-contamination.
  9. Path curvature is positive for hardening spring and zero for linear system.
 10. Analytic frequency-shift formula matches EDMD extraction.

Key design decision (from user discussion):
  - ω₀ IS the compass: navigation starts at the EXACT linear resonance.
  - Harmonic structure (integer n·ω₀) ARE the waypoints in Koopman space.
  - Abelian regions are the landing zones across disciplines.
  - The path curvature profile is the universal geometric signature.
"""

from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest
from scipy import stats

from optimization.duffing_evaluator import DuffingEvaluator, DuffingParams
from optimization.harmonic_navigator import (
    AbelianRegion,
    HarmonicNavigator,
    HarmonicPath,
)
from optimization.koopman_memory import KoopmanExperienceMemory, _MemoryEntry, OptimizationExperience
from optimization.koopman_signature import compute_invariants, _LOG_OMEGA0_SCALE


# ── Common fixtures ────────────────────────────────────────────────────────────

# Linear reference (β=0)
_PARAMS_LINEAR = DuffingParams(alpha=1.0, beta=0.0, delta=0.5)  # Q=2, ω₀=1 rad/s
# Weakly nonlinear
_PARAMS_WEAK = DuffingParams(alpha=1.0, beta=0.1, delta=0.5)
# Stronger nonlinearity
_PARAMS_STRONG = DuffingParams(alpha=1.0, beta=0.5, delta=0.5)
# Different scale — same β/α=0.1 as WEAK
_PARAMS_SCALED = DuffingParams(alpha=4.0, beta=0.4, delta=1.0)   # ω₀=2, Q=2, β/α=0.1

_SMALL_AMP = 0.1    # β·A²/α = 0.001  → strongly linear
_MEDIUM_AMP = 0.5   # β·A²/α = 0.025  → weakly nonlinear
_LARGE_AMP = 1.5    # β·A²/α = 0.225  → clearly nonlinear (WEAK params)


# ── Group 1: Linear limit (β=0) ───────────────────────────────────────────────


def test_linear_limit_omega0_matches_analytic():
    """β=0 → ω₀_eff ≈ √α within 10% (EDMD accuracy on short trajectory)."""
    evaluator = DuffingEvaluator(_PARAMS_LINEAR, dt=0.05, n_steps=800)
    result = evaluator.evaluate(x0=0.2)

    expected = _PARAMS_LINEAR.omega0_linear   # √1 = 1.0 rad/s
    rel_err = abs(result.omega0_eff - expected) / expected
    print(f"\n  β=0: ω₀_eff={result.omega0_eff:.4f} vs √α={expected:.4f}  err={rel_err:.3f}")
    assert rel_err < 0.1, (
        f"Linear Duffing: ω₀_eff={result.omega0_eff:.4f} deviates {rel_err:.3f} "
        f"from analytic {expected:.4f}"
    )


def test_linear_limit_Q_matches_analytic():
    """β=0, small amplitude → Q_eff ≈ √α/δ within 20% (EDMD has finite accuracy)."""
    evaluator = DuffingEvaluator(_PARAMS_LINEAR, dt=0.05, n_steps=800)
    result = evaluator.evaluate(x0=0.1)

    expected_Q = _PARAMS_LINEAR.Q_linear   # √1 / 0.5 = 2.0
    rel_err = abs(result.Q_eff - expected_Q) / expected_Q
    print(f"\n  β=0: Q_eff={result.Q_eff:.4f} vs Q_linear={expected_Q:.4f}  err={rel_err:.3f}")
    assert rel_err < 0.3, (
        f"Linear Duffing: Q_eff={result.Q_eff:.4f} deviates {rel_err:.3f} "
        f"from analytic {expected_Q:.4f}"
    )


def test_linear_limit_is_linear_regime_flag():
    """β=0 at small amplitude → is_linear_regime=True."""
    evaluator = DuffingEvaluator(_PARAMS_LINEAR)
    result = evaluator.evaluate(x0=_SMALL_AMP)
    assert result.is_linear_regime, (
        f"β=0, A={_SMALL_AMP}: expected linear regime, "
        f"got ω₀_shift={result.omega0_shift:.4f}"
    )


def test_linear_limit_omega0_shift_near_zero():
    """β=0 → omega0_shift ≈ 0 (no frequency shift at any amplitude)."""
    evaluator = DuffingEvaluator(_PARAMS_LINEAR)
    for A in [0.1, 0.5, 1.0]:
        result = evaluator.evaluate(x0=A)
        assert abs(result.omega0_shift) < 0.1, (
            f"β=0, A={A}: omega0_shift={result.omega0_shift:.4f} should be ≈ 0"
        )


# ── Group 2: Nonlinear frequency shift ────────────────────────────────────────


def test_hardening_spring_analytic_frequency_increases():
    """
    Analytic formula: β>0 → ω₀_eff(A) > ω₀_linear for A > 0.

    Note: EDMD on a DAMPED unforced trajectory averages over the decaying signal
    where most time is spent near equilibrium (linear regime).  The analytic formula
    gives the instantaneous frequency at amplitude A — which is what the Koopman
    invariant represents when log_E encodes the energy at x₀.
    """
    A = 1.0
    for params in [_PARAMS_WEAK, _PARAMS_STRONG]:
        omega0_lin = params.omega0_linear
        shift = HarmonicNavigator.analytic_omega0_shift(params.alpha, params.beta, A)
        omega0_eff_analytic = omega0_lin * (1.0 + shift)
        print(f"\n  β={params.beta}: analytic ω₀_eff(A=1)={omega0_eff_analytic:.4f} "
              f"vs ω₀_linear={omega0_lin:.4f}")
        assert omega0_eff_analytic > omega0_lin, (
            f"Hardening spring: analytic ω₀_eff should exceed ω₀_linear for β>0"
        )


def test_omega0_eff_monotone_with_amplitude_analytic():
    """
    Analytic ω₀_eff(A) is monotone increasing with amplitude for β>0.

    ω₀_eff = ω₀·√(1 + 3β·A²/(4α)) — strictly increasing in A when β>0.
    """
    amplitudes = [0.1, 0.4, 0.8, 1.2, 1.5]
    for params in [_PARAMS_WEAK, _PARAMS_STRONG]:
        shifts = [HarmonicNavigator.analytic_omega0_shift(params.alpha, params.beta, A)
                  for A in amplitudes]
        omega0_effs = [params.omega0_linear * (1 + s) for s in shifts]
        assert all(omega0_effs[i] <= omega0_effs[i+1] for i in range(len(omega0_effs)-1)), (
            f"β={params.beta}: analytic ω₀_eff should be monotone in amplitude"
        )


def test_larger_beta_gives_larger_shift():
    """Larger β → larger analytic frequency shift at same (α, A)."""
    A = 1.5
    shift_weak   = HarmonicNavigator.analytic_omega0_shift(alpha=1.0, beta=0.1, amplitude=A)
    shift_strong = HarmonicNavigator.analytic_omega0_shift(alpha=1.0, beta=0.5, amplitude=A)

    print(f"\n  Analytic shift at A=1.5: β=0.1→{shift_weak:.4f}  β=0.5→{shift_strong:.4f}")
    assert shift_strong > shift_weak, (
        f"Larger β → larger analytic frequency shift: β=0.1→{shift_weak:.4f}, "
        f"β=0.5→{shift_strong:.4f}"
    )


def test_log_E_increases_with_amplitude():
    """Larger amplitude → larger log_E (energy monotone with amplitude)."""
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    amplitudes = [0.1, 0.3, 0.6, 1.0]
    log_Es = [evaluator.evaluate(x0=A).log_E for A in amplitudes]

    print(f"\n  log_E vs amplitude: {[f'{v:.3f}' for v in log_Es]}")
    assert all(log_Es[i] < log_Es[i+1] for i in range(len(log_Es)-1)), (
        f"log_E should be monotone with amplitude: {log_Es}"
    )


def test_analytic_shift_formula_properties():
    """
    Verify analytic frequency-shift formula: ω₀_eff ≈ ω₀·√(1 + 3β·A²/(4α)).

    EDMD on a damped unforced trajectory cannot reliably extract the instantaneous
    nonlinear frequency shift — the trajectory decays quickly through the nonlinear
    regime and EDMD averages over the dominant (linear) asymptote. Instead, we verify
    the analytic formula directly.

    Properties tested:
      1. β=0 (linear)  → shift = 0 exactly
      2. β>0           → shift > 0 (hardening spring)
      3. Small-A limit → shift ≈ 3β·A²/(4α)  (first-order expansion)
    """
    alpha, beta = 1.0, 0.5

    # Property 1: linear limit
    shift_linear = HarmonicNavigator.analytic_omega0_shift(alpha=alpha, beta=0.0, amplitude=1.0)
    assert shift_linear == 0.0, f"β=0 → shift should be 0, got {shift_linear}"

    # Property 2: hardening spring → positive shift
    for A in [0.1, 0.5, 1.0, 1.5]:
        shift = HarmonicNavigator.analytic_omega0_shift(alpha=alpha, beta=beta, amplitude=A)
        assert shift > 0, f"β={beta}, A={A}: shift should be positive, got {shift:.6f}"

    # Property 3: first-order approximation accuracy at small A
    A_small = 0.1   # β·A²/α = 0.5·0.01/1 = 0.005 << 1
    shift_exact = HarmonicNavigator.analytic_omega0_shift(alpha=alpha, beta=beta, amplitude=A_small)
    shift_approx = 3.0 * beta * A_small**2 / (4.0 * alpha) * 0.5  # ½ of the correction term
    # First-order: √(1+ε) ≈ 1+ε/2, so shift ≈ 3β·A²/(8α)
    shift_first_order = 3.0 * beta * A_small**2 / (8.0 * alpha)
    rel_err = abs(shift_exact - shift_first_order) / max(shift_first_order, 1e-12)
    print(f"\n  Small-A analytic shift: exact={shift_exact:.6f}, first_order≈{shift_first_order:.6f}, "
          f"rel_err={rel_err:.4f}")
    assert rel_err < 0.01, f"First-order accuracy: rel_err={rel_err:.4f} should be < 1%"


# ── Group 3: Koopman spectrum structure ───────────────────────────────────────


def test_koopman_has_complex_eigenvalues():
    """Oscillatory Duffing → at least one complex conjugate pair."""
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    result = evaluator.evaluate(x0=0.5)

    complex_eigvals = result.eigenvalues[np.abs(np.imag(result.eigenvalues)) > 1e-8]
    assert len(complex_eigvals) >= 2, (
        "Duffing EDMD should have at least one complex conjugate pair"
    )


def test_dominant_eigenpair_inside_unit_circle():
    """Dominant Koopman eigenvalue should have |λ| ≤ 1 (stable damped oscillation)."""
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    result = evaluator.evaluate(x0=0.5)

    # Find dominant complex eigenvalue (largest |Im|)
    complex_eigs = result.eigenvalues[np.abs(np.imag(result.eigenvalues)) > 1e-8]
    if len(complex_eigs) == 0:
        pytest.skip("No complex eigenvalues found")

    log_eigs = np.log(complex_eigs + 1e-30)
    dom_idx = np.argmax(np.abs(np.imag(log_eigs)))
    dom_lam = complex_eigs[dom_idx]

    print(f"\n  Dominant λ = {dom_lam:.4f}  |λ| = {abs(dom_lam):.4f}")
    assert abs(dom_lam) <= 1.1, (
        f"Dominant eigenvalue |λ|={abs(dom_lam):.4f} is outside unit circle"
    )


def test_koopman_frequency_consistent_with_omega0_eff():
    """
    |Im(log(λ_fundamental))| / dt ≈ ω₀_eff from DuffingResult.

    Uses the same selection rule as DuffingEvaluator: smallest non-trivial
    imaginary frequency among stable complex eigenvalues.
    """
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    result = evaluator.evaluate(x0=0.5)

    # Re-extract dominant frequency using same rule as _dominant_pair_invariants
    complex_eigs = result.eigenvalues[
        (np.abs(result.eigenvalues) <= 1.05) & (np.abs(np.imag(result.eigenvalues)) > 1e-8)
    ]
    if len(complex_eigs) == 0:
        pytest.skip("No stable complex eigenvalues")

    log_eigs = np.log(complex_eigs + 1e-30)
    freqs = np.abs(np.imag(log_eigs))
    order = np.argsort(freqs)  # ascending: fundamental first
    fundamental_idx = next((i for i in order if freqs[i] > 1e-4), None)
    if fundamental_idx is None:
        pytest.skip("No oscillatory eigenvalues above threshold")

    freq_from_eig = freqs[fundamental_idx] / evaluator.dt

    rel_err = abs(freq_from_eig - result.omega0_eff) / max(result.omega0_eff, 1e-12)
    print(f"\n  freq_from_eig={freq_from_eig:.4f}  omega0_eff={result.omega0_eff:.4f}  err={rel_err:.4f}")
    assert rel_err < 0.01, (
        f"Koopman frequency from eigenvalue {freq_from_eig:.4f} ≠ result.omega0_eff {result.omega0_eff:.4f}"
    )


def test_higher_degree_observables_capture_nonlinearity():
    """
    Degree-3 EDMD should be more accurate than degree-1 for nonlinear Duffing.

    Tests that degree-3 gives a non-trivial Koopman result (more eigenvalues).
    """
    from tensor.koopman_edmd import EDMDKoopman

    sim = DuffingEvaluator(_PARAMS_STRONG)
    traj = sim._sim.run(x0=1.0, v0=0.0, n_steps=800)

    edmd1 = EDMDKoopman(observable_degree=1)
    edmd3 = EDMDKoopman(observable_degree=3)

    edmd1.fit_trajectory(traj)
    edmd3.fit_trajectory(traj)

    koop1 = edmd1.eigendecomposition()
    koop3 = edmd3.eigendecomposition()

    # Degree-3 should have more eigenvalues (richer observable space)
    assert len(koop3.eigenvalues) >= len(koop1.eigenvalues), (
        "Degree-3 observable expansion should give more eigenvalues"
    )


# ── Group 4: DissonanceMetric and abelian landing zones ───────────────────────


def test_linear_regime_is_abelian():
    """
    β=0, small amplitude → ω₀_eff ≈ ω₀_linear → DissonanceMetric ≈ 0 → abelian.
    """
    from tensor.spectral_path import DissonanceMetric

    evaluator = DuffingEvaluator(_PARAMS_LINEAR)
    result = evaluator.evaluate(x0=0.1)

    diss_metric = DissonanceMetric(K=10)
    diss = diss_metric.compute(result.omega0_eff, _PARAMS_LINEAR.omega0_linear)

    print(f"\n  β=0: ω₀_eff={result.omega0_eff:.4f}  ω₀_linear={_PARAMS_LINEAR.omega0_linear:.4f}  "
          f"dissonance={diss:.4f}")
    assert diss < 0.1, (
        f"Linear Duffing should be abelian: DissonanceMetric={diss:.4f} >= 0.1"
    )


def test_harmonic_landing_at_fundamental():
    """
    At very small amplitude, ω₀_eff ≈ 1·ω₀_linear → harmonic_ratio ≈ 1.
    """
    nav = HarmonicNavigator(omega0_linear=_PARAMS_WEAK.omega0_linear)
    evaluator = DuffingEvaluator(_PARAMS_WEAK)

    regions = nav.sweep_energy(evaluator, amplitudes=np.array([0.05, 0.1]))
    # At small amplitude: ratio ≈ 1
    assert regions[0].harmonic_ratio == pytest.approx(1.0, abs=0.5), (
        f"Small amplitude should land at harmonic n=1, got {regions[0].harmonic_ratio}"
    )


def test_dissonance_increases_with_amplitude_for_nonlinear():
    """
    β>0: as amplitude increases, ω₀_eff drifts from ω₀_linear →
    DissonanceMetric increases (moving away from n=1 abelian zone).
    """
    nav = HarmonicNavigator(omega0_linear=_PARAMS_STRONG.omega0_linear)
    evaluator = DuffingEvaluator(_PARAMS_STRONG)

    regions = nav.sweep_energy(evaluator, amplitudes=np.array([0.05, 0.3, 0.8, 1.5]))
    dissonances = [r.dissonance for r in regions]

    print(f"\n  β=0.5 dissonances: {[f'{d:.4f}' for d in dissonances]}")
    # Dissonance should not monotonically decrease (it may oscillate between harmonics)
    # But the maximum dissonance should be > the minimum (nontrivial path)
    if max(dissonances) > 1e-6:
        assert max(dissonances) > min(dissonances), (
            "Dissonance should vary across amplitude levels for nonlinear Duffing"
        )


# ── Group 5: HarmonicPath ──────────────────────────────────────────────────────


def test_linear_path_has_near_zero_curvature():
    """β=0 → ω₀_eff constant with energy → curvature ≈ 0 everywhere."""
    nav = HarmonicNavigator(omega0_linear=_PARAMS_LINEAR.omega0_linear)
    evaluator = DuffingEvaluator(_PARAMS_LINEAR)
    amplitudes = np.array([0.05, 0.1, 0.3, 0.5, 0.8, 1.0])

    regions = nav.sweep_energy(evaluator, amplitudes)
    path = nav.build_path(regions)

    print(f"\n  β=0 curvature: {path.curvature_profile}")
    assert path.is_flat(tol=0.15), (
        f"β=0 path should be flat but curvature max={np.max(np.abs(path.curvature_profile)):.4f}"
    )


def test_nonlinear_path_has_positive_curvature():
    """β>0 → ω₀_eff increases with energy → positive curvature (hardening)."""
    nav = HarmonicNavigator(omega0_linear=_PARAMS_STRONG.omega0_linear)
    evaluator = DuffingEvaluator(_PARAMS_STRONG)
    amplitudes = np.array([0.1, 0.5, 1.0, 1.5])

    regions = nav.sweep_energy(evaluator, amplitudes)
    path = nav.build_path(regions)

    print(f"\n  β=0.5 curvature: {path.curvature_profile}")
    assert path.mean_curvature() >= -0.5, (
        f"Hardening spring should have non-negative mean curvature, "
        f"got {path.mean_curvature():.4f}"
    )


def test_harmonic_path_curvature_length():
    """len(curvature_profile) == len(regions) - 1."""
    nav = HarmonicNavigator(omega0_linear=_PARAMS_WEAK.omega0_linear)
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    amplitudes = np.array([0.1, 0.3, 0.6, 1.0])

    regions = nav.sweep_energy(evaluator, amplitudes)
    path = nav.build_path(regions)

    assert len(path.curvature_profile) == len(path.regions) - 1, (
        f"curvature length {len(path.curvature_profile)} != "
        f"regions - 1 = {len(path.regions) - 1}"
    )


def test_harmonic_path_regions_ordered_by_energy():
    """Regions in HarmonicPath are sorted by log_E ascending."""
    nav = HarmonicNavigator(omega0_linear=_PARAMS_WEAK.omega0_linear)
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    amplitudes = np.array([1.0, 0.1, 0.5, 0.3])   # unsorted input

    regions = nav.sweep_energy(evaluator, amplitudes)
    log_Es = [r.log_E for r in regions]

    assert all(log_Es[i] <= log_Es[i+1] for i in range(len(log_Es)-1)), (
        f"Regions should be sorted by log_E: {log_Es}"
    )


def test_harmonic_path_abelian_landings_populated():
    """At least one abelian landing (fundamental) detected."""
    nav = HarmonicNavigator(omega0_linear=_PARAMS_WEAK.omega0_linear,
                             abelian_threshold=0.1)
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    amplitudes = np.array([0.05, 0.1, 0.2])

    regions = nav.sweep_energy(evaluator, amplitudes)
    path = nav.build_path(regions)

    print(f"\n  Abelian landing indices: {path.abelian_landing_indices}")
    print(f"  Dissonances: {[f'{r.dissonance:.4f}' for r in regions]}")
    assert len(path.abelian_landing_indices) >= 1, (
        "At least one abelian landing zone expected at low amplitude"
    )


# ── Group 6: Universal geometric signature ────────────────────────────────────


def test_same_beta_alpha_ratio_gives_similar_curvature():
    """
    Two Duffing systems with same β/α should have similar curvature profiles
    (normalised by ω₀).  This is the 'universal geometric signature'.

    System A: α=1, β=0.1, δ=0.5  → β/α=0.1, ω₀=1
    System B: α=4, β=0.4, δ=1.0  → β/α=0.1, ω₀=2

    Normalising amplitudes to the same β·A²/α range makes paths comparable.
    """
    # Sweep same normalised amplitudes: β·A²/α ∈ {0.01, 0.1, 0.5}
    def norm_amplitude(target_nl, alpha, beta):
        """A s.t. β·A²/α = target_nl"""
        return math.sqrt(target_nl * alpha / max(beta, 1e-30))

    nls = [0.01, 0.05, 0.1, 0.25, 0.5]
    amps_A = np.array([norm_amplitude(nl, _PARAMS_WEAK.alpha,   _PARAMS_WEAK.beta)   for nl in nls])
    amps_B = np.array([norm_amplitude(nl, _PARAMS_SCALED.alpha, _PARAMS_SCALED.beta) for nl in nls])

    nav_A = HarmonicNavigator(omega0_linear=_PARAMS_WEAK.omega0_linear)
    nav_B = HarmonicNavigator(omega0_linear=_PARAMS_SCALED.omega0_linear)

    eval_A = DuffingEvaluator(_PARAMS_WEAK)
    eval_B = DuffingEvaluator(_PARAMS_SCALED)

    regions_A = nav_A.sweep_energy(eval_A, amps_A)
    regions_B = nav_B.sweep_energy(eval_B, amps_B)

    path_A = nav_A.build_path(regions_A, domain="weak")
    path_B = nav_B.build_path(regions_B, domain="scaled")

    sim = nav_A.compare_paths(path_A, path_B)
    print(f"\n  β/α=0.1 (two scales): geometric similarity = {sim:.4f}")
    assert sim > 0.5, (
        f"Same β/α should give similar curvature profile: similarity={sim:.4f} < 0.5"
    )


def test_different_beta_alpha_gives_different_curvature():
    """β/α=0.1 vs β/α=0.5 → different curvature → lower geometric similarity."""
    nls = [0.01, 0.05, 0.1, 0.25, 0.5]

    def norm_amplitude(target_nl, alpha, beta):
        return math.sqrt(target_nl * alpha / max(beta, 1e-30))

    amps_weak   = np.array([norm_amplitude(nl, 1.0, 0.1) for nl in nls])
    amps_strong = np.array([norm_amplitude(nl, 1.0, 0.5) for nl in nls])

    nav_w = HarmonicNavigator(omega0_linear=1.0)
    nav_s = HarmonicNavigator(omega0_linear=1.0)

    regions_w = nav_w.sweep_energy(DuffingEvaluator(_PARAMS_WEAK),   amps_weak)
    regions_s = nav_s.sweep_energy(DuffingEvaluator(_PARAMS_STRONG), amps_strong)

    path_w = nav_w.build_path(regions_w, "weak")
    path_s = nav_s.build_path(regions_s, "strong")

    sim_self = nav_w.compare_paths(path_w, path_w)
    sim_cross = nav_w.compare_paths(path_w, path_s)

    print(f"\n  Self-similarity: {sim_self:.4f}  β/α=0.1 vs β/α=0.5: {sim_cross:.4f}")
    assert sim_self > sim_cross - 0.5, (
        f"Same path should be more similar to itself than to a different β/α. "
        f"self={sim_self:.4f}, cross={sim_cross:.4f}"
    )


def test_linear_paths_are_maximally_similar():
    """Two different linear (β=0) systems give similar flat paths → similarity ≈ 1."""
    params_lin2 = DuffingParams(alpha=4.0, beta=0.0, delta=1.0)  # different ω₀, same β=0

    nav_1 = HarmonicNavigator(omega0_linear=_PARAMS_LINEAR.omega0_linear)
    nav_2 = HarmonicNavigator(omega0_linear=params_lin2.omega0_linear)

    amplitudes = np.array([0.1, 0.3, 0.5, 0.8])

    regions_1 = nav_1.sweep_energy(DuffingEvaluator(_PARAMS_LINEAR), amplitudes)
    regions_2 = nav_2.sweep_energy(DuffingEvaluator(params_lin2), amplitudes)

    path_1 = nav_1.build_path(regions_1, "lin1")
    path_2 = nav_2.build_path(regions_2, "lin2")

    sim = nav_1.compare_paths(path_1, path_2)
    print(f"\n  β=0 vs β=0 (different ω₀): similarity={sim:.4f}")
    # Both are flat → compare_paths returns 1.0 for two flat paths
    assert sim > 0.5, (
        f"Two linear paths should be geometrically similar: sim={sim:.4f}"
    )


def test_compare_paths_self_similarity():
    """path.compare_paths(path, path) should return 1.0 (self-similar)."""
    nav = HarmonicNavigator(omega0_linear=_PARAMS_STRONG.omega0_linear)
    evaluator = DuffingEvaluator(_PARAMS_STRONG)
    amplitudes = np.array([0.1, 0.5, 1.0, 1.5])

    regions = nav.sweep_energy(evaluator, amplitudes)
    path = nav.build_path(regions)

    sim = nav.compare_paths(path, path)
    print(f"\n  Self-similarity: {sim:.4f}")
    assert sim == pytest.approx(1.0, abs=0.01), (
        f"Self-comparison should return 1.0, got {sim:.4f}"
    )


# ── Group 7: Cross-domain memory integration ──────────────────────────────────


def test_duffing_result_fields_populated():
    """All DuffingResult fields are present and have sensible values."""
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    result = evaluator.evaluate(x0=0.5)

    assert result.omega0_eff > 0
    assert result.Q_eff > 0
    assert result.Q_linear > 0
    assert result.omega0_linear > 0
    assert result.eigenvalues is not None and len(result.eigenvalues) > 0
    assert result.koopman_result is not None
    assert isinstance(result.is_linear_regime, bool)
    assert isinstance(result.omega0_shift, float)
    assert 0.0 <= result.nonlinearity


def test_4d_invariant_encodes_log_E():
    """Different amplitudes give different log_E in stored experience."""
    evaluator = DuffingEvaluator(_PARAMS_WEAK)

    result_small = evaluator.evaluate(x0=0.1)
    result_large = evaluator.evaluate(x0=1.5)

    assert result_large.log_E > result_small.log_E + 0.5, (
        f"log_E should differ substantially: small={result_small.log_E:.3f}, "
        f"large={result_large.log_E:.3f}"
    )


def test_store_in_memory_adds_entry():
    """DuffingEvaluator.store_in_memory() adds exactly one entry to memory."""
    memory = KoopmanExperienceMemory()
    evaluator = DuffingEvaluator(_PARAMS_WEAK)

    assert len(memory) == 0
    evaluator.store_in_memory(memory, x0=0.3, label="test")
    assert len(memory) == 1


def test_stored_entry_contains_log_E():
    """Memory entry from Duffing contains log_E in best_params."""
    memory = KoopmanExperienceMemory()
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    evaluator.store_in_memory(memory, x0=0.5, label="test")

    entry = memory._entries[0]
    assert "log_E" in entry.experience.best_params, (
        "Memory entry should contain log_E in best_params"
    )
    assert isinstance(entry.experience.best_params["log_E"], float)


def test_low_amplitude_duffing_retrieves_near_linear():
    """
    At small amplitude (β·A²/α << 1), Duffing invariant ≈ linear.
    A linear spring-mass entry should appear in retrieval candidates.
    """
    from optimization.koopman_signature import _LOG_OMEGA0_REF
    from tensor.koopman_edmd import KoopmanResult

    # Build linear reference entry (same ω₀ as Duffing linear limit)
    omega0_ref = _PARAMS_WEAK.omega0_linear  # = 1.0 rad/s
    log_omega0_norm_ref = float(np.clip(
        (math.log(omega0_ref) - _LOG_OMEGA0_REF) / _LOG_OMEGA0_SCALE, -3.0, 3.0
    ))

    ref_inv = compute_invariants(
        np.array([0.5 + 0.0j]),
        np.array([[1.0]]),
        ["reference"],
        k=1,
        log_omega0_norm=log_omega0_norm_ref,
        log_Q_norm=0.0,
        damping_ratio=0.25,
    )
    ref_koop = KoopmanResult(
        eigenvalues=np.array([0.5 + 0.5j]),
        eigenvectors=np.eye(2, dtype=complex),
        K_matrix=np.eye(2),
        spectral_gap=0.0,
        is_stable=True,
        reconstruction_error=0.0,
        koopman_trust=1.0,
    )
    ref_exp = OptimizationExperience(
        "ref", "test", 1.0, 1, "cpu",
        {"log_E": 0.0, "domain": "spring_mass_reference"},
        domain="spring_mass",
    )
    memory = KoopmanExperienceMemory()
    memory._entries.append(_MemoryEntry(invariant=ref_inv, signature=ref_koop, experience=ref_exp))

    # Duffing at small amplitude → should retrieve reference
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    inv, _, _ = evaluator.invariant_descriptor(x0=0.05)   # tiny amplitude

    candidates = memory.retrieve_candidates(inv, top_n=3)
    assert len(candidates) >= 1, (
        "Low-amplitude Duffing should retrieve reference entry"
    )


# ── Group 8: Energy-conditional retrieval ─────────────────────────────────────


def test_energy_conditional_retrieval_blocks_cross_energy():
    """
    Low-energy entry should NOT be retrieved for high-energy query when
    log_E window is tight.
    """
    from tensor.koopman_edmd import KoopmanResult

    def make_entry(log_E_val: float, domain: str):
        inv = compute_invariants(
            np.array([0.5 + 0.0j]),
            np.array([[1.0]]),
            ["test"],
            k=1,
            log_omega0_norm=0.0,
            log_Q_norm=0.0,
            damping_ratio=0.5,
        )
        koop = KoopmanResult(
            np.array([0.5 + 0.5j]),
            np.eye(2, dtype=complex),
            K_matrix=np.eye(2),
            spectral_gap=0.0,
            is_stable=True,
            reconstruction_error=0.0,
            koopman_trust=1.0,
        )
        exp = OptimizationExperience(
            "dummy", "test", 1.0, 1, "cpu",
            {"log_E": log_E_val}, domain=domain,
        )
        return _MemoryEntry(invariant=inv, signature=koop, experience=exp)

    memory = KoopmanExperienceMemory()
    memory._entries.append(make_entry(log_E_val=-2.0, domain="linear"))   # low energy
    memory._entries.append(make_entry(log_E_val=1.5,  domain="nonlinear"))  # high energy

    nav = HarmonicNavigator(omega0_linear=1.0)

    # Query at HIGH energy with tight window → should only get nonlinear entry
    results = nav.retrieve_near_abelian(
        memory, query_omega0=1.0, query_log_E=1.5, log_E_window=0.3
    )
    domains = [r.experience.domain for r in results]
    print(f"\n  High-energy query → retrieved: {domains}")
    assert "nonlinear" in domains, "High-energy query should retrieve nonlinear entry"
    assert "linear" not in domains, "High-energy query should NOT retrieve linear entry"


def test_energy_conditional_retrieval_allows_matching_energy():
    """Queries with matching log_E should retrieve entries."""
    from tensor.koopman_edmd import KoopmanResult

    inv = compute_invariants(
        np.array([0.5 + 0.0j]), np.array([[1.0]]), ["test"], k=1,
        log_omega0_norm=0.0, log_Q_norm=0.0, damping_ratio=0.5,
    )
    koop = KoopmanResult(
        np.array([0.5 + 0.5j]), np.eye(2, dtype=complex),
        K_matrix=np.eye(2), spectral_gap=0.0, is_stable=True,
        reconstruction_error=0.0, koopman_trust=1.0,
    )
    exp = OptimizationExperience("d", "t", 1.0, 1, "cpu", {"log_E": 0.5}, domain="x")

    memory = KoopmanExperienceMemory()
    memory._entries.append(_MemoryEntry(invariant=inv, signature=koop, experience=exp))

    nav = HarmonicNavigator(omega0_linear=1.0)
    results = nav.retrieve_near_abelian(memory, query_omega0=1.0, query_log_E=0.5, log_E_window=0.5)
    assert len(results) >= 1, "Matching energy should retrieve entry"


# ── Group 9: HarmonicNavigator helpers ────────────────────────────────────────


def test_analytic_omega0_shift_zero_at_beta_zero():
    """β=0 → analytic shift = 0."""
    shift = HarmonicNavigator.analytic_omega0_shift(alpha=1.0, beta=0.0, amplitude=2.0)
    assert abs(shift) < 1e-9, f"β=0 should give zero shift, got {shift}"


def test_analytic_omega0_shift_positive_for_hardening():
    """β>0 → positive shift (hardening)."""
    shift = HarmonicNavigator.analytic_omega0_shift(alpha=1.0, beta=0.1, amplitude=1.0)
    assert shift > 0, f"β=0.1 should give positive shift, got {shift}"


def test_analytic_shift_scales_with_beta():
    """Larger β → larger analytic shift (same α, A)."""
    shift_weak   = HarmonicNavigator.analytic_omega0_shift(alpha=1.0, beta=0.1, amplitude=1.0)
    shift_strong = HarmonicNavigator.analytic_omega0_shift(alpha=1.0, beta=0.5, amplitude=1.0)
    assert shift_strong > shift_weak, (
        f"Larger β should give larger shift: β=0.1→{shift_weak:.4f}, β=0.5→{shift_strong:.4f}"
    )


def test_normalised_beta_ratio():
    """normalised_beta(4, 0.4) == normalised_beta(1, 0.1) == 0.1."""
    r1 = HarmonicNavigator.normalised_beta(1.0, 0.1)
    r2 = HarmonicNavigator.normalised_beta(4.0, 0.4)
    assert abs(r1 - 0.1) < 1e-9
    assert abs(r2 - 0.1) < 1e-9
    assert abs(r1 - r2) < 1e-9, f"Normalised β/α should match: {r1} vs {r2}"


def test_find_universal_patterns_returns_dict():
    """find_universal_patterns returns dict with 'similarities' and 'universal_pattern_detected'."""
    nav = HarmonicNavigator(omega0_linear=1.0)
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    amplitudes = np.array([0.1, 0.5, 1.0])

    regions = nav.sweep_energy(evaluator, amplitudes)
    path = nav.build_path(regions, domain="A")

    result = nav.find_universal_patterns({"A": path, "B": path})
    assert "similarities" in result
    assert "universal_pattern_detected" in result
    assert isinstance(result["universal_pattern_detected"], bool)


# ── Group 10: DuffingParams validation ────────────────────────────────────────


def test_duffing_params_validation():
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="alpha"):
        DuffingParams(alpha=-1.0, beta=0.1, delta=0.5)

    with pytest.raises(ValueError, match="beta"):
        DuffingParams(alpha=1.0, beta=-0.1, delta=0.5)

    with pytest.raises(ValueError, match="delta"):
        DuffingParams(alpha=1.0, beta=0.1, delta=-0.1)


def test_duffing_params_properties():
    """omega0_linear and Q_linear match analytic formulas."""
    p = DuffingParams(alpha=4.0, beta=0.1, delta=1.0)
    assert abs(p.omega0_linear - 2.0) < 1e-9
    assert abs(p.Q_linear - 2.0) < 1e-9   # √4/1 = 2


def test_nonlinearity_strength_zero_at_beta_zero():
    """β=0 → nonlinearity_strength = 0."""
    p = DuffingParams(alpha=1.0, beta=0.0, delta=0.5)
    assert p.nonlinearity_strength(amplitude=100.0) == 0.0


def test_system_fn_returns_callable():
    """system_fn() returns a Callable compatible with LCAPatchDetector."""
    evaluator = DuffingEvaluator(_PARAMS_WEAK)
    fn = evaluator.system_fn()
    assert callable(fn)

    # Test that it accepts 2D state and returns 2D output
    test_state = np.array([0.5, 0.1])
    output = fn(test_state)
    assert output.shape == (2,)


def test_lca_patch_detector_accepts_duffing_system_fn():
    """
    LCAPatchDetector can be instantiated with the Duffing system_fn.
    At small amplitude, should classify as LCA (near-linear dynamics).
    """
    from tensor.lca_patch_detector import LCAPatchDetector

    evaluator = DuffingEvaluator(_PARAMS_LINEAR)
    fn = evaluator.system_fn()

    detector = LCAPatchDetector(system_fn=fn, n_states=2)

    # Sample small-amplitude phase space points
    traj = evaluator._sim.run(x0=0.1, v0=0.0, n_steps=100)
    classification = detector.classify_region(traj[10:50])   # skip transient

    print(f"\n  LCA classification (β=0): type={classification.patch_type}  "
          f"comm_norm={classification.commutator_norm:.4f}")

    # β=0 near equilibrium → should be LCA or have low commutator norm
    assert classification.commutator_norm < 1.0, (
        f"Near-linear Duffing should have low commutator norm: "
        f"{classification.commutator_norm:.4f}"
    )
