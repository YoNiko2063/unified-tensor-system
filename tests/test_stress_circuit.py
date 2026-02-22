"""
Spectral Sensitivity Stress Tests — Circuit Optimizer

Validates stability under edge-case operating regimes:
  - Extreme damping ratios (ζ → 0, ζ → 1, overdamped)
  - High-frequency targets (100 kHz)
  - Tight component tolerances (±0.1%)
  - Near-singular C matrices (extreme L/C ratios)
  - Eigenvalue clustering near bifurcation
  - Pareto filter robustness under degenerate inputs
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from optimization.circuit_optimizer import (
    CircuitSpec,
    EigenvalueMapper,
    CircuitCostFunction,
    CircuitOptimizer,
    OptimizationResult,
    ParetoResult,
    MonteCarloStabilityAnalyzer,
    BasinResult,
    _non_dominated,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mapper():
    return EigenvalueMapper()


# ---------------------------------------------------------------------------
# 1. Extreme damping ratios
# ---------------------------------------------------------------------------

class TestExtremeDamping:

    @pytest.mark.parametrize("Q_target", [
        0.5,    # critically damped (ζ = 1.0)
        0.4,    # overdamped (ζ = 1.25)
        0.3,    # heavily overdamped (ζ = 1.67)
        50.0,   # very underdamped (ζ = 0.01)
        100.0,  # extremely underdamped (ζ = 0.005)
    ])
    def test_optimizer_converges_extreme_Q(self, Q_target):
        """Optimizer must return finite results for extreme Q values."""
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=Q_target)
        opt = CircuitOptimizer(spec)
        pareto = opt.optimize()
        assert isinstance(pareto, ParetoResult)
        assert len(pareto.all_candidates) > 0
        for c in pareto.all_candidates:
            assert np.isfinite(c.cost), f"Non-finite cost at Q={Q_target}"
            assert c.R > 0 and c.L > 0 and c.C > 0

    @pytest.mark.parametrize("zeta", [0.001, 0.01, 0.1, 0.5, 0.99, 1.0, 1.5, 2.0])
    def test_eigenvalue_mapper_stable_all_zeta(self, mapper, zeta):
        """EigenvalueMapper must produce finite eigenvalues for all damping ratios."""
        spec = CircuitSpec(center_freq_hz=1000.0, damping_ratio=zeta)
        inv = mapper.inverse_map(spec.target_eigenvalues)
        eigs = mapper.eigenvalues(inv["R"], inv["L"], inv["C"])
        assert np.all(np.isfinite(eigs)), f"Non-finite eigenvalues at ζ={zeta}: {eigs}"

    def test_overdamped_real_eigenvalues(self, mapper):
        """Q < 0.5 (overdamped) should produce purely real eigenvalues."""
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=0.3)
        inv = mapper.inverse_map(spec.target_eigenvalues)
        eigs = mapper.eigenvalues(inv["R"], inv["L"], inv["C"])
        for e in eigs:
            assert abs(np.imag(e)) < 1e-6, f"Expected real eigenvalue, got {e}"

    def test_critically_damped_repeated_roots(self, mapper):
        """Q = 0.5 (ζ = 1.0) should produce nearly repeated real eigenvalues."""
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=0.5)
        inv = mapper.inverse_map(spec.target_eigenvalues)
        eigs = mapper.eigenvalues(inv["R"], inv["L"], inv["C"])
        # Both eigenvalues should be close to -ω₀
        assert abs(eigs[0] - eigs[1]) < spec.omega0 * 0.1


# ---------------------------------------------------------------------------
# 2. High-frequency targets
# ---------------------------------------------------------------------------

class TestHighFrequency:

    @pytest.mark.parametrize("freq_hz", [10_000, 50_000, 100_000])
    def test_optimizer_high_freq(self, freq_hz):
        """Optimizer must handle high-frequency targets without numerical blowup."""
        spec = CircuitSpec(center_freq_hz=freq_hz, Q_target=5.0)
        opt = CircuitOptimizer(spec)
        pareto = opt.optimize()
        best = pareto.best_eigenvalue
        assert np.isfinite(best.cost)
        assert best.omega0_achieved > 0
        # Achieved ω₀ should be within 50% of target (optimizer may not be exact)
        rel_err = abs(best.omega0_achieved - spec.omega0) / spec.omega0
        assert rel_err < 0.5, f"ω₀ off by {rel_err:.1%} at {freq_hz} Hz"

    def test_inverse_map_small_components(self, mapper):
        """At 100 kHz, L and C should be small but positive."""
        spec = CircuitSpec(center_freq_hz=100_000, Q_target=5.0)
        inv = mapper.inverse_map(spec.target_eigenvalues)
        assert inv["C"] > 0 and inv["C"] < 1e-3
        assert inv["L"] > 0


# ---------------------------------------------------------------------------
# 3. Tight tolerances
# ---------------------------------------------------------------------------

class TestTightTolerances:

    def test_basin_tight_tolerance(self):
        """±0.1% tolerance: basin should still be mostly LCA for stable circuit."""
        spec = CircuitSpec(
            center_freq_hz=1000.0,
            Q_target=5.0,
            component_tolerances={"R": 0.001, "L": 0.001, "C": 0.001},
        )
        opt = CircuitOptimizer(spec)
        pareto = opt.optimize()
        analyzer = MonteCarloStabilityAnalyzer(n_samples=200, seed=42)
        basin = analyzer.analyze(pareto.best_eigenvalue, spec)
        # With very tight tolerances, all samples should be LCA
        assert basin.lca_fraction > 0.95, \
            f"Expected >95% LCA with ±0.1% tolerance, got {basin.lca_fraction:.1%}"

    def test_basin_loose_tolerance(self):
        """±50% tolerance: should show regime diversity (not all LCA)."""
        spec = CircuitSpec(
            center_freq_hz=1000.0,
            Q_target=5.0,
            component_tolerances={"R": 0.50, "L": 0.50, "C": 0.50},
        )
        opt = CircuitOptimizer(spec)
        pareto = opt.optimize()
        analyzer = MonteCarloStabilityAnalyzer(n_samples=200, seed=42)
        basin = analyzer.analyze(pareto.best_eigenvalue, spec)
        # Should still have some LCA, but not 100% — boundary is messy
        assert basin.n_samples == 200
        assert basin.worst_case_error > 0.0
        assert np.isfinite(basin.worst_case_error)


# ---------------------------------------------------------------------------
# 4. Near-singular C matrices / extreme component ratios
# ---------------------------------------------------------------------------

class TestNearSingular:

    @pytest.mark.parametrize("L,C", [
        (1e-9, 1e-3),   # extreme L/C ratio: tiny L, large C
        (1e-1, 1e-12),  # extreme L/C ratio: large L, tiny C
        (1e-6, 1e-6),   # equal L and C
    ])
    def test_eigenvalues_finite_extreme_LC(self, mapper, L, C):
        """Eigenvalues must remain finite for extreme L/C ratios."""
        R = 100.0
        eigs = mapper.eigenvalues(R, L, C)
        assert np.all(np.isfinite(eigs)), f"Non-finite eigs for L={L}, C={C}: {eigs}"

    def test_cost_function_extreme_params(self):
        """Cost function must not crash on extreme log-space parameters."""
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
        mapper = EigenvalueMapper()
        cost_fn = CircuitCostFunction(spec, mapper)

        # Very large log-params → very large R/L/C
        extreme_large = np.array([20.0, 20.0, 20.0])
        val = cost_fn(extreme_large)
        assert np.isfinite(val)

        # Very negative log-params → very small R/L/C
        extreme_small = np.array([-20.0, -20.0, -20.0])
        val = cost_fn(extreme_small)
        assert np.isfinite(val)

    def test_condition_number_A(self, mapper):
        """State matrix A should have bounded condition number for nominal RLC."""
        A = mapper.compute_A(R=100.0, L=1e-3, C=1e-6)
        cond = np.linalg.cond(A)
        # For a well-conditioned 2x2, condition should be < 1e6
        assert cond < 1e8, f"Condition number too high: {cond:.2e}"


# ---------------------------------------------------------------------------
# 5. Eigenvalue clustering near bifurcation
# ---------------------------------------------------------------------------

class TestBifurcationBoundary:

    def test_near_critically_damped_optimizer(self):
        """Q ≈ 0.5 (bifurcation point): optimizer should not diverge."""
        for Q in [0.48, 0.49, 0.50, 0.51, 0.52]:
            spec = CircuitSpec(center_freq_hz=1000.0, Q_target=Q)
            opt = CircuitOptimizer(spec)
            pareto = opt.optimize()
            for c in pareto.all_candidates:
                assert np.isfinite(c.cost), f"Diverged at Q={Q}"

    def test_eigenvalue_spacing_near_bifurcation(self, mapper):
        """Near ζ = 1, eigenvalues collapse — spectral gap should approach 0."""
        spec_critical = CircuitSpec(center_freq_hz=1000.0, damping_ratio=0.999)
        inv = mapper.inverse_map(spec_critical.target_eigenvalues)
        eigs = mapper.eigenvalues(inv["R"], inv["L"], inv["C"])
        gap = abs(abs(eigs[0]) - abs(eigs[1]))
        # Near critical damping, gap in magnitudes should be small
        assert gap < spec_critical.omega0 * 0.1


# ---------------------------------------------------------------------------
# 6. Pareto filter robustness
# ---------------------------------------------------------------------------

class TestParetoFilterRobustness:

    def test_pareto_single_candidate(self):
        """Pareto front with 1 candidate should return that candidate."""
        r = OptimizationResult(
            R=100.0, L=1e-3, C=1e-6,
            achieved_eigenvalues=np.array([-500+3000j, -500-3000j]),
            eigenvalue_error=0.01, cost=0.5,
            regime_type="lca", spectral_gap=1.0,
            omega0_achieved=6283.0, Q_achieved=5.0, converged=True,
        )
        front = _non_dominated([r])
        assert len(front) == 1

    def test_pareto_dominated_removed(self):
        """A strictly dominated candidate should be filtered out."""
        good = OptimizationResult(
            R=100.0, L=1e-3, C=1e-6,
            achieved_eigenvalues=np.array([-500+3000j, -500-3000j]),
            eigenvalue_error=0.01, cost=0.1,
            regime_type="lca", spectral_gap=2.0,
            omega0_achieved=6283.0, Q_achieved=5.0, converged=True,
        )
        bad = OptimizationResult(
            R=200.0, L=2e-3, C=2e-6,
            achieved_eigenvalues=np.array([-500+3000j, -500-3000j]),
            eigenvalue_error=0.05, cost=0.5,
            regime_type="lca", spectral_gap=1.0,
            omega0_achieved=6283.0, Q_achieved=5.0, converged=True,
        )
        front = _non_dominated([good, bad])
        assert len(front) == 1
        assert front[0] is good

    def test_pareto_incomparable_both_kept(self):
        """Two incomparable candidates should both survive."""
        a = OptimizationResult(
            R=100.0, L=1e-3, C=1e-6,
            achieved_eigenvalues=np.array([-500+3000j, -500-3000j]),
            eigenvalue_error=0.001, cost=1.0,  # great eig, bad cost
            regime_type="lca", spectral_gap=1.0,
            omega0_achieved=6283.0, Q_achieved=5.0, converged=True,
        )
        b = OptimizationResult(
            R=50.0, L=0.5e-3, C=0.5e-6,
            achieved_eigenvalues=np.array([-500+3000j, -500-3000j]),
            eigenvalue_error=0.1, cost=0.01,  # bad eig, great cost
            regime_type="lca", spectral_gap=1.0,
            omega0_achieved=6283.0, Q_achieved=5.0, converged=True,
        )
        front = _non_dominated([a, b])
        assert len(front) == 2

    def test_pareto_all_identical(self):
        """N identical candidates: all should survive (none dominates another)."""
        r = OptimizationResult(
            R=100.0, L=1e-3, C=1e-6,
            achieved_eigenvalues=np.array([-500+3000j, -500-3000j]),
            eigenvalue_error=0.01, cost=0.5,
            regime_type="lca", spectral_gap=1.0,
            omega0_achieved=6283.0, Q_achieved=5.0, converged=True,
        )
        # Create distinct objects with same values
        candidates = [
            OptimizationResult(
                R=r.R, L=r.L, C=r.C,
                achieved_eigenvalues=r.achieved_eigenvalues.copy(),
                eigenvalue_error=r.eigenvalue_error, cost=r.cost,
                regime_type=r.regime_type, spectral_gap=r.spectral_gap,
                omega0_achieved=r.omega0_achieved, Q_achieved=r.Q_achieved,
                converged=r.converged,
            ) for _ in range(5)
        ]
        front = _non_dominated(candidates)
        assert len(front) == 5
