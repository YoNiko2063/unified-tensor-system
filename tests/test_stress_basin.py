"""
Basin Stability Validation

Quantitative validation of Monte Carlo stability basins:
  - Boundary messiness (too-clean basins are suspicious)
  - Spectral gap collapse regions
  - Chaotic boundary shape metrics
  - Regime distribution realism
  - Eigenvalue spread statistics
  - Sensitivity to tolerance parameter
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from optimization.circuit_optimizer import (
    CircuitSpec,
    EigenvalueMapper,
    CircuitOptimizer,
    MonteCarloStabilityAnalyzer,
    OptimizationResult,
    BasinResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mapper():
    return EigenvalueMapper()


def _make_result(mapper, spec):
    """Build an OptimizationResult from analytic inverse map."""
    inv = mapper.inverse_map(spec.target_eigenvalues)
    eigs = mapper.eigenvalues(inv["R"], inv["L"], inv["C"])
    sigma = float(np.real(eigs[0]))
    omega_d = abs(float(np.imag(eigs[0])))
    omega0 = float(np.sqrt(sigma**2 + omega_d**2))
    zeta = abs(sigma) / omega0 if omega0 > 0 else 0.0
    Q = 1.0 / (2.0 * max(zeta, 1e-12))
    gap = abs(abs(eigs[0]) - abs(eigs[1]))
    return OptimizationResult(
        R=inv["R"], L=inv["L"], C=inv["C"],
        achieved_eigenvalues=eigs,
        eigenvalue_error=0.0, cost=0.0,
        regime_type="lca", spectral_gap=float(gap),
        omega0_achieved=omega0, Q_achieved=Q, converged=True,
    )


# ---------------------------------------------------------------------------
# 1. Boundary messiness — basins should NOT be too clean
# ---------------------------------------------------------------------------

class TestBoundaryMessiness:

    def test_wide_tolerance_passive_rlc_all_lca(self, mapper):
        """Passive RLC with positive R/L/C is always stable → 100% LCA is correct.

        Regime diversity only appears in active/nonlinear circuits.
        For passive RLC, verify instead that eigenvalue error spread shows
        the tolerance effect (boundary visible in error, not regime).
        """
        spec = CircuitSpec(
            center_freq_hz=1000.0,
            Q_target=2.0,
            component_tolerances={"R": 0.30, "L": 0.30, "C": 0.30},
        )
        result = _make_result(mapper, spec)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=500, seed=42)
        basin = analyzer.analyze(result, spec)

        # Physically correct: passive RLC → 100% LCA
        assert basin.lca_fraction == 1.0, \
            "Passive RLC with positive components must always be LCA"
        # But the boundary IS visible in eigenvalue error distribution
        errors = [s["eig_error"] for s in basin.samples]
        error_std = np.std(errors)
        assert error_std > 0.0, \
            "Zero error variance with ±30% tolerance — tolerance not applied"
        # Wide tolerance should produce significant error variation
        assert basin.worst_case_error > 10.0, \
            f"Worst-case error suspiciously small: {basin.worst_case_error:.3f}"

    def test_eigenvalue_spread_nonzero(self, mapper):
        """Eigenvalue spread must be > 0 for any nonzero tolerance."""
        spec = CircuitSpec(
            center_freq_hz=1000.0,
            Q_target=5.0,
            component_tolerances={"R": 0.05, "L": 0.10, "C": 0.05},
        )
        result = _make_result(mapper, spec)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=200, seed=42)
        basin = analyzer.analyze(result, spec)

        assert basin.mean_eigenvalue_spread > 0.0, \
            "Zero eigenvalue spread implies no perturbation effect"

    def test_worst_case_error_larger_with_wider_tolerance(self, mapper):
        """Wider tolerances should produce larger worst-case errors."""
        spec_tight = CircuitSpec(
            center_freq_hz=1000.0, Q_target=5.0,
            component_tolerances={"R": 0.01, "L": 0.01, "C": 0.01},
        )
        spec_wide = CircuitSpec(
            center_freq_hz=1000.0, Q_target=5.0,
            component_tolerances={"R": 0.20, "L": 0.20, "C": 0.20},
        )
        result = _make_result(mapper, spec_tight)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=200, seed=42)

        basin_tight = analyzer.analyze(result, spec_tight)
        basin_wide = analyzer.analyze(result, spec_wide)

        assert basin_wide.worst_case_error > basin_tight.worst_case_error, \
            "Wider tolerance should produce larger worst-case error"


# ---------------------------------------------------------------------------
# 2. Spectral gap collapse regions
# ---------------------------------------------------------------------------

class TestSpectralGapCollapse:

    def test_near_critical_damping_gap_collapses(self, mapper):
        """Near ζ = 1, eigenvalue magnitudes merge — spectral gap → 0."""
        spec = CircuitSpec(
            center_freq_hz=1000.0,
            Q_target=0.51,  # just barely underdamped
            component_tolerances={"R": 0.05, "L": 0.05, "C": 0.05},
        )
        result = _make_result(mapper, spec)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=200, seed=42)
        basin = analyzer.analyze(result, spec)

        # Near critical damping, eigenvalue magnitudes are nearly equal
        # So spread in magnitudes should be small
        assert basin.mean_eigenvalue_spread < spec.omega0, \
            f"Spread {basin.mean_eigenvalue_spread:.1f} too large near critical damping"

    @pytest.mark.parametrize("Q_target", [0.3, 0.5, 1.0, 5.0, 20.0])
    def test_basin_counts_consistent(self, mapper, Q_target):
        """n_lca + n_nonabelian + n_chaotic == n_samples for all Q values."""
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=Q_target)
        result = _make_result(mapper, spec)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin = analyzer.analyze(result, spec)

        total = basin.n_lca + basin.n_nonabelian + basin.n_chaotic
        assert total == basin.n_samples, \
            f"Count mismatch at Q={Q_target}: {total} != {basin.n_samples}"


# ---------------------------------------------------------------------------
# 3. Regime distribution realism
# ---------------------------------------------------------------------------

class TestRegimeDistribution:

    def test_high_Q_mostly_lca(self, mapper):
        """High Q (underdamped, stable) should be overwhelmingly LCA."""
        spec = CircuitSpec(
            center_freq_hz=1000.0, Q_target=10.0,
            component_tolerances={"R": 0.05, "L": 0.10, "C": 0.05},
        )
        result = _make_result(mapper, spec)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=500, seed=42)
        basin = analyzer.analyze(result, spec)

        assert basin.lca_fraction > 0.90, \
            f"Expected >90% LCA for Q=10, got {basin.lca_fraction:.1%}"

    def test_lca_fraction_monotonic_with_Q(self, mapper):
        """LCA fraction should generally increase as Q increases (more stable)."""
        fractions = []
        for Q in [1.0, 3.0, 5.0, 10.0]:
            spec = CircuitSpec(
                center_freq_hz=1000.0, Q_target=Q,
                component_tolerances={"R": 0.10, "L": 0.10, "C": 0.10},
            )
            result = _make_result(mapper, spec)
            analyzer = MonteCarloStabilityAnalyzer(n_samples=200, seed=42)
            basin = analyzer.analyze(result, spec)
            fractions.append(basin.lca_fraction)

        # Allow some non-monotonicity due to sampling, but overall trend should hold
        # Check: last Q should have higher LCA fraction than first Q
        assert fractions[-1] >= fractions[0] - 0.05, \
            f"LCA fraction did not increase with Q: {fractions}"

    def test_all_regimes_have_valid_strings(self, mapper):
        """Every sample must have a valid regime string."""
        spec = CircuitSpec(
            center_freq_hz=1000.0, Q_target=2.0,
            component_tolerances={"R": 0.20, "L": 0.20, "C": 0.20},
        )
        result = _make_result(mapper, spec)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=200, seed=42)
        basin = analyzer.analyze(result, spec)

        valid_regimes = {"lca", "nonabelian", "chaotic"}
        for s in basin.samples:
            assert s["regime"] in valid_regimes, f"Invalid regime: {s['regime']}"


# ---------------------------------------------------------------------------
# 4. Tolerance sensitivity analysis
# ---------------------------------------------------------------------------

class TestToleranceSensitivity:

    def test_zero_tolerance_all_identical(self, mapper):
        """With zero tolerance, all samples should be identical to nominal."""
        spec = CircuitSpec(
            center_freq_hz=1000.0, Q_target=5.0,
            component_tolerances={"R": 0.0, "L": 0.0, "C": 0.0},
        )
        result = _make_result(mapper, spec)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=50, seed=42)
        basin = analyzer.analyze(result, spec)

        # All samples should be identical to nominal
        for s in basin.samples:
            assert abs(s["R"] - result.R) < 1e-9
            assert abs(s["L"] - result.L) < 1e-9
            assert abs(s["C"] - result.C) < 1e-9

        # All should be same regime
        assert basin.lca_fraction == 1.0
        assert basin.worst_case_error < 1e-9

    def test_spread_increases_with_tolerance(self, mapper):
        """Eigenvalue spread should increase as tolerance widens."""
        spreads = []
        for tol in [0.01, 0.05, 0.10, 0.20]:
            spec = CircuitSpec(
                center_freq_hz=1000.0, Q_target=5.0,
                component_tolerances={"R": tol, "L": tol, "C": tol},
            )
            result = _make_result(mapper, spec)
            analyzer = MonteCarloStabilityAnalyzer(n_samples=200, seed=42)
            basin = analyzer.analyze(result, spec)
            spreads.append(basin.mean_eigenvalue_spread)

        # Spreads should be monotonically increasing
        for i in range(1, len(spreads)):
            assert spreads[i] >= spreads[i-1] - 1e-6, \
                f"Spread decreased: {spreads}"

    def test_reproducible_with_same_seed(self, mapper):
        """Same seed must produce identical basin results."""
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
        result = _make_result(mapper, spec)

        analyzer1 = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin1 = analyzer1.analyze(result, spec)

        analyzer2 = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin2 = analyzer2.analyze(result, spec)

        assert basin1.n_lca == basin2.n_lca
        assert basin1.lca_fraction == basin2.lca_fraction
        assert basin1.worst_case_error == basin2.worst_case_error

    def test_different_seeds_different_results(self, mapper):
        """Different seeds should produce different sample sets."""
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
        result = _make_result(mapper, spec)

        analyzer1 = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin1 = analyzer1.analyze(result, spec)

        analyzer2 = MonteCarloStabilityAnalyzer(n_samples=100, seed=99)
        basin2 = analyzer2.analyze(result, spec)

        # Samples should differ (compare first sample R values)
        assert basin1.samples[0]["R"] != basin2.samples[0]["R"]


# ---------------------------------------------------------------------------
# 5. Cross-frequency basin comparison
# ---------------------------------------------------------------------------

class TestCrossFrequencyBasin:

    @pytest.mark.parametrize("freq_hz", [100, 1000, 10000])
    def test_basin_valid_across_frequencies(self, mapper, freq_hz):
        """Basin analysis must produce valid results across frequency range."""
        spec = CircuitSpec(
            center_freq_hz=freq_hz, Q_target=5.0,
            component_tolerances={"R": 0.05, "L": 0.10, "C": 0.05},
        )
        result = _make_result(mapper, spec)
        analyzer = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin = analyzer.analyze(result, spec)

        assert basin.n_samples == 100
        assert 0.0 <= basin.lca_fraction <= 1.0
        assert basin.mean_eigenvalue_spread >= 0.0
        assert np.isfinite(basin.worst_case_error)
        # Stable high-Q circuit should be mostly LCA regardless of frequency
        assert basin.lca_fraction > 0.5
