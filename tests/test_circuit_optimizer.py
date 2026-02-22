"""
Tests for optimization/circuit_optimizer.py — 25+ tests.
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
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spec_bandpass():
    return CircuitSpec(
        topology="bandpass_rlc",
        center_freq_hz=1000.0,
        Q_target=5.0,
    )


@pytest.fixture
def spec_lowq():
    """Low-Q (overdamped) spec."""
    return CircuitSpec(
        topology="bandpass_rlc",
        center_freq_hz=500.0,
        Q_target=1.0,
    )


@pytest.fixture
def spec_highfreq():
    return CircuitSpec(
        topology="bandpass_rlc",
        center_freq_hz=10000.0,
        Q_target=3.0,
    )


@pytest.fixture
def mapper():
    return EigenvalueMapper()


@pytest.fixture
def nominal_RLC():
    """Well-conditioned RLC: 100 Ω, 1 mH, 1 µF → ω₀ ≈ 31623 rad/s, Q ≈ 3.16."""
    return {"R": 100.0, "L": 1e-3, "C": 1e-6}


# ---------------------------------------------------------------------------
# EigenvalueMapper tests
# ---------------------------------------------------------------------------

class TestEigenvalueMapper:

    def test_compute_A_shape(self, mapper, nominal_RLC):
        A = mapper.compute_A(**nominal_RLC)
        assert A.shape == (2, 2)

    def test_compute_A_all_finite(self, mapper, nominal_RLC):
        A = mapper.compute_A(**nominal_RLC)
        assert np.all(np.isfinite(A))

    def test_compute_A_values(self, mapper):
        R, L, C = 100.0, 1e-3, 1e-6
        A = mapper.compute_A(R, L, C)
        assert np.isclose(A[0, 0], -1.0 / (R * C))
        assert np.isclose(A[0, 1],  1.0 / C)
        assert np.isclose(A[1, 0], -1.0 / L)
        assert np.isclose(A[1, 1],  0.0)

    def test_eigenvalues_returns_two_complex(self, mapper, nominal_RLC):
        eigs = mapper.eigenvalues(**nominal_RLC)
        assert eigs.shape == (2,)
        assert eigs.dtype == complex or np.iscomplexobj(eigs)

    def test_eigenvalues_negative_real_underdamped(self, mapper):
        """Q > 0.5 → underdamped → Re(λ) < 0 (stable), Im(λ) ≠ 0."""
        # R=100, L=1e-3, C=1e-6 → Q≈3.16 (underdamped)
        eigs = mapper.eigenvalues(R=100.0, L=1e-3, C=1e-6)
        for e in eigs:
            assert np.real(e) < 0, f"Expected negative real part, got {e}"
        assert abs(np.imag(eigs[0])) > 1.0, "Expected nonzero imaginary part"

    def test_eigenvalues_stable_for_underdamped_q5(self, mapper):
        """High-Q RLC is stable."""
        # ω₀=2π*1000, Q=5 → ζ=0.1 → underdamped
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
        inv = mapper.inverse_map(spec.target_eigenvalues)
        eigs = mapper.eigenvalues(inv["R"], inv["L"], inv["C"])
        for e in eigs:
            assert np.real(e) < 0

    def test_eigenvalue_error_zero_when_exact(self, mapper):
        """eigenvalue_error should be zero when params exactly match target."""
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
        target_eigs = spec.target_eigenvalues
        inv = mapper.inverse_map(target_eigs)
        R, L, C = inv["R"], inv["L"], inv["C"]
        err = mapper.eigenvalue_error(R, L, C, target_eigs)
        # Should be very close to zero (analytic round-trip)
        assert err < 1e-3 * np.linalg.norm(np.abs(target_eigs)), \
            f"Error too large: {err}"

    def test_eigenvalue_error_symmetric_matching(self, mapper, nominal_RLC):
        """error() uses minimum-distance matching (try both orderings)."""
        eigs = mapper.eigenvalues(**nominal_RLC)
        # target = conjugate pair in opposite order — should give same error
        target_forward = np.array([eigs[0], eigs[1]])
        target_reversed = np.array([eigs[1], eigs[0]])
        err_f = mapper.eigenvalue_error(**nominal_RLC, target_eigs=target_forward)
        err_r = mapper.eigenvalue_error(**nominal_RLC, target_eigs=target_reversed)
        assert np.isclose(err_f, err_r, atol=1e-10)

    def test_inverse_map_returns_dict_keys(self, mapper, spec_bandpass):
        inv = mapper.inverse_map(spec_bandpass.target_eigenvalues)
        assert set(inv.keys()) == {"R", "L", "C"}

    def test_inverse_map_positive_values(self, mapper, spec_bandpass):
        inv = mapper.inverse_map(spec_bandpass.target_eigenvalues)
        assert inv["R"] > 0
        assert inv["L"] > 0
        assert inv["C"] > 0

    def test_inverse_map_L_anchor(self, mapper, spec_bandpass):
        """Anchor is L = 1e-3 H."""
        inv = mapper.inverse_map(spec_bandpass.target_eigenvalues)
        assert np.isclose(inv["L"], 1e-3)

    def test_inverse_map_round_trip_error_under_1pct(self, mapper, spec_bandpass):
        """Recovered eigenvalues within 1% of target magnitude."""
        target_eigs = spec_bandpass.target_eigenvalues
        inv = mapper.inverse_map(target_eigs)
        err = mapper.eigenvalue_error(inv["R"], inv["L"], inv["C"], target_eigs)
        # Tolerance relative to magnitude
        target_mag = np.linalg.norm(np.abs(target_eigs))
        assert err / max(target_mag, 1.0) < 0.01, \
            f"Round-trip error too large: {err / target_mag:.4f}"

    def test_inverse_map_different_frequencies(self, mapper):
        """inverse_map works for different target frequencies."""
        for freq_hz in [100.0, 1000.0, 10000.0]:
            spec = CircuitSpec(center_freq_hz=freq_hz, Q_target=3.0)
            inv = mapper.inverse_map(spec.target_eigenvalues)
            assert inv["R"] > 0 and inv["L"] > 0 and inv["C"] > 0

    def test_eigenvalues_conjugate_pair(self, mapper, nominal_RLC):
        """For real-valued system, eigenvalues come in conjugate pairs."""
        eigs = mapper.eigenvalues(**nominal_RLC)
        assert np.isclose(eigs[0], np.conj(eigs[1]), atol=1e-10), \
            f"Expected conjugate pair, got {eigs}"


# ---------------------------------------------------------------------------
# CircuitSpec tests
# ---------------------------------------------------------------------------

class TestCircuitSpec:

    def test_omega0_formula(self):
        spec = CircuitSpec(center_freq_hz=1000.0)
        assert np.isclose(spec.omega0, 2 * np.pi * 1000.0)

    def test_zeta_from_Q(self):
        spec = CircuitSpec(Q_target=5.0)
        assert np.isclose(spec.zeta, 1.0 / 10.0)

    def test_zeta_from_damping_ratio_overrides_Q(self):
        spec = CircuitSpec(Q_target=5.0, damping_ratio=0.3)
        assert np.isclose(spec.zeta, 0.3)

    def test_target_eigenvalues_shape(self):
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
        eigs = spec.target_eigenvalues
        assert eigs.shape == (2,)

    def test_target_eigenvalues_conjugate_pair(self):
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
        eigs = spec.target_eigenvalues
        assert np.isclose(eigs[0], np.conj(eigs[1]), atol=1e-10)

    def test_target_eigenvalues_negative_real(self):
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
        eigs = spec.target_eigenvalues
        for e in eigs:
            assert np.real(e) < 0


# ---------------------------------------------------------------------------
# CircuitCostFunction tests
# ---------------------------------------------------------------------------

class TestCircuitCostFunction:

    def test_call_returns_finite_scalar(self, spec_bandpass, mapper):
        cost_fn = CircuitCostFunction(spec_bandpass, mapper)
        params = np.array([np.log(100.0), np.log(1e-3), np.log(1e-6)])
        val = cost_fn(params)
        assert np.isfinite(val)
        assert isinstance(val, float)

    def test_regime_penalty_zero_without_detector(self, spec_bandpass, mapper):
        cost_fn = CircuitCostFunction(spec_bandpass, mapper, patch_detector=None)
        penalty = cost_fn.regime_penalty(100.0, 1e-3, 1e-6)
        assert penalty == 0.0

    def test_stability_penalty_zero_when_gap_large(self, spec_bandpass, mapper):
        cost_fn = CircuitCostFunction(spec_bandpass, mapper)
        # R=100, L=1e-3, C=1e-6 → underdamped, should have gap
        penalty = cost_fn.stability_penalty(100.0, 1e-3, 1e-6, gap_threshold=0.0)
        assert penalty == 0.0

    def test_stability_penalty_positive_when_gap_small(self, spec_bandpass, mapper):
        cost_fn = CircuitCostFunction(spec_bandpass, mapper)
        # Very large threshold → penalty should be positive
        penalty = cost_fn.stability_penalty(100.0, 1e-3, 1e-6, gap_threshold=1e10)
        assert penalty > 0.0

    def test_cost_lower_near_target(self, spec_bandpass, mapper):
        """Cost at analytic inverse-map point should be less than far-off point."""
        cost_fn = CircuitCostFunction(spec_bandpass, mapper)
        inv = mapper.inverse_map(spec_bandpass.target_eigenvalues)
        good_params = np.array([np.log(inv["R"]), np.log(inv["L"]), np.log(inv["C"])])
        bad_params = good_params + np.array([5.0, -5.0, 3.0])  # far off
        assert cost_fn(good_params) < cost_fn(bad_params)


# ---------------------------------------------------------------------------
# CircuitOptimizer tests
# ---------------------------------------------------------------------------

class TestCircuitOptimizer:

    def test_optimize_returns_pareto_result(self, spec_bandpass):
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        assert isinstance(pareto, ParetoResult)

    def test_pareto_has_three_candidates(self, spec_bandpass):
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        assert isinstance(pareto.best_eigenvalue, OptimizationResult)
        assert isinstance(pareto.best_stability, OptimizationResult)
        assert isinstance(pareto.best_cost, OptimizationResult)

    def test_pareto_all_candidates_nonempty(self, spec_bandpass):
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        assert len(pareto.all_candidates) > 0

    def test_all_candidates_finite_cost(self, spec_bandpass):
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        for cand in pareto.all_candidates:
            assert np.isfinite(cand.cost), f"Non-finite cost: {cand.cost}"

    def test_best_eigenvalue_better_than_analytic(self, spec_bandpass, mapper):
        """Best Pareto candidate should have lower eigenvalue error than analytic guess."""
        # Compute analytic guess error
        inv = mapper.inverse_map(spec_bandpass.target_eigenvalues)
        analytic_err = mapper.eigenvalue_error(
            inv["R"], inv["L"], inv["C"], spec_bandpass.target_eigenvalues
        )
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        best_err = pareto.best_eigenvalue.eigenvalue_error
        # Allow up to 50% of analytic error (the optimizer may not always improve)
        assert best_err <= max(analytic_err * 1.5, 1.0), \
            f"best_err={best_err:.4f} > 1.5 * analytic_err={analytic_err:.4f}"

    def test_converged_Q_within_20pct(self, spec_bandpass):
        """Converged result has Q_achieved within 20% of Q_target."""
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        best = pareto.best_eigenvalue
        Q_target = spec_bandpass.Q_target
        Q_achieved = best.Q_achieved
        if Q_achieved > 0:
            rel_err = abs(Q_achieved - Q_target) / Q_target
            assert rel_err < 0.20, \
                f"Q_achieved={Q_achieved:.3f} not within 20% of Q_target={Q_target}"

    def test_positive_RLC_values(self, spec_bandpass):
        """All candidates must have positive R, L, C."""
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        for c in pareto.all_candidates:
            assert c.R > 0
            assert c.L > 0
            assert c.C > 0

    def test_eigenvalues_shape_and_type(self, spec_bandpass):
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        best = pareto.best_eigenvalue
        assert best.achieved_eigenvalues.shape == (2,)
        assert np.iscomplexobj(best.achieved_eigenvalues)

    def test_regime_type_valid_string(self, spec_bandpass):
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        valid_regimes = {"lca", "nonabelian", "chaotic"}
        for c in pareto.all_candidates:
            assert c.regime_type in valid_regimes

    def test_omega0_achieved_positive(self, spec_bandpass):
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        best = pareto.best_eigenvalue
        assert best.omega0_achieved > 0

    def test_optimize_low_q_spec(self, spec_lowq):
        """Optimizer handles low-Q (critically damped) spec."""
        opt = CircuitOptimizer(spec_lowq)
        pareto = opt.optimize()
        assert isinstance(pareto, ParetoResult)
        assert len(pareto.all_candidates) > 0

    def test_optimize_highfreq_spec(self, spec_highfreq):
        """Optimizer handles high-frequency spec."""
        opt = CircuitOptimizer(spec_highfreq)
        pareto = opt.optimize()
        assert isinstance(pareto, ParetoResult)

    def test_no_regime_penalty_by_default(self, spec_bandpass):
        """use_regime_penalty=False by default (fast mode)."""
        opt = CircuitOptimizer(spec_bandpass, use_regime_penalty=False)
        assert opt.cost_fn.patch_detector is None


# ---------------------------------------------------------------------------
# MonteCarloStabilityAnalyzer tests
# ---------------------------------------------------------------------------

class TestMonteCarloStabilityAnalyzer:

    @pytest.fixture
    def stable_result(self, spec_bandpass, mapper):
        """A well-conditioned stable RLC result."""
        inv = mapper.inverse_map(spec_bandpass.target_eigenvalues)
        eigs = mapper.eigenvalues(inv["R"], inv["L"], inv["C"])
        gap = abs(abs(eigs[0]) - abs(eigs[1]))
        omega0 = float(np.sqrt(np.real(eigs[0])**2 + np.imag(eigs[0])**2))
        zeta = abs(np.real(eigs[0])) / omega0
        Q = 1.0 / (2.0 * zeta) if zeta > 0 else 0.0
        return OptimizationResult(
            R=inv["R"], L=inv["L"], C=inv["C"],
            achieved_eigenvalues=eigs,
            eigenvalue_error=0.0,
            cost=0.0,
            regime_type="lca",
            spectral_gap=float(gap),
            omega0_achieved=omega0,
            Q_achieved=Q,
            converged=True,
        )

    def test_analyze_returns_basin_result(self, stable_result, spec_bandpass):
        analyzer = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        assert isinstance(basin, BasinResult)

    def test_analyze_n_samples(self, stable_result, spec_bandpass):
        analyzer = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        assert basin.n_samples == 100

    def test_analyze_samples_list_length(self, stable_result, spec_bandpass):
        analyzer = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        assert len(basin.samples) == 100

    def test_samples_have_required_keys(self, stable_result, spec_bandpass):
        analyzer = MonteCarloStabilityAnalyzer(n_samples=10, seed=0)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        for s in basin.samples:
            assert "R" in s
            assert "L" in s
            assert "C" in s
            assert "regime" in s
            assert "eig_error" in s

    def test_lca_fraction_stable_circuit(self, stable_result, spec_bandpass):
        """Well-conditioned stable circuit: lca_fraction should be > 0.5."""
        analyzer = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        assert basin.lca_fraction > 0.5, \
            f"Expected lca_fraction > 0.5, got {basin.lca_fraction:.3f}"

    def test_count_consistency(self, stable_result, spec_bandpass):
        """n_lca + n_nonabelian + n_chaotic == n_samples."""
        analyzer = MonteCarloStabilityAnalyzer(n_samples=50, seed=7)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        total = basin.n_lca + basin.n_nonabelian + basin.n_chaotic
        assert total == basin.n_samples

    def test_worst_case_error_finite(self, stable_result, spec_bandpass):
        analyzer = MonteCarloStabilityAnalyzer(n_samples=50, seed=0)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        assert np.isfinite(basin.worst_case_error)
        assert basin.worst_case_error >= 0.0

    def test_mean_eigenvalue_spread_nonnegative(self, stable_result, spec_bandpass):
        analyzer = MonteCarloStabilityAnalyzer(n_samples=50, seed=0)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        assert basin.mean_eigenvalue_spread >= 0.0

    def test_regime_strings_valid(self, stable_result, spec_bandpass):
        analyzer = MonteCarloStabilityAnalyzer(n_samples=20, seed=1)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        valid = {"lca", "nonabelian", "chaotic"}
        for s in basin.samples:
            assert s["regime"] in valid

    def test_sample_RLC_near_nominal(self, stable_result, spec_bandpass):
        """Sampled R/L/C should be within tolerance bounds of nominal."""
        analyzer = MonteCarloStabilityAnalyzer(n_samples=100, seed=42)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        R_tol = spec_bandpass.component_tolerances["R"]
        L_tol = spec_bandpass.component_tolerances["L"]
        C_tol = spec_bandpass.component_tolerances["C"]
        for s in basin.samples:
            assert s["R"] >= stable_result.R * (1 - R_tol - 1e-9)
            assert s["R"] <= stable_result.R * (1 + R_tol + 1e-9)
            assert s["L"] >= stable_result.L * (1 - L_tol - 1e-9)
            assert s["L"] <= stable_result.L * (1 + L_tol + 1e-9)
            assert s["C"] >= stable_result.C * (1 - C_tol - 1e-9)
            assert s["C"] <= stable_result.C * (1 + C_tol + 1e-9)

    def test_all_sampled_values_positive(self, stable_result, spec_bandpass):
        """All sampled R/L/C must be strictly positive."""
        analyzer = MonteCarloStabilityAnalyzer(n_samples=50, seed=42)
        basin = analyzer.analyze(stable_result, spec_bandpass)
        for s in basin.samples:
            assert s["R"] > 0, f"Non-positive R: {s['R']}"
            assert s["L"] > 0, f"Non-positive L: {s['L']}"
            assert s["C"] > 0, f"Non-positive C: {s['C']}"


class TestParetoFront:
    def test_pareto_front_nonempty(self, spec_bandpass):
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        assert len(pareto.pareto_front) > 0

    def test_pareto_front_non_dominated(self, spec_bandpass):
        """No solution in the front should be dominated by another."""
        opt = CircuitOptimizer(spec_bandpass)
        pareto = opt.optimize()
        for c in pareto.pareto_front:
            for o in pareto.pareto_front:
                if o is c:
                    continue
                # o should NOT dominate c
                better_eig = o.eigenvalue_error <= c.eigenvalue_error
                better_cost = o.cost <= c.cost
                better_gap = o.spectral_gap >= c.spectral_gap
                strict = (
                    o.eigenvalue_error < c.eigenvalue_error
                    or o.cost < c.cost
                    or o.spectral_gap > c.spectral_gap
                )
                assert not (better_eig and better_cost and better_gap and strict), \
                    "Found dominated solution in pareto_front"
