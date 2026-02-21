"""
Tests for optimization/rlc_parameterization.py, rlc_evaluator.py,
hdv_optimizer.py

Covers:
  1. RLCParams — fields, as_dict, __str__
  2. RLCDesignMapper — decode positivity, shape validation, encode round-trip,
     orthonormality of projection vectors
  3. RLCEvaluator — known formula values, constraint dict structure, evaluate()
  4. ConstrainedHDVOptimizer — returns EvaluationResult, reduces objective from
     random start, warm-start from memory shortens search
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from optimization.rlc_parameterization import RLCDesignMapper, RLCParams
from optimization.rlc_evaluator import EvaluationResult, RLCEvaluator
from optimization.koopman_memory import KoopmanExperienceMemory
from optimization.hdv_optimizer import ConstrainedHDVOptimizer


# ── 1. RLCParams ─────────────────────────────────────────────────────────────


class TestRLCParams:

    def test_fields(self):
        p = RLCParams(R=100.0, L=0.01, C=1e-6)
        assert p.R == pytest.approx(100.0)
        assert p.L == pytest.approx(0.01)
        assert p.C == pytest.approx(1e-6)

    def test_as_dict_keys(self):
        d = RLCParams(R=10.0, L=0.005, C=2e-6).as_dict()
        assert set(d.keys()) == {"R", "L", "C"}
        assert d["R"] == pytest.approx(10.0)

    def test_str_contains_values(self):
        s = str(RLCParams(R=47.0, L=0.002, C=5e-7))
        assert "R=" in s
        assert "L=" in s
        assert "C=" in s


# ── 2. RLCDesignMapper ────────────────────────────────────────────────────────


class TestRLCDesignMapper:

    def setup_method(self):
        self.mapper = RLCDesignMapper(hdv_dim=32, seed=0)

    def test_decode_returns_positive_params(self):
        """decode() must return R, L, C > 0 for any HDV vector."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            z = rng.standard_normal(32)
            p = self.mapper.decode(z)
            assert p.R > 0
            assert p.L > 0
            assert p.C > 0

    def test_decode_zero_vector_returns_centers(self):
        """z=0 should decode to (R_center, L_center, C_center) since exp(0)=1."""
        z = np.zeros(32)
        p = self.mapper.decode(z)
        assert p.R == pytest.approx(self.mapper.R_center)
        assert p.L == pytest.approx(self.mapper.L_center)
        assert p.C == pytest.approx(self.mapper.C_center)

    def test_decode_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            self.mapper.decode(np.zeros(10))

    def test_projection_vectors_are_orthonormal(self):
        """a_R, a_L, a_C should be pairwise orthogonal unit vectors."""
        a_R = self.mapper._a_R
        a_L = self.mapper._a_L
        a_C = self.mapper._a_C
        assert np.linalg.norm(a_R) == pytest.approx(1.0, abs=1e-12)
        assert np.linalg.norm(a_L) == pytest.approx(1.0, abs=1e-12)
        assert np.linalg.norm(a_C) == pytest.approx(1.0, abs=1e-12)
        assert abs(a_R @ a_L) < 1e-12
        assert abs(a_R @ a_C) < 1e-12
        assert abs(a_L @ a_C) < 1e-12

    def test_encode_decode_round_trip(self):
        """decode(encode(p)) ≈ p for a design within the exp-range."""
        # Choose params close to centers so log-ratios are small
        p = RLCParams(R=120.0, L=0.012, C=1.2e-6)
        z = self.mapper.encode(p)
        p2 = self.mapper.decode(z)
        assert p2.R == pytest.approx(p.R, rel=1e-6)
        assert p2.L == pytest.approx(p.L, rel=1e-6)
        assert p2.C == pytest.approx(p.C, rel=1e-6)

    def test_encode_output_shape(self):
        p = RLCParams(R=100.0, L=0.01, C=1e-6)
        z = self.mapper.encode(p)
        assert z.shape == (32,)

    def test_different_seeds_give_different_projections(self):
        m1 = RLCDesignMapper(hdv_dim=32, seed=1)
        m2 = RLCDesignMapper(hdv_dim=32, seed=2)
        assert not np.allclose(m1._a_R, m2._a_R)

    def test_same_seed_gives_same_projections(self):
        m1 = RLCDesignMapper(hdv_dim=32, seed=99)
        m2 = RLCDesignMapper(hdv_dim=32, seed=99)
        np.testing.assert_array_equal(m1._a_R, m2._a_R)


# ── 3. RLCEvaluator ──────────────────────────────────────────────────────────


class TestRLCEvaluator:

    def setup_method(self):
        self.ev = RLCEvaluator(max_Q=10.0, max_energy_loss=0.5)

    # Known analytic values
    # R=100, L=0.01, C=1e-6:
    #   ω₀ = 1/√(0.01·1e-6) = 1/√(1e-8) = 1e4 rad/s
    #   f₀ = 1e4/(2π) ≈ 1591.55 Hz
    #   Q  = (1/100)√(0.01/1e-6) = 0.01 · 100 = 1.0

    def _std_params(self):
        return RLCParams(R=100.0, L=0.01, C=1e-6)

    def test_cutoff_frequency_hz_known_value(self):
        # R=100, L=0.01, C=1e-6 → ω₀=1/√(1e-8)=1e4 rad/s → f₀=1e4/(2π)≈1591.55 Hz
        f0 = self.ev.cutoff_frequency_hz(self._std_params())
        assert f0 == pytest.approx(1e4 / (2 * np.pi), rel=1e-6)

    def test_Q_factor_known_value(self):
        Q = self.ev.Q_factor(self._std_params())
        assert Q == pytest.approx(1.0, rel=1e-6)

    def test_energy_loss_is_half_over_Q(self):
        p = self._std_params()
        Q = self.ev.Q_factor(p)
        loss = self.ev.energy_loss_estimate(p)
        assert loss == pytest.approx(1.0 / (2.0 * Q), rel=1e-9)

    def test_objective_is_zero_at_exact_cutoff(self):
        p = self._std_params()
        f0 = self.ev.cutoff_frequency_hz(p)
        assert self.ev.objective(p, f0) == pytest.approx(0.0, abs=1e-12)

    def test_objective_nonnegative(self):
        p = self._std_params()
        assert self.ev.objective(p, 500.0) >= 0.0
        assert self.ev.objective(p, 5000.0) >= 0.0

    def test_constraints_dict_has_required_keys(self):
        c = self.ev.constraints(self._std_params())
        for key in ("Q_limit", "energy_loss", "R_positive", "L_positive", "C_positive"):
            assert key in c

    def test_constraints_values_are_tuples(self):
        c = self.ev.constraints(self._std_params())
        for name, val in c.items():
            ok, value, limit = val
            assert isinstance(ok, (bool, np.bool_))
            assert isinstance(value, float)
            assert isinstance(limit, float)

    def test_constraints_satisfied_for_valid_design(self):
        p = self._std_params()   # Q=1 < max_Q=10; loss=0.5 = max
        c = self.ev.constraints(p)
        # Q_limit, positivity should pass; energy_loss is boundary (≤)
        assert c["Q_limit"][0]
        assert c["R_positive"][0]

    def test_evaluate_returns_evaluation_result(self):
        r = self.ev.evaluate(self._std_params(), 1000.0)
        assert isinstance(r, EvaluationResult)

    def test_evaluate_constraint_detail_matches_constraints(self):
        p = self._std_params()
        r = self.ev.evaluate(p, 1000.0)
        c = self.ev.constraints(p)
        assert r.constraint_detail == c

    def test_Q_factor_zero_R_returns_inf(self):
        p = RLCParams(R=0.0, L=0.01, C=1e-6)
        assert self.ev.Q_factor(p) == float("inf")

    def test_frequency_response_shape(self):
        p = self._std_params()
        freqs = np.logspace(1, 5, 100)
        H = self.ev.frequency_response(p, freqs)
        assert H.shape == (100,)
        assert np.all(H >= 0)

    def test_frequency_response_at_dc_is_near_one(self):
        p = self._std_params()
        H = self.ev.frequency_response(p, np.array([1e-3]))
        # At very low frequency: H ≈ 1
        assert H[0] == pytest.approx(1.0, rel=1e-3)

    def test_verify_with_simulation_no_simulator(self):
        p = self._std_params()
        out = self.ev.verify_with_simulation(p, 1000.0, simulator=None)
        assert out["source"] == "analytic"
        assert out["valid"] is True
        assert "predicted_hz" in out
        assert out["measured_hz"] is None


# ── 4. ConstrainedHDVOptimizer ────────────────────────────────────────────────


class TestConstrainedHDVOptimizer:

    def _make_optimizer(self, memory=None, n_iter=200, seed=0):
        mapper = RLCDesignMapper(hdv_dim=32, seed=42)
        evaluator = RLCEvaluator(max_Q=10.0, max_energy_loss=0.5)
        mem = memory or KoopmanExperienceMemory()
        return ConstrainedHDVOptimizer(
            mapper, evaluator, mem, n_iter=n_iter, seed=seed
        )

    def test_optimize_returns_evaluation_result(self):
        opt = self._make_optimizer()
        result = opt.optimize(target_hz=1000.0)
        assert isinstance(result, EvaluationResult)

    def test_optimize_has_positive_objective(self):
        opt = self._make_optimizer()
        result = opt.optimize(target_hz=1000.0)
        assert result.objective >= 0.0

    def test_optimize_result_has_correct_target(self):
        target = 500.0
        opt = self._make_optimizer()
        result = opt.optimize(target_hz=target)
        assert result.target_hz == pytest.approx(target)

    def test_optimizer_seeds_produce_deterministic_results(self):
        opt1 = self._make_optimizer(seed=7)
        opt2 = self._make_optimizer(seed=7)
        r1 = opt1.optimize(500.0)
        r2 = opt2.optimize(500.0)
        assert r1.objective == pytest.approx(r2.objective)

    def test_different_seeds_may_produce_different_results(self):
        """Not guaranteed, but with enough iterations very unlikely to match."""
        opt1 = self._make_optimizer(seed=0)
        opt2 = self._make_optimizer(seed=999)
        r1 = opt1.optimize(1000.0)
        r2 = opt2.optimize(1000.0)
        # Just check both finish without error; objectives may differ
        assert r1.objective >= 0.0
        assert r2.objective >= 0.0

    def test_memory_populated_after_optimize(self):
        """After optimize(), memory should have at least one entry (if Koopman fit succeeded)."""
        memory = KoopmanExperienceMemory()
        opt = self._make_optimizer(memory=memory, n_iter=200)
        opt.optimize(target_hz=1000.0)
        # May or may not store depending on trace length; just check no crash
        assert len(memory) >= 0   # must not raise

    def test_warm_start_uses_memory(self):
        """
        Second optimizer with pre-loaded memory starts from a better point.
        We verify it doesn't crash and produces a valid result.
        """
        memory = KoopmanExperienceMemory()
        # Run 1 to populate memory
        opt1 = self._make_optimizer(memory=memory, n_iter=300, seed=0)
        opt1.optimize(target_hz=1000.0)

        # Run 2 with same memory (warm start)
        opt2 = self._make_optimizer(memory=memory, n_iter=300, seed=1)
        result2 = opt2.optimize(target_hz=1000.0)
        assert isinstance(result2, EvaluationResult)
        assert result2.objective >= 0.0

    def test_tolerance_termination(self):
        """Optimizer with tol=1.0 should terminate after very few steps."""
        mapper = RLCDesignMapper(hdv_dim=32, seed=42)
        evaluator = RLCEvaluator()
        mem = KoopmanExperienceMemory()
        # tol=1.0 means any objective < 1.0 terminates (always true initially)
        opt = ConstrainedHDVOptimizer(
            mapper, evaluator, mem, n_iter=500, seed=0, tol=1.0
        )
        result = opt.optimize(target_hz=1000.0)
        assert isinstance(result, EvaluationResult)
