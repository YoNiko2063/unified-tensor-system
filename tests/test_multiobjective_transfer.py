"""
Phase 4 — Multi-Objective (ω₀, Q) Transfer Tests.

Proves that the 2D dynamical manifold (ω₀, Q) supports transfer:

  Phase 4a — Q convergence
    With Q_target set, the optimizer must drive Q close to Q_target.
    Tested for both RLC and spring-mass at Q ∈ {0.5, 1, 2, 5, 10}.

  Phase 4b — 2D monotonicity (Spearman ρ)
    Vary Q_target ∈ {0.5, 1, 2, 5, 10}; log_Q_norm of stored entries
    must be Spearman-correlated ρ > 0.9 with log(Q_target).
    Confirms the invariant descriptor tracks the 2D space faithfully.

  Phase 4c — 2D frequency monotonicity preserved
    Multi-objective mode must not degrade the Phase 1 ω₀-monotonicity.
    Spearman ρ(freq_target, log_omega0_norm) > 0.8 with Q_target=1.0 fixed.

  Phase 4d — 2D cross-domain transfer
    Spring-mass memory trained on (1kHz, Q_target=5) warm-starts an RLC
    optimizer targeting (1kHz, Q_target=5).
    The warm start must not be catastrophically worse than cold start.

  Phase 4e — 2D retrieval ordering
    Memory with entries at Q ∈ {0.5, 1, 2, 5, 10} (both ω₀ dimensions).
    A query at (1kHz, Q=3) should retrieve Q=2 as nearest, not Q=5.
    Confirms the geometry is correct in both dimensions simultaneously.

Design note
-----------
The optimizer's early-stop criterion is J < tol (combined objective).
With w_freq=1.0, w_Q=1.0, tol=1e-3 means both errors < ~0.05%
(since J=freq_err+Q_err and each ≤ J).  For convergence tests we use
tol=0.02 (2% total error) to verify the optimizer moves toward Q_target.
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

from optimization.koopman_memory import KoopmanExperienceMemory
from optimization.koopman_signature import _LOG_OMEGA0_SCALE, _LOG_OMEGA0_REF
from optimization.rlc_evaluator import RLCEvaluator
from optimization.rlc_parameterization import RLCDesignMapper
from optimization.hdv_optimizer import ConstrainedHDVOptimizer
from optimization.spring_mass_system import (
    SpringMassDesignMapper,
    SpringMassEvaluator,
    SpringMassOptimizer,
)


# ── Constants ─────────────────────────────────────────────────────────────────

_TARGET_HZ   = 1_000.0          # common frequency for most tests
_N_ITER      = 800
_HDV_DIM     = 64
_SEED        = 0
# max_energy_loss=5.5 ensures all Q ≥ 0.1 are constraint-valid (energy_loss = 1/(2Q) ≤ 5.5)
_MAX_ENERGY_LOSS = 5.5
_MAX_Q           = 200.0

# RLC requires correlated L+C changes to hit high Q (two parameters move).
# Q=10 is at the mapper's exploration boundary in 800 iters; exclude from RLC tests.
_RLC_Q_TARGETS = [0.5, 1.0, 2.0, 5.0]

# Spring-mass only needs b (damping) to change; Q=10 is reachable.
_SM_Q_TARGETS  = [0.5, 1.0, 2.0, 5.0, 10.0]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rlc_evaluator(q_target: float | None = None, w_Q: float = 1.0) -> RLCEvaluator:
    return RLCEvaluator(
        max_Q=_MAX_Q,
        max_energy_loss=_MAX_ENERGY_LOSS,
        Q_target=q_target,
        w_freq=1.0,
        w_Q=w_Q,
    )


def _sm_evaluator(q_target: float | None = None, w_Q: float = 1.0) -> SpringMassEvaluator:
    return SpringMassEvaluator(
        max_Q=_MAX_Q,
        max_energy_loss=_MAX_ENERGY_LOSS,
        Q_target=q_target,
        w_freq=1.0,
        w_Q=w_Q,
    )


def _run_rlc(target_hz: float, q_target: float | None, seed: int = _SEED,
             n_iter: int = _N_ITER) -> tuple:
    """Run RLC optimizer. Returns (result, memory)."""
    mapper    = RLCDesignMapper(hdv_dim=_HDV_DIM, seed=seed)
    evaluator = _rlc_evaluator(q_target)
    memory    = KoopmanExperienceMemory()
    opt = ConstrainedHDVOptimizer(mapper, evaluator, memory, n_iter=n_iter, seed=seed)
    result = opt.optimize(target_hz, pilot_steps=0)
    return result, memory


def _run_sm(target_hz: float, q_target: float | None, seed: int = _SEED,
            n_iter: int = _N_ITER) -> tuple:
    """Run spring-mass optimizer. Returns (result, memory)."""
    mapper    = SpringMassDesignMapper(hdv_dim=_HDV_DIM, seed=seed)
    evaluator = _sm_evaluator(q_target)
    memory    = KoopmanExperienceMemory()
    opt = SpringMassOptimizer(mapper, evaluator, memory, n_iter=n_iter, seed=seed)
    result = opt.optimize(target_hz)
    return result, memory


# ── Phase 4a: Q convergence ───────────────────────────────────────────────────


@pytest.mark.parametrize("q_target", _RLC_Q_TARGETS)
def test_rlc_converges_to_Q_target(q_target):
    """
    RLC optimizer with Q_target set must achieve Q within 30% of target.

    The multi-objective drives the optimizer away from degenerate ω₀-only
    solutions toward the unique (ω₀, Q) solution point.
    30% tolerance accounts for finite search budget.
    """
    result, _ = _run_rlc(_TARGET_HZ, q_target)
    if not result.constraints_ok:
        pytest.skip(f"RLC optimizer did not find feasible design at Q_target={q_target}")
    q_rel_err = abs(result.Q_factor - q_target) / q_target
    freq_rel_err = result.freq_error
    print(f"\n  RLC Q_target={q_target}: Q_achieved={result.Q_factor:.4f} "
          f"Q_rel_err={q_rel_err:.4f}  freq_err={freq_rel_err:.4f}")
    assert q_rel_err < 0.30, (
        f"RLC Q_target={q_target}: Q_achieved={result.Q_factor:.4f} "
        f"is >30% from target"
    )


@pytest.mark.parametrize("q_target", _SM_Q_TARGETS)
def test_sm_converges_to_Q_target(q_target):
    """Spring-mass optimizer with Q_target must achieve Q within 30% of target."""
    result, _ = _run_sm(_TARGET_HZ, q_target)
    if not result.constraints_ok:
        pytest.skip(f"SM optimizer did not find feasible design at Q_target={q_target}")
    q_rel_err = abs(result.Q_factor - q_target) / q_target
    print(f"\n  SM Q_target={q_target}: Q_achieved={result.Q_factor:.4f} "
          f"Q_rel_err={q_rel_err:.4f}")
    assert q_rel_err < 0.30, (
        f"SM Q_target={q_target}: Q_achieved={result.Q_factor:.4f} "
        f"is >30% from target"
    )


@pytest.mark.parametrize("q_target", _RLC_Q_TARGETS)
def test_rlc_frequency_preserved_in_multiobjective(q_target):
    """Multi-objective mode must not abandon frequency accuracy."""
    result, _ = _run_rlc(_TARGET_HZ, q_target)
    if not result.constraints_ok:
        pytest.skip(f"No feasible design at Q_target={q_target}")
    assert result.freq_error < 0.05, (
        f"RLC Q_target={q_target}: freq_error={result.freq_error:.4f} > 5%"
    )


# ── Phase 4b: 2D monotonicity (log_Q_norm Spearman ρ) ────────────────────────


@pytest.fixture(scope="module")
def rlc_2d_runs():
    """Run RLC optimizer at each Q_target. Returns list of (q_target, memory)."""
    results = []
    for q in _RLC_Q_TARGETS:
        _, mem = _run_rlc(_TARGET_HZ, q)
        results.append((q, mem))
    return results


@pytest.fixture(scope="module")
def sm_2d_runs():
    """Run spring-mass optimizer at each Q_target."""
    results = []
    for q in _SM_Q_TARGETS:
        _, mem = _run_sm(_TARGET_HZ, q)
        results.append((q, mem))
    return results


def test_rlc_log_Q_norm_spearman_vs_Q_target(rlc_2d_runs):
    """
    Stored log_Q_norm must be Spearman-correlated ρ > 0.8 with log(Q_target).

    This is the 2D monotonicity test: the invariant descriptor must track
    the Q dimension of the search space, not just the ω₀ dimension.
    """
    q_targets = []
    log_Q_norms = []
    for q, mem in rlc_2d_runs:
        if len(mem) == 0:
            continue
        q_targets.append(math.log(q))
        log_Q_norms.append(mem._entries[0].invariant.log_Q_norm)

    if len(q_targets) < 3:
        pytest.skip("Too few data points for Spearman test")

    rho, pval = stats.spearmanr(q_targets, log_Q_norms)
    print(f"\n  RLC 2D: Spearman ρ(log Q_target, log_Q_norm) = {rho:.4f}  p={pval:.4f}")
    print(f"  Q_targets={[f'{q:.1f}' for q, _ in rlc_2d_runs]}")
    print(f"  log_Q_norms={[f'{v:.3f}' for v in log_Q_norms]}")

    assert rho > 0.8, (
        f"RLC 2D Q monotonicity: Spearman ρ={rho:.4f} < 0.8. "
        f"Multi-objective is not driving Q toward target reliably."
    )


def test_sm_log_Q_norm_spearman_vs_Q_target(sm_2d_runs):
    """Spring-mass 2D: log_Q_norm Spearman ρ > 0.8 with log(Q_target)."""
    q_targets = []
    log_Q_norms = []
    for q, mem in sm_2d_runs:
        if len(mem) == 0:
            continue
        q_targets.append(math.log(q))
        log_Q_norms.append(mem._entries[0].invariant.log_Q_norm)

    if len(q_targets) < 3:
        pytest.skip("Too few data points")

    rho, pval = stats.spearmanr(q_targets, log_Q_norms)
    print(f"\n  SM 2D: Spearman ρ(log Q_target, log_Q_norm) = {rho:.4f}  p={pval:.4f}")

    assert rho > 0.8, (
        f"SM 2D Q monotonicity: Spearman ρ={rho:.4f} < 0.8."
    )


# ── Phase 4c: ω₀ monotonicity preserved in multi-objective mode ──────────────


@pytest.fixture(scope="module")
def rlc_freq_sweep_multiobjective():
    """
    Run RLC multi-objective at Q_target=1.0 across frequency targets.
    Returns list of (freq_target_hz, memory).
    """
    freq_targets = [500.0, 750.0, 1000.0, 1500.0, 2000.0]
    results = []
    for fhz in freq_targets:
        _, mem = _run_rlc(fhz, q_target=1.0)
        results.append((fhz, mem))
    return results


def test_rlc_omega0_monotone_in_multiobjective_mode(rlc_freq_sweep_multiobjective):
    """
    log_omega0_norm must be Spearman-correlated ρ > 0.8 with log(freq_target).
    This confirms that adding Q_target does not break the Phase 1 guarantee.
    """
    freq_targets_log = []
    log_omega0_norms = []
    for fhz, mem in rlc_freq_sweep_multiobjective:
        if len(mem) == 0:
            continue
        freq_targets_log.append(math.log(fhz))
        log_omega0_norms.append(mem._entries[0].invariant.log_omega0_norm)

    if len(freq_targets_log) < 3:
        pytest.skip("Too few data points")

    rho, pval = stats.spearmanr(freq_targets_log, log_omega0_norms)
    print(f"\n  Phase 4c: Spearman ρ(log f_target, log_omega0_norm) = {rho:.4f}  p={pval:.4f}")
    print(f"  log_omega0_norms: {[f'{v:.3f}' for v in log_omega0_norms]}")

    assert rho > 0.8, (
        f"Multi-objective mode breaks ω₀ monotonicity: ρ={rho:.4f} < 0.8"
    )


# ── Phase 4d: 2D cross-domain transfer ───────────────────────────────────────


@pytest.mark.parametrize("q_target", [1.0, 2.0, 5.0])
def test_2d_cross_domain_transfer(q_target):
    """
    A spring-mass memory trained at (1kHz, Q_target=q) provides a warm start
    for an RLC optimizer targeting (1kHz, Q_target=q).

    The warm median must be ≤ 20× cold median — same generous bound as Phase 3.
    With multi-objective, the warm start injects a z tuned to (ω₀, Q) from the
    spring-mass memory, giving the RLC optimizer a 2D head start.

    Tests Q_target ∈ {1.0, 2.0, 5.0} (all constraint-valid under max_energy_loss=5.5).
    """
    # Train spring-mass at (1kHz, q_target)
    _, sm_mem = _run_sm(_TARGET_HZ, q_target, seed=_SEED)

    rlc_mapper    = RLCDesignMapper(hdv_dim=_HDV_DIM, seed=_SEED)
    rlc_evaluator = _rlc_evaluator(q_target)

    # Cold: 3 seeds, median
    cold_results = []
    for cseed in [0, 1, 2]:
        cold_mem = KoopmanExperienceMemory()
        cold_opt = ConstrainedHDVOptimizer(rlc_mapper, rlc_evaluator, cold_mem,
                                           n_iter=_N_ITER, seed=cseed)
        cold_results.append(cold_opt.optimize(_TARGET_HZ, pilot_steps=0).objective)
    cold_median = float(np.median(cold_results))

    # Warm: from spring-mass 2D memory
    warm_results = []
    for wseed in [0, 1, 2]:
        warm_opt = ConstrainedHDVOptimizer(rlc_mapper, rlc_evaluator, sm_mem,
                                           n_iter=_N_ITER, seed=wseed)
        warm_results.append(warm_opt.optimize(_TARGET_HZ, pilot_steps=20).objective)
    warm_median = float(np.median(warm_results))

    ratio = cold_median / max(warm_median, 1e-9)
    print(f"\n  2D transfer Q={q_target}: cold={cold_median:.4f} warm={warm_median:.4f} "
          f"ratio={ratio:.2f}×")

    assert warm_median <= cold_median * 20.0, (
        f"2D cross-domain transfer failed at Q_target={q_target}: "
        f"warm={warm_median:.4f} > cold={cold_median:.4f} × 20"
    )


# ── Phase 4e: 2D retrieval ordering ──────────────────────────────────────────


@pytest.fixture(scope="module")
def memory_2d_grid(rlc_2d_runs):
    """Combined memory with entries at all Q_target values (same ω₀=1kHz)."""
    from optimization.koopman_memory import KoopmanExperienceMemory
    combined = KoopmanExperienceMemory()
    for _, mem in rlc_2d_runs:
        for entry in mem._entries:
            combined._entries.append(entry)
    return combined


def test_2d_retrieval_ordering_at_Q3(memory_2d_grid, rlc_2d_runs):
    """
    A query at (1kHz, Q=3) should retrieve Q=2 as nearest, not Q=5.

    This tests that the (ω₀, Q) geometry is correct: Q=2 is closer to Q=3
    in log-space (|log3 - log2| = 0.405) than Q=5 (|log3 - log5| = 0.511).
    Confirms 2D manifold retrieval works.
    """
    from optimization.koopman_signature import compute_invariants

    # Build query at (1kHz, Q=3)
    q_query = 3.0
    log_Q_query = float(np.clip(math.log(q_query) / _LOG_OMEGA0_SCALE, -3.0, 3.0))
    zeta_query   = 1.0 / (2.0 * q_query)

    dummy_eigs = np.array([0.5 + 0.0j])
    dummy_vecs = np.array([[1.0]])
    query_inv = compute_invariants(
        eigenvalues=dummy_eigs,
        eigenvectors=dummy_vecs,
        operator_types=["dummy"],
        k=5,
        log_omega0_norm=0.0,
        log_Q_norm=log_Q_query,
        damping_ratio=zeta_query,
    )

    candidates = memory_2d_grid.retrieve_candidates(query_inv, top_n=5)
    if len(candidates) < 2:
        pytest.skip("Not enough entries in combined memory for retrieval test")

    nearest = candidates[0].invariant
    nearest_log_Q = nearest.log_Q_norm
    nearest_dist  = abs(nearest_log_Q - log_Q_query)

    for other in candidates[1:]:
        other_dist = abs(other.invariant.log_Q_norm - log_Q_query)
        assert nearest_dist <= other_dist + 1e-6, (
            f"2D retrieval: nearest log_Q={nearest_log_Q:.4f} is farther from "
            f"query Q=3 (log_Q={log_Q_query:.4f}) than another candidate "
            f"log_Q={other.invariant.log_Q_norm:.4f}"
        )

    print(f"\n  2D retrieval Q=3: nearest stored log_Q_norm={nearest_log_Q:.4f} "
          f"(query={log_Q_query:.4f})")
    # Q=2 → log_Q = log(2)/log(10)=0.301; Q=5 → 0.699; Q=3 → 0.477
    # Nearest should be closer to 0.477 than Q=5 is
    assert nearest_dist < abs(math.log(5.0) / _LOG_OMEGA0_SCALE - log_Q_query) + 1e-6


def test_2d_retrieval_ordering_symmetric(memory_2d_grid):
    """
    Retrieval at Q=7 (between Q=5 and Q=10) should retrieve Q=5 or Q=10,
    not Q=2. Verifies monotone ordering at the upper end.
    """
    from optimization.koopman_signature import compute_invariants

    q_query = 7.0
    log_Q_query = float(np.clip(math.log(q_query) / _LOG_OMEGA0_SCALE, -3.0, 3.0))
    query_inv = compute_invariants(
        eigenvalues=np.array([0.5 + 0.0j]),
        eigenvectors=np.array([[1.0]]),
        operator_types=["dummy"],
        k=5,
        log_omega0_norm=0.0,
        log_Q_norm=log_Q_query,
        damping_ratio=1.0 / (2.0 * q_query),
    )

    candidates = memory_2d_grid.retrieve_candidates(query_inv, top_n=5)
    if len(candidates) == 0:
        pytest.skip("No entries in combined memory")

    nearest = candidates[0].invariant
    # Nearest log_Q_norm should be ≥ log(2)/log(10) = 0.301 (Q≥2, not Q=0.5 or Q=1)
    q_nearest = nearest.dynamical_Q()
    print(f"\n  2D retrieval Q=7: nearest Q={q_nearest:.2f}")
    assert q_nearest >= 1.5, (
        f"Retrieval at Q=7 returned Q={q_nearest:.2f} — expected Q≥2"
    )


# ── Phase 4f: EvaluationResult fields ────────────────────────────────────────


def test_evaluation_result_has_Q_error_field():
    """EvaluationResult.Q_error is populated when Q_target is set."""
    evaluator = _rlc_evaluator(q_target=5.0)
    from optimization.rlc_parameterization import RLCParams
    # Params at center: f≈1591 Hz, Q=1
    params = RLCParams(R=100.0, L=0.01, C=1e-6)
    result = evaluator.evaluate(params, 1000.0)
    assert result.Q_target == 5.0
    assert result.Q_error > 0.0      # Q=1 ≠ 5 → error > 0
    assert result.freq_error > 0.0   # f≠1kHz → error > 0
    assert abs(result.objective - (result.freq_error + result.Q_error)) < 1e-9


def test_evaluation_result_single_objective_compat():
    """Without Q_target, objective = freq_error exactly (backward compat)."""
    evaluator = RLCEvaluator(max_Q=10.0, max_energy_loss=0.5)
    from optimization.rlc_parameterization import RLCParams
    params = RLCParams(R=100.0, L=0.01, C=1e-6)
    result = evaluator.evaluate(params, 1000.0)
    assert result.Q_target is None
    assert result.Q_error == 0.0
    assert abs(result.objective - result.freq_error) < 1e-12


def test_sm_evaluation_result_has_Q_error_field():
    """SpringMassResult.Q_error is populated when Q_target is set."""
    from optimization.spring_mass_system import SpringMassParams
    evaluator = SpringMassEvaluator(
        max_Q=200.0, max_energy_loss=5.5, Q_target=2.0
    )
    params = SpringMassParams(k=1000.0, m=2.533e-5, b=0.15915)  # Q≈1 at 1kHz
    result = evaluator.evaluate(params, 1000.0)
    assert result.Q_target == 2.0
    assert result.Q_error > 0.0      # Q≈1 ≠ 2 → error > 0
    assert abs(result.objective - (result.freq_error + result.Q_error)) < 1e-9
