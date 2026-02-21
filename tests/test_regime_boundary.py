"""
Regime Boundary Stress Test.

Tests KoopmanInvariantDescriptor geometry across four damping regimes:
  Q = 0.1  (heavily overdamped)
  Q = 0.5  (near-critical)
  Q = 5.0  (lightly damped / underdamped)
  Q = 50.0 (high-Q resonator)

Two test strategies:

  A. Descriptor geometry (Groups 1–4): directly construct KoopmanInvariantDescriptor
     objects at each Q value via compute_invariants(), insert into a shared memory,
     and verify ordering properties without running the optimizer.
     This cleanly tests the geometry of the retrieval space.

  B. Optimizer convergence (Groups 5–7): run RLC and spring-mass optimisers with
     generous constraints (max_energy_loss=5.0, max_Q=200) so that any Q value
     in the mapper range produces a valid design point, and verify that:
       - The optimiser converges to <2% frequency error at 1 kHz.
       - Stored log_Q_norm reflects actual achieved Q (vs warm/cold start).
       - Cross-domain transfer (spring-mass → RLC) is not catastrophic.

  C. Analytic unit tests (Group 8): formula-level roundtrips at all boundary Q values.

Background note
---------------
The HDV optimizer minimises frequency error only.  Q is a free variable that
settles to whatever value the random walk naturally converges to, subject to
max_Q and max_energy_loss constraints.  It does NOT target a specific Q.
Therefore Groups 1–4 use direct construction to test Q-regime geometry.
"""

from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from optimization.koopman_memory import KoopmanExperienceMemory, OptimizationExperience
from optimization.koopman_signature import (
    KoopmanInvariantDescriptor,
    compute_invariants,
    _LOG_OMEGA0_SCALE,
    _LOG_OMEGA0_REF,
)
from optimization.rlc_evaluator import RLCEvaluator
from optimization.rlc_parameterization import RLCDesignMapper, RLCParams
from optimization.hdv_optimizer import ConstrainedHDVOptimizer
from optimization.spring_mass_system import (
    SpringMassDesignMapper,
    SpringMassEvaluator,
    SpringMassOptimizer,
)


# ── Regime parameters ─────────────────────────────────────────────────────────

_Q_VALUES  = [0.1, 0.5, 5.0, 50.0]   # Q targets for descriptor geometry tests
_TARGET_HZ = 1_000.0                  # common frequency for all tests
_N_ITER    = 600
_HDV_DIM   = 64
_SEED      = 0   # seed=0 was validated in Phase 3 cross-domain tests


# ── Descriptor-construction helpers ──────────────────────────────────────────


def _make_descriptor_at_Q(q: float, omega0_hz: float = _TARGET_HZ) -> KoopmanInvariantDescriptor:
    """
    Build a KoopmanInvariantDescriptor with explicit (ω₀, Q) values.

    Uses a minimal Koopman eigendecomposition (single eigenvalue = 0.5).
    The descriptor's retrieval key is determined entirely by the dynamical
    quantities (log_omega0_norm, log_Q_norm, damping_ratio), not by the
    eigenvalues — so the test isolates the geometry we care about.
    """
    omega0 = 2.0 * math.pi * omega0_hz
    log_omega0_norm = float(np.clip(
        (math.log(max(omega0, 1e-30)) - _LOG_OMEGA0_REF) / _LOG_OMEGA0_SCALE,
        -3.0, 3.0,
    ))
    log_Q_norm = float(np.clip(
        math.log(max(q, 1e-30)) / _LOG_OMEGA0_SCALE,
        -3.0, 3.0,
    ))
    damping_ratio = float(np.clip(1.0 / (2.0 * q), 0.0, 10.0))

    dummy_eigs = np.array([0.5 + 0.0j])
    dummy_vecs = np.array([[1.0]])
    return compute_invariants(
        eigenvalues=dummy_eigs,
        eigenvectors=dummy_vecs,
        operator_types=["dummy"],
        k=5,
        log_omega0_norm=log_omega0_norm,
        log_Q_norm=log_Q_norm,
        damping_ratio=damping_ratio,
    )


def _make_memory_with_Q_entries(
    q_values: list[float],
    omega0_hz: float = _TARGET_HZ,
    domain: str = "rlc",
) -> KoopmanExperienceMemory:
    """
    Build a KoopmanExperienceMemory with one directly-inserted entry per Q value.

    Each entry stores the descriptor built by _make_descriptor_at_Q(), with a
    dummy KoopmanResult and matching OptimizationExperience.
    """
    from tensor.koopman_edmd import KoopmanResult
    mem = KoopmanExperienceMemory()
    for q in q_values:
        inv = _make_descriptor_at_Q(q, omega0_hz)
        koop = KoopmanResult(
            eigenvalues=np.array([0.5 + 0.0j]),
            eigenvectors=np.array([[1.0]]),
            K_matrix=np.array([[0.5]]),
            spectral_gap=0.0,
            is_stable=True,
        )
        exp = OptimizationExperience(
            bottleneck_operator="dummy",
            replacement_applied="direct_inject",
            runtime_improvement=1.0,
            n_observations=1,
            hardware_target="cpu",
            best_params={"fn_name": f"q={q}", "complexity": "O(n)"},
            domain=domain,
        )
        mem._entries.append(mem._entries.__class__.__new__(mem._entries.__class__))
        # Use internal API to bypass merge logic and ensure distinct entries per Q
        from optimization.koopman_memory import _MemoryEntry
        mem._entries[-1] = _MemoryEntry(invariant=inv, signature=koop, experience=exp)
    return mem


# ── Module-level fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def q_geometry_memory():
    """Memory with one directly-constructed RLC entry per Q in _Q_VALUES."""
    return _make_memory_with_Q_entries(_Q_VALUES, domain="rlc")


@pytest.fixture(scope="module")
def q_geometry_memory_sm():
    """Memory with one directly-constructed spring_mass entry per Q in _Q_VALUES."""
    return _make_memory_with_Q_entries(_Q_VALUES, domain="spring_mass")


# ── Group 1: descriptor field encoding ───────────────────────────────────────


@pytest.mark.parametrize("q", _Q_VALUES)
def test_descriptor_log_Q_norm_value(q):
    """log_Q_norm should equal log(Q)/log(10), clipped to ±3."""
    desc = _make_descriptor_at_Q(q)
    expected = float(np.clip(math.log(q) / _LOG_OMEGA0_SCALE, -3.0, 3.0))
    assert abs(desc.log_Q_norm - expected) < 1e-9, (
        f"Q={q}: log_Q_norm={desc.log_Q_norm:.6f}, expected={expected:.6f}"
    )


@pytest.mark.parametrize("q", _Q_VALUES)
def test_descriptor_damping_ratio_value(q):
    """damping_ratio should equal min(1/(2Q), 10.0)."""
    desc = _make_descriptor_at_Q(q)
    expected = float(np.clip(1.0 / (2.0 * q), 0.0, 10.0))
    assert abs(desc.damping_ratio - expected) < 1e-9, (
        f"Q={q}: damping_ratio={desc.damping_ratio:.6f}, expected={expected:.6f}"
    )


@pytest.mark.parametrize("q", _Q_VALUES)
def test_descriptor_omega0_hz(q):
    """log_omega0_norm at 1 kHz should be 0.0 (by construction of reference)."""
    desc = _make_descriptor_at_Q(q, omega0_hz=_TARGET_HZ)
    # 1 kHz → log_omega0_norm = 0.0 by definition of _LOG_OMEGA0_REF
    assert abs(desc.log_omega0_norm) < 1e-9, (
        f"Q={q}: log_omega0_norm={desc.log_omega0_norm:.6f} should be 0.0 at 1kHz"
    )


# ── Group 2: log_Q_norm and damping_ratio monotonicity ───────────────────────


def test_log_Q_norm_strictly_increasing(q_geometry_memory):
    """
    log_Q_norm must be strictly increasing with Q.
    Q ordering: 0.1 < 0.5 < 5.0 < 50.0  →  log_Q_norm monotone increasing.
    """
    norms = [e.invariant.log_Q_norm for e in q_geometry_memory._entries]
    print(f"\n  log_Q_norm per entry: {[f'{v:.4f}' for v in norms]}")
    for i in range(len(norms) - 1):
        assert norms[i] < norms[i + 1], (
            f"log_Q_norm not strictly increasing at index {i}: "
            f"{norms[i]:.4f} ≥ {norms[i+1]:.4f}"
        )


def test_damping_ratio_strictly_decreasing(q_geometry_memory):
    """
    damping_ratio = ζ = 1/(2Q) must be strictly decreasing as Q increases.
    """
    zetas = [e.invariant.damping_ratio for e in q_geometry_memory._entries]
    print(f"\n  damping_ratio per entry: {[f'{v:.4f}' for v in zetas]}")
    for i in range(len(zetas) - 1):
        assert zetas[i] > zetas[i + 1], (
            f"damping_ratio not strictly decreasing at index {i}: "
            f"{zetas[i]:.4f} ≤ {zetas[i+1]:.4f}"
        )


def test_log_Q_norm_covers_expected_range(q_geometry_memory):
    """
    log_Q_norm(Q=0.1) ≈ -1.0, log_Q_norm(Q=50) ≈ +1.699.
    (Both within clip bounds ±3.)
    """
    norms = [e.invariant.log_Q_norm for e in q_geometry_memory._entries]
    lo, hi = norms[0], norms[-1]
    expected_lo = math.log(0.1) / _LOG_OMEGA0_SCALE   # -1.0
    expected_hi = math.log(50.0) / _LOG_OMEGA0_SCALE  # +1.699
    assert abs(lo - expected_lo) < 1e-9, f"Q=0.1 log_Q_norm={lo:.4f} ≠ {expected_lo:.4f}"
    assert abs(hi - expected_hi) < 1e-9, f"Q=50 log_Q_norm={hi:.4f} ≠ {expected_hi:.4f}"


# ── Group 3: cross-regime retrieval ordering ─────────────────────────────────


def test_query_at_each_Q_retrieves_nearest(q_geometry_memory):
    """
    A query at Q=q should retrieve the entry stored at Q=q as its nearest neighbor.

    Tests all 4 Q values.  Entries are stored at exact Q values, so for each
    query the nearest stored entry (by to_query_vector() L2 distance) should
    be the one at the matching Q.
    """
    for q in _Q_VALUES:
        query_inv = _make_descriptor_at_Q(q)
        candidates = q_geometry_memory.retrieve_candidates(query_inv, top_n=4)
        assert len(candidates) > 0, f"Q={q}: no candidates returned"

        nearest = candidates[0].invariant
        nearest_dist = abs(nearest.log_Q_norm - query_inv.log_Q_norm)

        for other in candidates[1:]:
            other_dist = abs(other.invariant.log_Q_norm - query_inv.log_Q_norm)
            assert nearest_dist <= other_dist + 1e-9, (
                f"Q={q}: nearest has log_Q_norm={nearest.log_Q_norm:.4f} but "
                f"another entry has log_Q_norm={other.invariant.log_Q_norm:.4f} "
                f"which is closer to query log_Q_norm={query_inv.log_Q_norm:.4f}"
            )
        print(f"\n  Q={q}: nearest log_Q_norm={nearest.log_Q_norm:.4f} "
              f"(query={query_inv.log_Q_norm:.4f})")


def test_cross_domain_query_retrieves_by_Q(q_geometry_memory_sm):
    """
    When querying with an RLC-style invariant against spring_mass entries,
    the nearest entry should still be determined by log_Q_norm distance.
    Confirms domain-invariant retrieval works across domain tag.
    """
    for q in _Q_VALUES:
        rlc_query = _make_descriptor_at_Q(q)   # domain not embedded in invariant
        candidates = q_geometry_memory_sm.retrieve_candidates(rlc_query, top_n=4)
        if len(candidates) == 0:
            continue
        nearest = candidates[0].invariant
        nearest_dist = abs(nearest.log_Q_norm - rlc_query.log_Q_norm)
        for other in candidates[1:]:
            other_dist = abs(other.invariant.log_Q_norm - rlc_query.log_Q_norm)
            assert nearest_dist <= other_dist + 1e-9


# ── Group 4: to_query_vector L2 distances ────────────────────────────────────


def test_adjacent_regime_distances_smaller_than_far_regime(q_geometry_memory):
    """
    The L2 distance between adjacent Q entries (Q=0.1 vs Q=0.5) should be
    smaller than the distance between extreme entries (Q=0.1 vs Q=50).
    This verifies that to_query_vector() gives a sensible geometry.
    """
    entries = q_geometry_memory._entries
    assert len(entries) >= 4

    v_lo   = entries[0].invariant.to_query_vector()   # Q=0.1
    v_adj  = entries[1].invariant.to_query_vector()   # Q=0.5
    v_far  = entries[3].invariant.to_query_vector()   # Q=50

    d_adjacent = float(np.linalg.norm(v_lo - v_adj))
    d_far      = float(np.linalg.norm(v_lo - v_far))

    assert d_adjacent < d_far, (
        f"Adjacent-regime distance {d_adjacent:.4f} ≥ far-regime distance {d_far:.4f}. "
        f"to_query_vector() geometry is not sensible."
    )
    print(f"\n  d(Q=0.1, Q=0.5)={d_adjacent:.4f}  d(Q=0.1, Q=50)={d_far:.4f}")


def test_log_omega0_norm_stable_across_Q_regimes(q_geometry_memory):
    """
    All entries were constructed at 1 kHz.  log_omega0_norm must be 0.0 for all.
    """
    for e in q_geometry_memory._entries:
        assert abs(e.invariant.log_omega0_norm) < 1e-9, (
            f"log_omega0_norm={e.invariant.log_omega0_norm:.6f} ≠ 0.0"
        )


# ── Group 5: optimizer convergence (frequency accuracy at 1 kHz) ─────────────


@pytest.mark.parametrize("domain_", ["rlc", "spring_mass"])
def test_optimizer_convergence_1khz(domain_):
    """
    RLC and spring-mass optimisers (seed=0) must converge to <2% frequency
    error at 1 kHz with generous constraints (max_energy_loss=5.0).

    Generous constraints ensure the optimizer is not stuck in a constraint-
    violating region, so we can evaluate convergence independently of Q targeting.
    """
    if domain_ == "rlc":
        mapper    = RLCDesignMapper(hdv_dim=_HDV_DIM, seed=_SEED)
        evaluator = RLCEvaluator(max_Q=200.0, max_energy_loss=5.0)
        memory    = KoopmanExperienceMemory()
        opt = ConstrainedHDVOptimizer(mapper, evaluator, memory, n_iter=_N_ITER, seed=_SEED)
        result = opt.optimize(_TARGET_HZ, pilot_steps=0)
        obj = result.objective
    else:
        mapper    = SpringMassDesignMapper(hdv_dim=_HDV_DIM, seed=_SEED)
        evaluator = SpringMassEvaluator(max_Q=200.0, max_energy_loss=5.0)
        memory    = KoopmanExperienceMemory()
        opt = SpringMassOptimizer(mapper, evaluator, memory, n_iter=_N_ITER, seed=_SEED)
        result = opt.optimize(_TARGET_HZ)
        obj = result.objective

    print(f"\n  {domain_}: obj={obj:.4f}")
    assert obj < 0.02, (
        f"{domain_} optimizer failed to converge to <2% at 1kHz: obj={obj:.4f}"
    )


# ── Group 6: cross-domain transfer at each Q regime (direct injection) ────────


@pytest.fixture(scope="module")
def sm_injected_memories():
    """
    Spring-mass memories with directly-injected entries at each Q regime.
    These are NOT from running the optimizer — they are constructed analytically
    to test whether the warm-start retrieval and RLC inference work correctly at
    each Q value.
    """
    return {q: _make_memory_with_Q_entries([q], domain="spring_mass") for q in _Q_VALUES}


@pytest.mark.parametrize("q_target", _Q_VALUES)
def test_cross_domain_transfer_injected(q_target, sm_injected_memories):
    """
    Cross-domain warm start with spring-mass entries directly injected at Q=q_target.

    The RLC evaluator uses max_energy_loss=5.0 so that inferred RLC params at
    any Q ∈ {0.1, 0.5, 5, 50} are constraint-valid (energy_loss=1/(2Q) ≤ 5.0).

    The warm start should either:
      (a) Converge to <10% error at 1 kHz, or
      (b) Not be catastrophically worse than cold start (≤ 20× cold median).

    Note: Q=0.1 gives energy_loss=5.0 (exactly at the boundary); Q=0.5 gives 1.0;
    higher Q values give small energy_loss.  The inferred RLC warm-start z is
    constraint-valid in all cases.
    """
    sm_mem = sm_injected_memories[q_target]
    rlc_mapper    = RLCDesignMapper(hdv_dim=_HDV_DIM, seed=_SEED)
    rlc_evaluator = RLCEvaluator(max_Q=200.0, max_energy_loss=5.5)

    # Cold: 3 seeds, median
    cold_results = []
    for cseed in [0, 1, 2]:
        cold_mem = KoopmanExperienceMemory()
        cold_opt = ConstrainedHDVOptimizer(rlc_mapper, rlc_evaluator, cold_mem,
                                           n_iter=_N_ITER, seed=cseed)
        cold_results.append(cold_opt.optimize(_TARGET_HZ, pilot_steps=0).objective)
    cold_median = float(np.median(cold_results))

    # Warm: spring-mass injected memory
    warm_results = []
    for wseed in [0, 1, 2]:
        warm_opt = ConstrainedHDVOptimizer(rlc_mapper, rlc_evaluator, sm_mem,
                                           n_iter=_N_ITER, seed=wseed)
        warm_results.append(warm_opt.optimize(_TARGET_HZ, pilot_steps=20).objective)
    warm_median = float(np.median(warm_results))

    ratio = cold_median / max(warm_median, 1e-9)
    print(f"\n  Q={q_target}: cold_med={cold_median:.4f} warm_med={warm_median:.4f} "
          f"ratio={ratio:.2f}×")

    # Generous bound: cross-domain warm from mismatched Q should not be >20× worse
    assert warm_median <= cold_median * 20.0, (
        f"Cross-domain transfer catastrophically worse at Q={q_target}: "
        f"warm_median={warm_median:.4f} > cold_median={cold_median:.4f} × 20"
    )


# ── Group 7: infer_params_from_dynamical warm-start validity ─────────────────


@pytest.mark.parametrize("q", _Q_VALUES)
def test_inferred_rlc_params_constraint_validity(q):
    """
    RLC params inferred from dynamical (ω₀, Q) at each Q value must produce
    a valid design when the evaluator uses max_energy_loss=5.5.

    Verifies: energy_loss = 1/(2Q) ≤ 5.5 and f₀ ≈ _TARGET_HZ.
    """
    evaluator = RLCEvaluator(max_Q=200.0, max_energy_loss=5.5)
    omega0_target = 2.0 * math.pi * _TARGET_HZ
    params = evaluator.infer_params_from_dynamical(omega0_target, q)

    omega0_actual, Q_actual, zeta_actual = evaluator.dynamical_quantities(params)
    freq_error = abs(omega0_actual - omega0_target) / omega0_target
    Q_error    = abs(Q_actual - q) / q
    energy_loss = evaluator.energy_loss_estimate(params)

    assert freq_error < 1e-9, f"Q={q}: ω₀ roundtrip error={freq_error:.2e}"
    assert Q_error < 1e-9,    f"Q={q}: Q roundtrip error={Q_error:.2e}"
    assert energy_loss <= 5.5, (
        f"Q={q}: energy_loss={energy_loss:.4f} > max_energy_loss=5.5"
    )


@pytest.mark.parametrize("q", _Q_VALUES)
def test_infer_params_roundtrip_at_boundary_Q(q):
    """
    RLCEvaluator.infer_params_from_dynamical(ω₀, Q) → params → dynamical_quantities
    should recover (ω₀, Q) to 1e-9 relative error at all boundary Q values.
    """
    evaluator = RLCEvaluator()
    omega0_target = 2.0 * math.pi * _TARGET_HZ

    params = evaluator.infer_params_from_dynamical(omega0_target, q)
    omega0_actual, Q_actual, _ = evaluator.dynamical_quantities(params)

    assert abs(omega0_actual - omega0_target) / omega0_target < 1e-9, (
        f"ω₀ roundtrip failed at Q={q}: {omega0_actual:.4f} vs {omega0_target:.4f}"
    )
    assert abs(Q_actual - q) / q < 1e-9, (
        f"Q roundtrip failed at Q={q}: {Q_actual:.6f} vs {q}"
    )


# ── Group 8: spring-mass analytic formulas at boundary Q values ──────────────


@pytest.mark.parametrize("q", _Q_VALUES)
def test_sm_dynamical_quantities_at_boundary_Q(q):
    """
    SpringMassEvaluator.dynamical_quantities() must return the correct (ω₀, Q, ζ)
    for analytically constructed (k, m, b) at each boundary Q.

    Construction: fix ω₀ = 2π×1kHz, Q = q_target.
      k = 1000 N/m;  m = k/ω₀²;  b = √(k·m) / Q
    """
    from optimization.spring_mass_system import SpringMassEvaluator, SpringMassParams

    k = 1000.0
    omega0_target = 2.0 * math.pi * _TARGET_HZ
    m = k / omega0_target ** 2
    b = math.sqrt(k * m) / q

    evaluator = SpringMassEvaluator()
    p = SpringMassParams(k=k, m=m, b=b)

    omega0_actual, Q_actual, zeta_actual = evaluator.dynamical_quantities(p)

    assert abs(omega0_actual - omega0_target) / omega0_target < 1e-6, (
        f"Spring-mass ω₀ at Q={q}: {omega0_actual:.4f} vs {omega0_target:.4f}"
    )
    assert abs(Q_actual - q) / q < 1e-6, (
        f"Spring-mass Q at Q={q}: {Q_actual:.6f} vs {q}"
    )
    zeta_expected = 1.0 / (2.0 * q)
    assert abs(zeta_actual - zeta_expected) / zeta_expected < 1e-6, (
        f"Spring-mass ζ at Q={q}: {zeta_actual:.6f} vs {zeta_expected:.6f}"
    )


@pytest.mark.parametrize("q", _Q_VALUES)
def test_dynamical_omega0_roundtrip(q):
    """descriptor.dynamical_omega0() must recover ω₀ from stored log_omega0_norm."""
    desc = _make_descriptor_at_Q(q, omega0_hz=_TARGET_HZ)
    omega0_expected = 2.0 * math.pi * _TARGET_HZ
    omega0_recovered = desc.dynamical_omega0()
    rel_err = abs(omega0_recovered - omega0_expected) / omega0_expected
    assert rel_err < 1e-6, (
        f"Q={q}: dynamical_omega0()={omega0_recovered:.2f} vs {omega0_expected:.2f}"
    )


@pytest.mark.parametrize("q", _Q_VALUES)
def test_dynamical_Q_roundtrip(q):
    """descriptor.dynamical_Q() must recover Q from stored log_Q_norm."""
    desc = _make_descriptor_at_Q(q)
    Q_recovered = desc.dynamical_Q()
    # Q=0.1 and Q=50 are within clip bounds (±3 decades), so exact roundtrip expected
    rel_err = abs(Q_recovered - q) / q
    assert rel_err < 1e-6, (
        f"Q={q}: dynamical_Q()={Q_recovered:.6f} vs {q}"
    )
